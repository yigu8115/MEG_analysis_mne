#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#@authors: Paul Sowman, Judy Zhu

#######################################################################################

# This resource by Christian Brodbeck is very useful:
# https://github.com/christianbrodbeck/Eelbrain/wiki/MNE-Pipeline
# https://github.com/christianbrodbeck/Eelbrain/wiki/Coregistration:-Structural-MRI
# etc.

'''
# Put these in .bashrc, so they will be executed on startup (is there an equivalent for Mac?)
# https://surfer.nmr.mgh.harvard.edu/fswiki/FS7_wsl_ubuntu
# (best not to be part of the script, as diff computers will have diff paths)
export FREESURFER_HOME=/Applications/freesurfer # also need to add FS license
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/home/jzhu/analysis_mne/processing/mri/ 
# output from recon-all will be stored here (each subject will have its own subfolder)
# a number of mne functions will also use this to determine path (could also pass in each time explicitly, 
# to avoid the need to rely on env var)

# Alternatively, these commands can be run as part of python script using the following: 
#import os
#os.system('your command here')

# The other OS commands below can also be run this way.
# or put into spawner script for batch processing.

my_subject=FTD0185_MEG1441 # a new folder with this name will be created inside $SUBJECTS_DIR, to contain the output from recon-all
my_nifti=/home/jzhu/analysis_mne/RawData/$my_subject/anat/FTD0185_T1a.nii # specify the input T1 scan
recon-all -i $my_nifti -s $my_subject -all -parallel #-openmp 6 # default 4 threads for parallel, can specify how many here

mne make_scalp_surfaces --overwrite -s $my_subject -d $SUBJECTS_DIR --force
mne watershed_bem -s $my_subject -d $SUBJECTS_DIR
mne setup_forward_model -s $my_subject -d $SUBJECTS_DIR --homog --ico 4

# convert any of the confiles to FIF format (to embed mrk & hsp), so we can do coreg later
# (to save disk space, can just use the empty room confile!)
mne kit2fiff --input fname.con --output fname.fif --mrk fname.mrk --elp fname.elp --hsp fname.hsp # should also include: --stim --stimthresh
'''

#######################################################################################

#NOTE: all plots will close when the script finishes running.
# In order to keep the figures open, use -i option when running, e.g.
# python3 -i ~/my_GH/MEG_analysis_mne/Source_analysis.py

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.beamformer import make_lcmv, apply_lcmv


# set up file and folder paths
exp_dir = "/home/jzhu/analysis_mne/processing/"
subjects_dir = exp_dir + "mri/"
subject = 'MMN_test' #'fsaverage' #'FTD0185_MEG1441' # specify subject MRI or use template (e.g. fsaverage)
subject_MEG = 'MMN_test' #'220112_p003' #'FTD0185_MEG1441'
meg_task = '_TSPCA' #'_1_oddball' #''

base_fname = exp_dir + "meg/" + subject_MEG + "/" + subject_MEG
raw_fname = base_fname + meg_task + "-raw.fif"
trans_fname = base_fname + meg_task + "-trans.fif"
epochs_fname = base_fname + meg_task + "-epo.fif"
fwd_fname = base_fname + "-fwd.fif"
filters_fname = base_fname + meg_task + "-filters-lcmv.h5"
filters_vec_fname = base_fname + meg_task + "-filters_vec-lcmv.h5"

# adjust mne options to fix rendering issues in Linux (not needed in Windows)
mne.viz.set_3d_options(antialias = 0, depth_peeling = 0) 


# Follow the steps here:
# https://mne.tools/stable/auto_tutorials/forward/30_forward.html

# plot the head surface (BEM) computed from MRI
plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white", # one or more brain surface to plot - should correspond to 
                            # files in the subject’s surf directory
    orientation="coronal",
    slices=[50, 100, 150, 200],
) # these args will be re-used below

mne.viz.plot_bem(**plot_bem_kwargs) # plot bem


# ===== Coregistration ===== #

# Coregister MRI with headshape from MEG digitisation 
# (embedded in .fif file, whereas for KIT data we have a separate .hsp file)

# Use the GUI for coreg, then save the results as -trans.fif
mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)
# Note: if this gives some issues with pyvista and vtk and the versions of python/mne,
# just install the missing packages as prompted (traitlets, pyvista, pyvistaqt, pyqt5).
# Also disable anti-aliasing if head model not rendering (see above); hence we 
# don't use "mne coreg" from command line (cannot set 3d options)

# Here we plot the dense head, which isn't used for BEM computations but
# is useful for checking alignment after coregistration
info = mne.io.read_info(raw_fname) # only supports fif file? For con file, you may need to read in the raw first (mne.io.read_raw_kit) then use raw.info
mne.viz.plot_alignment(
    info,
    trans_fname,
    subject=subject,
    dig=True, # include digitised headshape
    meg=["helmet", "sensors"], # include MEG helmet & sensors
    subjects_dir=subjects_dir,
    surfaces="head-dense", # include head surface from MRI
)

# also print out some info on distances
trans = mne.read_trans(trans_fname)
print(
    "Distance from head origin to MEG origin: %0.1f mm"
    % (1000 * np.linalg.norm(info["dev_head_t"]["trans"][:3, 3]))
)
print(
    "Distance from head origin to MRI origin: %0.1f mm"
    % (1000 * np.linalg.norm(trans["trans"][:3, 3]))
)

dists = mne.dig_mri_distances(info, trans, subject, subjects_dir=subjects_dir)
print(
    "Distance from %s digitized points to head surface: %0.1f mm"
    % (len(dists), 1000 * np.mean(dists))
)


# ===== Compute source space ===== #

# Note: beamformers are usually computed in a volume source space (3rd option below), 
# because estimating only cortical surface activation can misrepresent the data
# https://mne.tools/stable/auto_tutorials/inverse/50_beamformer_lcmv.html#the-forward-model

# create source space from cortical surface (by selecting a subset of vertices)
'''
spacing = "oct4" # use a recursively subdivided octahedron: 4 for speed, 6 for real analyses
src_fname = op.join(subjects_dir, subject, "bem", subject + "_" + spacing + "-src.fif")
if op.exists(src_fname):
    src = mne.read_source_spaces(src_fname)
else:
    src = mne.setup_source_space(
        subject, spacing=spacing, add_dist="patch", subjects_dir=subjects_dir
    )
    # save to mri folder
    mne.write_source_spaces(
        src_fname, src
    )
    
print(src)
mne.viz.plot_bem(src=src, **plot_bem_kwargs) # plot bem with source points
'''

# create volume source space using grid spacing (in a sphere)
'''
sphere = (0, 0.0, 0.0, 0.09)
src = mne.setup_volume_source_space(
    subject,
    subjects_dir=subjects_dir,
    sphere=sphere,
    sphere_units="m",
    add_interpolator=False,  # just for speed!
)

print(src)
mne.viz.plot_bem(src=src, **plot_bem_kwargs) # plot bem with source grid
'''

# create volume source space using grid spacing (bounded by the bem)
src_fname = op.join(subjects_dir, subject, "bem", subject + "_vol-src.fif")
if op.exists(src_fname):
    src = mne.read_source_spaces(src_fname)
else:
    surface = op.join(subjects_dir, subject, "bem", "inner_skull.surf")
    src = mne.setup_volume_source_space(
        subject, subjects_dir=subjects_dir, surface=surface, add_interpolator=True
    )
    # save to mri folder
    mne.write_source_spaces(
        src_fname, src
    )
    
print(src)
mne.viz.plot_bem(src=src, **plot_bem_kwargs) # plot bem with source grid


# check alignment after creating source space
fig = mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces="white",
    coord_frame="mri",
    src=src,
)
mne.viz.set_3d_view(
    fig,
    azimuth=173.78,
    elevation=101.75,
    distance=0.35,
    #focalpoint=(-0.03, -0.01, 0.03),
)


# ===== Compute forward solution / leadfield ===== #

if op.exists(fwd_fname):
    fwd = mne.read_forward_solution(fwd_fname)
else:
    conductivity = (0.3,)  # single layer: inner skull (good enough for MEG but not EEG)
    # conductivity = (0.3, 0.006, 0.3)  # three layers (use this for EEG)
    model = mne.make_bem_model( # BEM model describes the head geometry & conductivities of diff tissues
        subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)

    fwd = mne.make_forward_solution(
        raw_fname,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=1,
        verbose=True,
    )
    # save a copy
    mne.write_forward_solution(fwd_fname, fwd)
    
print(fwd)

# apply source orientation constraint (e.g. fixed orientation)
# note: not applicable to volumetric source space
'''
fwd = mne.convert_forward_solution(
    fwd, surf_ori=True, force_fixed=True, use_cps=True
)
'''

# we can explore the content of fwd to access the numpy array that contains the leadfield matrix
leadfield = fwd["sol"]["data"]
print("Leadfield size (free orientation): %d sensors x %d dipoles" % leadfield.shape)

# Forward computation can remove vertices that are too close to (or outside) the inner skull surface,
# so always use fwd["src"] (rather than just src) when passing to other functions.
# Let's compare before & after to see if any vertices were removed:
print(f'Before: {src}')
print(f'After:  {fwd["src"]}')

# save a bit of memory
src = fwd["src"]
#del fwd


# ===== Reconstruct source activity ===== #
# Note: if running this part in Windows, copy everything over 
# (from both the "mri" and "meg" folders), but can skip 
# the "-raw.fif" (if large) as we can just use the con file below

# Tutorial:
# https://mne.tools/stable/auto_tutorials/inverse/50_beamformer_lcmv.html

# Run sensor-space analysis script to obtain the epochs (or read from saved file)
epochs = mne.read_epochs(epochs_fname)
evoked_allconds = epochs.average()
#evoked_allconds.plot_joint() # average ERF across all conds

# compute evoked for each cond
evokeds = []
for cond in epochs.event_id:
    evokeds.append(epochs[cond].average())

# compute cov matrices
data_cov = mne.compute_covariance(epochs, tmin=-0.01, tmax=0.4,
                                  method='empirical')
noise_cov = mne.compute_covariance(epochs, tmin=-0.1, tmax=0,
                                   method='empirical')
#data_cov.plot(epochs.info)
#del epochs

# compute the spatial filter (LCMV beamformer) - use common filter for all conds?
filters = make_lcmv(evoked_allconds.info, fwd, data_cov, reg=0.05,
                    noise_cov=noise_cov, pick_ori='max-power', # 1 estimate per voxel (only preserve the axis with max power)
                    weight_norm='unit-noise-gain', rank=None)
filters_vec = make_lcmv(evoked_allconds.info, fwd, data_cov, reg=0.05,
                        noise_cov=noise_cov, pick_ori='vector', # 3 estimates per voxel, corresponding to the 3 axes
                        weight_norm='unit-noise-gain', rank=None)
# save the filters for later
filters.save(filters_fname, overwrite=True)
filters_vec.save(filters_vec_fname, overwrite=True)

# apply the spatial filter
stcs = dict()
stcs_vec = dict()
for index, evoked in enumerate(evokeds):
    cond = evoked.comment
    stcs[cond] = apply_lcmv(evoked, filters)
    stcs_vec[cond] = apply_lcmv(evoked, filters_vec)
    
    # plot the reconstructed source activity
    lims = [0.3, 0.45, 0.6] # set colour scale
    kwargs = dict(src=src, subject=subject, subjects_dir=subjects_dir, verbose=True,
                  initial_time=0.087)
    brain = stcs_vec[cond].plot_3d(
        clim=dict(kind='value', lims=lims), 
        hemi='both', size=(600, 600),
        #views=['sagittal'], # only show sag view
        view_layout='horizontal', views=['coronal', 'sagittal', 'axial'], # make a 3-panel figure showing all views
        brain_kwargs=dict(silhouette=True),
        **kwargs)

# Q: 
# 1. How do we choose an ROI, i.e. get source activity for A1 only? 
#    (need to apply the label from freesurfer to work out which vertices belong to A1?)
# https://mne.tools/stable/auto_examples/inverse/label_source_activations.html
# See also: 
# https://mne.tools/stable/auto_examples/visualization/parcellation.html
# https://mne.tools/stable/auto_tutorials/inverse/60_visualize_stc.html#volume-source-estimates

# 2. How to compare stcs between 2 conds? atm I'm just plotting each of them separately ...
#
# Compare evoked response across conds (can do the same to compare stcs?)
# https://mne.tools/stable/auto_examples/visualization/topo_compare_conditions.html
#
# Plotting stcs:
# https://mne.tools/stable/auto_tutorials/inverse/60_visualize_stc.html