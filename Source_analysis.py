#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 15:05:12 2022

@author: mq20096022
"""
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
recon-all -i $my_nifti -s $my_subject -all

mne make_scalp_surfaces --overwrite -s $my_subject -d $SUBJECTS_DIR --force
mne watershed_bem -s $my_subject -d $SUBJECTS_DIR
mne setup_forward_model -s $my_subject -d $SUBJECTS_DIR --homog --ico 4
'''

#######################################################################################

#NOTE: all plots will close when the script finishes running.
# In order to keep the figures open, use -i option when running, e.g.
# python3 -i ~/my_GH/MEG_analysis_mne/Source_analysis.py

import mne

import os.path as op
import numpy as np

# set up file and folder paths
exp_dir = "/home/jzhu/analysis_mne/processing/"
subjects_dir = exp_dir + "mri/"
subject = 'fsaverage' #'FTD0185_MEG1441' # specify subject MRI or use template (e.g. fsaverage)
subject_MEG = '220112_p003' #'FTD0185_MEG1441'
meg_task = '_1_oddball' #''
raw_fname = exp_dir + "meg/" + subject_MEG + "/" + subject_MEG + meg_task + "-raw.fif"
trans_fname = exp_dir + "meg/" + subject_MEG + "/" + subject_MEG + meg_task + "-trans.fif"

# adjust mne options to fix rendering issue (not needed in Windows)
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

# Use the GUI for coreg, then save it as -trans.fif
mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)
#mne coreg -s subject -d subjects_dir
# Note: if this gives some issues with pyvista and vtk and the versions of python/mne,
# just install the missing packages as prompted (traitlets, pyvista, pyvistaqt, pyqt5).
# Also disable anti-aliasing if head model not rendering (see above)

# Here we plot the dense head, which isn't used for BEM computations but
# is useful for checking alignment after coregistration
info = mne.io.read_info(raw_fname)
mne.viz.plot_alignment(
    info,
    trans_fname,
    subject=subject,
    dig=True,
    meg=["helmet", "sensors"],
    subjects_dir=subjects_dir,
    surfaces="head-dense",
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

# source space based on surface (by selecting a subset of vertices at specified spacing)
spacing = "oct6" # 'oct#' - use a recursively subdivided octahedron
src = mne.setup_source_space(
    subject, spacing=spacing, add_dist="patch", subjects_dir=subjects_dir
)
# save to mri folder
mne.write_source_spaces(
    op.join(subjects_dir, subject, "bem", subject_MEG + "_" + spacing + ".fif"), src
)
print(src)
mne.viz.plot_bem(src=src, **plot_bem_kwargs) # plot bem with source points

# try some other options
'''
# volume source space with grid spacing (based on a sphere)
sphere = (0, 0.0, 0.0, 0.09)
vol_src = mne.setup_volume_source_space(
    subject,
    subjects_dir=subjects_dir,
    sphere=sphere,
    sphere_units="m",
    add_interpolator=False,
)  # rough for speed!
print(vol_src)

mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs) # plot bem with source grid

# volume source space with grid spacing (based on bem)
surface = op.join(subjects_dir, subject, "bem", "inner_skull.surf")
vol_src = mne.setup_volume_source_space(
    subject, subjects_dir=subjects_dir, surface=surface, add_interpolator=False
)  # rough for speed
print(vol_src)

mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs) # plot bem with source grid
'''

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
print(fwd)

# Forward computation can remove vertices that are too close to (or outside) the inner skull surface,
# so always use fwd["src"] (rather than just src) when passing to other functions.
# Let's compare before & after to see if any vertices were removed:
print(f'Before: {src}')
print(f'After:  {fwd["src"]}')

# we can explore the content of fwd to access the numpy array that contains the leadfield matrix
leadfield = fwd["sol"]["data"]
print("Leadfield size (free orientation): %d sensors x %d dipoles" % leadfield.shape)

# apply source orientation constraint (e.g. fixed orientation)
fwd_fixed = mne.convert_forward_solution(
    fwd, surf_ori=True, force_fixed=True, use_cps=True
)
leadfield = fwd_fixed["sol"]["data"]
print("Leadfield size (fixed orientation): %d sensors x %d dipoles" % leadfield.shape)


# ===== Reconstruct source activity in A1 ===== #

