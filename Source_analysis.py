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
#Q: I ran the following in terminal. Should be able to use OS or mne functions though

#A: Can put these in .bashrc, so they will be executed on startup (is there an equivalent file for Mac?)
# https://surfer.nmr.mgh.harvard.edu/fswiki/FS7_wsl_ubuntu
# (best not to be part of the script, as diff computers will have diff paths)
export FREESURFER_HOME=/Applications/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# The other OS commands below can be run from within python script using the following: 
#import os
#os.system('your command here')


export SUBJECTS_DIR=/home/jzhu/analysis_mne/processing/mri/ # a number of mne functions will use this to determine path (could also pass in each time explicitly, to avoid the need to rely on env var?)
# output from recon-all will also be stored here (each subject will have their own subfolder)

my_subject=FTD0185_MEG1441 # a new folder with this name will be created inside $SUBJECTS_DIR, to contain the output from recon-all
my_nifti=/home/jzhu/analysis_mne/RawData/$my_subject/anat/FTD0185_T1a.nii # specify the input T1 scan
recon-all -i $my_nifti -s $my_subject -all

mne make_scalp_surfaces --overwrite -s $my_subject -d $SUBJECTS_DIR --force
mne watershed_bem -s $my_subject -d $SUBJECTS_DIR
mne setup_forward_model -s $my_subject -d $SUBJECTS_DIR --homog --ico 4
'''


#NOTE: all plots will close when the script finishes running.
# In order to keep the figures open, use -i option when running:
# python3 -i ~/my_GH/MEG_analysis_mne/Source_analysis.py

import mne

import os.path as op
import numpy as np

#subject = "xxxxx"
#subjects_dir = "/Users/mq20096022/Documents/xxxxx/Subjects"
#trans = "/Users/mq20096022/Documents/Data/ME202/Subjects/xxxxx/meg/0002_BH_ME152_18022022_rest_B1-trans.fif"
#raw = "/Users/mq20096022/Documents/Data/ME202/Subjects/xxxxx/meg/0002_BH_ME152_18022022_rest_B1-raw.fif"
exp_dir = "/home/jzhu/analysis_mne/processing/"
subjects_dir = exp_dir + "mri/"
subject = "FTD0185_MEG1441"
trans = exp_dir + "meg/" + subject + "/" + subject + "-trans.fif"
raw = exp_dir + "meg/" + subject + "/" + subject + "-raw.fif"


plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white",
    orientation="coronal",
    slices=[50, 100, 150, 200],
)

mne.viz.plot_bem(**plot_bem_kwargs)


# make the -trans.fif by doing coreg
mne.viz.set_3d_options(antialias=0) # disable anti-aliasing to fix rendering issue
mne.gui.coregistration()
#mne coreg -d subjects_dir
#Q: this always gives some issues with pyvista and vtk and the versions of python/mne
#A: just install the missing packages as prompted (traitlets, pyvista, pyvistaqt, pyqt5).
#   Also disable anti-aliasing if head model not rendering (see above)

info = mne.io.read_info(raw)

# Here we look at the dense head, which isn't used for BEM computations but
# is useful for coregistration.
mne.viz.plot_alignment(
    info,
    trans,
    subject=subject,
    dig=True,
    meg=["helmet", "sensors"],
    subjects_dir=subjects_dir,
    surfaces="head-dense",
)
src = mne.setup_source_space(
    subject, spacing="oct4", add_dist="patch", subjects_dir=subjects_dir
)
mne.write_source_spaces(
    op.join(subjects_dir, subject, "bem", "xxxxx_oct4-src.fif"), src
)

mne.viz.plot_bem(src=src, **plot_bem_kwargs)
trans = mne.read_trans(trans)
dists = mne.dig_mri_distances(info, trans, subject, subjects_dir=subjects_dir)
print(src)

print(
    "Distance from head origin to MEG origin: %0.1f mm"
    % (1000 * np.linalg.norm(info["dev_head_t"]["trans"][:3, 3]))
)

print(
    "Distance from head origin to MRI origin: %0.1f mm"
    % (1000 * np.linalg.norm(trans["trans"][:3, 3]))
)

print(
    "Distance from %s digitized points to head surface: %0.1f mm"
    % (len(dists), 1000 * np.mean(dists))
)

sphere = (0, 0.0, 0.0, 0.09)
vol_src = mne.setup_volume_source_space(
    subject,
    subjects_dir=subjects_dir,
    sphere=sphere,
    sphere_units="m",
    add_interpolator=False,
)  # rough for speed!
print(vol_src)

mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)

surface = op.join(subjects_dir, subject, "bem", "inner_skull.surf")
vol_src = mne.setup_volume_source_space(
    subject, subjects_dir=subjects_dir, surface=surface, add_interpolator=False
)  # rough for speed
print(vol_src)

mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)

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
    distance=0.30,
    focalpoint=(-0.03, -0.01, 0.03),
)

conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(
    subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
)

bem = mne.make_bem_solution(model)

fwd = mne.make_forward_solution(
    raw,
    trans=trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=1,
    verbose=True,
)

leadfield = fwd["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

fwd_fixed = mne.convert_forward_solution(
    fwd, surf_ori=True, force_fixed=True, use_cps=True
)
leadfield = fwd_fixed["sol"]["data"]
print(fwd)
print(f"Before: {src}")
print(f'After:  {fwd["src"]}')
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)
