#!/usr/bin/python3

import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import glob

import mne
from mne.beamformer import make_lcmv, apply_lcmv


# set up file and folder paths here
exp_dir = "/home/jzhu/analysis_mne/"
subject = 'fsaverage' #'FTD0185_MEG1441' # specify subject MRI or use template (e.g. fsaverage)
subject_MEG = 'antonio_002' #'220112_p003' #'FTD0185_MEG1441'
meg_task = '_TSPCA' #'_1_oddball' #''

# specify a name for this run (to save intermediate processing files)
run_name = "beamformer" #"beamformer_for_RNN_comparison"

# set to False if you just want to run the whole script & save results
SHOW_PLOTS = False 


# the paths below should be automatic
data_dir = exp_dir + "data/"
meg_dir = data_dir + subject_MEG + "/meg/"
processing_dir = exp_dir + "processing/"
subjects_dir = processing_dir + "mri/"

base_fname = processing_dir + "meg/" + subject_MEG + "/" + subject_MEG + meg_task
raw_fname = base_fname + "_emptyroom-raw.fif" #"-raw.fif" 
# just use empty room recording for kit2fiff (embedding hsp for coreg) & for mne.io.read_info()
trans_fname = base_fname + "-trans.fif"
epochs_fname = base_fname + "-epo.fif"

save_dir = processing_dir + "meg/" + subject_MEG + "/" + run_name + "/"
fwd_fname = save_dir + subject_MEG + "-fwd.fif"
filters_fname = save_dir + subject_MEG + meg_task + "-filters-lcmv.h5"
filters_vec_fname = save_dir + subject_MEG + meg_task + "-filters_vec-lcmv.h5"


# adjust mne options to fix rendering issues (only needed in Linux / WSL)
mne.viz.set_3d_options(antialias = 0, depth_peeling = 0) 


# Follow the steps here:
# https://mne.tools/stable/auto_tutorials/forward/30_forward.html


# ===== Compute head surfaces ===== #

# Note: these commands require both MNE & Freesurfer
if not op.exists(subjects_dir + subject + '/bem/inner_skull.surf'): # check one of the target files to see if these steps have been run already
    os.system('mne make_scalp_surfaces --overwrite -s ' + subject + ' -d ' + subjects_dir + ' --force')
    os.system('mne watershed_bem -s ' + subject + ' -d ' + subjects_dir)
    os.system('mne setup_forward_model -s ' + subject + ' -d ' + subjects_dir + ' --homog --ico 4')

# specify some settings for plots (will be re-used below)
plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white", # one or more brain surface to plot - should correspond to 
                            # files in the subject’s surf directory
    orientation="coronal",
    slices=[50, 100, 150, 200],
)

# plot the head surface (BEM) computed from MRI
if SHOW_PLOTS:
    mne.viz.plot_bem(**plot_bem_kwargs) # plot bem


# ===== Coregistration ===== #

# Coregister MRI scan with headshape from MEG digitisation 

# For FIF files, hsp info are embedded in it, whereas for KIT data we have a separate .hsp file.
# So, convert the confile to FIF format first (to embed mrk & hsp), which can then be loaded during coreg.
# (Note: to save disk space, we just use the empty room confile here!)
os.system('rm ' + raw_fname)
file_raw = glob.glob(data_dir + "MMN_test/meg/*empty*.con")
file_elp = glob.glob(meg_dir + "*.elp")
file_hsp = glob.glob(meg_dir + "*.hsp")
file_mrk = glob.glob(meg_dir + "*.mrk")
os.system('mne kit2fiff --input ' + file_raw[0] + ' --output ' + raw_fname + 
' --mrk ' + file_mrk[0] + ' --elp ' + file_elp[1] + ' --hsp ' + file_hsp[1])

# Use the GUI for coreg, then save the results as -trans.fif
mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)
# Note: if this gives some issues with pyvista and vtk and the versions of python/mne,
# just install the missing packages as prompted (traitlets, pyvista, pyvistaqt, pyqt5).
# Also disable anti-aliasing if head model not rendering (see above); hence we 
# don't use "mne coreg" from command line (cannot set 3d options)

trans = mne.read_trans(trans_fname)
print("done!")