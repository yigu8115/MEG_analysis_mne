import os.path as op
import mne


# set up file and folder paths here
#exp_dir = '/Volumes/Seagate' #'homejzhuanalysis_mne 
exp_dir = '/Users/45644438/Macquarie University/Macquarie_NIF_node - Data/FTD' #'homejzhuanalysis_mne 
subject = '1442_FTD0794' #'FTD0185_MEG1441' # specify subject MRI or use template (e.g. fsaverage)
subject_MEG = '1442' #'220112_p003' #'FTD0185_MEG1441'
#meg_task = '_B1' #'_1_oddball' #''
#run_name = '_TSPCA'


# All paths below should be automatic

data_dir = op.join(exp_dir, 'data')
meg_dir = op.join(data_dir, subject, 'meg')
processing_dir = op.join(exp_dir, 'processing')
subjects_dir = op.join(processing_dir, 'mri')
inner_skull = op.join(subjects_dir, subject, "bem", "inner_skull.surf")

subject_dir_meg = op.join(processing_dir, 'meg')



# adjust mne options to fix rendering issues (only needed in Linux / WSL)
#mne.viz.set_3d_options(antialias = False, depth_peeling = False) 



# ===== Coregistration ===== #

# Note: these commands require both MNE & Freesurfer
#if not op.exists(inner_skull): # check one of the target files to see if these steps have been run already
#    os.system('mne make_scalp_surfaces --overwrite -s ' + subject + ' -d ' + subjects_dir + ' --force')
#    os.system('mne watershed_bem -s ' + subject + ' -d ' + subjects_dir)
#    os.system('mne setup_forward_model -s ' + subject + ' -d ' + subjects_dir + ' --homog --ico 4')


# Coregister MRI scan with headshape from MEG digitisation 
mne.gui.coregistration(subject=subject, subjects_dir=subjects_dir)
# Note if this gives some issues with pyvista and vtk and the versions of pythonmne,
# just install the missing packages as prompted (traitlets, pyvista, pyvistaqt, pyqt5).
# Also disable anti-aliasing if head model not rendering (see above); hence we 
# don't use mne coreg from command line (cannot set 3d options)
