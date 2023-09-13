#!/usr/bin/python3
#
# batch process recon-all on all the individual T1 scans
# (can run this step on a fast computer)
#
# Can comment out the BEM surface creation & FIF conversion
# if you don't need these
#
# Author: Judy Zhu

#######################################################################################

import os
import os.path as op
import glob
import mne

# set up paths for FreeSurfer
#os.system('export FREESURFER_HOME=/Users/45644438/Downloads/freesurfer/')
#os.system('source $FREESURFER_HOME/SetUpFreeSurfer.sh')
#os.system('export FS_LICENSE=/Users/45644438/Downloads/license.txt')


# specify the path to your data folder
#data_dir = '/home/jzhu/analysis_mne/data/'
data_dir = '/Volumes/Seagate/data'

# specify where to store the output from recon-all (for all subjects)
#processing_dir = '/home/jzhu/analysis_mne/processing/'
processing_dir = '/Volumes/Seagate/processing'

subjects_dir = os.path.join(processing_dir, 'mri')
subject_dir_meg = os.path.join(processing_dir, 'meg')

# specify a list of subjects to process
#all_subjects = ['1441_FTD0185','1442_FTD0794','1560_FTD0887']
# or automatically detect all subject folders
all_subjects = [ f.name for f in os.scandir(data_dir) if f.is_dir() ]

for subject in all_subjects:
    print(subject)
    
    # a new folder with the subject name will be created inside SUBJECTS_DIR, 
    # to contain the output from recon-all for this subject

    # find the T1 scan
    nifti = glob.glob(os.path.join(data_dir, subject, 'anat', '*T1*.nii'))[0]

    # can specify how many threads to use for parallel (default is 4)
    # rule of thumb: use 1 less than the number of cores you have
    os.system('recon-all -i ' + nifti + ' -s ' + subject + ' -sd ' + subjects_dir 
               + ' -all -parallel -openmp 15')
    
    # create BEM surfaces
    # Note: these commands require both MNE & Freesurfer
    inner_skull = os.path.join(subjects_dir, subject, "bem", "inner_skull.surf")
    if not os.path.exists(inner_skull): # check one of the target files to see if these steps have been run already
        os.system('mne make_scalp_surfaces --overwrite -s ' + subject + ' -d ' + subjects_dir + ' --force')
        os.system('mne watershed_bem -s ' + subject + ' -d ' + subjects_dir)
        os.system('mne setup_forward_model -s ' + subject + ' -d ' + subjects_dir + ' --homog --ico 4')
        
    # kit2fiff conversion (for coreg)
    meg_dir = op.join(data_dir, subject, 'meg')

    raw_fname = op.join(subject_dir_meg, subject, subject + '_B1-raw.fif') # this is the output filename we want from kit2fiff
    os.system('mkdir -p ' + op.join(subject_dir_meg, subject)) # create the folder if it doesn't already exist
    
    # For FIF files, hsp info are embedded in it, whereas for KIT data we have a separate .hsp file.
    # So, convert the confile to FIF format first (to embed mrk & hsp), which can then be loaded during coreg.
    # (Note to save disk space, we just use the empty room confile here!)
    if not op.exists(raw_fname):
        file_raw = glob.glob(op.join(meg_dir, "*B1*.con"))
        file_elp = glob.glob(op.join(meg_dir, "*.elp"))
        file_hsp = glob.glob(op.join(meg_dir, "*.hsp"))
        file_mrk = glob.glob(op.join(meg_dir, "*.mrk"))
        os.system('mne kit2fiff --input ' + file_raw[0] + ' --output ' + raw_fname + 
        ' --mrk ' + file_mrk[0] + ' --elp ' + file_elp[0] + ' --hsp ' + file_hsp[0])
    
