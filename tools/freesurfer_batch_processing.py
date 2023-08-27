#!/usr/bin/python3
#
# batch process recon-all on all the individual T1 scans
# (can run this step on a fast computer)
#
# Author: Judy Zhu

#######################################################################################

import os
import glob

# specify the path to your data folder
data_dir = '/home/jzhu/analysis_mne/data/'

# specify where to store the output from recon-all (for all subjects)
subjects_dir = '/home/jzhu/analysis_mne/processing/mri/'

# specify a list of subjects to process
#all_subjects = ['1441','1442','1560']
# or automatically detect all subject folders
all_subjects = [ f.name for f in os.scandir(data_dir) if f.is_dir() ]

for subject in all_subjects:
    # a new folder with the subject name will be created inside SUBJECTS_DIR, 
    # to contain the output from recon-all for this subject

    # find the T1 scan
    nifti = glob.glob(os.path.join(data_dir, subject, 'anat', '*T1*.nii'))[0]

    # can specify how many threads to use for parallel (default is 4)
    # rule of thumb: use 1 less than the number of cores you have
    os.system('recon-all -i ' + nifti + ' -s ' + subject + ' -sd ' + subjects_dir 
               + ' -all -parallel -openmp 15')
