# MEG data analysis using mne-python
Analysis pipelines developed for data acquired at the KIT-Macquarie Brain Research (MEG) Laboratory.<br><br>


## Source analysis ##

### Using individual MRI scans for source analysis ###

*Note: the MRI processing step (recon-all in FreeSurfer) is time consuming. If you are analysing multiple participants, it may be useful to batch process the MRI scans on a fast computer.* <br><br>


1\. Install FreeSurfer

Instructions:
https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads

FreeSurfer is available for Linux and MacOS only. If you are using a Windows computer, you can install FreeSurfer on Windows Subsystem for Linux (WSL) or a Virtual Machine. Follow instructions in the link above.<br><br>


2\. Set up FreeSurfer for use

Run the following commands:
```
export FREESURFER_HOME=/usr/local/freesurfer/7-dev # path to your FreeSurfer installation
export FS_LICENSE=$HOME/Downloads/freesurfer/license.txt # path to FreeSurfer license file
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=$HOME/analysis_mne/processing/mri/ # path where all the processed MRI scans are to be stored
```
You can add these lines into your .bashrc file (usually located at ~/.bashrc), and these commands will be automatically executed each time you log in to the machine.<br><br>


3\. Run recon-all on the individual MRI scan
```
my_subject=FTD0185 # specify the subject name (a new folder with this name will be created inside $SUBJECTS_DIR, to store the output from recon-all)
my_nifti=$HOME/analysis_mne/data/$my_subject/anat/FTD0185_T1a.nii # specify the input T1 scan
recon-all -i $my_nifti -s $my_subject -all -parallel -openmp 6 # default 4 threads for parallel, can specify how many
```
<br>


4\. Organise files for the analysis

A recommended folder structure that works well with the scripts is shown below: <br>
![Alt](/folder_structure.png "Folder structure")

Raw data are stored under the "data" folder (each participant will have an "meg" folder, and optionally an "anat" folder if an individual MRI is available) - the idea is to follow the BIDS format. Intermediate files generated during the analysis will go under the "processing" folder. If running source analysis, a saved epoch file (-epo.fif) from the sensor-space analysis should be placed in processing/meg/[subject_name]/ first.<br><br>


5\. Run Source_analysis.py

Adjust settings at the top if needed:  
*exp_dir:* path to the analysis folder (e.g. "analysis_mne" in the screenshot above)  
*subject & subject_MEG:* subject name within the "mri" and "meg" processing folders

Note: some MNE commands in this script (e.g. creating head surfaces) requires FreeSurfer, so both packages need to be installed even if you have already done the recon-all step elsewhere.<br><br>



### Using template MRI for source analysis ###

Follow the same procedure as using individual MRI (from step 4).

Put the processed template MRI "fsaverage" into the MRI folder (see recommended folder structure above), and specify "fsaverage" as the MRI subject name in the script.
