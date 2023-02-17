# === visualise the result from freesurfer === #

import os
import mne

subject = 'FTD0185_MEG1441'
subjects_dir = '/home/jzhu/analysis_mne/processing/mri/'

mne.viz.set_3d_options(antialias = 0, depth_peeling = 0) # to fix rendering issue

Brain = mne.viz.get_brain_class()
brain = Brain(subject, hemi='lh', surf='pial', # view left hemisphere
              subjects_dir=subjects_dir, size=(800, 600))
brain.add_annotation('aparc.a2009s', borders=False) # add an overlay of parcellation


# === mne events file format === #

import mne
import os

sample_data_folder = mne.datasets.sample.data_path()
sample_data_events_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                       'sample_audvis_raw-eve.fif')
events_from_file = mne.read_events(sample_data_events_file)
