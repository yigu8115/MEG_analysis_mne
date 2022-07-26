import os
from datetime import timedelta
import mne

# load raw data
'''
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis_raw.fif')
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.crop(tmax=60).load_data()
'''
raw_fname = 'C:/Users/Judy/Downloads/0001_JC_ME200_11022022_resting_B1_TSPCA-raw.fif'
raw = mne.io.read_raw_fif(raw_fname, verbose=False)

# browse data to add annotations
fig = raw.plot(start=2, duration=6)
fig.fake_keypress('a') # press 'a' to start annotations mode

# always save straight after interactive session (as these annotations 
# can be easily lost in any subsequent calls to set_annotations())
raw.annotations.save('saved-annotations.csv', overwrite=True)
annot_from_file = mne.read_annotations('saved-annotations.csv')
print(annot_from_file)

# convert annotations to events array
events_from_annot, event_dict = mne.events_from_annotations(raw)
print(event_dict)
print(events_from_annot)
