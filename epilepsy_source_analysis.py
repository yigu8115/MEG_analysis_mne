import os
import mne

# load raw data
data_dir = '/home/jzhu/epilepsy_MEG/'
raw_fname = data_dir + '0001_JC_ME200_11022022_resting_B1_TSPCA-raw.fif'
raw = mne.io.read_raw_fif(raw_fname, verbose=False)

# filter the data
#raw.filter(l_freq=1, h_freq=None)

# read spikes from csv file
my_annot = mne.read_annotations(data_dir + 'saved-annotations-for_Judy_1Aug22.csv')
# remove 'BAD_' prefix in annotation descriptions
my_annot.rename({'BAD_sw_post' : 'sw_post', 
                        'BAD_postswbroad' : 'postswbroad', 
                        'BAD_sw_lesspost' : 'sw_lesspost', 
                        'BAD_sw_post' : 'sw_post', 
                        'BAD_polysw' : 'polysw', 
                        'BAD_smallpolysw' : 'smallpolysw'}, verbose=None)
print(my_annot)

# convert annotations to events array
raw.set_annotations(my_annot)
events_from_annot, event_dict = mne.events_from_annotations(raw)
print(event_dict)
print(events_from_annot)
#mne.write_events(data_dir + 'events_from_annot_eve.txt', events_from_annot)

# For more info:
# https://mne.tools/dev/auto_tutorials/raw/30_annotate_raw.html
# https://mne.tools/dev/auto_tutorials/intro/20_events_from_raw.html#the-events-and-annotations-data-structures


### Analysis starts from here ###

# epoching based on events
epochs = mne.Epochs(
    raw, events_from_annot, event_id=event_dict, tmin=-0.1, tmax=0.5, preload=True
)

# source localisation

