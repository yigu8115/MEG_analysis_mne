import os
#from datetime import timedelta
import mne

# load raw data
data_dir = 'C:/Users/mq43606024/Downloads/epilepsy_MEG/'
raw_fname = data_dir + '0001_JC_ME200_11022022_resting_B1_TSPCA-raw.fif'
raw = mne.io.read_raw_fif(raw_fname, verbose=False)

# browse data to mark the events manually
fig = raw.plot(start=0) #, duration=6)
#fig.fake_keypress('a') # press 'a' to toggle annotations mode

# always save straight after interactive session (as these annotations 
# can be easily lost in any subsequent calls to set_annotations())
raw.annotations.save(data_dir + 'saved-annotations.csv', overwrite=True)

# sanity check - print out the time span of each event
for ann in raw.annotations:
    descr = ann['description']
    start = ann['onset']
    end = ann['onset'] + ann['duration']
    print("'{}' goes from {} to {}".format(descr, start, end))



# send the csv file to us, we can then load it
my_annot = mne.read_annotations(data_dir + 'saved-annotations.csv')
print(my_annot)

# convert annotations to events array
raw.set_annotations(my_annot)
events_from_annot, event_dict = mne.events_from_annotations(raw)
print(event_dict)
print(events_from_annot)


# For more info:
# https://mne.tools/dev/auto_tutorials/raw/30_annotate_raw.html
# https://mne.tools/dev/auto_tutorials/intro/20_events_from_raw.html#the-events-and-annotations-data-structures
