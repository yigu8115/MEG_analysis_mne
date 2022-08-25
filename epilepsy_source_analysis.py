import os
import mne

# set up paths
data_dir = 'C:/Users/mq43606024/Downloads/epilepsy_MEG/'
raw_fname = data_dir + '0001_JC_ME200_11022022_resting_B1_TSPCA-raw.fif'
mri_dir = data_dir + 'p0001/'
fname_bem = mri_dir + 'bem/p0001-5120-bem-sol.fif' # obtained with: mne setup_forward_model -s $my_subject -d $SUBJECTS_DIR --homog --ico 4
fname_trans = data_dir + '0001_JC_ME200_11022022-trans.fif' # obtained with: mne coreg (GUI)

# load raw data
raw = mne.io.read_raw_fif(raw_fname, verbose=False)

# filter the data
raw.filter(l_freq=1, h_freq=80)

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
    raw, events_from_annot, event_id=event_dict, tmin=-0.1, tmax=0.6, preload=True
)

# average the epochs
evoked_polysw = epochs['polysw'].average()
evoked_sw_post = epochs['sw_post'].average()
evoked_sw_post.crop(tmin=-0.1, tmax=0.37) # crop based on average length of manually marked spikes in this cateogory

# compute noise covariance matrix
cov_polysw = mne.compute_covariance(epochs['polysw'])
cov_sw_post = mne.compute_covariance(epochs['sw_post'])

# source localisation

# method 1: fit a dipole
# https://mne.tools/stable/auto_tutorials/inverse/20_dipole_fit.html
dip = mne.fit_dipole(evoked_polysw, cov_polysw, fname_bem, fname_trans)[0]
dip.save(data_dir + 'polysw.dip')
dip_sw_post = mne.fit_dipole(evoked_sw_post, cov_sw_post, fname_bem, fname_trans)[0]
dip_sw_post.save(data_dir + 'sw_post.dip')

# Plot the result in 3D brain using individual T1
#dip = mne.read_dipole(data_dir + 'polysw.dip')
dip.plot_locations(fname_trans, 'p0001', data_dir, mode='orthoview')
#dip_sw_post = mne.read_dipole(data_dir + 'sw_post.dip')
dip_sw_post.plot_locations(fname_trans, 'p0001', data_dir, mode='orthoview')

fig = mne.viz.plot_alignment(
    subject='p0001',
    subjects_dir=data_dir,
    surfaces="white",
    coord_frame="mri"
)