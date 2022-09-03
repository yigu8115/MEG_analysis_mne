#!/usr/bin/python3
import os.path as op
import numpy as np
import copy

import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.beamformer import make_lcmv, apply_lcmv
from mne_features import univariate

# set up paths
exp_dir = '/home/jzhu/epilepsy_MEG/'
subjects_dir = op.join(exp_dir, 'mri')
subject = 'p0001' # subject name in mri folder
subjects_dir_MEG = op.join(exp_dir, 'meg')
subject_MEG = '0001_JC_ME200_11022022' # subject name in meg folder
meg_task = '_resting_B1_TSPCA'

path_MEG = op.join(subjects_dir_MEG, subject_MEG)
results_dir = op.join(path_MEG, 'source_analysis')
raw_fname = op.join(path_MEG, subject_MEG) + meg_task + '-raw.fif'
raw_emptyroom_fname = op.join(path_MEG, subject_MEG) + '_emptyroom.con'
fname_bem = op.join(subjects_dir, subject, 'bem', 'p0001-5120-bem-sol.fif') # obtained with: mne setup_forward_model -s $my_subject -d $SUBJECTS_DIR --homog --ico 4
fname_trans = op.join(path_MEG, subject_MEG) + "-trans.fif" # obtained with: mne coreg (GUI)
fname_fwd = op.join(results_dir, subject_MEG) + "-fwd.fif"
fname_filters = op.join(results_dir, subject_MEG) + meg_task + "-filters-lcmv.h5"
fname_annot = op.join(path_MEG, 'saved-annotations-for_Judy_1Aug22.csv')


# NOTE: filtering requires preload of raw data
# preloading & filtering need 12GB memory allocation for WSL (if "Killed", you need to allocate more)

# load raw data
raw = mne.io.read_raw_fif(raw_fname, verbose=False, preload=True)

# filter the data
#raw.filter(l_freq=1, h_freq=80)
raw.filter(l_freq=3, h_freq=70) # use this for kurtosis beamformer (see Rui's paper)

# read spikes from csv file
my_annot = mne.read_annotations(fname_annot)
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
#mne.write_events(exp_dir + 'events_from_annot_eve.txt', events_from_annot)

# For more info:
# https://mne.tools/dev/auto_tutorials/raw/30_annotate_raw.html
# https://mne.tools/dev/auto_tutorials/intro/20_events_from_raw.html#the-events-and-annotations-data-structures


'''
### Prepare for source analysis ###

# epoching based on events
epochs = mne.Epochs(
    raw, events_from_annot, event_id=event_dict, tmin=-0.1, tmax=0.6, preload=True
)

# average the epochs
evoked_polysw = epochs['polysw'].average()
evoked_sw_post = epochs['sw_post'].average()
evoked_sw_post.crop(tmin=-0.1, tmax=0.37) # crop based on average length of manually marked spikes in this cateogory

# compute noise covariance matrix
cov_polysw = mne.compute_covariance(epochs['polysw'], tmax=0., method=['shrunk', 'empirical'], rank=None)
cov_sw_post = mne.compute_covariance(epochs['sw_post'], tmax=0., method=['shrunk', 'empirical'], rank=None)


### Source localisation ###

# Method 1: fit a dipole
# https://mne.tools/stable/auto_tutorials/inverse/20_dipole_fit.html
dip = mne.fit_dipole(evoked_polysw, cov_polysw, fname_bem, fname_trans)[0]
dip.save(results_dir + 'polysw.dip')
dip_sw_post = mne.fit_dipole(evoked_sw_post, cov_sw_post, fname_bem, fname_trans)[0]
dip_sw_post.save(results_dir + 'sw_post.dip')

# Plot the result in 3D brain using individual T1
#dip = mne.read_dipole(exp_dir + 'polysw.dip')
dip.plot_locations(fname_trans, subject, subjects_dir, mode='orthoview')
#dip_sw_post = mne.read_dipole(exp_dir + 'sw_post.dip')
dip_sw_post.plot_locations(fname_trans, subject, subjects_dir, mode='orthoview')


# Prep for Methods 2 & 3 - create source space & forward model

# create source space
src_fname = op.join(subjects_dir, subject, "bem", subject + "_vol-src.fif")
if op.exists(src_fname):
    src = mne.read_source_spaces(src_fname)
else:
    surface = op.join(subjects_dir, subject, "bem", "inner_skull.surf")
    src = mne.setup_volume_source_space(
        subject, subjects_dir=subjects_dir, surface=surface, add_interpolator=True
    )
    # save to mri folder
    mne.write_source_spaces(
        src_fname, src
    )
print(src)

# plot bem with source points
plot_bem_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    brain_surfaces="white", # one or more brain surface to plot - should correspond to 
                            # files in the subject’s surf directory
    orientation="coronal",
    slices=[50, 100, 150, 200],
)
mne.viz.plot_bem(src=src, **plot_bem_kwargs) 

# check alignment after creating source space
fig = mne.viz.plot_alignment(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces="white",
    coord_frame="mri",
    src=src,
)
mne.viz.set_3d_view(
    fig,
    azimuth=173.78,
    elevation=101.75,
    distance=0.35,
    #focalpoint=(-0.03, -0.01, 0.03),
)

# compute forward model
if op.exists(fname_fwd):
    fwd = mne.read_forward_solution(fname_fwd)
else:
    conductivity = (0.3,)  # single layer: inner skull (good enough for MEG but not EEG)
    # conductivity = (0.3, 0.006, 0.3)  # three layers (use this for EEG)
    model = mne.make_bem_model( # BEM model describes the head geometry & conductivities of diff tissues
        subject=subject, ico=4, conductivity=conductivity, subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)

    trans = mne.read_trans(fname_trans)
    fwd = mne.make_forward_solution(
        raw_fname,
        trans=trans,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=1,
        verbose=True,
    )
    # save a copy
    mne.write_forward_solution(fname_fwd, fwd)
print(fwd)


# Method 2: minimum norm
# https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html#inverse-modeling-mne-dspm-on-evoked-and-raw-data

# compute inverse solution
inverse_operator = make_inverse_operator(
    evoked_sw_post.info, fwd, cov_sw_post, loose=0.2, depth=0.8)

method = "MNE" # “dSPM” | “sLORETA” | “eLORETA”
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked_sw_post, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)    

# plot the results:
# https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html#visualization

'''

# Method 3: kurtosis beamformer

# take 1-min segment around each manually marked spike, ensure there's no overlap

# select a particular condition
cond = 'sw_post'
cond_id = event_dict.get(cond)
rows = np.where(events_from_annot[:,2] == cond_id)
events_sw_post = events_from_annot[rows]
# calculate timing diff of two consecutive events, find where the diff is shorter than 1 min (i.e. too close)
t = events_sw_post[:,0]
diff = np.diff(t)
too_close = np.where(diff < 60000) # note: indices start from 0
too_close = np.asarray(too_close).flatten() # convert tuple to array
# when 2 events are too close, combine them into one event
for i in reversed(too_close): # do in reversed order as indices will change after each iteration
    #t[i] = (t[i] + t[i+1]) / 2
    #t = np.delete(t, i+1)
    events_sw_post[i,0] = (t[i] + t[i+1]) / 2 # average the timing of the two events
    events_sw_post = np.delete(events_sw_post, i+1, axis=0) # delete the second event

# create 1-min epochs around these events
epochs = mne.Epochs(
    raw, events_sw_post, event_id={cond: cond_id}, tmin=-30, tmax=30, preload=True
)
# downsample to 100Hz, otherwise stc will be too large 
# (browsed data to check - spikes are still pretty obvious)
epochs.resample(100) 
# this step is done here as it's better not to epoch after downsampling:
# https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample

# average the epochs
evoked = epochs[cond].average()
#evoked.save(op.join(results_dir, subject_MEG + '_' + cond + '-ave.fif'))

# compute data cov
data_cov = mne.compute_covariance(epochs) # use the whole epochs
#data_cov.plot(epochs.info)

# compute noise cov from empty room data
# https://mne.tools/dev/auto_tutorials/forward/90_compute_covariance.html
raw_empty_room = mne.io.read_raw_kit(raw_emptyroom_fname)
raw_empty_room.resample(100) # just to be consistent (not sure if required)
noise_cov = mne.compute_raw_covariance(
    raw_empty_room, tmin=0, tmax=None)


# LCMV beamformer
# https://mne.tools/stable/auto_tutorials/inverse/50_beamformer_lcmv.html

fwd = mne.read_forward_solution(fname_fwd)
src = fwd["src"]

# compute the spatial filter (LCMV beamformer) - use common filter for all conds?
if op.exists(fname_filters):
    filters = mne.beamformer.read_beamformer(fname_filters)
else:
    filters = make_lcmv(evoked.info, fwd, data_cov, reg=0.05,
        noise_cov=noise_cov, pick_ori='max-power', # 1 estimate per voxel (only preserve the axis with max power)
        weight_norm='unit-noise-gain', rank=None)
    # save the filters for later
    filters.save(fname_filters)

# save some memory
del raw, raw_empty_room, epochs, fwd

# apply the spatial filter
stcs = dict()
#evoked = mne.read_evokeds(op.join(results_dir, subject_MEG + '_' + cond + '-ave.fif'))
stcs[cond] = apply_lcmv(evoked, filters)

# plot the reconstructed source activity
# (Memory intensive - cannot run if we didn't downsample from 1000Hz)
'''
lims = [0.3, 0.45, 0.6] # set colour scale
kwargs = dict(src=src, subject=subject, subjects_dir=subjects_dir, verbose=True)
brain = stcs[cond].plot_3d(   
    #clim=dict(kind='value', lims=lims), 
    hemi='both', size=(600, 600),
    #views=['sagittal'], # only show sag view
    view_layout='horizontal', views=['coronal', 'sagittal', 'axial'], # make a 3-panel figure showing all views
    brain_kwargs=dict(silhouette=True),
    **kwargs)
'''


# compute kurtosis for each vertex
kurtosis = univariate.compute_kurtosis(stcs[cond].data)

#selected_funcs = ['kurtosis']
#('fe', FeatureExtractor(sfreq=sfreq, selected_funcs=selected_funcs))

# find the vertex (or cluster of vertices) with maximum kurtosis
print(np.max(kurtosis))
VS_list = np.where(kurtosis > 3.7) 
# note: Rui's paper uses all the local maxima on the kurtosis map, we are just using an absolute cutoff here

# TODO: how do I know where these vertices are? check src - how to read out the coordinates for each vertex?
# we can plot the kurtosis value for each vertex on the source model - but this looks weird!
tmp = copy.copy(stcs[cond]) # make a fake stc by copying the structure
tmp.data = kurtosis.reshape(-1,1) # convert 1d array to 2d
kwargs = dict(src=src, subject=subject, subjects_dir=subjects_dir, verbose=True)
tmp.plot_3d(   
    #hemi='both', size=(600, 600),
    #views=['sagittal'], # only show sag view
    view_layout='horizontal', views=['coronal', 'sagittal', 'axial'], # make a 3-panel figure showing all views
    brain_kwargs=dict(silhouette=True),
    **kwargs)
