#!/usr/bin/python3
#
# MEG sensor space analysis for auditory roving MMF
#
# Authors: Paul Sowman, Judy Zhu

#######################################################################################

import os
import mne
import glob
import matplotlib.pyplot as plt
import numpy as np
from mne.preprocessing import find_bad_channels_maxwell
from autoreject import get_rejection_threshold  # noqa
from autoreject import Ransac  # noqa
from autoreject.utils import interpolate_bads  # noqa
from mne.decoding import EMS

# We can use the `decim` parameter to only take every nth time slice.
# This speeds up the computation time. Note however that for low sampling
# rates and high decimation parameters, you might not detect "peaky artifacts"
# (with a fast timecourse) in your data. A low amount of decimation however is
# almost always beneficial at no decrease of accuracy.

#os.chdir("/Users/mq20096022/Downloads/MD_pilot1/")
#os.chdir("/Users/mq20096022/Downloads/220112_p003/")
#os.chdir("/home/jzhu/analysis_mne/data/220112_p003/meg/")

# set up file and folder paths here
exp_dir = "/home/jzhu/analysis_mne/"
subject_MEG = 'MMN_test' #'220112_p003' #'FTD0185_MEG1441'
meg_task = '_TSPCA' #'_1_oddball' #''

# the paths below should be automatic
data_dir = exp_dir + "data/"
processing_dir = exp_dir + "processing/"
meg_dir = data_dir + subject_MEG + "/meg/"
epochs_fname = processing_dir + "meg/" + subject_MEG + "/" + subject_MEG + meg_task + "-epo.fif"

#print(glob.glob("*_oddball.con"))
fname_raw = glob.glob(meg_dir + "*" + meg_task + ".con")
fname_elp = glob.glob(meg_dir + "*.elp")
fname_hsp = glob.glob(meg_dir + "*.hsp")
fname_mrk = glob.glob(meg_dir + "*.mrk")

#%% Raw extraction ch misc 23-29 = triggers
# ch misc 007 = audio
raw = mne.io.read_raw_kit(
    fname_raw[0],  # change depending on file i want
    mrk=fname_mrk[0],
    elp=fname_elp[0],
    hsp=fname_hsp[0],
    stim=[*[166], *range(182, 190)],
    slope="+",
    stim_code="channel",
    stimthresh=2,  # 2 for adult (1 for child??)
    preload=True,
    allow_unknown_format=False,
    verbose=True,
)

#%% Finding events
events = mne.find_events(
    raw,
    output="onset",
    consecutive=False,
    min_duration=0,
    shortest_event=1,  # 5 for adult
    mask=None,
    uint_cast=False,
    mask_type="and",
    initial_event=False,
    verbose=None,
)

# get rid of audio trigger for now
events = np.delete(events, np.where(events[:, 2] == 166), 0)

# re-code standard & deviant trials as '1' and '2'
#events = copy.deepcopy(events)
std_dev_bool = np.insert(np.diff(events[:, 2]) != 0, 0, "True")
for idx, event in enumerate(std_dev_bool):
    if event and idx > 0:
        events[idx, 2] = 2
        if events[idx - 1, 2] != 2:
            events[idx - 1, 2] = 1
event_ids = {
    "standard": 1,
    "deviant": 2,
}

# adjust timing based on audio triggers (see bottom of "..._VEP" script, or Jamie script) 
# OR use a fixed delay (~160ms)
import copy
events_corrected = copy.copy(events)
events_corrected[:,0] = events[:,0] + 160

epochs = mne.Epochs(raw, events_corrected, event_id=event_ids, tmin=-0.1, tmax=0.4, preload=True)
conds_we_care_about = ["standard", "deviant"]
epochs.equalize_event_counts(conds_we_care_about)

# downsample to 100Hz
print("Original sampling rate:", epochs.info["sfreq"], "Hz")
epochs_resampled = epochs.copy().resample(100, npad="auto")
print("New sampling rate:", epochs_resampled.info["sfreq"], "Hz")

# save for use in Source_analysis script
epochs_resampled.save("MMN_test_TSPCA-epo.fif")

mne.viz.plot_evoked(epochs_resampled.average(), gfp="only")

mne.viz.plot_compare_evokeds(
    [
        epochs_resampled["standard"].average(),
        epochs_resampled["deviant"].average(),
    ]
)

epochs_resampled.pick_types(meg=True, exclude="bads")
X = epochs_resampled.get_data()  # MEG signals: n_epochs, n_channels, n_times

# pca = UnsupervisedSpatialFilter(PCA(0.95), average=False)
# X = pca.fit_transform(X)

y = epochs_resampled.events[:, 2]  # target: Standard or Deviant
# # Setup the data to use it a scikit-learn way:
# X = epochs.get_data()  # The MEG data
# y = epochs.events[:, 2]  # The conditions indices
n_epochs, n_channels, n_times = X.shape

#############################################################################

# Initialize EMS transformer
ems = EMS()

# Initialize the variables of interest
X_transform = np.zeros((n_epochs, n_times))  # Data after EMS transformation
filters = list()  # Spatial filters at each time point

# In the original paper, the cross-validation is a leave-one-out. However,
# we recommend using a Stratified KFold, because leave-one-out tends
# to overfit and cannot be used to estimate the variance of the
# prediction within a given fold.

for train, test in StratifiedKFold().split(X, y):
    # In the original paper, the z-scoring is applied outside the CV.
    # However, we recommend to apply this preprocessing inside the CV.
    # Note that such scaling should be done separately for each channels if the
    # data contains multiple channel types.
    X_scaled = X / np.std(X[train])

    # Fit and store the spatial filters
    ems.fit(X_scaled[train], y[train])

    # Store filters for future plotting
    filters.append(ems.filters_)

    # Generate the transformed data
    X_transform[test] = ems.transform(X_scaled[test])

# Average the spatial filters across folds
filters = np.mean(filters, axis=0)

# # Plot individual trials
# plt.figure()
# plt.title('single trial surrogates')
# plt.imshow(X_transform[y.argsort()], origin='lower', aspect='auto',
#            extent=[epochs_resampled.times[0], epochs_resampled.times[-1], 1, len(X_transform)],
#            cmap=None)
# plt.xlabel('Time (ms)')
# plt.ylabel('Trials (reordered by condition)')
# plt.clim(vmin=None, vmax=None)

# Plot average response
plt.figure()
plt.title("Average EMS signal")
mappings = [(key, value) for key, value in event_ids.items()]
for key, value in mappings:
    ems_ave = X_transform[y == value]
    plt.plot(epochs_resampled.times, ems_ave.mean(0), label=key)
plt.xlabel("Time (ms)")
plt.ylabel("a.u.")
plt.legend(loc="best")
plt.show()

# # Visualize spatial filters across time
# evoked = EvokedArray(filters, epochs_resampled.info, tmin=epochs_resampled.tmin)
# evoked.plot_topomap(time_unit="s")

# s = os.path.splitext(folder)[0].split("/")
# np.savetxt(s[-2] + "-filters.csv", filters, delimiter=",")
# np.savetxt(s[-2] + "-trials.csv", X_transform, delimiter=",")
# epochs_resampled.save(s[-2] + "-epo.fif", overwrite=True)
# epochs_resampled.pick_types(meg=True, exclude="bads")

X = epochs_resampled.get_data()  # MEG signals: n_epochs, n_channels, n_times

pca = UnsupervisedSpatialFilter(PCA(0.95), average=False)
X = pca.fit_transform(X)
y = epochs_resampled.events[:, 2]  # target: Standard or Deviant

clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
time_decod = SlidingEstimator(clf, n_jobs=1, scoring="accuracy")

# scores = cross_val_multiscore(time_decod, X, y, cv=int(X.shape[0]/2),
#                               n_jobs=1) #LOO

scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=1)  # k=5

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# # Plot
# fig, ax = plt.subplots()
# ax.plot(epochs_resampled.times, scores, label="score")
# ax.axhline(0.5, color="k", linestyle="--", label="chance")
# ax.set_xlabel("Times")
# ax.set_ylabel("AUC")  # Area Under the Curve
# ax.legend()
# ax.axvline(0.0, color="k", linestyle="-")
# ax.set_title("Sensor space decoding")
# plt.show()

# You can retrieve the spatial filters and spatial patterns if you explicitly
# use a LinearModel - DOesn't work for PCA reduced data though as need all chs
# clf = make_pipeline(StandardScaler(), LinearModel(LinearDiscriminantAnalysis()))
# time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc')

# time_decod.fit(X, y)

# coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
# evoked = mne.EvokedArray(coef, epochs_resampled.info, tmin=epochs.times[0])
# joint_kwargs = dict(ts_args=dict(time_unit='s'),
#                     topomap_args=dict(time_unit='s'))
# evoked.plot_joint(times=np.arange(0., .500, .100), title='patterns',
#                   **joint_kwargs)

s = os.path.splitext(confile)[0].split("/")
np.savetxt(s[-2] + "-filters.csv", filters, delimiter=",")
np.savetxt(s[-2] + "-trials.csv", X_transform, delimiter=",")
np.savetxt(s[-2] + "-decoding.csv", scores, delimiter=",")
epochs_resampled.save(s[-2] + "-epo.fif", overwrite=True)
# except:
#     print(" error with " + confile)
#     errors.append(confile)
#     continue
# # epochs_resampled.p
