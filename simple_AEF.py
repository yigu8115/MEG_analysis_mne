#!/usr/bin/python3
#
# Simple AEF & AEP analysis (for sanity check)
#
# Authors: Paul Sowman, Judy Zhu

#######################################################################################

import os
import mne
import glob
import matplotlib.pyplot as plt
import numpy as np
import copy

from mne.preprocessing import find_bad_channels_maxwell
from autoreject import get_rejection_threshold  # noqa
from autoreject import Ransac  # noqa
from autoreject.utils import interpolate_bads  # noqa

import my_preprocessing

# set up file and folder paths here
exp_dir = "/mnt/d/Work/analysis_ME206/"; #"/home/jzhu/analysis_mne/"
subject_MEG = 'G03'; #'gopro_test'; #'MMN_test' #'220112_p003' #'FTD0185_MEG1441'
meg_task = '_localiser'; #'_TSPCA' #'_1_oddball' #''

# the paths below should be automatic
data_dir = exp_dir + "data/"
processing_dir = exp_dir + "processing/"
meg_dir = data_dir + subject_MEG + "/meg/"
eeg_dir = data_dir + subject_MEG + "/eeg/"
save_dir_meg = processing_dir + "meg/" + subject_MEG + "/"
save_dir_eeg = processing_dir + "eeg/" + subject_MEG + "/"
epochs_fname_meg = save_dir_meg + subject_MEG + meg_task + "-epo.fif"
epochs_fname_eeg = save_dir_eeg + subject_MEG + meg_task + "-epo.fif"
os.system('mkdir -p ' + save_dir_meg) # create the folder if needed
os.system('mkdir -p ' + save_dir_eeg)


#%% === Read raw MEG data === #

#print(glob.glob("*_oddball.con"))
fname_raw = glob.glob(meg_dir + "*" + meg_task + ".con")
fname_elp = glob.glob(meg_dir + "*.elp")
fname_hsp = glob.glob(meg_dir + "*.hsp")
fname_mrk = glob.glob(meg_dir + "*.mrk")

# Raw extraction ch misc 23-29 = triggers
# ch misc 007 = audio
raw = mne.io.read_raw_kit(
    fname_raw[0],  # change depending on file i want
    mrk=fname_mrk[0],
    elp=fname_elp[0],
    hsp=fname_hsp[0],
    stim=[*[166], *range(176, 190)],
    slope="+",
    stim_code="channel",
    stimthresh=2,  # 2 for adult (1 for child??)
    preload=True,
    allow_unknown_format=False,
    verbose=True,
)

# Browse the raw data
#raw.plot()

# Filtering & ICA
raw = my_preprocessing.reject_artefact(raw, 1, 40, 0)


#%% === Trigger detection & timing correction === #

# Find events
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

# specify the event IDs
event_ids = {
    "ba": 181,
    "da": 182,
}


# Adjust trigger timing based on audio channel signal 

# get rid of audio triggers for now
events = np.delete(events, np.where(events[:, 2] == 166), 0)

# get raw audio signal from ch166
aud_ch_data_raw = raw.get_data(picks="MISC 007")

def getEnvelope(inputSignal):
    # Taking the absolute value
    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append(abs(sample))
    absoluteSignal = absoluteSignal[0]

    # Peak detection
    intervalLength = 15  # Experiment with this number!
    outputSignal = []

    # Like a sample and hold filter
    for baseIndex in range(intervalLength, len(absoluteSignal)):
        maximum = 0
        for lookbackIndex in range(intervalLength):
            maximum = max(absoluteSignal[baseIndex - lookbackIndex], maximum)
        outputSignal.append(maximum)

    outputSignal = np.concatenate(
        (
            np.zeros(intervalLength),
            np.array(outputSignal)[:-intervalLength],
            np.zeros(intervalLength),
        )
    )
    # finally binarise the output at a threshold of 2.5  <-  adjust this 
    # threshold based on diagnostic plot below!
    return np.array([1 if np.abs(x) > 0.2 else 0 for x in outputSignal])

#raw.load_data().apply_function(getEnvelope, picks="MISC 006")
envelope = getEnvelope(aud_ch_data_raw)
envelope = envelope.tolist() # convert ndarray to list
# detect the beginning of each envelope (set the rest of the envelope to 0)
new_stim_ch = np.clip(np.diff(envelope),0,1)
# find all the 1s (i.e. audio triggers)
stim_tps = np.where(new_stim_ch==1)[0]

# compare number of events from trigger channels & from AD
print("Number of events from trigger channels:", events.shape[0])
print("Number of events from audio channel (166) signal:", stim_tps.shape[0])

# plot any problematic time period to aid diagnosis
'''
test_time = 20000
span = 5000
plt.figure()
plt.plot(aud_ch_data_raw[0], 'b')
#plt.plot(outputSignal, 'r')
for i in range(events.shape[0]):
   plt.axvline(events[i,0], color='b', lw=2, ls='--')
   plt.axvline(stim_tps[i], color='r', lw=2, ls='--')
plt.xlim(test_time-span, test_time+span)
plt.show()
'''

# apply timing correction onto the events array
events_corrected = copy.copy(events) # work on a copy so we don't affect the original

# Missing AD triggers can be handled:
# if there's an AD trigger within 50ms following the normal trigger
# (this ensures we've got the correct trial), update to AD timing;
# if there's no AD trigger in this time range, discard the trial
AD_delta = []
missing = [] # keep track of the trials to discard (due to missing AD trigger)
for i in range(events.shape[0]):
    idx = np.where((stim_tps >= events[i,0]) & (stim_tps < events[i,0]+50))
    if len(idx[0]): # if an AD trigger exists within 200ms of trigger channel
        idx = idx[0][0] # use the first AD trigger (if there are multiple)
        AD_delta.append(stim_tps[idx] - events[i,0]) # keep track of audio delay values (for histogram)
        events_corrected[i,0] = stim_tps[idx] # update event timing
    else:
        missing.append(i)
# discard events which could not be corrected
events_corrected = np.delete(events_corrected, missing, 0)
print("Could not correct", len(missing), "events - these were discarded!")

# histogram showing the distribution of audio delays
n, bins, patches = plt.hist(
    x=AD_delta, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
)
plt.grid(axis="y", alpha=0.75)
plt.xlabel("Delay (ms)")
plt.ylabel("Frequency")
plt.title("Audio Detector Delays")
plt.text(
    70,
    50,
    r"$mean="
    + str(round(np.mean(AD_delta)))
    + ", std="
    + str(round(np.std(AD_delta)))
    + "$",
)
maxfreq = n.max()
# set a clean upper y-axis limit
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


#%% === Epoching === #
if not os.path.exists(epochs_fname_meg):
    epochs = mne.Epochs(raw, events_corrected, event_id=event_ids, tmin=-0.1, tmax=0.41, preload=True)

    conds_we_care_about = ["ba", "da"]
    epochs.equalize_event_counts(conds_we_care_about)

    # downsample to 100Hz
    print("Original sampling rate:", epochs.info["sfreq"], "Hz")
    epochs_resampled = epochs.copy().resample(100, npad="auto")
    print("New sampling rate:", epochs_resampled.info["sfreq"], "Hz")

    # save for later use (e.g. in Source_analysis script)
    epochs_resampled.save(epochs_fname_meg)

    # plot ERFs
    mne.viz.plot_evoked(epochs_resampled.average(), gfp="only")
    fig = mne.viz.plot_compare_evokeds(
        [
            epochs_resampled["ba"].average(),
            epochs_resampled["da"].average(),
        ]
    )
    fig[0].savefig(processing_dir + 'meg/' + subject_MEG + '_AEF.png')



#%% === Analyse EEG data === #

# Read raw EEG data
fname_eeg = glob.glob(eeg_dir + "*" + meg_task + ".eeg")
fname_vhdr = glob.glob(eeg_dir + "*" + meg_task + ".vhdr")
raw_eeg = mne.io.read_raw_brainvision(fname_vhdr[0], preload=True)

# Filtering & ICA
raw_eeg = my_preprocessing.reject_artefact(raw_eeg, 1, 40, 0)

# Browse the raw data
raw_eeg.plot()


# Find events embedded in data
eeg_events, _ = mne.events_from_annotations(raw_eeg)
#print(eeg_events[:5])
eeg_events = np.delete(eeg_events, [0,1], 0) # remove first 2 triggers (new segment start & one extra trigger sent by PTB script)

# specify the event IDs
eeg_event_ids = {
    "ba": 53,
    "da": 54,
}

assert len(eeg_events) == len(AD_delta) # sanity check

# Adjust trigger timing based on delay values from MEG data
eeg_events_corrected = copy.copy(eeg_events) # work on a copy so we don't affect the original
for i in range(len(eeg_events)):
    eeg_events_corrected[i,0] += AD_delta[i] # update event timing


# Epoch & calculate ERPs
epochs_eeg = mne.Epochs(raw_eeg, eeg_events_corrected, event_id=eeg_event_ids, tmin=-0.1, tmax=0.41, preload=True)

conds_we_care_about = ["ba", "da"]
epochs_eeg.equalize_event_counts(conds_we_care_about)

# downsample to 100Hz
epochs_eeg_resampled = epochs_eeg.copy().resample(100, npad="auto")

# save for later use
epochs_eeg_resampled.save(epochs_fname_eeg)

# plot ERPs
mne.viz.plot_evoked(epochs_eeg_resampled.average(), gfp="only")
fig = mne.viz.plot_compare_evokeds(
    [
        epochs_eeg_resampled["ba"].average(),
        epochs_eeg_resampled["da"].average(),
    ]
)
fig[0].savefig(processing_dir + 'eeg/' + subject_MEG + '_AEP.png')



#%% === Make alternative plots === #

normal = mne.read_epochs(save_dir_meg + subject_MEG + "_localiser_normal-epo.fif")
gopro = mne.read_epochs(save_dir_meg + subject_MEG + "_localiser_gopro-epo.fif")

# plot 'da' only (normal vs with_gopro)
mne.viz.plot_compare_evokeds(
    [
        normal["da"].average(),
        gopro["da"].average(),
    ]
)
