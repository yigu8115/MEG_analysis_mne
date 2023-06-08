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

import my_preprocessing


# set up file and folder paths here
exp_dir = "/mnt/d/Work/analysis_ME206/"; #"/home/jzhu/analysis_mne/"
subject_MEG = 'G01';
task = 'B1'; #'_TSPCA' #''

# the paths below should be automatic
data_dir = exp_dir + "data/"
processing_dir = exp_dir + "processing/"
meg_dir = data_dir + subject_MEG + "/meg/"
eeg_dir = data_dir + subject_MEG + "/eeg/"
save_dir_meg = processing_dir + "meg/" + subject_MEG + "/" # where to save the epoch files for each subject
save_dir_eeg = processing_dir + "eeg/" + subject_MEG + "/"
figures_dir_meg = processing_dir + 'meg/Figures/' + task + '/' # where to save the figures for all subjects
figures_dir_eeg = processing_dir + 'eeg/Figures/' + task + '/'
epochs_fname_meg = save_dir_meg + subject_MEG + "_" + task + "-epo.fif"
epochs_fname_eeg = save_dir_eeg + subject_MEG + "_" + task + "-epo.fif"
# create the folders if needed
os.system('mkdir -p ' + save_dir_meg)
os.system('mkdir -p ' + save_dir_eeg)
os.system('mkdir -p ' + figures_dir_meg)
os.system('mkdir -p ' + figures_dir_eeg)


#%% === Read raw MEG data === #

#print(glob.glob("*_oddball.con"))
fname_raw = glob.glob(meg_dir + "*" + task + ".con")
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
raw = my_preprocessing.reject_artefact(raw, 1, 10, 0)


#%% === Extract the sections where the participant is purely listening === #

# This is defined as where interviewer (ch166/MISC007) is talking && 
# participant (ch167/MISC008) is not talking

# TODO: maybe use Jasmin 2019 approach (see their supplementary materials); 
# our envelope method requires a lot of manual adjustments of threshold 
# as there is too much variation in speech volume 
# Also there are quite a lot of short segments / mixed speech from both people
# - probably good idea to adopt the "4 second rule" used by Jasmin 2019)


# get raw audio signal from ch166
aud_ch_data_raw = raw.get_data(picks="MISC 007")

def getEnvelope(inputSignal):
    # Taking the absolute value
    absoluteSignal = []
    for sample in inputSignal:
        absoluteSignal.append(abs(sample))
    absoluteSignal = absoluteSignal[0]

    # Peak detection
    intervalLength = 1000  # how many ms to look back - experiment with this number!
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
    return np.array([1 if np.abs(x) > 0.5 else 0 for x in outputSignal])

#raw.load_data().apply_function(getEnvelope, picks="MISC 006")
envelope = getEnvelope(aud_ch_data_raw)
envelope = envelope.tolist() # convert ndarray to list
# detect the beginning of each envelope (set the rest of the envelope to 0)
new_stim_ch = np.clip(np.diff(envelope),0,1)
# find all the 1s (i.e. audio triggers)
stim_tps = np.where(new_stim_ch==1)[0]

# plot any problematic time period to aid diagnosis

test_time = 20000
span = 30000
plt.figure()
plt.plot(aud_ch_data_raw[0], 'b')
#plt.plot(outputSignal, 'r')
for i in range(stim_tps.shape[0]):
   plt.axvline(stim_tps[i], color='r', lw=2, ls='--')
#plt.xlim(test_time-span, test_time+span)
plt.show()



#%% === Epoching === #

# cut into 2-second epochs


if not os.path.exists(epochs_fname_meg):
    epochs = mne.Epochs(raw, events_corrected, event_id=event_ids, tmin=-0.1, tmax=0.41, preload=True)

    conds_we_care_about = ["ba", "da"]
    epochs.equalize_event_counts(conds_we_care_about)

    # downsample to 100Hz
    print("Original sampling rate:", epochs.info["sfreq"], "Hz")
    epochs_resampled = epochs.copy().resample(100, npad="auto")
    print("New sampling rate:", epochs_resampled.info["sfreq"], "Hz")

    # save for later use (e.g. in Source_analysis script)
    epochs_resampled.save(epochs_fname_meg, overwrite=True)

# plot ERFs
if not os.path.exists(figures_dir_meg + subject_MEG + '_AEF_butterfly.png'):
    epochs_resampled = mne.read_epochs(epochs_fname_meg)

    fig = epochs_resampled.average().plot(spatial_colors=True, gfp=True)
    fig.savefig(figures_dir_meg + subject_MEG + '_AEF_butterfly.png')

    fig2 = mne.viz.plot_compare_evokeds(
        [
            epochs_resampled["ba"].average(),
            epochs_resampled["da"].average(),
        ],
        #combine = 'mean' # combine channels by taking the mean (default is GFP)
    )
    fig2[0].savefig(figures_dir_meg + subject_MEG + '_AEF_gfp.png')



#%% === Analyse EEG data === #

# Read raw EEG data
fname_eeg = glob.glob(eeg_dir + "*" + task + ".eeg")
fname_vhdr = glob.glob(eeg_dir + "*" + task + ".vhdr")
raw_eeg = mne.io.read_raw_brainvision(fname_vhdr[0], preload=True)

# set channel types explicitly as these are not read in automatically
raw_eeg.set_channel_types({'32': 'ecg', '63': 'eog'})

# Filtering & ICA
raw_eeg = my_preprocessing.reject_artefact(raw_eeg, 1, 40, 0)

# Browse the raw data
#raw_eeg.plot()


# Find events embedded in data
eeg_events, _ = mne.events_from_annotations(raw_eeg)
#print(eeg_events[:5])

# remove first 2 triggers (new segment start & one extra trigger sent by PTB script)
eeg_events = np.delete(eeg_events, [0,1], 0) 
if subject_MEG == 'G03':
    eeg_events = np.delete(eeg_events, 198, 0) # MEG data is missing the final trigger, so remove it from EEG data too

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


# Epoching
if not os.path.exists(epochs_fname_eeg):
    epochs_eeg = mne.Epochs(raw_eeg, eeg_events_corrected, event_id=eeg_event_ids, tmin=-0.1, tmax=0.41, preload=True)

    conds_we_care_about = ["ba", "da"]
    epochs_eeg.equalize_event_counts(conds_we_care_about)

    # downsample to 100Hz
    epochs_eeg_resampled = epochs_eeg.copy().resample(100, npad="auto")

    # save for later use
    epochs_eeg_resampled.save(epochs_fname_eeg, overwrite=True)

# plot ERPs
if not os.path.exists(figures_dir_eeg + subject_MEG + '_AEP_butterfly.png'):
    epochs_eeg_resampled = mne.read_epochs(epochs_fname_eeg)

    fig = epochs_eeg_resampled.average().plot(spatial_colors=True, gfp=True)
    fig.savefig(figures_dir_eeg + subject_MEG + '_AEP_butterfly.png')
    # Note: it seems like channel locations are not available from vhdr file;
    # you need to specify these explicitly using epochs.set_montage()

    fig2 = mne.viz.plot_compare_evokeds(
        [
            epochs_eeg_resampled["ba"].average(),
            epochs_eeg_resampled["da"].average(),
        ],
        #combine = 'mean' # combine channels by taking the mean (default is GFP)
    )
    fig2[0].savefig(figures_dir_eeg + subject_MEG + '_AEP_gfp.png')



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
