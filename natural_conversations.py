#!/usr/bin/python3
#
# Analysis of natural conversations and repetitions
#
# Authors: Paul Sowman, Judy Zhu

#######################################################################################

import os
import mne
import meegkit # for TSPCA
import glob
import matplotlib.pyplot as plt
import numpy as np
import copy

import my_preprocessing


# set up file and folder paths here
exp_dir = "/mnt/d/Work/analysis_ME206/"; #"/home/jzhu/analysis_mne/"
subject_MEG = 'G01';
task = 'B1' #''

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
ica_fname_meg = save_dir_meg + subject_MEG + "_" + task + "-ica.fif"
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


# Apply TSPCA for noise reduction
# https://github.com/pealco/python-meg-denoise/blob/master/example5.py
noisy_data = raw.get_data(picks="meg").transpose()
noisy_ref = raw.get_data(picks=[160,161,162]).transpose()
#shifts = r_[-50:51]
data_after_tspca, idx = meegkit.tspca.tsr(noisy_data, noisy_ref)[0:2]

# check the results
raw.plot()
raw_clean = copy.deepcopy(raw)
raw_clean._data[0:160] = data_after_tspca.transpose()
raw_clean.plot()

# compare TSPCA from MEG160 and from meegkit
'''
plt.figure()
plt.plot(raw_MEG160.get_data(picks=14)[0], 'b')
plt.plot(raw_clean.get_data(picks=14)[0], 'r')
plt.xlim(100, 200)
'''

# can also try SSS:
# https://mne.tools/stable/auto_tutorials/preprocessing/60_maxwell_filtering_sss.html
# or DSS in the meegkit package


# Filtering & ICA
raw_clean = my_preprocessing.reject_artefact(raw_clean, 1, 10, False, '')
#raw_clean = my_preprocessing.reject_artefact(raw_clean, 1, 10, True, ica_fname_meg)


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

# TODO: cut into 2-second epochs


if not os.path.exists(epochs_fname_meg):
    epochs = mne.Epochs(raw_clean, events_corrected, event_id=event_ids, tmin=-0.1, tmax=0.41, preload=True)

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
