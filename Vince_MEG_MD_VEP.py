#!/usr/bin/python3
#
# MEG sensor space analysis for visual LTP (vertical & horizontal gratings)
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

from mne.preprocessing import find_bad_channels_maxwell
#from autoreject import get_rejection_threshold  # noqa
#from autoreject import Ransac  # noqa
#from autoreject.utils import interpolate_bads  # noqa

from scipy import stats

import my_preprocessing


# Make plots interactive when running in interactive window in vscode
#plt.switch_backend('TkAgg') You can use this backend if needed
#plt.ion() 
# %matplotlib qt 


# set up file and folder paths here
exp_dir = "/mnt/d/Work/analysis_ME197/"
subject_MEG = '230301_72956_S1' #'220112_p003'
meg_tasks = ['_ltp2', '_ltp3'] #'_oddball' #''

# the paths below should be automatic
data_dir = exp_dir + "data/"
processing_dir = exp_dir + "processing/"
meg_dir = data_dir + subject_MEG + "/meg/"
save_dir = processing_dir + "meg/" + subject_MEG + "/"
os.system('mkdir -p ' + save_dir) # create the folder if needed


fname_elp = glob.glob(meg_dir + "*.elp")
fname_hsp = glob.glob(meg_dir + "*.hsp")
fname_mrk = glob.glob(meg_dir + "*.mrk")

#%% Loop over tasks: pre, 2min post, 30min post
for counter, task in enumerate(meg_tasks):
    fname_raw = glob.glob(meg_dir + "*" + task + ".con")

    epochs_fname = save_dir + subject_MEG + task + "-epo.fif"
    ica_fname = save_dir + subject_MEG + task + "-ica.fif"

    raw = mne.io.read_raw_kit(
        fname_raw[0], 
        mrk=fname_mrk[0],
        elp=fname_elp[0],
        hsp=fname_hsp[0],
        stim=[*range(177, 179)], # these triggers (177 and 178) indicate vertical or horizontal
        slope="+",
        stim_code="channel",
        stimthresh=1,  # 2 for adults
        preload=True,
        allow_unknown_format=False,
        verbose=True,
    )
    
    # Apply TSPCA for noise reduction
    noisy_data = raw.get_data(picks="meg").transpose()
    noisy_ref = raw.get_data(picks=[160,161,162]).transpose()
    data_after_tspca, idx = meegkit.tspca.tsr(noisy_data, noisy_ref)[0:2]
    raw._data[0:160] = data_after_tspca.transpose()

    # browse data to identify bad sections & bad channels
    raw.plot()


    # Filtering & ICA
    raw = my_preprocessing.reject_artefact(raw, 0.1, 40, True, ica_fname)


    #%% ch misc 23-29 are trigger channels
    # MISC 18 and 19 == horizontal / vertical gratings
    # MISC 6 or 10 == Photodetector

    #%% Finding events
    events = mne.find_events(
        raw,
        output="onset",
        consecutive=False,
        min_duration=0,
        shortest_event=1,  # 5 for adults
        mask=None,
        uint_cast=False,
        mask_type="and",
        initial_event=False,
        verbose=None,
    )

    for index, event in enumerate(events):
        if event[2] == 177: # ch177 == MISC 18
            events[index, 2] = 2 # horizontal
        elif event[2] == 178: # ch178 == MISC 19
            events[index, 2] = 3 # vertical

    # Find times of PD triggers
    # Ensure correct PD channel is entered here, might sometimes be 165
    events_PD = mne.find_events(
        raw, stim_channel=[raw.info["ch_names"][x] for x in [169]], output="onset"
    )

    combined_events = np.concatenate([events, events_PD])
    combined_events = combined_events[np.argsort(combined_events[:, 0])]

    #%% find the difference between PD time and trigger time
    pd_delta = []
    for index, event in enumerate(combined_events):
        if (
            index > 0  # PD can't be first event
            and combined_events[index, 2] == 1 # current trigger is PD trigger
            and combined_events[index - 1, 2] != 1 # previous trigger is not PD trigger
        ):
            pd_delta.append(
                combined_events[index, 0] - combined_events[index - 1, 0] # find the time difference
            )
    # show histogram of PD delays
    n, bins, patches = plt.hist(
        x=pd_delta, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85
    )
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Delay (ms)")
    plt.ylabel("Frequency")
    plt.title("Photo Detector Delays")
    plt.text(
        70,
        50,
        r"$mean="
        + str(round(np.mean(pd_delta)))
        + ", std="
        + str(round(np.std(pd_delta)))
        + "$",
    )
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

    # Use target events to align triggers & avoid outliers using z of 3
    z = np.abs(stats.zscore(pd_delta))
    #TODO: check this part works correctly when we do have outliers!
    if [pd_delta[i] for i in np.where(z > 3)[0]]:
        tmax = -max([pd_delta[i] for i in np.where(z > 3)[0]]) / 1000
    else:
        tmax = 0

    events_to_find = [2, 3] # target events
    sfreq = raw.info["sfreq"]  # sampling rate
    tmin = -0.4  # PD occurs after trigger, hence negative
    fill_na = None  # the fill value for non-target
    reference_id = 1  # PD
    
    # loop through events and replace PD events with event class identifier i.e. trigger number
    events_target = {}
    for event in events_to_find:
        new_id = 20 + event
        events_target["event" + str(event)], lag = mne.event.define_target_events(
            combined_events,
            reference_id,
            event,
            sfreq,
            tmin,
            tmax,
            new_id,
            fill_na,
        )
    events = np.concatenate((events_target["event2"], events_target["event3"]))

    event_ids = {
        "horizontal": 22,
        "vertical": 23,
    }


    #%% === Epoching === #

    epochs = mne.Epochs(
        raw, events, event_id=event_ids, tmin=-0.1, tmax=0.5, preload=True
    )

    conds_we_care_about = ["horizontal", "vertical"]
    epochs.equalize_event_counts(conds_we_care_about)

    # sanity check - PD triggers occur at 0ms
    mne.viz.plot_evoked(
        epochs_resampled.average(picks="MISC 010")
    ) 

    # downsample to 100Hz
    print("Original sampling rate:", epochs.info["sfreq"], "Hz")
    epochs_resampled = epochs.copy().resample(100, npad="auto")
    print("New sampling rate:", epochs_resampled.info["sfreq"], "Hz")

    # save the epochs to file
    epochs.save(epochs_fname)


    # plot ERFs
    mne.viz.plot_evoked(epochs_resampled.average(), gfp="only")
    fig = mne.viz.plot_compare_evokeds(
        [
            epochs_resampled["horizontal"].average(),
            epochs_resampled["vertical"].average(),
        ]
    )
    fig[0].savefig(save_dir + subject_MEG + task + ".png")
 
           

# TODO #

# load the saved epochs for ltp1, ltp2 & ltp3
epochs_all = {} # initialise dict to store the epochs from all blocks
task_labels = ["pre", "post_2min", "post_30min"] # corresponding to the 3 meg tasks above


# calculate difference wave (tetanised minus non-tetanised) for each block
tetanised = "horizontal" # these need to be set dynamically for each subject
non-tetanised = "vertical"

# plot 3 lines (one line per block) from this session




# report = mne.Report(title=fname_raw[0])
# report.add_evokeds(
#     evokeds=evoked, titles=["VEP"], n_time_points=25  # Manually specify titles
# )
# report.save(fname_raw[0] + "_report_evoked.html", overwrite=True)
