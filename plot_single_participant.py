#!/usr/bin/python3
#
# Make some plots from a single participant's data
#
# Author: Judy Zhu

#######################################################################################

import mne
import numpy as np
import matplotlib.pyplot as plt



# === Plotting for auditory roving oddball task === #

# plot difference wave (deviant minus standard)
# 2 lines representing S1 vs S2


conds = ["standard", "deviant"]

# load the previously saved epoch files
S1 = mne.read_epochs('/mnt/d/Work/analysis_ME197/processing/meg/230301_72956_S1/230301_72956_S1_oddball-epo.fif')
S2 = mne.read_epochs('/mnt/d/Work/analysis_ME197/processing/meg/230426_72956_S2/230426_72956_S2_oddball-epo.fif')

# average the epochs in each cond to get the ERF
S1_evoked = dict()
S2_evoked = dict()
for cond in conds:
    S1_evoked[cond] = S1[cond].average()
    S2_evoked[cond] = S2[cond].average()


# Opt 1: Compute GFP for each cond, then take the diff between conds

S1_GFP = dict()
S2_GFP = dict()
times = S1_evoked['standard'].times
for cond in conds:
    # How to compute GFP: https://github.com/mne-tools/mne-python/pull/5796
    S1_GFP[cond] = S1_evoked[cond].copy().pick_types(meg=True).data.std(axis=0)
    S2_GFP[cond] = S2_evoked[cond].copy().pick_types(meg=True).data.std(axis=0)
    # baseline correct the GFPs? shouldn't need to - baseline correction already applied by default when creating the epochs (verified!)
    #S1_GFP[cond] = mne.baseline.rescale(S1_GFP[cond], times, baseline=(None, 0))
    #S2_GFP[cond] = mne.baseline.rescale(S2_GFP[cond], times, baseline=(None, 0))

# deviant GFP minus standard GFP
S1_diff = np.subtract(S1_GFP['deviant'], S1_GFP['standard'])
S2_diff = np.subtract(S2_GFP['deviant'], S2_GFP['standard'])

fig, ax = plt.subplots()
ax.plot(times * 1000, S1_diff * 1e15) # convert time units to ms
ax.plot(times * 1000, S2_diff * 1e15) # convert amplitude units to fT
ax.axhline(0, linestyle="--", color="grey", linewidth=2) # draw a horizontal line at y=0
plt.legend(['S1: deviant - standard', 'S2: deviant - standard'])
plt.xlabel("Time (ms)")
plt.ylabel("fT")
plt.show()


# Opt 2: Compute difference wave at each sensor, then plot the GFP

# difference wave = deviant minus standard
S1_MMF = mne.combine_evoked([S1_evoked['deviant'], S1_evoked['standard']], 
                            weights=[1, -1])
S2_MMF = mne.combine_evoked([S2_evoked['deviant'], S2_evoked['standard']], 
                            weights=[1, -1])
S1_MMF.comment = 'S1: deviant - standard' # legend for the plot
S2_MMF.comment = 'S2: deviant - standard'

# plot GFPs
mne.viz.plot_compare_evokeds(
    [
        S1_MMF,
        S2_MMF,
    ]
)




# === Plotting for visual LTP task === #

# plot difference wave (tetanised minus non-tetanised)
# each graph has 3 lines representing the 3 task blocks: ltp1, lpt2, ltp3
# make separate graphs for S1 & S2


# specify which subject & session to plot:
subject_MEG = '230426_72956_S2' #'230301_72956_S1'

#TODO# the mappings below need to be set dynamically for each subject
tetanised = "horizontal" 
non_tetanised = "vertical"


save_dir = "/mnt/d/Work/analysis_ME197/processing/meg/" + subject_MEG + "/"
meg_tasks = ['_ltp1', '_ltp2', '_ltp3'] #'_oddball' #''
task_labels = ["pre", "post_2min", "post_30min"] # corresponding to the 3 meg tasks above

# Loop over blocks (pre, 2min post, 30min post) & read in the saved ERFs
evokeds = {} # initialise the dict
for counter, task in enumerate(meg_tasks):
    evokeds_fname = save_dir + subject_MEG + task + "-ave.fif"
    evokeds[task_labels[counter]] = mne.read_evokeds(evokeds_fname)


# Opt 1: Compute GFP for each cond, then take the diff between conds

GFPs = {}
GFP_diffs = {}

for counter, task_label in enumerate(task_labels):
    GFPs[task_label] = {} # declare as a dict again in order to access the next dimension (i.e. cond)
    for index, evoked in enumerate(evokeds[task_label]):
        cond = evoked.comment
        GFPs[task_label][cond] = evoked.copy().pick_types(meg=True).data.std(axis=0)
        # the evokeds are stored in a list, cannot access by indexing the cond name,
        # so we convert back to using a dict to store the GFPs

    # calculate difference wave (tetanised minus non-tetanised) within each task
    GFP_diffs[task_label] = np.subtract(GFPs[task_label][tetanised], GFPs[task_label][non_tetanised])

# plot 3 lines (one line per block) from this session
fig, ax = plt.subplots()
times = evokeds['pre'][0].times
for counter, task_label in enumerate(task_labels):
    ax.plot(times * 1000, GFP_diffs[task_label] * 1e15) # convert time units to ms, convert amplitude units to fT
#ax.plot(times * 1000, GFP_diffs['pre'] * 1e15) # convert time units to ms, 
#ax.plot(times * 1000, GFP_diffs['post_30min'] * 1e15) # convert amplitude units to fT
ax.axhline(0, linestyle="--", color="grey", linewidth=2) # draw a horizontal line at y=0
plt.legend(['pre: tenanised - non-tetanised', 
            '2min post: tenanised - non-tetanised', 
            '30min post: tenanised - non-tetanised'])
plt.xlabel("Time (ms)")
plt.ylabel("fT")
plt.show()
