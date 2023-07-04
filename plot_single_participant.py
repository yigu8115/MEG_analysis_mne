#!/usr/bin/python3
#
# Make some plots from a single participant's data
#
# Author: Judy Zhu

#######################################################################################

import mne
import numpy as np
import matplotlib.pyplot as plt

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
    S1_GFP[cond] = S1_evoked[cond].copy().pick_types(meg=True).data.std(axis=0)
    S2_GFP[cond] = S2_evoked[cond].copy().pick_types(meg=True).data.std(axis=0)
    S1_GFP[cond] = mne.baseline.rescale(S1_GFP[cond], times, baseline=(None, 0))
    S2_GFP[cond] = mne.baseline.rescale(S2_GFP[cond], times, baseline=(None, 0))

# deviant GFP minus standard GFP
S1_diff = np.subtract(S1_GFP['deviant'], S1_GFP['standard'])
S2_diff = np.subtract(S2_GFP['deviant'], S2_GFP['standard'])

#gfp = np.sum(average.data**2, axis=0)
#gfp = mne.baseline.rescale(gfp, times, baseline=(None, 0))
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
