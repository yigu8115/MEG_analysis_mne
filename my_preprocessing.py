# MEG preprocessing: filtering & ICA
#
# (autoreject & Ransac are not finding much?? 
# maybe they only target super large artefact - that makes sense; 
# in most cases, ICA alone seems to suffice - it can fix bad sensors &
# some LFN as well)
#
# Authors: Paul Sowman, Judy Zhu

def reject_artefact(raw, l_freq, h_freq, do_ICA):
    import mne

    # disable the following imports for now as they are throwing errors 
    # after upgrading to mne v1.3.1
    '''
    from mne.preprocessing import find_bad_channels_maxwell, ICA
    from autoreject import get_rejection_threshold  # noqa
    from autoreject import Ransac  # noqa
    from autoreject.utils import interpolate_bads  # noqa
    '''

    if do_ICA:
        # filtering for ICA
        raw_for_ICA = raw.copy()
        raw_for_ICA.filter(l_freq=1.0, h_freq=h_freq) # use 1Hz for ICA
        
        # 'autoreject' requires epoched data
        # here we create arbitrary epochs (making use of all data - useful for short recordings)
        tstep = 1.0 # make 1 second epochs
        events_ICA = mne.make_fixed_length_events(raw_for_ICA, duration=tstep)
        epochs_ICA = mne.Epochs(
            raw_for_ICA, events_ICA, tmin=0.0, tmax=tstep, baseline=None, preload=True
        )
        
        # use 'autoreject' to compute a threshold for removing large noise
        reject = get_rejection_threshold(epochs_ICA)
        reject # print the result
        # remove large noise before running ICA
        #epochs_ICA.load_data() # to avoid reading epochs from disk multiple times
        epochs_ICA.drop_bad(reject=reject)

        # could also try Ransac
        #ransac = Ransac(verbose=True, n_jobs=1)
        #epochs_ICA_clean = ransac.fit_transform(epochs_ICA)
        
        # run ICA
        ica = ICA(n_components=60, max_iter="auto", random_state=97)
        ica.fit(epochs_ICA, tstep=tstep) # 'reject' param here only works for continuous data, so we use drop_bad() above instead
        ica.plot_sources(raw) # plot IC time series
        ica.plot_components() # plot IC topography

        # manually select which components to reject
        select_comps = input("Enter the IC components to remove (e.g. [0, 3, 5]): ")
        #ica.exclude = [8, 17, 23, 58]
        ica.exclude = select_comps
        # can also use automatic methods:
        # https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#using-a-simulated-channel-to-select-ica-components
        # https://github.com/LanceAbel/MQ_MEG_Analysis (selecting channels to simulate EOG)


    # filtering
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    if do_ICA:
        # Compare raw data before & after ICA
        raw.plot(n_channels=160, title='before ICA') # before IC rejection
        raw_clean = ica.apply(raw) # apply component rejection onto raw (continuous) data
        raw_clean.plot(n_channels=160, title='after ICA') # after IC rejection
        return raw_clean
    else:
        return raw
