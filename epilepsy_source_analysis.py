import os.path as op
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse

# set up paths
exp_dir = 'C:/Users/mq43606024/Downloads/epilepsy_MEG/'
subjects_dir = exp_dir + 'mri/'
subject = 'p0001' # subject name in mri folder
subjects_dir_MEG = exp_dir + 'meg/'
subject_MEG = '0001_JC_ME200_11022022' # subject name in meg folder
meg_task = '_resting_B1_TSPCA'
results_dir = subjects_dir_MEG + 'source_analysis/'

base_fname = subjects_dir_MEG + subject_MEG + "/" + subject_MEG
raw_fname = base_fname + meg_task + '-raw.fif'
fname_bem = subjects_dir + subject + '/bem/p0001-5120-bem-sol.fif' # obtained with: mne setup_forward_model -s $my_subject -d $SUBJECTS_DIR --homog --ico 4
fname_trans = base_fname + "-trans.fif" # obtained with: mne coreg (GUI)
fname_fwd = results_dir + subject_MEG + "-fwd.fif"

# load raw data
raw = mne.io.read_raw_fif(raw_fname, verbose=False, preload=True)

# filter the data
raw.filter(l_freq=1, h_freq=80)

# read spikes from csv file
my_annot = mne.read_annotations(exp_dir + 'saved-annotations-for_Judy_1Aug22.csv')
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
cov_polysw = mne.compute_covariance(epochs['polysw'])
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


# Method 2: minimum norm
# https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html#inverse-modeling-mne-dspm-on-evoked-and-raw-data

# create source space
spacing = "oct6" # use a recursively subdivided octahedron: 4 for speed, 6 for real analyses
src_fname = op.join(subjects_dir, subject, "bem", subject_MEG + "_" + spacing + "-src.fif")
if op.exists(src_fname):
    src = mne.read_source_spaces(src_fname)
else:
    src = mne.setup_source_space(
        subject, spacing=spacing, add_dist="patch", subjects_dir=subjects_dir
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

    trans = mne.read_trans(trans_fname)
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

# compute inverse solutoion
inverse_operator = make_inverse_operator(
    evoked_sw_post.info, fwd, cov_sw_post, loose=0.2, depth=0.8)

method = "MNE" # “dSPM” | “sLORETA” | “eLORETA”
snr = 3.
lambda2 = 1. / snr ** 2
stc, residual = apply_inverse(evoked_sw_post, inverse_operator, lambda2,
                              method=method, pick_ori=None,
                              return_residual=True, verbose=True)    

