# this is the first computing exercise for ERP data
# tasks:
# 1) execute this code step by step and try to understand what is going on
# 2) in the last two graphs that are shown (7.5 and 7.6) we are looking at
# the difference of the two condifions. Instead of taking the difference,
# calculate ROC-AUC between all univariate features and display them instead
# of the difference
# 3) create a cross-validation scheme, where you esimate
# a) an NCC classifier
# b) an LDA and RLDA classifier
# c) a logistic regression classifier
# In addition, you should experiment with different types of features:
# a) purely temporal features
# b) purely spatial features
# c) spatio-temporal features
# When using spatio-temporal feature, you may need to think about how to
# reduce the dimensionality of your features. (you can find some hint in
# our lecture material..)
# Also, you need to carefully think about whether both classes have the same
# number of samples (they do not). Come up with a way of taking this into account.



#%%
import mne
import matplotlib.pyplot as plt
import numpy as np
from mne.io import read_raw_brainvision
from mne.decoding import Vectorizer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.utils import resample

#%%

# 0. Load and Preprocess Data

loadpath = 'data/'
filename = 'calibration_CenterSpellerMVEP_VPiac'

# Replace with the path to your BrainVision .vhdr file
raw = read_raw_brainvision(loadpath + filename + '.vhdr', preload=True)


# 1. Set Montage and plot a small segment of the data

# After loading your raw BrainVision data, add a standard montage.
# This will provide the necessary digitization points for scalp plotting.
montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage)
# Plot the sensor locations in 2D using the info from raw.
mne.viz.plot_sensors(raw.info, ch_type='eeg', show_names=True)
plt.show()

#%%

# 2.  Apply band-pass filtering (common ERP range is 1-40 Hz)
raw.filter(l_freq=1., h_freq=40., fir_design='firwin')


# 3. Extract events (using all available markers)
events, _ = mne.events_from_annotations(raw)

# Create epochs without filtering by event_id
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5,
                    baseline=(None, 0), preload=True)

# 4. Define marker ranges
target_ids = list(range(31, 47))    # markers 31 to 46
nontarget_ids = list(range(11, 27))   # markers 11 to 26

# 5. Filter epochs based on marker codes
target_epochs = epochs[np.isin(epochs.events[:, 2], target_ids)]
nontarget_epochs = epochs[np.isin(epochs.events[:, 2], nontarget_ids)]

# 6. Compute evoked responses (averaging the epochs)
evoked_target = target_epochs.average()
evoked_nontarget = nontarget_epochs.average()


#%%

# ---------------------------
# 7. Plotting
# ---------------------------
# 7.1. Raw Data Plot
raw_fig = raw.plot(duration =30, n_channels=30, show=False)
raw_fig.suptitle('Raw EEG Data')
plt.show()

#%%

# 7.2. ERP Plot for Target Condition
target_fig = evoked_target.plot(show=False)
target_fig.axes[0].set_title("")
target_fig.suptitle('ERP (Averaged Epochs) - Target')
plt.show()

#%%

# 7.3. ERP Plot for Non-Target Condition
nontarget_fig = evoked_nontarget.plot(show=False)
nontarget_fig.axes[0].set_title("")
nontarget_fig.suptitle('ERP (Averaged Epochs) - Non-Target')
plt.show()

#%%

# 7.4. Joint Plot: Combines ERP waveform with topographies at key time points
joint_fig = evoked_target.plot_joint(times=[0.1, 0.15, 0.18, 0.3, 0.4], show=False)
joint_fig.suptitle('Joint Plot of ERP with Topographies - Target')
plt.show()

joint_fig = evoked_nontarget.plot_joint(times=[0.1, 0.15, 0.18, 0.3, 0.4], show=False)
joint_fig.suptitle('Joint Plot of ERP with Topographies - Non-Target')
plt.show()

#%%

# 7.5 ERP matrix

from sklearn.metrics import roc_auc_score, make_scorer, f1_score

# Get target and non-target data
target_data = target_epochs.get_data()  # (n_target, n_channels, n_times)
nontarget_data = nontarget_epochs.get_data()  # (n_nontarget, n_channels, n_times)

# Initialize AUC matrix
n_channels = target_data.shape[1]
n_times = target_data.shape[2]
auc_scores = np.zeros((n_channels, n_times))

# Compute AUC for each channel and time point
for ch in range(n_channels):
    for t in range(n_times):
        target_vals = target_data[:, ch, t]
        nontarget_vals = nontarget_data[:, ch, t]
        y_true = np.concatenate([np.ones(len(target_vals)), np.zeros(len(nontarget_vals))])
        y_scores = np.concatenate([target_vals, nontarget_vals])
        auc_scores[ch, t] = roc_auc_score(y_true, y_scores)

# Plot AUC image
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
evoked_target.plot_image(show=False, axes=axes[0])
axes[0].set_title('ERP Image Plot - Target')
evoked_nontarget.plot_image(show=False, axes=axes[1])
axes[1].set_title('ERP Image Plot - Non-Target')
im = axes[2].imshow(auc_scores,
                   aspect='auto',
                   origin='lower',
                   extent=[epochs.times[0], epochs.times[-1], 0, n_channels],
                   vmin=0.3,
                   vmax=0.7,
                    cmap='RdBu_r')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Channel')
plt.colorbar(im, ax=axes[2], label='AUC Score')
axes[2].set_title('AUC Scores (Target vs Non-Target)')

plt.tight_layout()
plt.show()


#%%

# 7.6 Comparison of ERP data in a scalp map

# Define typical time points for ERP components (in seconds, adjust as needed)
n100_time = 0.1  # ~100 ms for N100
p300_time = 0.3  # ~300 ms for P300

times = evoked_target.times
t_idx_n100 = np.argmin(np.abs(times - n100_time))
t_idx_p300 = np.argmin(np.abs(times - p300_time))

# Create a figure with 3 rows and 2 columns of subplots.
fig, axes = plt.subplots(3, 2, figsize=(12, 16))

# Row 0: Target condition
evoked_target.plot_topomap(times=n100_time, ch_type='eeg', axes=axes[0, 0],
                           colorbar=False, show=False)
axes[0, 0].set_title('N100 Topography (Target)')

evoked_target.plot_topomap(times=p300_time, ch_type='eeg', axes=axes[0, 1],
                           colorbar=False, show=False)
axes[0, 1].set_title('P300 Topography (Target)')

# Row 1: Non-target condition
evoked_nontarget.plot_topomap(times=n100_time, ch_type='eeg', axes=axes[1, 0],
                              colorbar=False, show=False)
axes[1, 0].set_title('N100 Topography (Non-Target)')

evoked_nontarget.plot_topomap(times=p300_time, ch_type='eeg', axes=axes[1, 1],
                              colorbar=False, show=False)
axes[1, 1].set_title('P300 Topography (Non-Target)')

mne.viz.plot_topomap(auc_scores[:, t_idx_n100], evoked_target.info, show=False, axes=axes[2, 0], cmap='RdBu_r',  vlim=(0.3, 0.7))
axes[2, 0].set_title(f'ROC-AUC at {n100_time*1000:.0f}ms')

mne.viz.plot_topomap(auc_scores[:, t_idx_p300], evoked_target.info, show=False, axes=axes[2, 1], cmap='RdBu_r',  vlim=(0.3, 0.7))
axes[2, 1].set_title(f'ROC-AUC at {p300_time*1000:.0f}ms')

plt.tight_layout()
plt.show()


X_target = target_epochs.get_data()
X_nontarget = nontarget_epochs.get_data()

X = np.concatenate([X_target, X_nontarget])
y = np.concatenate([np.ones(len(X_target)), np.zeros(len(X_nontarget))])

def extract_features(X):
    n_samples, n_channels, n_timepoints = X.shape
    X_temp = np.mean(X, axis=1)
    z_temp = np.abs((X_temp - np.mean(X_temp, axis=0)) / np.std(X_temp, axis=0))
    X_temp[z_temp > 3] = 0
    X_temp_feat = X_temp


    X_spat = np.mean(X, axis=2)
    z_spat = np.abs((X_spat - np.mean(X_spat, axis=0)) / np.std(X_spat, axis=0))
    X_spat[z_spat > 3] = 0
    X_spat_feat = X_spat


    X_flat = X.reshape(n_samples, -1)
    pca_st = PCA(n_components=0.999)
    X_flat_pca = pca_st.fit_transform(X_flat)
    selector_st = SelectKBest(score_func=f_classif, k=min(200, X_flat_pca.shape[1]))
    X_st_feat = selector_st.fit_transform(X_flat_pca, y)

    return {
        'temporal': X_temp_feat,
        'spatial': X_spat_feat,
        'spatio-temporal': X_st_feat
    }

classifiers = {
    'NCC': make_pipeline(StandardScaler(), NearestCentroid()),
    'LDA': make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
    'RLDA': make_pipeline(StandardScaler(),
                          LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
    'LogReg': make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight='balanced'))
}

#ensures balanced fold to train our models
cv = StratifiedKFold(n_splits=20, shuffle=True, random_state=42)
feature_sets = extract_features(X)

results = {}
for feat_name, X_feat in feature_sets.items():
    results[feat_name] = {}

    for clf_name, clf in classifiers.items():
        try:
            scoring = {
                'accuracy': 'accuracy'
            }

            scores = cross_validate(clf, X_feat, y, cv=cv, scoring=scoring, n_jobs=-1)
            mean_acc = np.mean(scores['test_accuracy'])
            std_acc = np.std(scores['test_accuracy'])

            results[feat_name][clf_name] = {
                'accuracy': (mean_acc, std_acc)
            }
        except Exception as e:
            print(f"{clf_name} failed on {feat_name}: {str(e)}")
            results[feat_name][clf_name] = None

print("\n=== Final Results ===")
for feat_name in feature_sets:
    print(f"\n{feat_name.upper()} FEATURES:")
    for clf_name in classifiers:
        res = results[feat_name][clf_name]
        if res:
            acc = res['accuracy']
            print(f"{clf_name}: ACC={acc[0]:.3f}±{acc[1]:.3f}")
        else:
            print(f"{clf_name}: Failed")