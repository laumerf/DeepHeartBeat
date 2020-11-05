from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from utils import *
from ecg_utils import get_results, run_af_experiment


# Load Physionet data
print('Load Physionet data...')
labels_train, files_train = get_physionet_data(split='training')
labels_val, files_val = get_physionet_data(split='validation')
print('Physionet data loaded.')


print('')
print('Evaluate Physionet data')
print('-----------------------')

full_results_train = get_results(files_train, mode='full')
full_results_val = get_results(files_val, mode='full')
slice_results_train = get_results(files_train, mode='slices')
slice_results_val = get_results(files_val, mode='slices')


print('')
print('Anomaly detection')
print('-----------------')
noise_labels = [1 if labels_train[rec_id] == '~' else 0 for rec_id in full_results_train.keys()]
rec_error = [full_results_train[rec_id]['reconstruction_error'] for rec_id in full_results_train.keys()]
print("ROC AUC score: {}".format(roc_auc_score(noise_labels, rec_error)))


print('')
print('AF detection')
print('------------')

def get_af_dataset(results, labels):
    X = np.array([np.append(results[id]['parameters'], results[id]['reconstruction_error']) for id in results.keys()])
    y = np.array([labels[id] for id in results.keys()])
    y = np.where(y == 'A', 1.0, 0.0)
    return X, y

X_train, y_train = get_af_dataset(slice_results_train, labels_train)
X_test, y_test = get_af_dataset(slice_results_val, labels_val)

c_values = np.logspace(-1, 1, 50)
run_af_experiment(X_train, y_train, X_test, y_test, lambda param: SVC(C=param, class_weight='balanced', probability=True), c_values)