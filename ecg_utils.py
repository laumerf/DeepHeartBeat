import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm.auto import tqdm

from utils import load_physionet_ecg_model


# Downsampling rate for the signal
sampling_rate = 4


# data loading and evaluation

def gen_factory(data_dict, mode="slices"):
    def gen():
        for subject in data_dict.keys():
            measurements = data_dict[subject]['measurements'].astype('float32')

            if mode == "full":
                # Embed the whole sequence
                length = len(measurements)
                start = 0
            elif mode == "slices":
                # Select a random subsequence of the signal to encode
                length = np.random.randint(450, 1200)
                start = np.random.randint(measurements.shape[0] - length)
            else:
                raise ValueError("Expected mode: 'slices' or 'full', got: {}".format(mode))

            length -= length % sampling_rate
            sub_measurements = measurements[start:start + length].copy()
            sub_measurements = np.mean(np.reshape(sub_measurements, [-1, sampling_rate]), axis=-1)
            frequency = data_dict[subject]['frequency']
            sub_times = np.array([sampling_rate * t / frequency for t in range(sub_measurements.shape[0])], dtype='float32')
            yield subject, sub_times, sub_measurements

    return gen

def create_dataset(generator):

    batch_size = 64

    dataset_output_types = (tf.string, tf.float32, tf.float32)
    dataset_output_shapes = (tf.TensorShape(()), tf.TensorShape([None]), tf.TensorShape([None]))
    dataset = tf.data.Dataset.from_generator(generator, dataset_output_types, dataset_output_shapes)

    # messy batching, as RaggedTensors not fully supported by Tensorflow's Dataset
    dataset = dataset.map(lambda id, times, frames: (id, tf.expand_dims(times, 0), tf.expand_dims(frames, 0)))
    dataset = dataset.map(
        lambda id, times, frames: (id, tf.RaggedTensor.from_tensor(times), tf.RaggedTensor.from_tensor(frames)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda ids, times, frames: (ids, tf.squeeze(times, axis=1), tf.squeeze(frames, axis=1)))
    return dataset

# Evaluation: pass the dataset to the (trained) model
def encode_datasest(dataset):

    # Load Echo models
    model = load_physionet_ecg_model()

    model_results = dict()
    for ids, times, frames in tqdm(dataset):

        params, _, reconstructions = model((times, frames))

        for i, id in enumerate(ids.numpy()):
            heart_rate = 60 * np.exp(params[i][0])
            phase = params[i][1].numpy()
            reconstruction_error = np.mean((frames[i] - reconstructions[i]) ** 2)
            reconstruction_pixel_stddev = np.mean(np.std(reconstructions[i], axis=0))
            model_results[str(id, 'ascii')] = {
                'heart_rate': heart_rate,
                'phase': phase,
                'reconstruction_error': reconstruction_error,
                'reconstruction_pixel_stddev': reconstruction_pixel_stddev,
                'parameters': params[i].numpy()
            }
    return model_results

def get_results(files, mode):
    return encode_datasest(create_dataset(gen_factory(files, mode)))


# AF experiment

def run_af_experiment(X_train, y_train, X_test, y_test, classifier_factory, hyperparam_list):
    results = []

    test_roc_auc = []
    test_accuracies = []
    test_recall = []
    test_precision = []
    test_f1 = []
    opt_hparam_values = []
    val_roc_auc = []

    tune_kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=104)
    for train_index_tune, val_index_tune in tqdm(tune_kf.split(X_train, y_train), total=3):

        X_train_tune = X_train[train_index_tune]
        y_train_tune = y_train[train_index_tune]

        X_val_tune = X_train[val_index_tune]
        y_val_tune = y_train[val_index_tune]

        # standardise
        scaler = StandardScaler()
        X_train_tune_s = scaler.fit_transform(X_train_tune)
        X_val_tune_s = scaler.transform(X_val_tune)

        val_roc_auc_split = []
        for hyperparam in hyperparam_list:
            classifier = classifier_factory(hyperparam)
            classifier.fit(X_train_tune_s, y_train_tune)

            y_pred_proba_val_tune = classifier.predict_proba(X_val_tune_s)[:, 1]
            val_roc_auc_split.append(roc_auc_score(y_val_tune, y_pred_proba_val_tune))

        val_roc_auc.append(val_roc_auc_split)

    # mean score per hyperparameter
    val_roc_auc = np.mean(val_roc_auc, axis=0)
    print(hyperparam_list)
    print(val_roc_auc)

    # optimal hyperparameter(s)
    opt_hparam = hyperparam_list[np.argmax(val_roc_auc)]
    opt_hparam_values.append(opt_hparam)

    # train & predict for test set
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    classifier = classifier_factory(opt_hparam)
    classifier.fit(X_train_s, y_train)

    y_pred_test = classifier.predict(X_test_s)
    y_pred_proba_test = classifier.predict_proba(X_test_s)[:, 1]

    test_roc_auc.append(roc_auc_score(y_test, y_pred_proba_test))
    test_accuracies.append(balanced_accuracy_score(y_test, y_pred_test))
    test_recall.append(recall_score(y_test, y_pred_test))
    test_precision.append(precision_score(y_test, y_pred_test))
    test_f1.append(f1_score(y_test, y_pred_test))

    print("Optimal hyperparameter value: {}".format(opt_hparam))
    print("ROC AUC: {}".format(roc_auc_score(y_test, y_pred_test)))
    print("Balanced accuracy: {}".format(balanced_accuracy_score(y_test, y_pred_test)))
    print("Recall (sensitivity): {}".format(recall_score(y_test, y_pred_test)))
    print("Precision: {}".format(precision_score(y_test, y_pred_test)))
    print("F1-measure: {}".format(f1_score(y_test, y_pred_test)))
    print("\n")

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    print("TN: {}, FN: {}, TP: {}".format(tn, fp, fn, tp))
    print("Specificity: {}".format(tn * 1.0 / (tn + fp)))

    results.append({'test_roc_auc': test_roc_auc, 'test_accuracies': test_accuracies, 'test_recall': test_recall,
                    'test_precision': test_precision, 'test_f1': test_f1, 'opt_C': opt_hparam_values})

    return results