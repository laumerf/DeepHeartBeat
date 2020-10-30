from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tqdm.auto import tqdm

from utils import *


# Load EchoNet-Dynamic data
print('Load EchoNet-Dynamic data...')
echonet_info, files = load_data_echonet()
print('EchoNet-Dynamic data loaded.')

# Load Echo models
models = [load_echonet_dynamic_model(i) for i in range(5)]

# For each model and each subject: determine latent parameters
def get_results(model, files):

    batch_size = 64
    
    def gen():
        for id, filepath in files.items():
            data = np.load(filepath)
            times = data['times']
            frames = data['frames']
            frames = (frames/255).astype('float32')

            yield id, times, frames
            
    dataset_output_types = (tf.string, tf.float32, tf.float32)
    dataset_output_shapes = (tf.TensorShape(()), tf.TensorShape([None]), tf.TensorShape([None, 112, 112]))
    dataset = tf.data.Dataset.from_generator(gen, dataset_output_types, dataset_output_shapes)

    # messy batching, as RaggedTensors not fully supported by Tensorflow's Dataset
    dataset = dataset.map(lambda id, times, frames: (id, tf.expand_dims(times, 0), tf.expand_dims(frames, 0)))
    dataset = dataset.map(lambda id, times, frames: (id, tf.RaggedTensor.from_tensor(times), tf.RaggedTensor.from_tensor(frames)))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda ids, times, frames: (ids, tf.squeeze(times, axis=1), tf.squeeze(frames, axis=1)))
    
    model_results = dict()
    for ids, times, frames in tqdm(dataset, total=int(np.ceil(len(files)/batch_size))):

        params, _, reconstructions = model((times, frames))

        for i, id in enumerate(ids.numpy()):
            reconstruction_error = np.mean((frames[i] - reconstructions[i])**2)
            model_results[str(id, 'ascii')] = {
                'parameters': params[i].numpy()
            }
            
    return model_results

results = [get_results(model, files) for model in models]


# Heart rate - RNMF comparison
print('')
print('RNMF heart rate comparison')
print('--------------------------')

rnmf_results = pd.read_csv('./data/rnmf_heart_rates_echonet.csv', index_col=0)
ids = rnmf_results.index.values

for i, model_results in enumerate(results):

    rnmf_rates = np.array([rnmf_results.loc[id]['bpm'] for id in ids])
    model_rates = np.array([60*np.exp(model_results[id]['parameters'][0]) for id in ids])

    errs = np.abs(rnmf_rates - model_rates)/np.minimum(rnmf_rates, model_rates)
    print(f'Model {i} | Deviation (Mean): {np.mean(errs):.4f} - Deviation (Median): {np.median(errs):.4f}')


# Alignment
print('')
print('Alignment')
print('---------')

volume_tracings = pd.read_csv('./data/EchoNet-Dynamic/VolumeTracings.csv', index_col='FileName')

def get_end_diastole_index(id):
    return min(volume_tracings[volume_tracings.index == id].Frame.unique())

def get_end_systole_index(id):
    return max(volume_tracings[volume_tracings.index == id].Frame.unique())

end_systole_times = dict()
end_diastole_times = dict()

print('Determine ground-truth end-systole and end-diastole times...')
for id, filepath in tqdm(files.items()):
    
    if id in [
        '0X5DD5283AC43CCDD1',
        '0X234005774F4CB5CD',
        '0X2DC68261CBCC04AE',
        '0X35291BE9AB90FB89',
        '0X6C435C1B417FDE8A',
        '0X5515B0BD077BE68A'
    ]:
        continue
    
    data = np.load(filepath)
    times = data['times']
    
    end_systole_times[id] = times[get_end_systole_index(id)]
    end_diastole_times[id] = times[get_end_diastole_index(id)]

for i, model_results in enumerate(results):

    print(f'Model {i}')
    
    ids = end_systole_times.keys()
    
    offsets = np.array([model_results[id]['parameters'][1] for id in ids])
    heart_rates = np.array([60*np.exp(model_results[id]['parameters'][0]) for id in ids])
    end_systole_time_values = np.array([end_systole_times[id] for id in ids])
    end_diastole_time_values = np.array([end_diastole_times[id] for id in ids])
    
    s_end_systole = np.mod(heart_rates/60*(end_systole_time_values - offsets), 1.0)
    s_end_diastole = np.mod(heart_rates/60*(end_diastole_time_values - offsets), 1.0)
    
    s_end_systole_discrete = np.rint(100*s_end_systole).astype('int')
    s_end_systole_discrete[s_end_systole_discrete == 100] = 0
    values, counts = np.unique(s_end_systole_discrete, return_counts=True)
    selected_end_systole = values[np.argmax(counts)]/100
    print('Selected end systole:', selected_end_systole)
    
    s_end_diastole_discrete = np.rint(100*s_end_diastole).astype('int')
    s_end_diastole_discrete[s_end_diastole_discrete == 100] = 0
    values, counts = np.unique(s_end_diastole_discrete, return_counts=True)
    selected_end_diastole = values[np.argmax(counts)]/100
    print('Selected end diastole:', selected_end_diastole)
    
    end_systole_diffs = np.abs(s_end_systole - selected_end_systole)
    end_systole_diffs = np.where(end_systole_diffs > 0.5, 1 - end_systole_diffs, end_systole_diffs)

    print('Mean Systole Error:', np.mean(end_systole_diffs))
    print('Median Systole Error:', np.median(end_systole_diffs))
    
    end_diastole_diffs = np.abs(s_end_diastole - selected_end_diastole)
    end_diastole_diffs = np.where(end_diastole_diffs > 0.5, 1 - end_diastole_diffs, end_diastole_diffs)
    
    print('Mean Diastole Error:', np.mean(end_diastole_diffs))
    print('Median Diastole Error:', np.median(end_diastole_diffs))


# Ejection fraction prediction of the left ventricle
print('')
print('Ejection Fraction Prediction')
print('----------------------------')

# train-val-test split as in EchoNet-Dynamic paper
echonet_train_ids = echonet_info[echonet_info.Split == 'TRAIN'].index.values
echonet_val_ids = echonet_info[echonet_info.Split == 'VAL'].index.values
echonet_test_ids = echonet_info[echonet_info.Split == 'TEST'].index.values

def echonet_split(results):
    X_train = np.array([results[str(id)]['parameters'] for id in echonet_train_ids])
    y_train = np.array([echonet_info.loc[id].EF for id in echonet_train_ids])

    X_val = np.array([results[str(id)]['parameters'] for id in echonet_val_ids])
    y_val = np.array([echonet_info.loc[id].EF for id in echonet_val_ids])

    X_test = np.array([results[str(id)]['parameters'] for id in echonet_test_ids])
    y_test = np.array([echonet_info.loc[id].EF for id in echonet_test_ids])
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# EF predictor model
def build_model(n_features):
    
    inputs = Input(shape=(n_features,))
    h = inputs
    h = Dense(1024, activation='relu')(h)
    h = Dense(1024, activation='relu')(h)
    h = Dense(1, activation='sigmoid')(h)
    h = Lambda(lambda x: 100*x)(h)
    outputs = h
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Train EF predictors and evaluate on test split
for i, model_results in enumerate(results):
    
    X_train, y_train, X_val, y_val, X_test, y_test = echonet_split(model_results)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    
    es = EarlyStopping(patience=20, restore_best_weights=True)
    optimizer = Adam(learning_rate=1e-4)
    regressor = build_model(n_features=X_train.shape[1])
    regressor.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    regressor.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), batch_size=8, epochs=1000, callbacks=[es], verbose=0)

    y_pred_test = regressor.predict(X_test_s)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print(f'Model {i} | RMSE: {rmse:.5f} - MAE: {mae:.5f} - R2 Score: {r2:.5f}')