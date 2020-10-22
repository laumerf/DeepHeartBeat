from datetime import datetime
from itertools import chain
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import Progbar



class Encoder(Layer):
    
    def __init__(self, latent_space_dim, hidden_dim, name='encoder', **kwargs):
        
        super(Encoder, self).__init__(name=name, **kwargs)

        self._supports_ragged_inputs = True
                
        self.latent_space_dim = latent_space_dim
        self.hidden_dim = hidden_dim

        self.dense_obs1 = Dense(16, activation='relu', name='dense_obs1')
        
        # Bidirectional LSTM
        self.lstm1 = Bidirectional(LSTM(hidden_dim, return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), name='lstm1')
        self.lstm2 = Bidirectional(LSTM(hidden_dim, activation='tanh', recurrent_activation='sigmoid'), name='lstm2')
        self.dense1 = Dense(latent_space_dim, activation='linear', name='dense1')
        
    def call(self, inputs, training=False):

        """
        inputs is expected to be a RaggedTensor
        """
        
        times, sequences = inputs

        # Convolution
        h = tf.expand_dims(sequences.values, -1)
        h = self.dense_obs1(h)

        h = tf.concat([h, tf.expand_dims(times.values, axis=-1)], axis=1)
        h = tf.RaggedTensor.from_row_splits(h, row_splits=times.row_splits)
        
        # LSTM
        h_padded = h.to_tensor(default_value=0.0)
        h_mask = tf.sequence_mask(h.row_lengths(), maxlen=tf.shape(h_padded)[1])
        h_padded = self.lstm1(h_padded, mask=h_mask, training=training)
        h = self.lstm2(h_padded, mask=h_mask, training=training)

        h = self.dense1(h)
        output = h
        
        return output

class Trajectory(Layer):
    
    def __init__(self, name='trajectory', **kwargs):
        
        super(Trajectory, self).__init__(name=name, **kwargs)
        
    def call(self, inputs):
        
        times, parameters = inputs
        
        pace = parameters[:, 0, None]
        shift = parameters[:, 1, None]

        t = tf.exp(pace)*(times - shift)*2*np.pi
        e1 = tf.sin(t)
        e2 = tf.cos(t)
        l = tf.concat([e1, e2, parameters[:, 2:]], axis=1)

        return l

class Decoder(Layer):
    
    def __init__(self, latent_space_dim, name='decoder', **kwargs):
        
        super(Decoder, self).__init__(name=name, **kwargs)

        self.latent_space_dim = latent_space_dim
        
        self.dense1 = Dense(128, activation='relu', name='dense1', input_shape=(latent_space_dim,))
        self.dense2 = Dense(128, activation='relu', name='dense2')
        self.dense3 = Dense(1, activation='linear', name='dense3')
        
    def call(self, inputs, training=False):
        
        h = inputs
        h = self.dense1(h)
        h = self.dense2(h)
        h = self.dense3(h)
        h = tf.squeeze(h, axis=-1)       
        output = h
        
        return output

class ECGModel(Model):
    
    def __init__(self, latent_space_dim, batch_size=8, hidden_dim=32, learning_rate=5e-4, log_dir=None, name='ecg_model', **kwargs):
        
        super(ECGModel, self).__init__(name=name, **kwargs)

        self._supports_ragged_inputs = True
        
        self.params_dim = latent_space_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        self.encoder = Encoder(latent_space_dim=latent_space_dim, hidden_dim=hidden_dim)
        self.trajectory = Trajectory()
        self.decoder = Decoder(latent_space_dim=latent_space_dim)

        self._sigma_squared = tf.Variable(1.0, trainable=False)
        self._train_stddev = tf.Variable(1.0, trainable=False)

        self.log_dir = log_dir

    def call(self, inputs, training=False):
        
        times, sequences = inputs

        sequences /= self._train_stddev # divide by training data standard deviation

        # subject parameters
        params = self.encoder((times, sequences), training=training)

        times_list = tf.expand_dims(times.values, axis=-1)
        params_list = tf.repeat(params, times.row_lengths(), axis=0)

        # latent trajectories
        latents = self.trajectory((times_list, params_list))
        latent_trajectories = tf.RaggedTensor.from_row_splits(latents, row_splits=times.row_splits)

        # reconstructions
        reconstructions = self.decoder(latents, training=training)
        reconstructions = tf.RaggedTensor.from_row_splits(reconstructions, row_splits=times.row_splits)

        reconstructions *= self._train_stddev # multiply by training data standard deviation
        
        return params, latent_trajectories, reconstructions

    def decode(self, inputs):

        x_rec = self._train_stddev*self.decoder(inputs, training=False)
        return x_rec

    def _parameter_regularisation(self, params):

        return tf.reduce_mean(params[:,2:]**2, axis=-1)
    
    def _reconstruction_error(self, y_true, y_pred):

        mae = tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)
        return mae

    def _heart_rate_regularisation(self, params):

        # penalise heart rates below 40bpm
        hr = 60*tf.exp(params[:,0])
        return tf.nn.relu(40 - hr)
    
    def _loss(self, true_sequences, pred_sequences, params):
        
        # flatten sequences and calculate reconstruction error
        reconstruction_error = self._reconstruction_error(true_sequences, pred_sequences)

        # regularisation
        regularisation = self._parameter_regularisation(params)

        # heart rate regularisation
        hr_regularisation = self._heart_rate_regularisation(params)

        # total loss
        loss = reconstruction_error/self._sigma_squared + regularisation + hr_regularisation
        
        return loss, reconstruction_error, regularisation, hr_regularisation
    
    @tf.function
    def _train_step(self, times, sequences, optimizer):
        
        with tf.GradientTape() as tape:
            params = self.encoder((times, sequences), training=True)
            times_list = tf.expand_dims(times.values, axis=-1)
            params_list = tf.repeat(params, times.row_lengths(), axis=0)
            latents = self.trajectory((times_list, params_list))
            reconstructed_sequences_seq = self.decoder(latents, training=True)
            reconstructed_sequences = tf.RaggedTensor.from_row_splits(reconstructed_sequences_seq, row_splits=times.row_splits)

            loss, reconstruction_error, regularisation, hr_regularisation = self._loss(sequences, reconstructed_sequences, params)

        heart_rates = tf.exp(params[:, 0])*60
            
        # update model weights
        variables = self.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        # logging
        self._train_metrics['loss'](loss)
        self._train_metrics['reconstruction_error'](reconstruction_error)
        self._train_metrics['regularisation'](regularisation)
        self._train_metrics['heart_rate'](heart_rates)
        self._train_metrics['heart_rate_regularisation'](hr_regularisation)
        self._train_metrics['regularisation_strength'](self._sigma_squared)
        
        means = tf.expand_dims(tf.reduce_mean(reconstructed_sequences, axis=1), axis=1)
        reconstruction_var = tf.reduce_mean((reconstructed_sequences - means)**2, axis=1)
        reconstruction_stddev = tf.sqrt(reconstruction_var)
        self._train_metrics['reconstruction_stddev'](reconstruction_stddev)

        # update regularisation weight
        update_rate = 0.99
        self._sigma_squared.assign(update_rate*self._sigma_squared + (1 - update_rate)*tf.reduce_mean(reconstruction_error))
    
    
    @tf.function
    def _evaluate(self, times, sequences):
        
        params = self.encoder((times, sequences), training=False)
        times_list = tf.expand_dims(times.values, axis=-1)
        params_list = tf.repeat(params, times.row_lengths(), axis=0)
        latents = self.trajectory((times_list, params_list))
        reconstructed_sequences = self.decoder(latents, training=False)

        reconstruction_error = self._reconstruction_error(sequences.values, reconstructed_sequences)
        heart_rates = tf.exp(params[:, 0])*60
        
        # logging
        self._validation_metrics['reconstruction_error'](reconstruction_error)
        self._validation_metrics['heart_rate'](heart_rates)
    
    def fit(self, train_data, val_data):

        # determine train data stddev
        self._train_stddev.assign(np.mean([np.std(x['measurements']) for x in train_data]))

        n_train_subjects = len(train_data)        
        n_val_subjects = len(val_data)

        val_batch_size = 1024

        sampling_rate = 4

        def train_dataset_generator():
            n = len(train_data)
            while True:
                ix = np.random.randint(0, n)
                measurements = train_data[ix]['measurements'].astype('float32')

                # use random subsequence of with 450-1199 measurements and take every 2nd measurement
                length = np.random.randint(450, 1200)
                length -= length % sampling_rate
                start = np.random.randint(measurements.shape[0]-length)
                measurements = measurements[start:start+length].copy()
                measurements = np.mean(np.reshape(measurements, [-1, sampling_rate]), axis=-1)
                measurements /= self._train_stddev

                frequency = train_data[ix]['frequency']
                times = np.array([sampling_rate*i/frequency for i in range(measurements.shape[0])], dtype='float32')
                yield times, measurements

        def val_dataset_generator():
            for subject in val_data:
                measurements = subject['measurements'].astype('float32')

                for _ in range(10):
                    length = np.random.randint(450, 1200)
                    length -= length % sampling_rate
                    start = np.random.randint(measurements.shape[0]-length)
                    sub_measurements = measurements[start:start+length].copy()
                    sub_measurements = np.mean(np.reshape(sub_measurements, [-1, sampling_rate]), axis=-1)
                    sub_measurements /= self._train_stddev

                    frequency = subject['frequency']
                    times = np.array([sampling_rate*i/frequency for i in range(sub_measurements.shape[0])], dtype='float32')
                    yield times, sub_measurements
        
        dataset_output_types = (tf.float32, tf.float32)
        dataset_output_shapes = (tf.TensorShape([None]), tf.TensorShape([None]))
        train_dataset = tf.data.Dataset.from_generator(train_dataset_generator, dataset_output_types, dataset_output_shapes)
        val_dataset = tf.data.Dataset.from_generator(val_dataset_generator, dataset_output_types, dataset_output_shapes)

        # messy batching, as RaggedTensors not fully supported by Tensorflow's Dataset
        train_dataset = train_dataset.map(lambda x, y: (tf.expand_dims(x, 0), tf.expand_dims(y, 0)))
        train_dataset = train_dataset.map(lambda x, y: (tf.RaggedTensor.from_tensor(x), tf.RaggedTensor.from_tensor(y)))
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.map(lambda x, y: (tf.squeeze(x, axis=1), tf.squeeze(y, axis=1)))

        val_dataset = val_dataset.map(lambda x, y: (tf.expand_dims(x, 0), tf.expand_dims(y, 0)))
        val_dataset = val_dataset.map(lambda x, y: (tf.RaggedTensor.from_tensor(x), tf.RaggedTensor.from_tensor(y)))
        val_dataset = val_dataset.batch(val_batch_size)
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.map(lambda x, y: (tf.squeeze(x, axis=1), tf.squeeze(y, axis=1)))
        
        # setup summary writers
        train_log_dir = self.log_dir + '/train'
        val_log_dir = self.log_dir + '/validation'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        
        optimizer = Adam(learning_rate=self.learning_rate)

        opt_weights = self.get_weights()
        opt_early_stopping_metric = np.inf
        patience = 30
        count = 0
        epoch = 0
        max_epochs = 200

        # setup train metrics
        self._train_metrics = dict()
        train_metrics = ['loss', 'reconstruction_error', 'regularisation', 'heart_rate', 'heart_rate_regularisation', 'regularisation_strength', 'reconstruction_stddev']
        for metric in train_metrics:
            self._train_metrics[metric] = tf.keras.metrics.Mean(metric)
        
        # setup validation metrics
        self._validation_metrics = dict()
        val_metrics = ['reconstruction_error', 'heart_rate']
        for metric in val_metrics:
            self._validation_metrics[metric] = tf.keras.metrics.Mean(metric)

        while count < patience and epoch <= max_epochs:
            epoch += 1
            count += 1
            steps_per_epoch = 1000

            # train
            for times, sequences in train_dataset.take(steps_per_epoch):
                self._train_step(times, sequences, optimizer)
            
            # log train metrics
            with train_summary_writer.as_default():
                for metric in train_metrics:
                    tf.summary.scalar(metric, self._train_metrics[metric].result(), step=epoch)

            # print train metrics
            train_strings = ['%s: %.3e' % (k, self._train_metrics[k].result()) for k in train_metrics]
            print('{}: Epoch {}: Train: '.format(datetime.now(), epoch) + ' - '.join(train_strings), flush=True)

            # validate
            for times, sequences in val_dataset:
                self._evaluate(times, sequences)

            # log validation metrics
            with val_summary_writer.as_default():
                for metric in val_metrics:
                    tf.summary.scalar(metric, self._validation_metrics[metric].result(), step=epoch)

            # reset early stopping counter on improvement
            if self._validation_metrics['reconstruction_error'].result() < opt_early_stopping_metric:
                opt_early_stopping_metric = self._validation_metrics['reconstruction_error'].result()
                opt_weights = self.get_weights()
                count = 0

            # print validation metrics
            val_strings = ['%s: %.3e' % (k, self._validation_metrics[k].result()) for k in val_metrics]
            print('{}: Epoch {}: Validation: '.format(datetime.now(), epoch) + ' - '.join(val_strings), flush=True)
            
            # reset metrics
            for metric in chain(self._train_metrics.values(), self._validation_metrics.values()):
                metric.reset_states()

        # reset to optimal weights
        self.set_weights(opt_weights)
