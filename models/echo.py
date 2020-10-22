from datetime import datetime
from itertools import chain
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import Progbar

from utils import *


# utility function
        
def random_subsequence_start_end(N, min_length):
    """
    Random start and end indices for a random subsequence for a sequences of length N.
    """
    n_subsequences = ((N-min_length)**2 + N - min_length)/2
    start = np.random.choice(np.arange(N - min_length), p=np.arange(N - min_length, 0, step=-1)/n_subsequences)
    end = np.random.choice(np.arange(start+min_length, N))
    return start, end


# Model definition

class Conv(Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, batch_normalisation=False, name=None):
        super(Conv, self).__init__()

        self.activation = activation
        self.batch_normalisation = batch_normalisation

        self.conv = Conv2D(filters, kernel_size, activation=None, strides=strides, padding=padding, name=name, use_bias=(not batch_normalisation))
        
        if self.batch_normalisation:
            self.bn = BatchNormalization()

    def call(self, inputs, training=False):

        h = inputs
        h = self.conv(h)
        if self.batch_normalisation:
            h = self.bn(h, training=training)
        h = Activation(self.activation)(h)
        return h

class DeConv(Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, batch_normalisation=False, output_padding=None, name=None):
        super(DeConv, self).__init__()

        self.activation = activation
        self.batch_normalisation = batch_normalisation

        self.conv = Conv2DTranspose(filters, kernel_size, strides=strides, activation=None, padding=padding, output_padding=output_padding, use_bias=(not batch_normalisation), name=name)
        
        if self.batch_normalisation:
            self.bn = BatchNormalization()

    def call(self, inputs, training=False):

        h = inputs
        h = self.conv(h)
        if self.batch_normalisation:
            h = self.bn(h, training=training)
        h = Activation(self.activation)(h)
        return h

class Encoder(Layer):
    
    def __init__(self, latent_space_dim, hidden_dim, input_noise=None, name='encoder', **kwargs):
        
        super(Encoder, self).__init__(name=name, **kwargs)

        self._supports_ragged_inputs = True
                
        self.latent_space_dim = latent_space_dim
        self.input_noise = input_noise
        
        # Convolution
        self.conv1 = Conv(8, 4, 2, activation='relu', batch_normalisation=False, name='conv1')
        self.conv2 = Conv(16, 4, 2, activation='relu', batch_normalisation=False, name='conv2')
        self.conv3 = Conv(16, 4, 2, activation='relu', batch_normalisation=False, name='conv3')
        self.conv4 = Conv(16, 4, 2, activation='relu', batch_normalisation=False, name='conv4')
        
        # Elman Net
        self.lstm = Bidirectional(LSTM(hidden_dim, activation='tanh', recurrent_activation='sigmoid'))
        self.dense = Dense(latent_space_dim, activation='linear', name='dense')
        
    def call(self, inputs, training=False):

        """
        inputs is expected to be a RaggedTensor
        """
        
        times, sequences = inputs
        
        # Convolution
        h = tf.expand_dims(sequences.values, -1)

        # Add Gaussian noise
        if self.input_noise is not None:
            h = GaussianNoise(self.input_noise)(h, training=training)

        h = self.conv1(h, training=training)
        h = self.conv2(h, training=training)
        h = self.conv3(h, training=training)
        h = self.conv4(h, training=training)
        h = tf.reshape(h, shape=[-1, 5*5*16])

        h = tf.concat([h, tf.expand_dims(times.values, axis=-1)], axis=1)
        h = tf.RaggedTensor.from_row_splits(h, row_splits=times.row_splits)
                
        # RNN
        h_padded = h.to_tensor(default_value=0.0)
        h_mask = tf.sequence_mask(h.row_lengths(), maxlen=tf.shape(h_padded)[1])
        h = self.lstm(h_padded, mask=h_mask, training=training)
        h = self.dense(h)
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
        
        self.dense = Dense(5*5*16, activation='relu', name='dense', input_shape=(latent_space_dim,))
        self.deconv1 = DeConv(32, 4, 2, activation='relu', batch_normalisation=False, name='deconv1')
        self.deconv2 = DeConv(16, 4, 2, activation='relu', batch_normalisation=False, name='deconv2')
        self.deconv3 = DeConv(8, 4, 2, activation='relu', batch_normalisation=False, output_padding=1, name='deconv3')
        self.deconv4 = DeConv(1, 4, 2, activation='sigmoid', batch_normalisation=False, name='deconv4')
        
    def call(self, inputs, training=False):
        
        h = inputs
        h = self.dense(h)

        h = tf.reshape(h, shape=[-1, 5, 5, 16])
        h = self.deconv1(h, training=training)
        h = self.deconv2(h, training=training)
        h = self.deconv3(h, training=training)
        h = self.deconv4(h, training=training)

        h = tf.squeeze(h, axis=-1)
        
        output = h
        
        return output

class EchocardioModel(Model):
    
    def __init__(self, latent_space_dim, batch_size=8, hidden_dim=32, learning_rate=5e-4, input_noise=None, log_dir=None, name='cardio', **kwargs):
        
        super(EchocardioModel, self).__init__(name=name, **kwargs)

        self._supports_ragged_inputs = True
        
        self.params_dim = latent_space_dim
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.input_noise = input_noise
        
        self.encoder = Encoder(latent_space_dim=latent_space_dim, hidden_dim=hidden_dim, input_noise=input_noise)
        self.trajectory = Trajectory()
        self.decoder = Decoder(latent_space_dim=latent_space_dim)

        self._sigma_squared = tf.Variable(1.0, trainable=False)

        self.log_dir = log_dir

    def call(self, inputs, training=False):
        
        times, sequences = inputs

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
        
        return params, latent_trajectories, reconstructions

    def decode(self, inputs):

        x_rec = self.decoder(inputs, training=False)
        return x_rec

    def _parameter_regularisation(self, params):

        return tf.reduce_mean(params[:,2:]**2, axis=-1)
    
    def _reconstruction_error(self, y_true, y_pred):
        
        mse = tf.reduce_mean((y_true - y_pred)**2, axis=[-3, -2, -1])
        return mse
    
    def _loss(self, true_sequences, pred_sequences, params):
        
        # flatten sequences and calculate reconstruction error
        reconstruction_error = self._reconstruction_error(true_sequences, pred_sequences)

        # regularisation
        regularisation = self._parameter_regularisation(params)

        # total loss
        loss = reconstruction_error/self._sigma_squared + regularisation
        
        return loss, reconstruction_error, regularisation
    
    @tf.function
    def _train_step(self, times, sequences, optimizer):
        
        with tf.GradientTape() as tape:
            params = self.encoder((times, sequences), training=True)
            times_list = tf.expand_dims(times.values, axis=-1)
            params_list = tf.repeat(params, times.row_lengths(), axis=0)
            latents = self.trajectory((times_list, params_list))
            reconstructed_sequences_seq = self.decoder(latents, training=True)
            reconstructed_sequences = tf.RaggedTensor.from_row_splits(reconstructed_sequences_seq, row_splits=times.row_splits)

            loss, reconstruction_error, regularisation = self._loss(sequences, reconstructed_sequences, params)

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
        self._train_metrics['regularisation_strength'](self._sigma_squared)
        
        # reconstruction_stddev = tf.reduce_mean(tf.math.reduce_std(reconstructed_sequences, axis=1, keepdims=False).values)
        means = tf.expand_dims(tf.reduce_mean(reconstructed_sequences, axis=1), axis=1)
        reconstruction_var = tf.reduce_mean(tf.reduce_mean((reconstructed_sequences - means)**2, axis=1))
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
    
    def fit(self, train_files, val_files):
        
        n_train_subjects = len(train_files)        
        n_val_subjects = len(val_files)

        val_batch_size = 16

        # Dataset generator
        def train_dataset_generator(files):

            n = len(files)
            while True:
                ix = np.random.randint(0, n)
                filepath = files[ix]
                data = np.load(filepath)
                times = data['times']
                frames = data['frames']

                minimum_length = np.ceil(2.0/times[1]).astype('int')
                if not times.shape[0] <= minimum_length:
                    start, end = random_subsequence_start_end(times.shape[0], minimum_length)
                    times = times[start:end] - times[start]
                    frames = frames[start:end]
                
                frames = (frames/255).astype('float32')
                yield times, frames

        def val_dataset_generator(files):

            for filepath in files:
                data = np.load(filepath)
                times = data['times']
                frames = data['frames']
                frames = (frames/255).astype('float32')
                yield times, frames
        
        dataset_output_types = (tf.float32, tf.float32)
        dataset_output_shapes = (tf.TensorShape([None]), tf.TensorShape([None, 112, 112]))
        train_dataset = tf.data.Dataset.from_generator(train_dataset_generator, dataset_output_types, dataset_output_shapes, args=(train_files,))
        val_dataset = tf.data.Dataset.from_generator(val_dataset_generator, dataset_output_types, dataset_output_shapes, args=(val_files,))

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
        
        optimizer = Adam(lr=self.learning_rate)

        opt_weights = self.get_weights()
        opt_early_stopping_metric = np.inf
        patience = 10
        count = 0
        epoch = 0
        max_epochs = 200

        # setup train metrics
        self._train_metrics = dict()
        train_metrics = ['loss', 'reconstruction_error', 'regularisation', 'heart_rate', 'regularisation_strength', 'reconstruction_stddev']
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
            steps_per_epoch = 200

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