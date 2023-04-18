# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Models.py

import util
import os
import numpy as np
import layers
import logging
import tensorflow as tf
import loss_plot_callback
import file_read_log_callback
from my_loss_functions import *


# Speech Denoising Wavenet Model


class DenoisingWavenet():

    def __init__(self, config, load_checkpoint=None, input_length=None, target_field_length=None,
                 print_model_summary=False):

        self.config = config
        self.verbosity = config['training']['verbosity']

        self.num_stacks = self.config['model']['num_stacks']
        if type(self.config['model']['dilations']) is int:
            self.dilations = [
                2 ** i for i in range(0, self.config['model']['dilations'] + 1)]
        elif type(self.config['model']['dilations']) is list:
            self.dilations = self.config['model']['dilations']

        self.num_condition_classes = config['dataset']['num_condition_classes']

        self.condition_input_length = self.get_condition_input_length(
            self.config['model']['condition_encoding'])
        self.receptive_field_length = util.compute_receptive_field_length(config['model']['num_stacks'], self.dilations,
                                                                          config['model']['filters']['lengths']['res'],
                                                                          1)

        if input_length is not None:
            self.input_length = input_length
            self.target_field_length = self.input_length - \
                                       (self.receptive_field_length - 1)
        if target_field_length is not None:
            self.target_field_length = target_field_length
            self.input_length = self.receptive_field_length + \
                                (self.target_field_length - 1)
        else:
            self.target_field_length = config['model']['target_field_length']
            self.input_length = self.receptive_field_length + \
                                (self.target_field_length - 1)

        self.target_padding = config['model']['target_padding']
        self.padded_target_field_length = int(
            self.target_field_length + 2 * self.target_padding)
        self.half_target_field_length = int(self.target_field_length / 2)
        self.half_receptive_field_length = int(self.receptive_field_length / 2)
        self.num_residual_blocks = len(self.dilations) * self.num_stacks
        self.activation = tf.keras.layers.Activation('relu')
        self.samples_of_interest_indices = self.get_padded_target_field_indices()
        self.target_sample_indices = self.get_target_field_indices()

        self.optimizer = self.get_optimizer()
        self.out_1_loss = self.get_out_1_loss()
        self.out_2_loss = self.get_out_2_loss()
        self.metrics = self.get_metrics()
        self.epoch_num = 0
        self.checkpoints_path = ''
        self.samples_path = ''
        self.history_filename = ''

        self.config['model']['num_residual_blocks'] = self.num_residual_blocks
        self.config['model']['receptive_field_length'] = self.receptive_field_length
        self.config['model']['input_length'] = self.input_length
        self.config['model']['target_field_length'] = self.target_field_length

        if config['model'].get('no_conditioning') is True:
            self.build_model = self.build_model_without_conditioning
        else:
            self.build_model = self.build_model_with_conditioning

        self.use_dropout = bool(config['model']['dropout']['use'])

        self.model = self.setup_model(load_checkpoint, print_model_summary)

    def setup_model(self, load_checkpoint=None, print_model_summary=False):

        self.checkpoints_path = os.path.join(
            self.config['training']['path'], 'checkpoints')
        self.samples_path = os.path.join(
            self.config['training']['path'], 'samples')
        self.history_filename = 'history_' + self.config['training']['path'][
                                             self.config['training']['path'].rindex('/') + 1:] + '.csv'

        model = self.build_model()

        if os.path.exists(self.checkpoints_path) and util.dir_contains_files(self.checkpoints_path):

            if load_checkpoint is not None:
                last_checkpoint_path = load_checkpoint
                self.epoch_num = 0
            else:
                checkpoints = os.listdir(self.checkpoints_path)
                checkpoints.sort(key=lambda x: os.stat(
                    os.path.join(self.checkpoints_path, x)).st_mtime)
                last_checkpoint = checkpoints[-1]
                last_checkpoint_path = os.path.join(
                    self.checkpoints_path, last_checkpoint)
                self.epoch_num = int(last_checkpoint[11:16])
            print('Loading model from epoch: %d' % self.epoch_num)
            model.load_weights(last_checkpoint_path)

        else:
            print('Building new model...')

            if not os.path.exists(self.config['training']['path']):
                os.mkdir(self.config['training']['path'])

            if not os.path.exists(self.checkpoints_path):
                os.mkdir(self.checkpoints_path)

            self.epoch_num = 0

        if not os.path.exists(self.samples_path):
            os.mkdir(self.samples_path)

        if print_model_summary:
            model.summary()

        model.compile(optimizer=self.optimizer,
                      loss={'data_output_1': self.out_1_loss, 'data_output_2': self.out_2_loss}, metrics=self.metrics)
        self.config['model']['num_params'] = model.count_params()

        config_path = os.path.join(
            self.config['training']['path'], 'config.json')
        if not os.path.exists(config_path):
            util.pretty_json_dump(self.config, config_path)

        if print_model_summary:
            util.pretty_json_dump(self.config)
        return model

    def get_optimizer(self):

        return tf.keras.optimizers.legacy.Adam(learning_rate=self.config['optimizer']['lr'],
                                               decay=self.config['optimizer']['decay'],
                                               epsilon=self.config['optimizer']['epsilon'])

    def get_out_1_loss(self):

        if self.config['training']['loss']['out_1']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        spec_loss, spec_conv_loss, weighted_spec_loss, rms_loss = self.prepare_additional_loss_functions("out_1")

        return lambda y_true, y_pred: self.config['training']['loss']['out_1'][
                                          'weight'] * util.l1_l2_combined_loss(
            y_true, y_pred, self.config['training']['loss']['out_1']['l1'],
            self.config['training']['loss']['out_1']['l2']) + \
                                      spec_loss(y_true, y_pred) + spec_conv_loss(y_true, y_pred) + \
                                      weighted_spec_loss(y_true, y_pred) + rms_loss(y_true, y_pred)

    def get_out_2_loss(self):

        if self.config['training']['loss']['out_2']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        spec_loss, spec_conv_loss, weighted_spec_loss, rms_loss = self.prepare_additional_loss_functions("out_2")

        return lambda y_true, y_pred: self.config['training']['loss']['out_2'][
                                          'weight'] * util.l1_l2_combined_loss(
            y_true, y_pred, self.config['training']['loss']['out_2']['l1'],
            self.config['training']['loss']['out_2']['l2']) + spec_loss(y_true, y_pred) + \
                                      spec_conv_loss(y_true, y_pred) + \
                                      weighted_spec_loss(y_true, y_pred) + \
                                      rms_loss(y_true, y_pred)

    def prepare_additional_loss_functions(self, output):
        def spec_loss(y_true, y_pred):
            return 0

        def spec_conv_loss(y_true, y_pred):
            return 0

        def weighted_spec_loss(y_true, y_pred):
            return 0

        def rms_loss(y_true, y_pred):
            return 0

        nfft = self.config['training']['loss'][output]['spec_param']['nfft']
        frame_len = self.config['training']['loss'][output]['spec_param']['frame_len']
        frame_step = self.config['training']['loss'][output]['spec_param']['frame_step']
        center_freq = self.config['training']['loss'][output]['weighted_spectrogram']['center_frequency']
        std = self.config['training']['loss'][output]['weighted_spectrogram']['std']

        if self.config['training']['loss'][output]['spectrogram']['weight'] != 0:
            def spec_loss(y_true, y_pred):
                return spectrogram_loss(y_true, y_pred, frame_len, frame_step, nfft) * \
                    self.config['training']['loss']['out_2']['spectrogram']['weight']

        if self.config['training']['loss'][output]['spectral_convergence']['weight'] != 0:
            def spec_conv_loss(y_true, y_pred):
                return spectral_convergence_loss(y_true, y_pred, frame_len, frame_step, nfft) * \
                    self.config['training']['loss'][output]['spectral_convergence']['weight']

        if self.config['training']['loss'][output]['weighted_spectrogram']['weight'] != 0:
            weights = gaussian_spectrogram_weights(nfft, center_freq, std, self.config['dataset']['sample_rate'])

            def weighted_spec_loss(y_true, y_pred):
                return weighted_spectrogram_loss(y_true, y_pred, weights, frame_len, frame_step, nfft) * \
                    self.config['training']['loss']['out_2']['weighted_spectrogram']['weight']

        if self.config['training']['loss'][output]['rms']['weight'] != 0:
            def rms_loss(y_true, y_pred):
                return tf_rms_loss(y_true, y_pred) * self.config['training']['loss']['out_2']['rms']['weight']

        return spec_loss, spec_conv_loss, weighted_spec_loss, rms_loss

    def get_callbacks(self):

        return [
            tf.keras.callbacks.ReduceLROnPlateau(patience=self.config['training']['early_stopping_patience'] / 2,
                                                 cooldown=self.config['training']['early_stopping_patience'] / 4,
                                                 verbose=1),
            tf.keras.callbacks.EarlyStopping(patience=self.config['training']['early_stopping_patience'], verbose=1,
                                             monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.checkpoints_path, 'checkpoint.{epoch:05d}-{val_loss:.3f}.hdf5')),
            tf.keras.callbacks.CSVLogger(os.path.join(self.config['training']['path'], self.history_filename),
                                         append=True),
            loss_plot_callback.LossPlotCallback(os.path.join(
                self.config['training']['path'], 'loss_plots.png')),
        ]

    def fit_model(self, train_set_generator, num_train_samples, test_set_generator, num_test_samples, num_epochs):

        print('Fitting model with %d training samples and %d test samples...' %
              (num_train_samples, num_test_samples))

        # self.model.fit_generator(train_set_generator,
        #                          num_train_samples,
        #                          epochs=num_epochs,
        #                          validation_data=test_set_generator,
        #                          validation_steps=num_test_samples,
        #                          callbacks=self.get_callbacks(),
        #                          verbose=self.verbosity,
        #                          initial_epoch=self.epoch_num)

        with open("./file_read_log.txt", "w") as frlf:
            frlf.write("")

        batch_size = self.config["training"]["batch_size"]

        self.model.fit(train_set_generator,
                       epochs=num_epochs,
                       steps_per_epoch=int(num_train_samples / batch_size),
                       validation_data=test_set_generator,
                       validation_steps=int(num_test_samples / batch_size),
                       callbacks=self.get_callbacks(),
                       verbose=self.verbosity,
                       initial_epoch=self.epoch_num)

    def denoise_batch(self, inputs):
        return self.model.predict_on_batch(inputs)

    def get_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(int(target_sample_index - self.half_target_field_length),
                     int(target_sample_index + self.half_target_field_length + 1))

    def get_padded_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(int(target_sample_index - self.half_target_field_length - self.target_padding),
                     int(target_sample_index + self.half_target_field_length + self.target_padding + 1))

    def get_target_sample_index(self):
        return int(np.floor(self.input_length / 2.0))

    def get_metrics(self):

        return [
            tf.keras.metrics.mean_absolute_error,
            self.valid_mean_absolute_error
        ]

    def valid_mean_absolute_error(self, y_true, y_pred):
        return tf.keras.backend.mean(
            tf.keras.backend.abs(y_true[:, 1:-2] - y_pred[:, 1:-2]))

    def get_condition_input_length(self, representation):

        if representation == 'binary':
            return int(np.max((np.ceil(np.log2(self.num_condition_classes)), 1)))
        else:
            return self.num_condition_classes

    def build_model_with_conditioning(self):

        data_input = tf.keras.layers.Input(
            shape=(int(self.input_length),),
            name='data_input')

        condition_input = tf.keras.layers.Input(shape=(int(self.condition_input_length),),
                                                name='condition_input')

        data_expanded = layers.AddSingletonDepth()(data_input)
        data_input_target_field_length = layers.Slice(
            (slice(
                self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1), Ellipsis),
            (self.padded_target_field_length, 1),
            name='data_input_target_field_length')(data_expanded)

        data_out = tf.keras.layers.Convolution1D(self.config['model']['filters']['depths']['res'],
                                                 self.config['model']['filters']['lengths']['res'], padding='same',
                                                 use_bias=False,
                                                 name='initial_causal_conv')(data_expanded)

        condition_out = tf.keras.layers.Dense(self.config['model']['filters']['depths']['res'],
                                              name='initial_dense_condition',
                                              use_bias=False)(condition_input)
        condition_out = tf.keras.layers.RepeatVector(int(self.input_length),
                                                     name='initial_condition_repeat')(condition_out)
        data_out = tf.keras.layers.Add(name='initial_data_condition_merge')(
            [data_out, condition_out])

        skip_connections = []
        res_block_i = 0
        for stack_i in range(self.num_stacks):
            layer_in_stack = 0
            for dilation in self.dilations:
                res_block_i += 1
                data_out, skip_out = self.dilated_residual_block(data_out, condition_input, res_block_i, layer_in_stack,
                                                                 dilation, stack_i)
                if skip_out is not None:
                    skip_connections.append(skip_out)
                layer_in_stack += 1

        data_out = tf.keras.layers.Add()(skip_connections)
        data_out = self.activation(data_out)

        data_out = tf.keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][0],
                                                 self.config['model']['filters']['lengths']['final'][0],
                                                 padding='same',
                                                 use_bias=False)(data_out)

        condition_out = tf.keras.layers.Dense(self.config['model']['filters']['depths']['final'][0],
                                              use_bias=False,
                                              name='penultimate_conv_1d_condition')(condition_input)

        condition_out = tf.keras.layers.RepeatVector(self.padded_target_field_length,
                                                     name='penultimate_conv_1d_condition_repeat')(condition_out)

        data_out = tf.keras.layers.Add(name='penultimate_conv_1d_condition_merge')([
            data_out, condition_out])

        data_out = self.activation(data_out)
        data_out = tf.keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][1],
                                                 self.config['model']['filters']['lengths']['final'][1], padding='same',
                                                 use_bias=False)(data_out)

        condition_out = tf.keras.layers.Dense(self.config['model']['filters']['depths']['final'][1], use_bias=False,
                                              name='final_conv_1d_condition')(condition_input)

        condition_out = tf.keras.layers.RepeatVector(self.padded_target_field_length,
                                                     name='final_conv_1d_condition_repeat')(condition_out)

        data_out = tf.keras.layers.Add(name='final_conv_1d_condition_merge')([
            data_out, condition_out])

        data_out = tf.keras.layers.Convolution1D(1, 1)(data_out)

        data_out_speech = data_out
        data_out_noise = layers.Subtract(name='subtract_layer')(
            [data_input_target_field_length, data_out_speech])

        data_out_speech = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2),
                                                 output_shape=lambda shape: (shape[0], shape[1]), name='data_output_1')(
            data_out_speech)

        data_out_noise = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2),
                                                output_shape=lambda shape: (shape[0], shape[1]), name='data_output_2')(
            data_out_noise)

        return tf.keras.models.Model(inputs=[data_input, condition_input], outputs=[data_out_speech, data_out_noise])

    def dilated_residual_block(self, data_x, condition_x, res_block_i, layer_i, dilation, stack_i):

        original_x = data_x

        # Data sub-block
        # data_out = tf.keras.layers.AtrousConvolution1D(2 * self.config['model']['filters']['depths']['res'],
        #                                             self.config['model']['filters']['lengths']['res'],
        #                                             atrous_rate=dilation, padding='same',
        #                                             use_bias=False,
        #                                             name='res_%d_dilated_conv_d%d_s%d' % (
        #                                             res_block_i, dilation, stack_i),
        #                                             activation=None)(data_x)

        data_out = tf.keras.layers.Convolution1D(2 * self.config['model']['filters']['depths']['res'],
                                                 self.config['model']['filters']['lengths']['res'],
                                                 dilation_rate=dilation, padding='same',
                                                 use_bias=False,
                                                 name='res_%d_dilated_conv_d%d_s%d' % (
                                                     res_block_i, dilation, stack_i),
                                                 activation=None)(data_x)

        data_out_1 = layers.Slice(
            (Ellipsis, slice(
                0, self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']
            ['filters']['depths']['res']),
            name='res_%d_data_slice_1_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        data_out_2 = layers.Slice(
            (Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                             2 * self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']
            ['filters']['depths']['res']),
            name='res_%d_data_slice_2_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        # Condition sub-block
        condition_out = tf.keras.layers.Dense(2 * self.config['model']['filters']['depths']['res'],
                                              name='res_%d_dense_condition_%d_s%d' % (
                                                  res_block_i, layer_i, stack_i),
                                              use_bias=False)(condition_x)

        condition_out = tf.keras.layers.Reshape((self.config['model']['filters']['depths']['res'], 2),
                                                name='res_%d_condition_reshape_d%d_s%d' % (
                                                    res_block_i, dilation, stack_i))(condition_out)

        condition_out_1 = layers.Slice((Ellipsis, 0), (self.config['model']['filters']['depths']['res'],),
                                       name='res_%d_condition_slice_1_d%d_s%d' % (
                                           res_block_i, dilation, stack_i))(condition_out)

        condition_out_2 = layers.Slice((Ellipsis, 1), (self.config['model']['filters']['depths']['res'],),
                                       name='res_%d_condition_slice_2_d%d_s%d' % (
                                           res_block_i, dilation, stack_i))(condition_out)

        condition_out_1 = tf.keras.layers.RepeatVector(int(self.input_length),
                                                       name='res_%d_condition_repeat_1_d%d_s%d' % (
                                                           res_block_i, dilation, stack_i))(condition_out_1)
        condition_out_2 = tf.keras.layers.RepeatVector(int(self.input_length),
                                                       name='res_%d_condition_repeat_2_d%d_s%d' % (
                                                           res_block_i, dilation, stack_i))(condition_out_2)

        data_out_1 = tf.keras.layers.Add(name='res_%d_merge_1_d%d_s%d' %
                                              (res_block_i, dilation, stack_i))([data_out_1, condition_out_1])
        data_out_2 = tf.keras.layers.Add(name='res_%d_merge_2_d%d_s%d' % (
            res_block_i, dilation, stack_i))([data_out_2, condition_out_2])

        tanh_out = tf.keras.layers.Activation('tanh')(data_out_1)
        sigm_out = tf.keras.layers.Activation('sigmoid')(data_out_2)

        data_x = tf.keras.layers.Multiply(name='res_%d_gated_activation_%d_s%d' % (res_block_i, layer_i, stack_i))(
            [tanh_out, sigm_out])

        data_x = tf.keras.layers.Convolution1D(
            self.config['model']['filters']['depths']['res'] +
            self.config['model']['filters']['depths']['skip'], 1,
            padding='same', use_bias=False)(data_x)

        res_x = layers.Slice((Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
                             (self.input_length,
                              self.config['model']['filters']['depths']['res']),
                             name='res_%d_data_slice_3_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = layers.Slice((Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                                               self.config['model']['filters']['depths']['res'] +
                                               self.config['model']['filters']['depths']['skip'])),
                              (self.input_length,
                               self.config['model']['filters']['depths']['skip']),
                              name='res_%d_data_slice_4_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = layers.Slice((slice(self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1),
                               Ellipsis),
                              (self.padded_target_field_length,
                               self.config['model']['filters']['depths']['skip']),
                              name='res_%d_keep_samples_of_interest_d%d_s%d' % (res_block_i, dilation, stack_i))(skip_x)

        res_x = tf.keras.layers.Add()([original_x, res_x])

        return res_x, skip_x

    def build_model_without_conditioning(self):

        data_input = tf.keras.layers.Input(
            shape=(int(self.input_length),),
            name='data_input')

        data_expanded = layers.AddSingletonDepth()(data_input)
        data_input_target_field_length = layers.Slice(
            (slice(
                self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1), Ellipsis),
            (self.padded_target_field_length, 1),
            name='data_input_target_field_length')(data_expanded)

        data_out = tf.keras.layers.Convolution1D(self.config['model']['filters']['depths']['res'],
                                                 self.config['model']['filters']['lengths']['res'], padding='same',
                                                 use_bias=False,
                                                 name='initial_causal_conv')(data_expanded)

        skip_connections = []
        res_block_i = 0
        for stack_i in range(self.num_stacks):
            layer_in_stack = 0
            for dilation in self.dilations:
                res_block_i += 1
                data_out, skip_out = self.dilated_residual_block_without_conditioning(data_out, res_block_i,
                                                                                      layer_in_stack, dilation, stack_i)
                if skip_out is not None:
                    skip_connections.append(skip_out)
                layer_in_stack += 1

        data_out = tf.keras.layers.Add()(skip_connections)
        data_out = self.activation(data_out)

        data_out = tf.keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][0],
                                                 self.config['model']['filters']['lengths']['final'][0],
                                                 padding='same',
                                                 use_bias=False)(data_out)

        data_out = self.activation(data_out)
        data_out = tf.keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][1],
                                                 self.config['model']['filters']['lengths']['final'][1], padding='same',
                                                 use_bias=False)(data_out)

        data_out = tf.keras.layers.Convolution1D(1, 1)(data_out)

        data_out_speech = data_out
        data_out_noise = layers.Subtract(name='subtract_layer')(
            [data_input_target_field_length, data_out_speech])

        data_out_speech = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2),
                                                 output_shape=lambda shape: (shape[0], shape[1]), name='data_output_1')(
            data_out_speech)

        data_out_noise = tf.keras.layers.Lambda(lambda x: tf.keras.backend.squeeze(x, 2),
                                                output_shape=lambda shape: (shape[0], shape[1]), name='data_output_2')(
            data_out_noise)

        return tf.keras.models.Model(inputs=[data_input], outputs=[data_out_speech, data_out_noise])

    def dilated_residual_block_without_conditioning(self, data_x, res_block_i, layer_i, dilation, stack_i):

        original_x = data_x

        # Data sub-block
        # data_out = tf.keras.layers.AtrousConvolution1D(2 * self.config['model']['filters']['depths']['res'],
        #                                             self.config['model']['filters']['lengths']['res'],
        #                                             atrous_rate=dilation, padding='same',
        #                                             use_bias=False,
        #                                             name='res_%d_dilated_conv_d%d_s%d' % (
        #                                             res_block_i, dilation, stack_i),
        #                                             activation=None)(data_x)

        data_out = tf.keras.layers.Convolution1D(2 * self.config['model']['filters']['depths']['res'],
                                                 self.config['model']['filters']['lengths']['res'],
                                                 dilation_rate=dilation, padding='same',
                                                 use_bias=False,
                                                 name='res_%d_dilated_conv_d%d_s%d' % (
                                                     res_block_i, dilation, stack_i),
                                                 activation=None)(data_x)

        data_out_1 = layers.Slice(
            (Ellipsis, slice(
                0, self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']
            ['filters']['depths']['res']),
            name='res_%d_data_slice_1_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        data_out_2 = layers.Slice(
            (Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                             2 * self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']
            ['filters']['depths']['res']),
            name='res_%d_data_slice_2_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        tanh_out = tf.keras.layers.Activation('tanh')(data_out_1)
        sigm_out = tf.keras.layers.Activation('sigmoid')(data_out_2)

        data_x = tf.keras.layers.Multiply(name='res_%d_gated_activation_%d_s%d' % (res_block_i, layer_i, stack_i))(
            [tanh_out, sigm_out])

        data_x = tf.keras.layers.Convolution1D(
            self.config['model']['filters']['depths']['res'] +
            self.config['model']['filters']['depths']['skip'], 1,
            padding='same', use_bias=False)(data_x)

        res_x = layers.Slice((Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
                             (self.input_length,
                              self.config['model']['filters']['depths']['res']),
                             name='res_%d_data_slice_3_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = layers.Slice((Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                                               self.config['model']['filters']['depths']['res'] +
                                               self.config['model']['filters']['depths']['skip'])),
                              (self.input_length,
                               self.config['model']['filters']['depths']['skip']),
                              name='res_%d_data_slice_4_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = layers.Slice((slice(self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1),
                               Ellipsis),
                              (self.padded_target_field_length,
                               self.config['model']['filters']['depths']['skip']),
                              name='res_%d_keep_samples_of_interest_d%d_s%d' % (res_block_i, dilation, stack_i))(skip_x)

        res_x = tf.keras.layers.Add()([original_x, res_x])

        return res_x, skip_x
