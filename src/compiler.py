#!/usr/bin/env python
#
# Copyright 2016-present Tuan Le.
#
# Licensed under the MIT License.
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://opensource.org/licenses/mit-license.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------
#
# Module compiler
# Description - Time series forecast regression / classification model compiler using keras/tensorflow.
#
# Author Tuan Le (tuan.t.lei@gmail.com)
#
# ------------------------------------------------------------------------

import os
import time
import argparse
import math
import warnings

from keras.models import Sequential, Model
from keras.layers.core import Activation, Dense, Dropout, RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers import Input

from keras import regularizers
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax
from keras.initializers import Constant
from keras import metrics

from src.utils.keras_model_encoder import Encoder

# ------------------------------------------------------------------------

MIN_WINDOW_SIZE = 2
MIN_DENSE_LAYER_COUNT = 2

MIN_FRAME_SIZE = 1
MIN_FEATURE_SIZE = 1
MIN_PREDICTION_SIZE = 1

# ------------------------------------------------------------------------


def compile_feature_extractor(
    *,
    extracted_feature_size=0,
    optimizer='adam',
    window_size=MIN_FRAME_SIZE,
    feature_size=MIN_FEATURE_SIZE,
    verbose=True
):
    """
    Decription: Compling a simple feature extractor using autoencoder.
        The autoencoder will compress the original features by 75%. The extracted features are the compress latent output tensor.
        See readme for architecture details.
    Arguments:
        extracted_feature_size - number of features extracted out via autoencoder
        optimizer - optimizer choises: sgd, adam, ...
        window_size - window size from current to past
        feature_size - number of features
    """

    if extracted_feature_size == 0 or extracted_feature_size >= feature_size:
        warnings.warn('compile_feature_extractor - Number of extracted features must be =< %d and >= 1. Skip compiling feature extractor model.' % feature_size, Warning)
        return (None, 0)

    if window_size < MIN_FRAME_SIZE:
        window_size = MIN_FRAME_SIZE
        warnings.warn('compile_feature_extractor - Current to past window size must be >= %d.' % MIN_FRAME_SIZE, Warning)
        warnings.warn('Using %d as default.' % MIN_FRAME_SIZE, Warning)

    if feature_size < MIN_FEATURE_SIZE:
        feature_size = MIN_FEATURE_SIZE
        warnings.warn('compile_feature_extractor - Number of features must be >= %d.' % MIN_FEATURE_SIZE, Warning)
        warnings.warn('Using %d as default.' % MIN_FEATURE_SIZE, Warning)

    encoder_input_unit_size = feature_size
    decoder_output_unit_size = encoder_input_unit_size

    encoder_hidden_unit_size = extracted_feature_size
    decoder_hidden_unit_size = encoder_hidden_unit_size

    tsf_feature_extractor = Sequential()
    lstm_encoder_input = Input((window_size, feature_size), name='autoencoder_input_layer')
    lstm_encoded_hidden_output = LSTM(
        name='encoder_input_layer',
        units=encoder_input_unit_size,
        return_sequences=True,
        recurrent_activation='hard_sigmoid',
        kernel_initializer='glorot_normal',
        activation='tanh'
    )(lstm_encoder_input)
    lstm_encoded_output = LSTM(
        name='encoder_output_layer',
        units=encoder_hidden_unit_size,
        return_sequences=False,
        recurrent_activation='hard_sigmoid',
        kernel_initializer='glorot_normal',
        activation='sigmoid'  # NOTE: encoder output activation must be sigmoid to ensure that the output is normalized between [0, 1]
    )(lstm_encoded_hidden_output)
    lstm_decoded_input = RepeatVector(window_size, name='autoencoder_latent_output_layer')(lstm_encoded_output)
    lstm_decoded_hidden_output = LSTM(
        name='decoder_input_layer',
        units=decoder_hidden_unit_size,
        return_sequences=True,
        recurrent_activation='hard_sigmoid',
        kernel_initializer='glorot_normal',
        activation='tanh'
    )(lstm_decoded_input)
    lstm_decoded_output = LSTM(
        name='decoder_output_layer',
        units=decoder_output_unit_size,
        return_sequences=True,
        recurrent_activation='hard_sigmoid',
        kernel_initializer='glorot_normal',
        activation='linear'
    )(lstm_decoded_hidden_output)
    lstm_encoder = Model(lstm_encoder_input, lstm_encoded_output, name='encoder')
    lstm_autoencoder = Model(lstm_encoder_input, lstm_decoded_output, name='autoencoder')
    tsf_feature_extractor.add(
        Model(lstm_encoder.inputs, lstm_autoencoder(lstm_encoder.inputs), name='feature_extractor')
    )

    if optimizer == 'sgd':
        optimizer = SGD(lr=0)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr=0)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(lr=0)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(lr=0)
    elif optimizer == 'adam':
        optimizer = Adam(lr=0)
    elif optimizer == 'adamax':
        optimizer = Adamax(lr=0)

    start_timestamp = time.time()
    tsf_feature_extractor.compile(
        loss='mean_squared_error',
        optimizer=optimizer
    )
    done_timestamp = time.time()

    if verbose:
        print('Model feature extractor compilation time: %0.3fs' % (done_timestamp - start_timestamp))
        print(lstm_autoencoder.summary())

    return (tsf_feature_extractor, extracted_feature_size)


def compile(
    objective,
    recurrent_layer_size,
    dense_layer_size, *,
    optimizer='adam',
    window_size=MIN_FRAME_SIZE,
    feature_size=MIN_FEATURE_SIZE,
    extracted_feature_size=0,
    prediction_size=MIN_PREDICTION_SIZE,
    dropout_rate=0,
    input_l1_l2=0,
    batch_normalize=False,
    verbose=True
):
    """
    Decription: Compling time series model. See readme for architecture details.
    Arguments:
        objective - objective choices: can be regression (mse, mae, ...) or classification (binary or categorical)
        recurrent_layer_size - number of recurrent (lstm) layers
        dense_layer_size - number of dense fully connected layers
        optimizer - optimizer choises: sgd, adam, ...
        window_size - window size from current to past
        feature_size - number of features
        extracted_feature_size - number of extracted feature size
        prediction_size - number of predictions
        dropout_rate - set dropout propbability
        input_l1_l2 - set input layer L1L2 regularization
        batch_normalize - enable batch normalize
        verbose - enable verbose mode
    """

    if recurrent_layer_size < MIN_WINDOW_SIZE:
        recurrent_layer_size = MIN_WINDOW_SIZE
        warnings.warn('compile - Number of recurrent layer must be >= %d.' % MIN_WINDOW_SIZE, Warning)
        warnings.warn('Using %d as default.' % MIN_WINDOW_SIZE, Warning)

    if dense_layer_size < MIN_DENSE_LAYER_COUNT:
        dense_layer_size = MIN_DENSE_LAYER_COUNT
        warnings.warn('compile - Number of dense layer must be >= %d.' % MIN_DENSE_LAYER_COUNT, Warning)
        warnings.warn('Using %d as default.' % MIN_DENSE_LAYER_COUNT, Warning)

    if window_size < MIN_FRAME_SIZE:
        window_size = MIN_FRAME_SIZE
        warnings.warn('compile - Current to past window size must be >= %d.' % MIN_FRAME_SIZE, Warning)
        warnings.warn('Using %d as default.' % MIN_FRAME_SIZE, Warning)

    if feature_size < MIN_FEATURE_SIZE:
        feature_size = MIN_FEATURE_SIZE
        warnings.warn('compile - Number of features must be >= %d.' % MIN_FEATURE_SIZE, Warning)
        warnings.warn('Using %d as default.' % MIN_FEATURE_SIZE, Warning)

    if extracted_feature_size >= feature_size:
        extracted_feature_size = 0
        warnings.warn('compile_feature_extractor - Number of extracted features must be < %d.' % feature_size, Warning)
        warnings.warn('Using 0 as default.', Warning)

    if prediction_size < MIN_PREDICTION_SIZE:
        prediction_size = MIN_PREDICTION_SIZE
        warnings.warn('compile - Number of prediction must be >= %d.' % MIN_PREDICTION_SIZE, Warning)
        warnings.warn('Using %d as default.' % MIN_PREDICTION_SIZE, Warning)

    if dropout_rate < 0 or dropout_rate > 1:
        dropout_rate = 0
        warnings.warn('compile - Hidden layers dropout rate should be between >= 0 and <= 1.', Warning)
        warnings.warn('Using 0 as default.', Warning)

    if input_l1_l2 < 0 or input_l1_l2 > 1:
        input_l1_l2 = 0
        warnings.warn('compile - Input layer L1 & L2 regularizer value should be between >= 0 and <= 1.', Warning)
        warnings.warn('Using 0 as default.', Warning)

    input_unit_size = (2 * window_size) * (feature_size + extracted_feature_size)
    output_unit_size = prediction_size

    hidden_recurrent_unit_size = input_unit_size

    hidden_dense_unit_size = hidden_recurrent_unit_size
    if dense_layer_size > 2:
        hidden_dense_first_unit_size = math.ceil(hidden_dense_unit_size * 1.2)
    else:
        hidden_dense_first_unit_size = math.ceil(hidden_dense_unit_size * 0.75)
    hidden_inner_dense_unit_size = math.ceil(hidden_dense_first_unit_size * 0.75)
    hidden_outer_dense_unit_size = math.ceil(hidden_inner_dense_unit_size * 0.5)
    if hidden_outer_dense_unit_size < output_unit_size * 4:
        hidden_outer_dense_unit_size = output_unit_size * 4

    tsf_model = Sequential()

    for i in range(recurrent_layer_size):
        # Input lstm layer
        if i == 0:
            if input_l1_l2 > 0:
                tsf_model.add(
                    LSTM(
                        name='lstm_input_layer',
                        units=input_unit_size,
                        recurrent_activation='hard_sigmoid',
                        kernel_initializer='glorot_normal',
                        kernel_regularizer=regularizers.l1_l2(input_l1_l2),
                        return_sequences=True,
                        input_shape=(window_size, feature_size + extracted_feature_size)
                    )
                )
            else:
                tsf_model.add(
                    LSTM(
                        name='lstm_input_layer',
                        units=input_unit_size,
                        recurrent_activation='hard_sigmoid',
                        kernel_initializer='glorot_normal',
                        return_sequences=True,
                        input_shape=(window_size, feature_size + extracted_feature_size)
                    )
                )
            tsf_model.add(Activation('tanh'))
            if batch_normalize:
                tsf_model.add(BatchNormalization())
        # Hidden lstm layers
        elif i == recurrent_layer_size - 1:
            tsf_model.add(
                LSTM(
                    name='lstm_output_layer',
                    units=hidden_recurrent_unit_size,
                    recurrent_activation='hard_sigmoid',
                    kernel_initializer='glorot_normal',
                    return_sequences=False,
                )
            )
            tsf_model.add(Activation('tanh'))
            if batch_normalize:
                tsf_model.add(BatchNormalization())
        else:
            tsf_model.add(
                LSTM(
                    name='lstm_hidden_layer_%d' % i,
                    units=hidden_recurrent_unit_size,
                    recurrent_activation='hard_sigmoid',
                    kernel_initializer='glorot_normal',
                    return_sequences=True,
                )
            )
            tsf_model.add(Activation('tanh'))
            if batch_normalize:
                tsf_model.add(BatchNormalization())
    # Hidden dense layers
    for i in range(dense_layer_size):
        if i == 0:
            tsf_model.add(
                Dense(
                    name='dense_input_layer',
                    units=hidden_dense_unit_size,
                    kernel_initializer='glorot_normal',
                    bias_initializer=Constant(0.01)
                )
            )
            tsf_model.add(Activation('relu'))
            if batch_normalize:
                tsf_model.add(BatchNormalization())
            if dropout_rate > 0:
                tsf_model.add(Dropout(rate=dropout_rate))
        elif i == 1:
            tsf_model.add(
                Dense(
                    name='dense_hidden_layer_1',
                    units=hidden_dense_first_unit_size,
                    kernel_initializer='glorot_normal',
                    bias_initializer=Constant(0.01)
                )
            )
            tsf_model.add(Activation('relu'))
            if batch_normalize:
                tsf_model.add(BatchNormalization())
            if dropout_rate > 0:
                tsf_model.add(Dropout(rate=dropout_rate))
        elif i == dense_layer_size - 1:
            tsf_model.add(
                Dense(
                    name='dense_hidden_layer_%d' % i,
                    units=hidden_outer_dense_unit_size,
                    kernel_initializer='glorot_normal',
                    bias_initializer=Constant(0.01)
                )
            )
            tsf_model.add(Activation('relu'))
            if batch_normalize:
                tsf_model.add(BatchNormalization())
            if dropout_rate > 0:
                tsf_model.add(Dropout(rate=dropout_rate))
        else:
            tsf_model.add(
                Dense(
                    name='dense_hidden_layer_%d' % i,
                    units=hidden_inner_dense_unit_size,
                    kernel_initializer='glorot_normal',
                    bias_initializer=Constant(0.01)
                )
            )
            tsf_model.add(Activation('relu'))
            if batch_normalize:
                tsf_model.add(BatchNormalization())
            if dropout_rate > 0:
                tsf_model.add(Dropout(rate=dropout_rate))
    # Output layer
    tsf_model.add(
        Dense(
            name='dense_output_layer',
            units=output_unit_size,
            kernel_initializer='glorot_normal',
            bias_initializer=Constant(0.01)
        )
    )

    if optimizer == 'sgd':
        optimizer = SGD(lr=0)
    elif optimizer == 'rmsprop':
        optimizer = RMSprop(lr=0)
    elif optimizer == 'adagrad':
        optimizer = Adagrad(lr=0)
    elif optimizer == 'adadelta':
        optimizer = Adadelta(lr=0)
    elif optimizer == 'adam':
        optimizer = Adam(lr=0)
    elif optimizer == 'adamax':
        optimizer = Adamax(lr=0)

    if objective == 'mean_squared_error' or objective == 'mse':
        tsf_model.add(Activation('linear'))
        start_timestamp = time.time()
        tsf_model.compile(
            loss='mean_squared_error',
            optimizer=optimizer
        )
    elif objective == 'mean_absolute_error' or objective == 'mae':
        tsf_model.add(Activation('linear'))
        start_timestamp = time.time()
        tsf_model.compile(
            loss='mean_absolute_error',
            optimizer=optimizer
        )
    elif objective == 'logcosh_loss' or objective == 'lcl':
        tsf_model.add(Activation('linear'))
        start_timestamp = time.time()
        tsf_model.compile(
            loss='logcosh',
            optimizer=optimizer
        )
    elif objective == 'binary_crossentropy' or objective == 'bce':
        tsf_model.add(Activation('sigmoid'))
        start_timestamp = time.time()
        tsf_model.compile(
            loss='binary_crossentropy',
            metrics=[metrics.binary_accuracy],
            optimizer=optimizer
        )
    elif objective == 'category_classification' or objective == 'cce':
        tsf_model.add(Activation('softmax'))
        start_timestamp = time.time()
        tsf_model.compile(
            loss='categorical_crossentropy',
            metrics=[metrics.categorical_accuracy],
            optimizer=optimizer
        )
    else:
        raise Exception('ERROR: compile - Unknow learning objective %s.' % objective)

    done_timestamp = time.time()

    if verbose:
        print('Time series model compilation time: %0.3fs' % (done_timestamp - start_timestamp))
        print(tsf_model.summary())

    return tsf_model

# ------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=False, help='Output compiled model filename')
    parser.add_argument('-o', '--objective', type=str, required=True, help='Learning objective, regression or classification')
    parser.add_argument('-opt', '--optimizer', type=str, default='adam', required=False, help='Optimization method. Default is Adam')
    parser.add_argument('-rl', '--recurrent_layer_size', type=int, required=True, help='Number of recurrent layers')
    parser.add_argument('-dl', '--dense_layer_size', type=int, required=True, help='Number of dense layers')
    parser.add_argument('-w', '--window_size', type=int, required=True, help='Current to past window in number of time step periods')
    parser.add_argument('-f', '--feature_size', type=int, required=True, help='Number of features')
    parser.add_argument('-ef', '--extracted_feature_size', type=int, default=0, required=False, help='Number of features extraction via autoencoder. Default is 0')
    parser.add_argument('-p', '--prediction_size', type=int, required=True, help='Number of predictions')
    parser.add_argument('-dr', '--dropout_rate', type=float, default=0, required=False, help='Hidden layers dropout rate. Default is 0')
    parser.add_argument('-ir', '--input_l1_l2', type=float, default=0, required=False, help='Input layer L1 & L2 regularization. Default is 0')
    parser.add_argument('-bn', '--batch_normalize', action='store_true', default=False, required=False, help='Enable batch normalization at every layer output')
    parser.add_argument('-bem', '--binary_encoded_model', action='store_true', default=True, required=False, help='Encode output model HDF5 to BIN format')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, required=False, help='Verbose mode - show summary and plots')
    args = parser.parse_args()

    if args.model is None or len(args.model) == 0:
        raise Exception('ERROR: compiler - Model file path/name must not be empty.')

    extracted_feature_size = args.extracted_feature_size
    objective = args.objective
    recurrent_layer_size = args.recurrent_layer_size
    dense_layer_size = args.dense_layer_size
    optimizer = args.optimizer
    window_size = args.window_size
    feature_size = args.feature_size
    prediction_size = args.prediction_size
    dropout_rate = args.dropout_rate
    input_l1_l2 = args.input_l1_l2
    batch_normalize = args.batch_normalize
    (tsf_model_filepath, tsf_model_filename) = os.path.split(args.model)
    binary_encoded_model = args.binary_encoded_model
    verbose = args.verbose

    if extracted_feature_size >= 1:
        (tsf_model_feature_extractor, extracted_feature_size) = compile_feature_extractor(
            extracted_feature_size=extracted_feature_size,
            optimizer=optimizer,
            window_size=window_size,
            feature_size=feature_size
        )
        # serialize tsf_model_feature_extractor to JSON
        tsf_model_feature_extractor_json = tsf_model_feature_extractor.to_json()
        with open(os.path.join(tsf_model_filepath, tsf_model_filename + '_fext.json'), 'w') as json_file:
            json_file.write(tsf_model_feature_extractor_json)
        # serialize tsf_model to HDF5
        tsf_model_feature_extractor.save(os.path.join(tsf_model_filepath, tsf_model_filename + '_fext.h5'))

        if binary_encoded_model:
            encoder = Encoder(
                os.path.join(tsf_model_filepath, tsf_model_filename + '_fext.h5'),
                name=tsf_model_filename + '_fext',
                quantize=False
            )
            encoder.serialize()
            encoder.save()
    else:
        extracted_feature_size = 0

    tsf_model = compile(
        objective,
        recurrent_layer_size,
        dense_layer_size,
        optimizer=optimizer,
        window_size=window_size,
        feature_size=feature_size,
        extracted_feature_size=extracted_feature_size,
        prediction_size=prediction_size,
        dropout_rate=dropout_rate,
        input_l1_l2=input_l1_l2,
        batch_normalize=batch_normalize,
        verbose=verbose
    )

    # serialize tsf_model to JSON
    tsf_model_json = tsf_model.to_json()
    with open(os.path.join(tsf_model_filepath, tsf_model_filename + '.json'), 'w') as json_file:
        json_file.write(tsf_model_json)
    # serialize tsf_model to HDF5
    tsf_model.save(os.path.join(tsf_model_filepath, tsf_model_filename + '.h5'))

    if binary_encoded_model:
        encoder = Encoder(
            os.path.join(tsf_model_filepath, tsf_model_filename + '.h5'),
            name=tsf_model_filename,
            quantize=False
        )
        encoder.serialize()
        encoder.save()
    print('Saved compiled HDF5 time series ml model to %s' % tsf_model_filepath)
