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
# Module trainer
# Description - Time series forecast model trainer using keras/tensorflow.
#
# Author Tuan Le (tuan.t.lei@gmail.com)
#
# ------------------------------------------------------------------------

import os
import time
import math
import argparse
import warnings

import numpy as np

from matplotlib import pyplot
import seaborn as sns

from keras.models import load_model
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras import backend as K

from src.utils.keras_model_encoder import Encoder
from src.utils.ts_dataset_loader import load_and_prepare_ts_dataset

# ------------------------------------------------------------------------

MIN_FRAME_SIZE = 1

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_VALIDATION_SPLIT = 0.2

DEFAULT_INITIAL_LRATE = 0.002
DEFAULT_LRATE_DROP = 0.15
LRATE_EPOCH_DROP_PERCENTAGE = 0.05

sns.set(context='notebook', style='white', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)

# ------------------------------------------------------------------------


def train(
    tsf_model,
    training_ts_dataset_x, training_ts_dataset_y, *,
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    validation_split=DEFAULT_VALIDATION_SPLIT,
    initial_lrate=DEFAULT_INITIAL_LRATE,
    lrate_drop=DEFAULT_LRATE_DROP,
    verbose=True,
):
    """
    Decription: Start model training
    Arguments:
        tsf_model
        training_ts_dataset_x, training_ts_dataset_y
        epochs
        batch_size
        verbose
    Returns:
    """

    if epochs < 1:
        epochs = DEFAULT_EPOCHS
        warnings.warn('train - Number of training epochs must be > 0.', Warning)
        warnings.warn('Using %d as default.' % DEFAULT_EPOCHS, Warning)

    if batch_size < 1:
        batch_size = DEFAULT_BATCH_SIZE
        warnings.warn('train - Training batch size must be > 0.', Warning)
        warnings.warn('Using %d as default.' % DEFAULT_BATCH_SIZE, Warning)

    if validation_split < 0 or validation_split > 0.5:
        validation_split = DEFAULT_VALIDATION_SPLIT
        warnings.warn('train - Training vs validating split ratio size must be >= 0 and <= 0.5.', Warning)
        warnings.warn('Using %d as default.' % DEFAULT_VALIDATION_SPLIT, Warning)

    if initial_lrate <= 0 or initial_lrate > 1:
        initial_lrate = DEFAULT_INITIAL_LRATE
        warnings.warn('train - Learning rate must be > 0 and < 1.', Warning)
        warnings.warn('Using %d as default.' % DEFAULT_INITIAL_LRATE, Warning)

    if lrate_drop < 0 or lrate_drop > 1:
        lrate_drop = DEFAULT_LRATE_DROP
        warnings.warn('train - Learning rate drop or decay must be >= 0 and < 1.', Warning)
        warnings.warn('Using %d as default.' % DEFAULT_LRATE_DROP, Warning)

    if verbose:
        print('Training batch size: %d' % batch_size)
        print('Training epochs: %d' % epochs)

    start_timestamp = time.time()

    callbacks = []

    # learning rate anealling callbacks
    if lrate_drop > 0:
        lrate_epoch_drop = math.ceil(LRATE_EPOCH_DROP_PERCENTAGE * epochs)

        def lrate_step_decay(epoch):
            """
            Decription: Learning rate step decap based on epochs
            Arguments:
                epochs
            Returns:
            """
            lrate = initial_lrate * math.pow(1 - lrate_drop, math.floor((1 + epoch) / lrate_epoch_drop))
            print('Learning rate: %f' % lrate)
            return lrate
        callbacks.append(LearningRateScheduler(lrate_step_decay))
    else:

        def lrate_step_const(epoch):
            """
            Decription: Constant learning rate
            Arguments:
                epochs
            Returns:
            """
            lrate = initial_lrate
            print('Learning rate: %f' % lrate)
            return lrate
        callbacks.append(LearningRateScheduler(lrate_step_const))
    # early stopping callbacks
    callbacks.append(EarlyStopping(monitor='loss', patience=math.ceil(epochs * 0.15), verbose=0, mode='auto'))

    # start training...
    history = tsf_model.fit(
        training_ts_dataset_x, training_ts_dataset_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=verbose and 0 or 1,
        shuffle=False
    )
    done_timestamp = time.time()

    if verbose:
        print('Model training time: %0.3fs' % (done_timestamp - start_timestamp))

    return (tsf_model, history)

# ------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ds', '--ts_dataset', type=str, required=True, help='Input time series training dataset filename')
    parser.add_argument('-m', '--model', type=str, required=False, help='Input model filename')
    parser.add_argument('-w', '--window_size', type=int, required=True, help='Current to past window in number of time step periods')
    parser.add_argument('-f', '--feature_size', type=int, required=True, help='Number of features')
    parser.add_argument('-p', '--prediction_size', type=int, required=True, help='Number of predictions')
    parser.add_argument('-ep', '--epochs', type=int, default=DEFAULT_EPOCHS, required=False, help='Number of training epochs. Default is 50')
    parser.add_argument('-b', '--batch_size', type=int, default=DEFAULT_BATCH_SIZE, required=False, help='Size of training batch. Default is 32')
    parser.add_argument('-vs', '--validation_split', type=float, default=DEFAULT_VALIDATION_SPLIT, required=False, help='Training vs validating split ratio. Default is 0.2')
    parser.add_argument('-ilr', '--initial_learning_rate', type=float, default=DEFAULT_INITIAL_LRATE, required=False, help='Initial learning rate. Default is 0.02')
    parser.add_argument('-lrd', '--learning_rate_drop', type=int, default=DEFAULT_LRATE_DROP, required=False, help='Learning rate step drop/decay. Default is 0.15')
    parser.add_argument('-bem', '--binary_encoded_model', action='store_true', default=True, required=False, help='Encode output model HDF5 to BIN format')
    parser.add_argument('-v', '--verbose', action='store_true', default=False, required=False, help='Verbose mode - show summary and plots')
    args = parser.parse_args()

    if args.ts_dataset is None or len(args.ts_dataset) == 0:
        raise Exception('ERROR: trainer - Time series dataset file path/name must not be empty.')
    elif args.model is None or len(args.model) == 0:
        raise Exception('ERROR: trainer - Model file path/name must not be empty.')

    window_size = args.window_size
    feature_size = args.feature_size
    prediction_size = args.prediction_size
    epochs = args.epochs
    batch_size = args.batch_size
    validation_split = args.validation_split
    initial_lrate = args.initial_learning_rate
    lrate_drop = args.learning_rate_drop

    input_ts_dataset_filepath, input_ts_dataset_filename = os.path.split(args.ts_dataset)
    tsf_model_filepath, tsf_model_filename = os.path.split(args.model)
    (tsf_model_filename, tsf_model_ext) = os.path.splitext(tsf_model_filename)
    binary_encoded_model = args.binary_encoded_model
    verbose = args.verbose

    if validation_split < 0 or validation_split > 0.5:
        validation_split = DEFAULT_VALIDATION_SPLIT
        warnings.warn('train - Training vs validating split ratio size must be >= 0 and <= 0.5.', Warning)
        warnings.warn('Using %d as default.' % DEFAULT_VALIDATION_SPLIT, Warning)

    (ts_dataset,
     ts_dataset_feature_size, ts_dataset_prediction_size,
     ts_dataset_feature_labels, ts_dataset_prediction_labels,
     ts_dataset_x, ts_dataset_y,
     ts_normed_dataset_x, ts_normed_dataset_y,
     ts_dataset_x_scaler, ts_dataset_y_scaler) = load_and_prepare_ts_dataset(
        os.path.join(input_ts_dataset_filepath, input_ts_dataset_filename),
        cp_frame_size=window_size,
        cf_frame_size=1,
        feature_size=feature_size,
        prediction_size=prediction_size,
        normalize=True,
        verbose=verbose
    )

    enable_validation = validation_split > 0
    sample_size = ts_dataset.shape[0]
    if enable_validation:
        training_sample_size = math.floor(sample_size * (1 - validation_split))
        validation_sample_size = sample_size - training_sample_size
    else:
        training_sample_size = sample_size

    if verbose:
        figure1 = pyplot.figure(figsize=(8, 6))
        figure1.suptitle('Training Time Series Dataset', fontsize=16)
        for j in range(1, feature_size + 1):
            pyplot.subplot(feature_size + prediction_size + 1, 1, j)
            pyplot.plot(ts_dataset[:training_sample_size, j - 1], color='deepskyblue')
            pyplot.title(ts_dataset_feature_labels[j - 1], fontsize=10, x=-0.15, y=0.25, loc='right', rotation=0)
            pyplot.grid()
            pyplot.tick_params(axis='x', bottom='off', labelbottom='off')
        for j in range(feature_size, feature_size + prediction_size):
            pyplot.subplot(feature_size + prediction_size + 1, 1, j + 1)
            pyplot.plot(ts_dataset[:training_sample_size, j], linewidth=2, color='salmon')
            pyplot.title(ts_dataset_prediction_labels[j - feature_size], fontsize=10, x=-0.15, y=0.25, loc='right', rotation=0)
            pyplot.grid()
            pyplot.tick_params(axis='x', bottom='off', labelbottom='off')
        pyplot.subplots_adjust(top=0.85, bottom=0.05, left=0.3, right=0.95, hspace=0.2, wspace=0.35)
        pyplot.tick_params(axis='x', bottom='on', labelbottom='on')
        pyplot.xlabel('Sample')
        pyplot.show()

        if enable_validation:
            figure2 = pyplot.figure(figsize=(8, 6))
            figure2.suptitle('Validation Time Series Dataset', fontsize=16)
            for j in range(1, feature_size + 1):
                pyplot.subplot(feature_size + prediction_size + 1, 1, j)
                pyplot.plot(ts_dataset[training_sample_size + window_size:, j - 1], color='deepskyblue')
                pyplot.title(ts_dataset_feature_labels[j - 1], fontsize=10, x=-0.15, y=0.25, loc='right', rotation=0)
                pyplot.grid()
                pyplot.tick_params(axis='x', bottom='off', labelbottom='off')

            for j in range(feature_size, feature_size + prediction_size):
                pyplot.subplot(feature_size + prediction_size + 1, 1, j + 1)
                pyplot.plot(ts_dataset[training_sample_size + window_size:, j], linewidth=2, color='salmon')
                pyplot.title(ts_dataset_prediction_labels[j - feature_size], fontsize=10, x=-0.15, y=0.25, loc='right', rotation=0)
                pyplot.grid()
                pyplot.tick_params(axis='x', bottom='off', labelbottom='off')

            pyplot.subplots_adjust(top=0.85, bottom=0.05, left=0.3, right=0.95, hspace=0.2, wspace=0.35)
            pyplot.tick_params(axis='x', bottom='on', labelbottom='on')
            pyplot.xlabel('Sample')
            pyplot.show()

    if os.path.isfile(os.path.join(tsf_model_filepath, tsf_model_filename + '_fext' + tsf_model_ext)):
        tsf_model_feature_extractor = load_model(os.path.join(tsf_model_filepath, tsf_model_filename + '_fext' + tsf_model_ext))
        print('Begin training feature extractor model!')
        (tsf_model_feature_extractor, history) = train(
            tsf_model_feature_extractor,
            ts_normed_dataset_x, ts_normed_dataset_x,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            initial_lrate=initial_lrate,
            lrate_drop=lrate_drop,
            verbose=verbose
        )

        get_feature_extractor_output = K.function(
            [tsf_model_feature_extractor.get_layer('feature_extractor').get_layer('autoencoder').get_layer('autoencoder_input_layer').input],
            [tsf_model_feature_extractor.get_layer('feature_extractor').get_layer('autoencoder').get_layer('autoencoder_latent_output_layer').output])
        feature_extractor_output = get_feature_extractor_output([ts_normed_dataset_x])[0]

        if verbose:
            figure2 = pyplot.figure(figsize=(8, 6))
            figure2.suptitle('Feature Extractor Model Training Results', fontsize=16)
            pyplot.yscale('linear')
            pyplot.ylabel('Loss')
            pyplot.xlabel('Epoch')
            pyplot.plot(history.history['loss'], label='Training Loss', color='orangered')
            pyplot.plot(history.history['val_loss'], label='Validation Loss', color='deepskyblue')
            pyplot.grid()
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
            pyplot.subplots_adjust(top=0.85, bottom=0.1, hspace=0.2, wspace=1.05)
            pyplot.show()

            figure3 = pyplot.figure(figsize=(8, 6))
            figure3.suptitle('Extracted Features', fontsize=16)
            for i in range(feature_extractor_output.shape[2]):
                pyplot.subplot(feature_extractor_output.shape[2], 1, i + 1)
                pyplot.plot(feature_extractor_output[:, 0, i], color='deepskyblue')
                pyplot.grid()
                pyplot.tick_params(axis='x', bottom='off', labelbottom='off')
            pyplot.tick_params(axis='x', bottom='on', labelbottom='on')
            pyplot.xlabel('Sample')
            pyplot.show()

        ts_normed_dataset_x = np.concatenate((ts_normed_dataset_x, feature_extractor_output), axis=2)

    tsf_model = load_model(os.path.join(tsf_model_filepath, tsf_model_filename + tsf_model_ext))
    print('Begin training time series model!')
    (tsf_model, history) = train(
        tsf_model,
        ts_normed_dataset_x, ts_normed_dataset_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        initial_lrate=initial_lrate,
        lrate_drop=lrate_drop,
        verbose=verbose
    )

    if verbose:
        figure4 = pyplot.figure(figsize=(8, 6))
        figure4.suptitle('Model Training Results', fontsize=16)
        if 'accuracy' in history.history:
            pyplot.subplot(2, 1, 1)
            pyplot.yscale('linear')
            pyplot.ylabel('Loss')
            pyplot.xlabel('Epoch')
            pyplot.plot(history.history['loss'], label='Training Loss', color='orangered')
            pyplot.plot(history.history['val_loss'], label='Validation Loss', color='deepskyblue')
            pyplot.grid()
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
            pyplot.subplot(2, 1, 2)
            pyplot.yscale('linear')
            pyplot.ylabel('Acc')
            pyplot.xlabel('Epoch')
            pyplot.plot(history.history['accuracy'], linewidth=2, label='Training Acc', color='orangered')
            if 'val_accuracy' in history.history:
                pyplot.plot(history.history['val_accuracy'], linewidth=2, label='Validation Acc', color='deepskyblue')
            pyplot.grid()
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
        elif 'acc' in history.history:
            pyplot.subplot(2, 1, 1)
            pyplot.yscale('linear')
            pyplot.ylabel('Loss')
            pyplot.xlabel('Epoch')
            pyplot.plot(history.history['loss'], label='Training Loss', color='orangered')
            pyplot.plot(history.history['val_loss'], label='Validation Loss', color='deepskyblue')
            pyplot.grid()
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
            pyplot.subplot(2, 1, 2)
            pyplot.yscale('linear')
            pyplot.ylabel('Acc')
            pyplot.xlabel('Epoch')
            pyplot.plot(history.history['acc'], linewidth=2, label='Training Acc', color='orangered')
            if 'val_acc' in history.history:
                pyplot.plot(history.history['val_acc'], linewidth=2, label='Validation Acc', color='deepskyblue')
            pyplot.grid()
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
        elif 'categorical_accuracy' in history.history:
            pyplot.subplot(2, 1, 1)
            pyplot.yscale('linear')
            pyplot.ylabel('Loss')
            pyplot.xlabel('Epoch')
            pyplot.plot(history.history['loss'], label='Training Loss', color='orangered')
            pyplot.plot(history.history['val_loss'], label='Validation Loss', color='deepskyblue')
            pyplot.grid()
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
            pyplot.subplot(2, 1, 2)
            pyplot.yscale('linear')
            pyplot.ylabel('Acc')
            pyplot.xlabel('Epoch')
            pyplot.plot(history.history['categorical_accuracy'], linewidth=2, label='Training Acc', color='orangered')
            if 'val_categorical_accuracy' in history.history:
                pyplot.plot(history.history['val_categorical_accuracy'], linewidth=2, label='Validation Acc', color='deepskyblue')
            pyplot.grid()
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
        elif 'binary_accuracy' in history.history:
            pyplot.subplot(2, 1, 1)
            pyplot.yscale('linear')
            pyplot.ylabel('Loss')
            pyplot.xlabel('Epoch')
            pyplot.plot(history.history['loss'], label='Training Loss', color='orangered')
            pyplot.plot(history.history['val_loss'], label='Validation Loss', color='deepskyblue')
            pyplot.grid()
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
            pyplot.subplot(2, 1, 2)
            pyplot.yscale('linear')
            pyplot.ylabel('Acc')
            pyplot.xlabel('Epoch')
            pyplot.plot(history.history['binary_accuracy'], linewidth=2, label='Training Acc', color='orangered')
            if 'val_binary_accuracy' in history.history:
                pyplot.plot(history.history['val_binary_accuracy'], linewidth=2, label='Validation Acc', color='deepskyblue')
            pyplot.grid()
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
        else:
            pyplot.yscale('linear')
            pyplot.ylabel('Loss')
            pyplot.xlabel('Epoch')
            pyplot.plot(history.history['loss'], label='Training Loss', color='orangered')
            pyplot.plot(history.history['val_loss'], label='Validation Loss', color='deepskyblue')
            pyplot.grid()
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
        pyplot.subplots_adjust(top=0.85, bottom=0.1, hspace=0.2, wspace=1.05)
        pyplot.show()

    if enable_validation:
        prediction_ts_normed_dataset_y = tsf_model.predict(
            ts_normed_dataset_x[training_sample_size:],
            verbose=1
        )

        validation_ts_dataset_y = ts_dataset_y[training_sample_size:]
        prediction_ts_dataset_y = ts_dataset_y_scaler.inverse_transform(prediction_ts_normed_dataset_y)
        validation_ts_dataset_y = validation_ts_dataset_y.reshape((ts_dataset_y[training_sample_size:].shape[0], 1, prediction_size))
        prediction_ts_dataset_y = prediction_ts_dataset_y.reshape((prediction_ts_dataset_y.shape[0], 1, prediction_size))

        figure5 = pyplot.figure(figsize=(8, 6))
        figure5.suptitle('Time Series Prediction Results', fontsize=16)
        for j in range(1, feature_size + 1):
            pyplot.subplot(feature_size + prediction_size + 1, 1, j)
            pyplot.plot(ts_dataset[training_sample_size + window_size:, j - 1], color='deepskyblue')
            pyplot.title(ts_dataset_feature_labels[j - 1], fontsize=10, x=-0.15, y=0.25, loc='right', rotation=0)
            pyplot.grid()
            pyplot.tick_params(axis='x', bottom='off', labelbottom='off')

        for j in range(1, prediction_size + 1):
            pyplot.subplot(feature_size + prediction_size + 1, 1, feature_size + j)
            pyplot.plot(validation_ts_dataset_y[:, 0, j - 1], linewidth=2, label='Expected', color='salmon')
            pyplot.plot(prediction_ts_dataset_y[:, 0, j - 1], label='Predicted', linestyle='--', marker='o', markersize=3, markeredgecolor='none', markerfacecolor='firebrick', color='orangered')
            pyplot.title(ts_dataset_prediction_labels[j - 1], fontsize=10, x=-0.15, y=0.25, loc='right', rotation=0)
            pyplot.grid()
            pyplot.tick_params(axis='x', bottom='off', labelbottom='off')
            pyplot.legend(loc='best', bbox_to_anchor=(1, 0.5))
        pyplot.subplots_adjust(top=0.85, bottom=0.05, left=0.25, right=0.85, hspace=0.2, wspace=1.05)
        pyplot.tick_params(axis='x', bottom='on', labelbottom='on')
        pyplot.xlabel('Sample')
        pyplot.show()

    # serialize tsf_model to JSON
    tsf_model_json = tsf_model.to_json()
    with open(os.path.join(tsf_model_filepath, tsf_model_filename + '.json'), 'w') as json_file:
        json_file.write(tsf_model_json)
    # serialize tsf_model to HDF5
    tsf_model.save(os.path.join(tsf_model_filepath, tsf_model_filename + '.h5'))
    tsf_model.save_weights(os.path.join(tsf_model_filepath, tsf_model_filename + '-weights' + '.h5'))

    if binary_encoded_model:
        encoder = Encoder(
            os.path.join(tsf_model_filepath, tsf_model_filename + '.h5'),
            name=tsf_model_filename,
            quantize=False
        )
        encoder.serialize()
        encoder.save()
    print('Saved trained HDF5 time series ml model to %s' % tsf_model_filepath)
