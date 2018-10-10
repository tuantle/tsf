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
# Module ts_dataset_loader
# Description - Time series dataset loader
#
# Author Tuan Le (tuan.t.lei@gmail.com)
#
# ------------------------------------------------------------------------

import re
import warnings

from pandas import concat, read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------------------

MIN_FRAME_SIZE = 1
MIN_FEATURE_SIZE = 1
MIN_PREDICTION_SIZE = 1

# ------------------------------------------------------------------------


def frame(
    ts_dataset,
    ts_dataset_feature_labels,
    ts_dataset_prediction_labels, *,
    cp_frame_size=1,
    cf_frame_size=1,
    dropnan=True
):
    """
    Decription: Frame a time series dataset into current to past and current to future frame window.
    [[X - CPFn], [Y - CPFn], ..., [X - CPF1], [Y - CPF1], [X], [Y], [X + CFF1], [Y + CFF1], ..., [X + CFFn], [Y + CFFn]]
    where X are input features, Y are output predictions.
    CPF = current to past frame size
    CFF = current to future frame size
    Arguments:
        ts_dataset - sequence of observations as a list or NumPy array.
        ts_dataset_feature_labels - feature labels or names
        ts_dataset_prediction_labels - prediction labels or names
        cp_frame_size - number of lag observations, or current to past frame size, as input (x) size.
        cf_frame_size - number of observations, or current to future frame size, as output (y) size.
        dropnan - enable to drop rows with NaN values.
    https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
    """
    feature_size = len(ts_dataset_feature_labels)
    prediction_size = len(ts_dataset_prediction_labels)

    if (feature_size + prediction_size) != ts_dataset.shape[1]:
        raise Exception('ERROR: reframe - Time series dataset does not have the same number of expected of features and prediction labels.')

    df = DataFrame(ts_dataset)
    cols, names = list(), list()
    # input feature and output prediction sequence to the past frame (t-n, ..., t-1, t)
    for i in range(cp_frame_size, 0, -1):
        cols.append(df.shift(i))
        names += [('%s%d(t - %d)' % (ts_dataset_feature_labels[j], j + 1, i)) for j in range(feature_size)]
        names += [('%s%d(t - %d)' % (ts_dataset_prediction_labels[j], j + 1, i)) for j in range(prediction_size)]
    # input feature and output prediction sequence to the future frame (t, ..., t+1, t+n)
    for i in range(0, cf_frame_size):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('%s%d(t)' % (ts_dataset_feature_labels[j], j + 1)) for j in range(feature_size)]
            names += [('%s%d(t)' % (ts_dataset_prediction_labels[j], j + 1)) for j in range(prediction_size)]
        else:
            names += [('%s%d(t + %d)' % (ts_dataset_feature_labels[j], j + 1, i)) for j in range(feature_size)]
            names += [('%s%d(t + %d)' % (ts_dataset_prediction_labels[j], j + 1, i)) for j in range(prediction_size)]
    # put it all together
    framed_ts_dataset = concat(cols, axis=1)
    framed_ts_dataset.columns = names
    # drop rows with NaN values
    if dropnan:
        framed_ts_dataset.dropna(inplace=True)

    return framed_ts_dataset


def load_and_prepare_ts_dataset(
    ts_dataset_filepath, *,
    cp_frame_size=MIN_FRAME_SIZE,
    cf_frame_size=MIN_FRAME_SIZE,
    feature_size=MIN_FEATURE_SIZE,
    prediction_size=MIN_PREDICTION_SIZE,
    normalize=True,
    verbose=True
):
    """
    Decription: Load and prepare time datset into proper framed tensor for training / inference
    Arguments:
        ts_dataset_filepath - time series dataset filename/path
        cp_frame_size - number of lag observations, or current to past frame size, as input (x) size.
        cf_frame_size - number of observations, or current to future frame size, as output (y) size.
        feature_size - number of features
        prediction_size - number of predictions
        normalize - enable to normalize dataset
        verbose - enable verbose mode
    """

    if cp_frame_size < MIN_FRAME_SIZE:
        cp_frame_size = MIN_FRAME_SIZE
        warnings.warn('ts_dataset_loader - Current to past horizon window frame periods must be >= %d.' % MIN_FRAME_SIZE, Warning)
        warnings.warn('Using %d as default.' % MIN_FRAME_SIZE, Warning)

    if cf_frame_size < MIN_FRAME_SIZE:
        cf_frame_size = MIN_FRAME_SIZE
        warnings.warn('ts_dataset_loader - Current to future horizon window frame periods must be >= %d.' % MIN_FRAME_SIZE, Warning)
        warnings.warn('Using %d as default.' % MIN_FRAME_SIZE, Warning)

    if feature_size < MIN_FEATURE_SIZE:
        feature_size = MIN_FEATURE_SIZE
        warnings.warn('ts_dataset_loader - Number of features must be >= %d.' % MIN_FEATURE_SIZE, Warning)

        warnings.warn('Using %d as default.' % MIN_FEATURE_SIZE, Warning)

    if prediction_size < MIN_FEATURE_SIZE:
        prediction_size = MIN_FEATURE_SIZE
        warnings.warn('ts_dataset_loader - Number of predictions must be >= %d.' % MIN_PREDICTION_SIZE, Warning)
        warnings.warn('Using %d as default.' % MIN_PREDICTION_SIZE, Warning)

    raw_ts_dataset = read_csv(
        ts_dataset_filepath,
        header=0,
        index_col=0
    )

    ts_dataset = raw_ts_dataset.values.astype('float32')

    ts_dataset_labels = raw_ts_dataset.columns.tolist()
    ts_dataset_col_size = len(ts_dataset_labels)

    ts_dataset_feature_labels = ts_dataset_labels[:feature_size]
    ts_dataset_feature_labels = [*map(lambda label: re.sub(r'(^[ \t]+|[ \t]+(?=:))', '', label, flags=re.M), ts_dataset_feature_labels)]
    ts_dataset_feature_size = len(ts_dataset_feature_labels)

    if ts_dataset_feature_size != feature_size:
        raise Exception('ERROR: ts_dataset_loader - Time series dataset feature count is not equal to the expected of %d.' % feature_size)

    ts_dataset_prediction_labels = ts_dataset_labels[feature_size:]
    ts_dataset_prediction_labels = [*map(lambda label: re.sub(r'(^[ \t]+|[ \t]+(?=:))', '', label, flags=re.M), ts_dataset_prediction_labels)]
    ts_dataset_prediction_size = len(ts_dataset_prediction_labels)

    if ts_dataset_prediction_size != prediction_size:
        raise Exception('ERROR: ts_dataset_loader - Time series dataset prediction count is not equal to the expected of %d.' % prediction_size)

    # frame as supervised learning
    framed_ts_dataset = frame(
        ts_dataset,
        ts_dataset_feature_labels,
        ts_dataset_prediction_labels,
        cp_frame_size=cp_frame_size,
        cf_frame_size=cf_frame_size,
        dropnan=True
    )

    # print(framed_ts_dataset)

    # create training and testing time series dataset
    prepared_ts_dataset = framed_ts_dataset.values
    col_skip = ts_dataset_col_size * cp_frame_size + ts_dataset_feature_size
    prepared_ts_dataset_y_cols = []
    for i in range(cf_frame_size):
        for j in range(ts_dataset_prediction_size):
            prepared_ts_dataset_y_cols.append(col_skip + j)
        col_skip = ts_dataset_col_size * (i + 1)

    col_skip = 0
    prepared_ts_dataset_x_cols = []
    for i in range(cp_frame_size):
        for j in range(ts_dataset_feature_size):
            prepared_ts_dataset_x_cols.append(col_skip + j)
        col_skip = ts_dataset_col_size * (i + 1)

    # print(prepared_ts_dataset_x_cols)
    # print(prepared_ts_dataset_y_cols)

    # split into x input and y output time series dataset
    prepared_ts_dataset_x = prepared_ts_dataset[:, prepared_ts_dataset_x_cols]
    prepared_ts_dataset_y = prepared_ts_dataset[:, prepared_ts_dataset_y_cols]

    # print(prepared_ts_dataset_x)
    # print(prepared_ts_dataset_y)

    prepared_ts_normed_dataset_x = None
    prepared_ts_normed_dataset_y = None
    ts_dataset_x_scaler = None
    ts_dataset_y_scaler = None

    if normalize:
        # load and normalize time series dataset
        ts_dataset_x_scaler = MinMaxScaler()
        ts_dataset_y_scaler = MinMaxScaler()

        prepared_ts_normed_dataset_x = ts_dataset_x_scaler.fit_transform(prepared_ts_dataset_x)
        prepared_ts_normed_dataset_y = ts_dataset_y_scaler.fit_transform(prepared_ts_dataset_y)
        # reshape input to be [samples, cp_frame_size, features] or [samples, cp_frame_size, features, channels]
        prepared_ts_normed_dataset_x = prepared_ts_normed_dataset_x.reshape((prepared_ts_normed_dataset_x.shape[0], cp_frame_size, ts_dataset_feature_size))
        # reshape output to be [samples, cf_frame_size, predictions]
        prepared_ts_normed_dataset_y = prepared_ts_normed_dataset_y.reshape((prepared_ts_normed_dataset_y.shape[0], cf_frame_size * ts_dataset_prediction_size))

    # reshape input to be [samples, cp_frame_size, features] or [samples, cp_frame_size, features, channels]
    prepared_ts_dataset_x = prepared_ts_dataset_x.reshape((prepared_ts_dataset_x.shape[0], cp_frame_size, ts_dataset_feature_size))
    # reshape output to be [samples, cf_frame_size, predictions]
    prepared_ts_dataset_y = prepared_ts_dataset_y.reshape((prepared_ts_dataset_y.shape[0], cf_frame_size * ts_dataset_prediction_size))

    return (ts_dataset,
            ts_dataset_feature_size, ts_dataset_prediction_size,
            ts_dataset_feature_labels, ts_dataset_prediction_labels,
            prepared_ts_dataset_x, prepared_ts_dataset_y,
            prepared_ts_normed_dataset_x, prepared_ts_normed_dataset_y,
            ts_dataset_x_scaler, ts_dataset_y_scaler)
