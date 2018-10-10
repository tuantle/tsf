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
# Module ewma
# Description - Exponental moving average in numpu
#
# https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
#
# ------------------------------------------------------------------------

import numpy as np

# ------------------------------------------------------------------------

def ewma(x, *, window=10):
    """
    Decription: Exponentialy weighted moving average specified by a decay ``window``
    to provide better adjustments for small windows via:
        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).
    Arguments:
        x - a single dimenisional numpy array
        window - the decay window, or 'span'
    Returns:
        np.ndarray
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    n = x.shape[0]
    ewma = np.empty(n, dtype=float)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = x[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1 - alpha)**i
        ewma_old = ewma_old * (1 - alpha) + x[i]
        ewma[i] = ewma_old / w
    return ewma
