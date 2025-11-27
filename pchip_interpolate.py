# coding=utf-8
"""
Pchip implementation in pure numpy
.. author:: Michael Taylor <mtaylor@atlanticsciences.com>
.. author:: Mathieu Virbel <mat@meltingrocks.com>
Copyright (c) 2016 Michael Taylor and Mathieu Virbel
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# Library of mathematical functions from scipy, numpy that do not call
# scipy, numpy.
from numpy import (zeros, where, diff, floor, minimum, maximum, array,
                   concatenate, logical_or, logical_xor)
from numpy.fft import rfft, irfft


def pchip_interpolate(xi, yi, x, mode="mono"):
    """
    Interpolation using piecewise cubic Hermite polynomial.
    """
    if mode not in ("mono", "quad"):
        raise ValueError("Unrecognized mode string")

    # Search for [xi,xi+1] interval for each x
    xi = xi.astype("double")
    yi = yi.astype("double")

    x_index = zeros(len(x), dtype="int")
    xi_steps = diff(xi)
    if not all(xi_steps > 0):
        raise ValueError("x-coordinates are not in increasing order.")

    x_steps = diff(x)
    if xi_steps.max() / xi_steps.min() < 1.000001:
        # uniform input grid
        if __debug__:
            print("pchip: uniform input grid")
        xi_start = xi[0]
        xi_step = (xi[-1] - xi[0]) / (len(xi) - 1)
        x_index = minimum(
            maximum(
                floor((x - xi_start) / xi_step).astype(int), 0), len(xi) - 2)

        # Calculate gradients d
        h = (xi[-1] - xi[0]) / (len(xi) - 1)
        d = zeros(len(xi), dtype="double")
        if mode == "quad":
            # quadratic polynomial fit
            d[[0]] = (yi[1] - yi[0]) / h
            d[[-1]] = (yi[-1] - yi[-2]) / h
            d[1:-1] = (yi[2:] - yi[0:-2]) / 2 / h
        else:
            # mode=='mono', Fritsch-Carlson algorithm from fortran numerical
            # recipe
            delta = diff(yi) / h
            d = concatenate((delta[0:1], 2 / (1 / delta[0:-1] + 1 / delta[1:]),
                             delta[-1:]))
            d[concatenate((array([False]), logical_xor(delta[0:-1] > 0, delta[
                1:] > 0), array([False])))] = 0
            d[logical_or(
                concatenate((array([False]), delta == 0)), concatenate((
                    delta == 0, array([False]))))] = 0
        # Calculate output values y
        dxxi = x - xi[x_index]
        dxxid = x - xi[1 + x_index]
        dxxi2 = pow(dxxi, 2)
        dxxid2 = pow(dxxid, 2)
        y = (2 / pow(h, 3) *
             (yi[x_index] * dxxid2 * (dxxi + h / 2) - yi[1 + x_index] * dxxi2 *
              (dxxid - h / 2)) + 1 / pow(h, 2) *
             (d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid))
    else:
        # not uniform input grid
        if (x_steps.max() / x_steps.min() < 1.000001 and
                x_steps.max() / x_steps.min() > 0.999999):
            # non-uniform input grid, uniform output grid
            if __debug__:
                print("pchip: non-uniform input grid, uniform output grid")
            x_decreasing = x[-1] < x[0]
            if x_decreasing:
                x = x[::-1]
            x_start = x[0]
            x_step = (x[-1] - x[0]) / (len(x) - 1)
            x_indexprev = -1
            for xi_loop in range(len(xi) - 2):
                x_indexcur = max(
                    int(floor((xi[1 + xi_loop] - x_start) / x_step)), -1)
                x_index[1 + x_indexprev:1 + x_indexcur] = xi_loop
                x_indexprev = x_indexcur
            x_index[1 + x_indexprev:] = len(xi) - 2
            if x_decreasing:
                x = x[::-1]
                x_index = x_index[::-1]
        elif all(x_steps > 0) or all(x_steps < 0):
            # non-uniform input/output grids, output grid monotonic
            if __debug__:
                print("pchip: non-uniform in/out grid, output grid monotonic")
            x_decreasing = x[-1] < x[0]
            if x_decreasing:
                x = x[::-1]
            x_len = len(x)
            x_loop = 0
            for xi_loop in range(len(xi) - 1):
                while x_loop < x_len and x[x_loop] < xi[1 + xi_loop]:
                    x_index[x_loop] = xi_loop
                    x_loop += 1
            x_index[x_loop:] = len(xi) - 2
            if x_decreasing:
                x = x[::-1]
                x_index = x_index[::-1]
        else:
            # non-uniform input/output grids, output grid not monotonic
            if __debug__:
                print("pchip: non-uniform in/out grids, "
                      "output grid not monotonic")
            for index in range(len(x)):
                loc = where(x[index] < xi)[0]
                if loc.size == 0:
                    x_index[index] = len(xi) - 2
                elif loc[0] == 0:
                    x_index[index] = 0
                else:
                    x_index[index] = loc[0] - 1
        # Calculate gradients d
        h = diff(xi)
        d = zeros(len(xi), dtype="double")
        delta = diff(yi) / h
        if mode == "quad":
            # quadratic polynomial fit
            d[[0, -1]] = delta[[0, -1]]
            d[1:-1] = (delta[1:] * h[0:-1] + delta[0:-1] * h[1:]) / (
                h[0:-1] + h[1:])
        else:
            # mode=='mono', Fritsch-Carlson algorithm from fortran numerical
            # recipe
            d = concatenate((delta[0:1], 3 * (h[0:-1] + h[1:]) / ((h[
                0:-1] + 2 * h[1:]) / delta[0:-1] + (2 * h[0:-1] + h[
                    1:]) / delta[1:]), delta[-1:]))
            d[concatenate((array([False]), logical_xor(delta[0:-1] > 0, delta[
                1:] > 0), array([False])))] = 0
            d[logical_or(
                concatenate((array([False]), delta == 0)), concatenate((
                    delta == 0, array([False]))))] = 0
        dxxi = x - xi[x_index]
        dxxid = x - xi[1 + x_index]
        dxxi2 = pow(dxxi, 2)
        dxxid2 = pow(dxxid, 2)
        y = (2 / pow(h[x_index], 3) *
             (yi[x_index] * dxxid2 *
              (dxxi + h[x_index] / 2) - yi[1 + x_index] * dxxi2 *
              (dxxid - h[x_index] / 2)) + 1 / pow(h[x_index], 2) *
             (d[x_index] * dxxid2 * dxxi + d[1 + x_index] * dxxi2 * dxxid))
    return y


def convolve(x, y, mode="full"):
    """
    Convolve function implemented using FFT.
    .. note::
        Works only with real x and y.
        Not a full replacement for numpy.convolve.
    """
    if mode not in ["full", "same", "valid"]:
        raise ValueError("Unrecognized mode string")
    result = irfft(
        rfft(x, len(x) + len(y) - 1) * rfft(y, len(x) + len(y) - 1),
        len(x) + len(y) - 1)
    if mode == "same":
        if len(y) > 1:
            result = result[(len(y) - 1) / 2:(1 - len(y)) / 2]
    elif mode == "valid":
        if len(y) > 1:
            result = result[len(y) - 1:1 - len(y)]
    return result