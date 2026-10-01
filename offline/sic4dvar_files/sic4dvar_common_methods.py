"""
SIC4DVAR-LC
Copyright (C) 2025 INRAE

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import pandas as pd

def check_na(value):
    """ check if the specified value is None, '', pd.na, np.nan, is_empty or masked """
    if value is None:
        return True
    if value == '':
        return True
    if value == '--':
        return True
    if value is np.ma.masked:
        return True
    if np.ma.is_masked(value):
        return True
    if isinstance(value, np.ma.core.MaskedConstant):
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    try:
        if np.isnan(value):
            return True
    except TypeError:
        pass
    try:
        if value.mask:
            return True
    except AttributeError:
        pass
    try:
        if value.is_empty:
            return True
    except AttributeError:
        pass
    try:
        if value < -100000000.0:
            return True
    except TypeError:
        pass
    return False

def fnc_APR(_z, _Ws, _Zs):
    """Computes hydraulic parameters from a water elevation and a cross section
    description. Can be run for 1z scaler value, or also vactor to process several
    z in the same call.
    NOTE Elevations Zs must be increasing
    NOTE Elevations Zs must be strictly > in the interpolation zone
    NOTE Ws and Zs must be the same length
    NOTE Ws must be >= 0

    Example input:
    results = fnc_APR([1.5], [1, 1, 2, 2], [0, 1, 1, 2])
    results = fnc_APR([1.1, 2], [1, 1, 2, 2], [0, 1, 1, 2])

    Parameters
    ----------
    _z : float vector
        a given water elevation (m). Zobs.
    _Ws : float vector
        Width of the section (m)
    _Zs : float vector
        Elevation of the section (m)

    Returns
    -------
    A : float scaler/vector
        wetted area (m^2)
    P : real scaler/vector
        wetted perimeter (m)
    R : real scaler/vector
        Hydraulic radius (m) w/ R = A/P
    w : real scaler/vector
        Width corresponding to z according to the bathymetry

    """
    nz = len(_z)
    ns = len(_Zs) - 1
    _Ws = _Ws / 2
    A = np.zeros(nz)
    W = np.zeros(nz)
    P = np.zeros(nz)
    R = np.zeros(nz)
    w = np.zeros(nz)
    for i in range(nz):
        A[i] = 0
        w[i] = _Ws[0]
        if _z[i] >= _Zs[0]:
            P[i] = _Ws[0]
        else:
            P[i] = _Ws[0]
        j = 0
        if i == 60:
            pass
        while j < ns and _z[i] > _Zs[j + 1]:
            A[i] = A[i] + (_Ws[j] + _Ws[j + 1]) * (_Zs[j + 1] - _Zs[j])
            P[i] = P[i] + np.sqrt((_Ws[j + 1] - _Ws[j]) ** 2 + (_Zs[j + 1] - _Zs[j]) ** 2)
            j += 1
        if _z[i] > _Zs[j]:
            if j < ns:
                w[i] = _Ws[j] + (_Ws[j + 1] - _Ws[j]) * (_z[i] - _Zs[j]) / (_Zs[j + 1] - _Zs[j])
            else:
                w[i] = _Ws[ns]
            A[i] = A[i] + (_Ws[j] + w[i]) * (_z[i] - _Zs[j])
            P[i] = P[i] + np.sqrt((w[i] - _Ws[j]) ** 2 + (_z[i] - _Zs[j]) ** 2)
        if P[i] <= 0:
            pass
        P[i] = P[i] * 2
        w[i] = w[i] * 2
        if P[i] > 0:
            R[i] = A[i] / P[i]
        else:
            R[i] = 0
    return (A, P, R, w)

def call_func_APR(node_w, node_z, node_xr, node_yr, params, param_dict, coeff_array=[], t=-1.0):
    node_a = node_w.copy()
    node_p = node_w.copy()
    node_r = node_w.copy()
    node_w_simp = node_w.copy()
    depth_mean = []
    A0_mean = []
    P0_mean = []
    W0_mean = []
    if np.array(coeff_array).size < 1.0:
        coeff_array = np.ones(len(node_z[0]))
    for i in range(len(node_w)):
        if t >= 0.0:
            node_a[i], node_p[i], node_r[i], node_w_simp[i] = fnc_APR(node_z[i, t] * coeff_array, node_xr[i], node_yr[i])
        else:
            node_a[i], node_p[i], node_r[i], node_w_simp[i] = fnc_APR(node_z[i] * coeff_array, node_xr[i], node_yr[i])
        depth_section = 0.0
        if param_dict['bb_computation']:
            depth_section = np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])
            A0 = 0
            P0 = node_xr[i][0]
            for j in range(0, len(node_xr[i]) - 1):
                A0 = A0 + (node_xr[i][j] + node_xr[i][j + 1]) / 2 * (node_yr[i][j + 1] - node_yr[i][j])
                P0 = P0 + 2 * np.sqrt((node_xr[i][j + 1] / 2 - node_xr[i][j] / 2) ** 2 + (node_yr[i][j + 1] - node_yr[i][j]) ** 2)
                if node_yr[i][j] - np.nanmin(node_yr[i]) > (np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])) / 2:
                    depth_section = (np.nanmax(node_yr[i]) - np.nanmin(node_yr[i])) / 2
                    break
            if False:
                results2 = f_approx_sections_v6(node_w[i], node_z[i], params.approx_section_params[0], params.approx_section_params[1], params.approx_section_params[2])
                plt.plot(node_xr[i], node_yr[i])
                plt.plot(results2[0], results2[1])
                plt.plot(node_w[i], node_z[i], marker='.', linestyle='None')
                print('A0, P0, R, depth, depth/R:', A0, P0, A0 / P0, depth_section, depth_section / (A0 / P0))
                plt.show()
                plt.clf()
            A0_mean.append(A0)
            P0_mean.append(P0)
            W0_mean.append(node_xr[i][0])
        index_temp = np.where(node_w_simp[i] == 0.0)
        node_w_simp[i][index_temp] = np.nanmin(node_xr[i][np.where(node_xr[i] > 0.0)])
        depth_mean.append(depth_section)
    if not param_dict['bb_computation']:
        bb = 9999.0
    else:
        depth_mean = np.nanmean(depth_mean)
        A0_mean = np.nanmean(A0_mean)
        P0_mean = np.nanmean(P0_mean)
        W0_mean = np.nanmean(W0_mean)
        print('depth_mean, A0_mean, P0_mean, W0_mean:', depth_mean, A0_mean, P0_mean, W0_mean)
        print('mean Radius:', A0_mean / P0_mean, depth_mean / (A0_mean / P0_mean))
        bb = 9999.0
        print('Final computed bb:', bb)
        if bb < 1:
            bb = 1
        print('Final bb:', bb)
    return (node_a, node_p, node_r, node_w_simp, bb)