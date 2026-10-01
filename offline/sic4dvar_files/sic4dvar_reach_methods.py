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

from copy import deepcopy
from types import SimpleNamespace

from .sic4dvar_common_methods import check_na, call_func_APR

MISSING_VALUE_FLT = -999999999999

def compute_sic4dvar_discharge_reach(sic4dvar_model, reach_height, reach_width, reach_slope, reach_d_x_area, debug=False):
    valid = True

    if debug:
        pass
        #print(f"Processing reach: {sic4dvar_model['reach_id']}")
        #print(bugg)

    sic4dvar_Abar = sic4dvar_model["Abar"]
    sic4dvar_n = sic4dvar_model["n"]
    sic4dvar_reach_bathy_z = sic4dvar_model["reach_yr"]
    sic4dvar_reach_bathy_w = sic4dvar_model["reach_xr"]

    #Bathy remove padded NaNs
    tmp_bathy_z = []
    tmp_bathy_w = []
    for pt in range(0,len(sic4dvar_reach_bathy_z)):
        if not check_na(sic4dvar_reach_bathy_z[pt]):
            tmp_bathy_z.append(sic4dvar_reach_bathy_z[pt])
            tmp_bathy_w.append(sic4dvar_reach_bathy_w[pt])
    sic4dvar_reach_bathy_z = np.array([tmp_bathy_z])
    sic4dvar_reach_bathy_w = np.array([tmp_bathy_w])

    if debug:
        sic4dvar_correlation_dynamic_slope = sic4dvar_model["correlation_dynamic_slope"]
        Q_da = sic4dvar_model["Q_da"]
        Q_mm = sic4dvar_model["Q_mm"]
    final_Q = MISSING_VALUE_FLT

    if check_na(sic4dvar_Abar) or check_na(sic4dvar_n):
        #logging.warning("Missing required SIC4DVar parameters: Abar or n")
        valid = False
        return final_Q, sic4dvar_model, valid

    reach_height = np.array([reach_height])
    reach_width = np.array([reach_width])
    # sic4dvar_reach_bathy_z = np.array([sic4dvar_reach_bathy_z])
    # sic4dvar_reach_bathy_w = np.array([sic4dvar_reach_bathy_w])

    params = SimpleNamespace()
    param_dict = {'bb_computation': False}

    #Option use their dA or compute ours
    reach_a, _, _, _, _= call_func_APR([reach_width], [reach_height], \
        sic4dvar_reach_bathy_w, sic4dvar_reach_bathy_z, params, param_dict)
    masked_data = np.ma.masked_values(np.array(reach_a[0]), value=-9999.0)
    reach_d_x_area = masked_data

    reach_w_projected = np.full(reach_height.shape, np.nan)
    valid_idx = reach_height > -1e11

    valid_bathy = (
        np.isfinite(sic4dvar_reach_bathy_z)
        & (sic4dvar_reach_bathy_z > -1e11)
        & np.isfinite(sic4dvar_reach_bathy_w)
    )
    bathy_z = sic4dvar_reach_bathy_z[valid_bathy]
    bathy_w = sic4dvar_reach_bathy_w[valid_bathy]
    sort_idx = np.argsort(bathy_z)
    bathy_z = bathy_z[sort_idx]
    bathy_w = bathy_w[sort_idx]

    reach_w_projected[valid_idx] = np.interp(
        reach_height[valid_idx],
        bathy_z,
        bathy_w,
    )
    reach_width = deepcopy(reach_w_projected)

    if ((reach_width > 0 and not check_na(reach_width)) and (reach_slope > 0 and not check_na(reach_slope)) \
        and ((sic4dvar_Abar+reach_d_x_area >= 0) and (not check_na(reach_d_x_area))) and (not check_na(sic4dvar_n)) \
        and (not check_na(sic4dvar_Abar) and (sic4dvar_Abar > 0) and (sic4dvar_n > 0 and not check_na(sic4dvar_n)))): 
        sic4dvar_q = (
                ((reach_d_x_area+sic4dvar_Abar)**(5/3) * reach_width**(-2/3) *
                (reach_slope)**(1/2)) / sic4dvar_n
            )
    else:
        print("Invalid data for discharge computation. Setting Q to NaN.")
        print(f"reach_width: {reach_width}, reach_slope: {reach_slope}, reach_d_x_area: {reach_d_x_area}, sic4dvar_n: {sic4dvar_n}, sic4dvar_Abar: {sic4dvar_Abar}")
        sic4dvar_q = [np.nan]
    final_Q = sic4dvar_q[0]
        
    return final_Q, sic4dvar_model, valid