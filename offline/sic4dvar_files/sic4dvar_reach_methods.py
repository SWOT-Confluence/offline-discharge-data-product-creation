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

    #Additional checks ?
    # count = {'invalid_width': 0, 'invalid_slope': 0, 'invalid_d_x_area': 0, 'invalid_height': 0}
    # for t in range(0,len(reach_height)):
    #     if reach_width[t] <= 0. or check_na(reach_width[t]):
    #         count['invalid_width'] += 1
    #     if reach_slope[t] <= 0. or check_na(reach_slope[t]):
    #         count['invalid_slope'] += 1
    #     if check_na(reach_d_x_area[t]) or (reach_d_x_area[t]+sic4dvar_Abar <= 0.):
    #         count['invalid_d_x_area'] += 1
    #     if check_na(reach_height[t]):
    #         count['invalid_height'] += 1

    # for key in count.keys():
    #     if count[key] == len(reach_height):
    #         logging.warning(f"Invalid data for {key}: {count[key]} occurrences")
    #         valid = False
    #         return final_Q, reach, valid

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