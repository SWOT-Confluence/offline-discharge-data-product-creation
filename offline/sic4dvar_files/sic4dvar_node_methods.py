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
import math
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
from warnings import simplefilter
from .sic4dvar_common_methods import check_na
from .F70 import AB, AC

def create_node_x(dist_out, node_length, start_from_downstream=False):
    dist_out_1 = []
    if dist_out.size > 0:
        dist_out_1.append(float(dist_out[0]))
        if not start_from_downstream:
            direction = -1
        else:
            direction = 1
        for i in range(1, len(node_length)):
            dist_out_1.append(dist_out_1[i - 1] + direction * node_length[i - 1])
        dist_out_1 = dist_out_1[0] - np.array(dist_out_1)
        node_x = np.array(dist_out_1)
    else:
        node_x = np.array([])
    return node_x

def reorder_ids_with_indices(ids, sword_node_order=None, start_from_downstream=False):
    indexed_ids = list(enumerate(ids))
    if sword_node_order is not None:
        indexed_ids = list(zip(indexed_ids, sword_node_order))
        indexed_ids.sort(key=lambda x: x[1])
        sorted_ids = [x[0] for x in indexed_ids]
    else:
        sorted_ids = sorted(indexed_ids, key=lambda x: x[1])
    if not start_from_downstream:
        sorted_ids.reverse()
    indexes = [index for index, _ in sorted_ids]
    values = [id_value for _, id_value in sorted_ids]
    return (indexes, values)

def change_node_ids_order(node_ids, nodes_reach_ids, reach_id, node_order=None):
    sword_dict = {}
    reach_ids_nodes = np.array(nodes_reach_ids[:])
    index_node_ids = np.array(np.where(reach_ids_nodes == int(reach_id)))[0]
    sword_nodes_id = node_ids[index_node_ids]
    sword_node_order = None
    if node_order is not None:
        sword_node_order = node_order[index_node_ids]
    sword_dict['node_order'] = sword_node_order
    print(f'Reach {reach_id}: SWORD node order before reordering: {sword_node_order}')
    sword_nodes_id_reordered_index, sword_nodes_id_reordered = reorder_ids_with_indices(sword_nodes_id, sword_node_order=sword_node_order, start_from_downstream=False)
    print(f'Reach {reach_id}: SWORD node order after reordering: {sword_nodes_id_reordered_index}')
    index_node_ids = index_node_ids[sword_nodes_id_reordered_index]
    sword_dict['node_order_modified'] = sword_nodes_id_reordered_index
    sword_dict['node_order_ids'] = sword_nodes_id
    sword_dict['index_node_ids'] = index_node_ids
    return sword_dict

def preprocessing(array_of_interest, model, node_x, reach_t, reach_id, options, params):
    for t in range(0, len(array_of_interest)):
        if check_na(array_of_interest[t]):
            array_of_interest[t] = np.nan
    array_to_work = deepcopy(array_of_interest)
    array_of_interest_2D2 = []
    array_of_interest_2D2.append(array_to_work)
    mean_elevation_profile_array = model['mean_elevation_profile_array']
    quantile_matrix = model['quantile_matrix']
    if not options['use_quantile_matrix']:
        array_of_interest_2D2.append(mean_elevation_profile_array)
        array_of_interest_2D2 = np.array(array_of_interest_2D2)
        array_of_interest_2D2 = np.transpose(np.array(array_of_interest_2D2))
        array_of_interest_2D2 = np.ma.masked_array(array_of_interest_2D2)
    else:
        array_of_interest_2D2 = np.ones((array_to_work.shape[0], 4)) * np.nan
        array_of_interest_2D2[:, 0] = array_to_work[:, 0]
        array_of_interest_2D2[:, 1] = quantile_matrix[:, 0]
        array_of_interest_2D2[:, 2] = quantile_matrix[:, 1]
        array_of_interest_2D2[:, 3] = quantile_matrix[:, 2]
        for n in range(0, array_of_interest_2D2.shape[0]):
            if check_na(array_of_interest_2D2[n, 0]):
                array_of_interest_2D2[n, 0] = np.nan
    if True:
        tmp_reach_t = []
        tmp_reach_t.append(reach_t)
        if not options['use_quantile_matrix']:
            tmp_reach_t.append(reach_t + 7 * 24 * 3600)
        else:
            tmp_reach_t.append(reach_t + 7 * 24 * 3600)
            tmp_reach_t.append(reach_t + 7 * 2 * 24 * 3600)
            tmp_reach_t.append(reach_t + 7 * 3 * 24 * 3600)
        tmp_reach_t = np.transpose(np.array(tmp_reach_t))
    corx = 200.0
    corx_array = np.ones(len(node_x)) * corx
    cort_wse = 1 * 6 * 3600.0
    cort_width = cort_wse
    if not options['use_quantile_matrix']:
        array_of_interest_2D2 = array_of_interest_2D2[0]
        large_deviations = False
    else:
        large_deviations = True
    node_z1, node_w1, optionnal_output_dict = AB(node_z=array_of_interest_2D2, node_w=array_of_interest_2D2, node_x=node_x, reach_t=tmp_reach_t, corx=corx, corx_array=corx_array, cort_wse=cort_wse, cort_width=cort_width, gnuplot_saving=False, reach_id=reach_id, output_dir='', run_type='seq', use_large_deviations=large_deviations, start_from_downstream=params.start_from_downstream, run_preprocessing=params.run_preprocessing, run_extrapolation=params.run_extrapolation, pooling=params.pooling, regularization_profile=False, use_quantiles_for_densification=False)
    return node_z1

def estimate(bathy_width, bathy_elevation, array_of_interest, node_x, acc_Zb, acc_friction, SLOPEM1, options):
    node_a_test = np.zeros((array_of_interest.shape[0], 1))
    node_p_test = np.zeros((array_of_interest.shape[0], 1))
    Zb2 = -0.05
    ZB_update = acc_Zb
    Zb = np.zeros(len(array_of_interest))
    t = 0
    Wmean_tmp = []
    for i in range(len(bathy_width)):
        Wmean_tmp.append(min(bathy_width[i]))
    Wmean = np.average(Wmean_tmp)
    dict_to_pass = {}
    dict_to_pass['filtered_data'] = {}
    dict_to_pass['filtered_data']['node_z'] = deepcopy(array_of_interest)
    discharge, _, _, _ = AC(bathy_width, bathy_elevation, node_a_test, node_p_test, Wmean, len(bathy_width), Zb2, ZB_update, Zb, t, node_x, SLOPEM1, ZM=1.0, KMI=1.0, option_recompute_area=options['option_recompute_area'], sic4dvar_dict=dict_to_pass)
    final_value = discharge * acc_friction
    return np.atleast_1d(final_value)

def compute_sic4dvar_discharge_node(sic4dvar_model, node_height, node_width, node_slope, node_d_x_area, node_ids, sword_data, debug=False):
    sword_dict = change_node_ids_order(sword_data['nodes_node_id'], sword_data['nodes_reach_id'], reach_id=sword_data['reach_id'], node_order=sword_data.get('nodes_node_order', None))
    swot_node_ids = node_ids
    swot_nodes_id_reordered_index, swot_nodes_id_reordered = reorder_ids_with_indices(swot_node_ids, sword_node_order=sword_dict['node_order'], start_from_downstream=False)
    swot_node_ids = swot_node_ids[swot_nodes_id_reordered_index]
    node_height = node_height[swot_nodes_id_reordered_index]
    node_width = node_width[swot_nodes_id_reordered_index]
    node_slope = node_slope[swot_nodes_id_reordered_index]
    node_d_x_area = node_d_x_area[swot_nodes_id_reordered_index]
    index_node_ids = sword_dict['index_node_ids']
    bathy_width = sic4dvar_model['node_bathymetry_width'][swot_nodes_id_reordered_index]
    bathy_elevation = sic4dvar_model['node_bathymetry_elevation'][swot_nodes_id_reordered_index]
    array_of_interest = node_height
    array_of_interest = array_of_interest.reshape(-1, 1)
    node_x = create_node_x(sword_data['nodes_dist_out'][index_node_ids], sword_data['nodes_node_length'][index_node_ids], start_from_downstream=False)
    reach_t = 0.0
    options = {'use_quantile_matrix': True, 'option_recompute_area': True}
    params = SimpleNamespace(corx_option=0, cort=1 * 6 * 3600.0, override_cort=False, start_from_downstream=False, run_preprocessing=True, run_extrapolation=True, pooling=False)
    densified_data = preprocessing(array_of_interest, sic4dvar_model, node_x, reach_t, str(sword_data['reach_id']), options, params)
    densified_data = densified_data[:, 0]
    densified_data = densified_data.reshape(-1, 1)
    SLOPEM1 = np.array([densified_data[0] - densified_data[-1]])
    if SLOPEM1 < 0:
        SLOPEM1 = np.array([1e-06])
    if 'q_integrator' in sword_data:
        if not check_na(np.nanmean(sword_data['q_integrator'])) and np.nanmean(sword_data['q_integrator']) > 0:
            K = sic4dvar_model['accumulated_friction'] * np.nanmean(sword_data['q_integrator']) / sic4dvar_model['prior_used']
            print('Calibrating K based on prior and mean Q from sword_data')
        else:
            K = sic4dvar_model['accumulated_friction'] * 1.0
            print('Prior Q NA or mean < 0., using default K value')
    else:
        K = sic4dvar_model['accumulated_friction'] * 1.0
        print('No prior Q available, using default K value')
    Q = estimate(bathy_width, bathy_elevation, densified_data, node_x, sic4dvar_model['accumulated_Zb'], K, SLOPEM1, options)
    return Q