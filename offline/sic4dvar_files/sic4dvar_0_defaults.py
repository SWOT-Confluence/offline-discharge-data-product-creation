#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

Created on September 15th 2023 at 16:00
by @Isadora Silva

Last modified on February 7th 2025 at 22:00
by @Isadora Silva

@authors: Isadora Silva
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np


@dataclass
class SIC4DVarLowCostDefaults:
    # algo version
    def_algo_version: str = "31_v4"

    # the way to handle reaches, either separately (reach by reach) or collectively (all reaches as one).
    def_handle_reaches: str = "separately"

    # default reference datetime
    def_ref_datetime: datetime = datetime(2000, 1, 1, 0, 0, 0, 0)
    # the time of observations is approximated every freq when loading and adding the data, set as "" to skip this
    # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    def_freq_datetime: str = "6h"  # multiplier and str "D", "h", "min", "s"
    # distance difference between the observations so they are considered at the same distance when loading the data
    def_dist_dif_obs: int = 10  # default is 10 meters

    ################################################ OUTLIER PARAMETERS ################################################
    # number of iter to remove outliers in wse considering:
    def_outl_z_median_n_iters: int = 0  # deviations from median surface per time
    def_outl_z_space_n_iters: int = 0  # deviations from spatial patterns per time
    def_outl_z_time_n_iters: int = 0  # deviations from patterns per node across time
    # the threshold for the deviations in wse outliers considering:
    def_outl_z_median_thr: float = 0.1  # deviations from median surface per time
    def_outl_z_space_thr: float = 0.1  # deviations from spatial patterns per time
    def_outl_z_time_thr: float = 0.1  # deviations from patterns per node across time
    # outlier window size for the local regression of wse considering spatial patterns per time
    def_outl_z_space_w_size: int = 5

    # number of iter to remove outliers in width considering:
    def_outl_w_median_n_iters: int = 0  # deviations from median surface per time
    def_outl_w_space_n_iters: int = 0  # deviations from spatial patterns per time
    def_outl_w_time_n_iters: int = 0  # deviations from patterns per node across time
    # the threshold for the deviations in width outliers considering:
    def_outl_w_median_thr: float = 10.  # deviations from median surface per time
    def_outl_w_space_thr: float = 10.  # deviations from spatial patterns per time
    def_outl_w_time_thr: float = 10.  # deviations from patterns per node across time
    # outlier window size for the local regression of width considering spatial patterns per time
    def_outl_w_space_w_size: int = 5

    ############################################### SMOOTHING PARAMETERS ###############################################
    def_lsm_z_x: int = 10  # wse smoothing parameters, max number of iterations for smoothing observations in space
    # maximum number of iterations for smoothing elevation observations in time
    def_lsm_z_t: int = 1
    # spatial correlation coefficient in meters for smoothing elev. observations in space
    def_lsm_cor_x: float | None = None  # Igor suggested to keep it the average dist between cross-sections
    # the check_behavior option for smoothing node z in space. Igor uses sort, I prefer not to do it (or use force).
    def_lsm_z_x_behavior: str | Literal["force", "sort", "print", "none", ""] = ""
    # temporal correlation coefficient in seconds for smoothing elev. observations in time -> sensor dependent!
    def_lsm_cor_t: float = 0.25 * 24 * 3600  # Igor suggested value
    # required min change in elevation for next smoothing iter.
    def_lsm_z_min_dz: float = 0.01
    # whether to interchange z values while smoothing elevation observations
    def_lsm_z_inter: bool = False  # modified by Isa, big source of error propagation while smoothing!
    # minimum threshold for interchanging elevation values to force decrease while smoothing elevation in space
    def_lsm_z_inter_min_dz: float = def_lsm_z_min_dz
    # maximum threshold for interchanging elevation values to force decrease while smoothing elevation in space
    def_lsm_z_inter_max_dz: float = 0.5
    # the first sweep to perform (either backward or forward)
    def_lsm_z_first_sweep: str = "backward"
    # whether to remove the bias added by the sweep inside the relaxation sweeps loop or outside, at the end
    def_lsm_z_rem_bias_in_loop: bool = True

    # width smoothing parameters
    def_lsm_w_x: int = 3  # max. number of iter. for smoothing width obs. in space
    # max. number of iter. for smoothing width obs. in time
    def_lsm_w_t: int = 1
    # maximum number of iterations for smoothing the width in the elevation space to build the cross-sections
    def_lsm_w_z: int = 10
    # elevation spatial correlation coefficient in meters for building the cross-sections
    def_lsm_cor_z: float | None = None  # (if left as None, compute avg difference between z obs)
    # required min change in width for next smoothing iter.
    def_lsm_w_min_dw: float | None = 0.005
    # the value of Beta in the exponential function which determines the weight of the observations
    def_lsm_w_weight_exp_beta: float = 0.5
    # the first sweep to perform (either backward or forward)
    def_lsm_w_first_sweep: str = "forward"
    # whether to remove the bias added by the sweep inside the relaxation sweeps loop or outside, at the end
    def_lsm_w_rem_bias_in_loop: bool = False

    ############################################# INTERPOLATION PARAMETERS #############################################
    def_fill_missing_z_bool: bool = True  # elevation interpolation parameters, whether to fill missing z observations
    # whether to fill nodes that were fully unobserved
    def_fill_unobserved_nodes_z_bool: bool = False  # Igor said in his version this does not happen
    # max distance between missing points in other time instances in interpolation case for elevation
    def_dx_max_z_in: float = np.inf  # Igor suggested that, for interpolation, no limit is needed
    # max distance between missing points in other time instances in extrapolation case for elevation
    def_dx_max_z_out: float = 1000.  # Igor suggested to keep it smaller (previous was 5km) and change if necessary

    # width interpolation parameters
    def_fill_missing_w_bool: bool = True  # whether to fill missing w observations
    # the weight for filling the missing width
    def_fill_missing_w_weight: str = "mean_cs"  # either "z" or "mean_cs"
    # Whether to linear extrapolate the width values under the lowest observed width up to the lowest observed z
    def_fill_missing_w_extrap_min: bool = True
    # Whether to linear extrapolate the width values under the highest observed width up to the highest observed z
    def_fill_missing_w_extrap_max: bool = True
    # max distance between missing points in other time instances in interpolation case for width
    def_dx_max_w_in: float = np.inf  # Igor suggested that, for interpolation, no limit is needed
    # max distance between missing points in other time instances in extrapolation case for width
    def_dx_max_w_out: float = 300.

    ############################################# CROSS-SECTION PARAMETERS #############################################
    # Parameters for simplifying cross-section from observations
    def_cs_max_n_points: int = 8  # max number of points in section approximation, in meters
    # max distance in section simplification, in meters
    def_cs_max_dist: float = 0.15
    # absolute maximum depth for all the cross-sections, in meters (ref Congo)
    def_cs_depth_max: float = 250.
    # minimum depth of the cross-section before DA, in meters
    def_cs_depth_min: float = 0.25  # If smaller, exclude node
    # tolerance for rounding width and elevation values of the cross-section, in meters
    def_cs_float_atol: float = 1e-2  # default is 0.01 meters

    # whether to verify the top width with the SWORD width
    def_cs_width_top_check: bool = True
    # whether to modify the (in theory) medium/major bed to capture high flows
    def_cs_width_top_modify: bool = True
    # whether to shrink the cross-section so the bottom depth corresponds to the lowest observed z
    def_cs_zb_shrink: bool = True
    # in the continuity burning method, whether to bound the argmin to the mean flow width from SWORD
    def_cs_width_bottom_check: bool = True

    # the method to use to build the cross-section, "sort" or "relax_sweeps"
    def_cs_build_method: str = "relax_sweeps"
    # the method to extend the bottom of the cross-section either "keep", "burn", "breakpoint", "inflection",
    def_cs_zb_method: str = "burn"  # "debitance", "continuity", "bp-smooth", "bp-debit", or "bp-cont"

    # the maximum angle (in degrees) to propagate the medium/major bed to capture high flows
    def_cs_width_top_max_angle: float = 30.
    # the maximum distance (times * x) to propagate the medium/major bed to capture high flows
    def_cs_width_top_max_dx: float = 3.

    # args for cross-section bottom modify
    def_cs_wb_n_iters: float = 5  # the number of iterations for the bottom width when cs_zb_method is continuity
    # the float tolerance for the bottom width when cs_zb_method is continuity
    def_cs_wb_float_atol: float = 0.5
    # the minimum required depth to compute the debitance when cs_zb_method is continuity or debitance
    def_cs_wb_depth_min: float = 0.1
    # the relative minimum required depth to compute the debitance when cs_zb_method is continuity
    def_cs_wb_depth_rel_min: float = 0.1
    # the quantile to be selected when aggregating discharge in space when cs_zb_method is continuity
    def_cs_wb_q_quant: float = 0.5
    # the relative tolerance in debitance for numerical solve when cs_zb_method is continuity or debitance
    def_cs_wb_rel_tol_deb_solver: float = 0.5
    # the relative difference tolerance in debitance to keep the node without modifying the bottom width when
    # cs_zb_method is debitance
    def_cs_wb_rel_diff_debitance_tol: float = 0.1
    # the relative difference tolerance in discharge to keep the node without modifying the bottom width when
    #  cs_zb_method is continuity
    def_cs_wb_rel_diff_q_tol: float = 0.25
    # the relative difference tolerance in discharge to select the time instances for each discharge iteration when
    #  cs_zb_method is continuity
    def_cs_wb_rel_diff_q_t_thr: float = 0.05

    # cross-section from output discharge method. Options are: candidates or numerical
    def_cs_from_q_out_method: str = "numerical"

    ############################################### FILTERING PARAMETERS ###############################################
    def_q_min_n_nodes: int = 1
    def_q_min_n_times: int = 1
    def_q_min_per_nodes: float = 5  # minimum percentage only used if more than 10 nodes
    def_q_min_per_times: float = 5  # minimum percentage only used if more than 10 time instances

    ################################################# SLOPE PARAMETERS #################################################
    def_slope_max_dx: float = 2.0 * 1e4  # maximum distance between nodes to be included in the time averaging in meters
    # minimum distance between nodes to be included in the time averaging
    def_slope_min_dx: float = 1.2 * 1e3  # in meters
    # maximum allowed value for slope
    def_slope_max: float = 5e-3  # in meters/meters source: from DAHITI database
    # minimum allowed value for slope
    def_slope_min: float = 1e-7  # in meters/meters source: from core algo script + DAHITI database
    # number of smoothing passes for slope
    def_slope_lsm: int = 0
    # method for computing the slope using observed water surface elevations, options are "pair" or "lin"
    def_slope_from_nodes_method: str = "lin"  # lin is much faster than pair, but more sensitive to outliers

    ################################################# ALGO31 PARAMETERS ################################################
    # IMPORTANT: do not change the parameters below without contacting INRAE SWOT team directly
    def_q_algo31_pdf: str = "beta"  # the probability distribution function to use in algo 31, either beta or normal
    # discharge (Q) upper/lower bound coefficient for beta distribution
    def_q_bound_coef: float = 5.0
    # discharge std for normal distribution (times * q_mean)
    def_q_pdf_std: float = 1.
    # whether to use discharge prior from q_mean or as a weighted mean from q_monthly
    def_q_prior_from_monthly = False
    # what to return if the discharge compute from the dry bathymetry is higher than the prior discharge
    def_q_dry_out_of_bounds: str = "NaN"  # options are "NaN", "shift"
    # friction (Km) lower bound parameter
    def_km_low: float = 10.0
    # friction (Km) upper bound parameter
    def_km_up: float = 60.0
    # friction (Km) number of candidates
    def_km_n_iters: int = 20
    # bottom elevation (Zb) lower percentile from prior parameter if PDF is beta
    def_zb_p_low_beta: float = 0.
    # bottom elevation (Zb) upper percentile from prior parameter if PDF is beta
    def_zb_p_up_beta: float = 100.
    # bottom elevation (Zb) number of candidates
    def_zb_n_iters: int = 20
    # dimensions of PDF table
    def_k_dim: float = 101
    # discharge coefficient for shape
    def_shape03: float = 5.0
    # Local qm1 bound
    def_local_qm1: float = 5.0
    # float absolute tolerance for algo 31 (friction, elevation and discharge, not slope!)
    def_float_atol: float = 1e-8
    # What is this? used in algo32
    def_thres: float = 1.0
    # What is this? used in algo32
    def_use_ext_bool = True

    def_val_swot_q_flag = (0, 1)

    ################################################# ALGO5 PARAMETERS ################################################
    def_swot_q_min_n_nodes: int = 3  # Parameters for computing SWOT-like discharge
    def_swot_q_min_n_times: int = 10
    def_manning_n_min: float = 0.01
    def_manning_n_max: float = 0.25


sic_def = SIC4DVarLowCostDefaults()

_attr_dict = {
    "Simulated reach(es)":
        ("reach_id",),
    "General parameters":
        ("algo_version",
         "start_datetime_param",
         "end_datetime_param",
         "handle_reaches_param",
         "ref_datetime_param",
         "freq_datetime_param",
         "float_atol_param",),
    "Data sources":
        ("observations_source",
         "cross_sections_source",
         "q_prior_source",
         "slope_source",
         ),
    "Outlier z params":
        ("node_z_outlier_removed_bool",
         "outlier_z_median_n_iters_param",
         "outlier_z_space_n_iters_param",
         "outlier_z_time_n_iters_param",
         "outlier_z_median_thr_param",
         "outlier_z_space_thr_param",
         "outlier_z_time_thr_param",
         "outlier_z_space_w_size_param",
         ),
    "Outlier w params":
        ("node_w_outlier_removed_bool",
         "outlier_w_median_n_iters_param",
         "outlier_w_space_n_iters_param",
         "outlier_w_time_n_iters_param",
         "outlier_w_median_thr_param",
         "outlier_w_space_thr_param",
         "outlier_w_time_thr_param",
         "outlier_w_space_w_size_param",
         ),
    "Smoothing z parameters":
        ("node_z_smooth_bool",
         "smooth_z_in_space_bool",
         "lsm_z_x_param",
         "lsm_cor_x_param",
         "smooth_z_in_time_bool",
         "lsm_z_t_param",
         "lsm_cor_t_param",
         "lsm_z_min_dz_param",
         "lsm_z_inter_bool",
         "lsm_z_inter_max_dz_param",
         "lsm_z_inter_min_dz_param",
         "lsm_z_first_sweep_param",
         "lsm_z_remove_bias_in_loop_bool",
         ),
    "Smoothing w parameters":
        ("node_w_obs_bool",
         "node_w_smooth_bool",
         "smooth_w_in_space_bool",
         "lsm_w_x_param",
         "smooth_w_in_time_bool",
         "lsm_w_t_param",
         "lsm_w_min_dw_param",
         "lsm_w_first_sweep_param",
         "lsm_w_remove_bias_in_loop_bool"
         ),
    "Filling missing z parameters":
        ("node_z_fill_bool",
         "fill_missing_z_bool",
         "fill_missing_z_unobserved_nodes_bool",
         "dx_max_z_in_param",
         "dx_max_z_out_param",
         ),
    "Filling missing w parameters":
        ("node_w_fill_bool",
         "fill_missing_w_bool",
         "fill_missing_w_weight_param",
         "fill_missing_w_extrap_min_bool",
         "fill_missing_w_extrap_max_bool",
         "dx_max_w_in_param",
         "dx_max_w_out_param",
         ),
    "Slope parameters":
        ("reach_s_obs_bool",
         "slope_node_bool",
         "reach_s_fixed_bool",
         "slope_load_bool",
         "slope_bool",
         "slope_from_nodes_method_param",
         "slope_lsm_param",
         "slope_max_dx_param",
         "slope_min_dx_param",
         "slope_max_param",
         "slope_min_param",),
    "Cross-Section building parameters":
        ("cross_section_obs_bool",
         "cross_section_ref_bool",
         "cross_section_comb_bool",
         "cross_section_reg_bool",
         "cross_section_modif_zb_bool",
         "cross_section_shrink_zb_bool",
         "cross_section_use_obs_bool",
         "cross_section_build_method_param",
         "cross_section_bottom_method_param",
         "cross_section_bottom_shrink_bool",
         "cross_section_depth_min_param",
         "cross_section_depth_max_param",
         "cross_section_width_bottom_check_bool",
         "cross_section_width_top_check_bool",
         "cross_section_width_top_modify_bool",
         "cross_section_width_top_max_angle_param",
         "cross_section_width_top_max_dx_param",
         "cross_section_float_atol_param",
         "cross_section_simp_max_dist_param",
         "cross_section_simp_max_n_points_param",
         "lsm_w_z_param",
         "lsm_cor_z_param",
         "lsm_w_weight_exp_beta_param",
         "cross_section_width_bottom_n_iters_param",
         "cross_section_width_bottom_float_atol_param",
         "cross_section_width_bottom_depth_min_param",
         "cross_section_width_bottom_depth_rel_min_param",
         "cross_section_width_bottom_q_quant_param",
         "cross_section_width_bottom_rel_tol_deb_solver_param",
         "cross_section_width_bottom_rel_diff_debitance_tol_param",
         "cross_section_width_bottom_rel_diff_q_tol_param",
         "cross_section_width_bottom_rel_diff_q_t_thr_param",
         ),
    "Filtering data parameters":
        ("q_min_n_nodes_param",
         "q_min_n_times_param",
         "q_min_per_nodes_param",
         "q_min_per_times_param",),
    "Algo 31 parameters":
        ("q_algo31_pdf_param",
         "q_bound_coef_param",
         "q_pdf_std_param",
         "q_prior_from_monthly_bool",
         "q_dry_out_of_bounds_param",
         "km_low_param",
         "km_up_param",
         "km_n_iters_param",
         "zb_p_low_beta_param",
         "zb_p_up_beta_param",
         "zb_n_iters_param",
         "k_dim_param",
         "shape03_param",
         # "shape13_param",
         # "shape23_param",
         # "val1_param",
         # "val2_param",
         "local_qm1_param",),
    # "Algo 32 parameters":
    #     ("thres_param",
    #      "use_ext_bool",),
    # "SWOT-like discharge parameters":
    #     ("swot_q_obs_bool",
    #      "swot_q_run_bool",
    #      "swot_q_min_n_nodes_param",
    #      "swot_q_min_n_times_param",
    #      "manning_n_min_param",
    #      "manning_n_max_param",),
    "Number of nodes, reaches and time instances":
        ("total_n_nodes",
         "total_n_reaches",
         "total_n_times",),
    "Arrays loaded inputs":
        ("dist_description",
         "time_description",
         "q_prior_description",
         "node_z_init_description",
         "node_w_init_description",
         "reach_da_description",
         "reach_s_description",
         "reach_w_description",
         ),
    "Arrays modified inputs":
        ("node_z_fill_description",
         "node_z_smooth_description",
         "node_w_fill_description",
         "node_w_smooth_description",
         ),
    "Arrays uncertainties":
        ("node_z_init_low_bound_description",
         "node_z_init_up_bound_description",
         "node_z_fill_low_bound_description",
         "node_z_fill_up_bound_description",
         "node_z_smooth_low_bound_description",
         "node_z_smooth_up_bound_description",
         "node_w_init_low_bound_description",
         "node_w_init_up_bound_description",
         "node_w_fill_low_bound_description",
         "node_w_fill_up_bound_description",
         "node_w_smooth_low_bound_description",
         "node_w_smooth_up_bound_description",
         ),
    "Arrays cross-Section":
        ("cross_section_w_obs_description",
         "cross_section_w_ref_description",
         "cross_section_w_comb_description",
         "cross_section_w_reg_description",
         "cross_section_w_description",
         "cross_section_w_out_description",
         "cross_section_w_min_description",
         "cross_section_z_obs_description",
         "cross_section_z_ref_description",
         "cross_section_z_comb_description",
         "cross_section_z_reg_description",
         "cross_section_z_description",
         "cross_section_z_out_description",
         "cross_section_z_min_description",
         "cross_section_z_out_min_description",
         ),
    "Array slope":
        ("slope_description",),
    "Area":
        ("wet_area_added_zb_loop_description",
         "wet_area_max_dry_description",),
    "Roughness":
        ("roughness_out_description",),
    "Array discharge output":
        ("reach_q_out_description",),
}

_attr_dict_summary = {
    "Simulated reach(es)":
        ("reach_id",),
    "General parameters":
        ("algo_version",
         "start_datetime_param",
         "end_datetime_param",
         "handle_reaches_param",
         "freq_datetime_param",
         ),
    "Data sources":
        ("observations_source",
         "cross_sections_source",
         "q_prior_source",
         "slope_source",
         ),
    "Outlier z params":
        ("node_z_outlier_removed_bool",
         ),
    "Outlier w params":
        ("node_w_outlier_removed_bool",
         ),
    "Smoothing z parameters":
        ("node_z_smooth_bool",
         "lsm_z_x_param",
         "lsm_z_t_param",
         ),
    "Smoothing w parameters":
        ("node_w_obs_bool",
         "lsm_w_x_param",
         "lsm_w_t_param",
         ),
    "Filling missing z parameters":
        ("node_z_fill_bool",
         "fill_missing_z_bool",
         "fill_missing_z_unobserved_nodes_bool",
         ),
    "Filling missing w parameters":
        ("node_w_fill_bool",
         "fill_missing_w_bool",
         ),
    "Slope parameters":
        ("reach_s_obs_bool",
         "slope_node_bool",
         "reach_s_fixed_bool",
         "slope_load_bool",
         "slope_bool",
         ),
    "Cross-Section building parameters":
        ("cross_section_obs_bool",
         "cross_section_ref_bool",
         "cross_section_comb_bool",
         "cross_section_reg_bool",
         "cross_section_modif_zb_bool",
         "cross_section_shrink_zb_bool",
         "cross_section_use_obs_bool",
         "cross_section_build_method_param",
         "cross_section_bottom_method_param",
         "cross_section_bottom_shrink_bool",
         "cross_section_simp_max_n_points_param",
         ),
    "Algo 31 parameters":
        ("q_algo31_pdf_param",
         "q_bound_coef_param",
         "q_pdf_std_param",
         "q_prior_from_monthly_bool",
         "km_low_param",
         "km_up_param",
         "km_n_iters_param",
         "zb_p_low_beta_param",
         "zb_p_up_beta_param",
         "zb_n_iters_param",
         "local_qm1_param",),
    "Number of nodes, reaches and time instances":
        ("total_n_nodes",
         "total_n_reaches",
         "total_n_times",),
}
