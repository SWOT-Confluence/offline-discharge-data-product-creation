import numpy as np
import logging
import copy
import warnings
import itertools
import operator
import math

from typing import Iterable, Tuple, Any, Literal
from copy import deepcopy
from pathlib import Path

from .sic4dvar_common_methods import check_na, fnc_APR
from .sic4dvar_0_defaults import SIC4DVarLowCostDefaults

try:
    pairwise = itertools.pairwise  # only available from python 3.10
except ImportError:
    def pairwise(iterable):
        """ pairwise('ABCDEFG') --> AB BC CD DE EF FG """
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

def get_relax_factor(
        value_0: float,
        value_1: float,
        cor: float,
):
    """
    Compute the relaxation factor from the values and the correlation.
    The expression is evaluated as -exp(-(value_1 - value_0) / cor)
    """
    return 1. - np.exp(-(value_1 - value_0) / cor)


def _relaxation_check_update(
        value_init: float,
        value_smoothed: float,
        min_change_v_thr: float = SIC4DVarLowCostDefaults().def_float_atol,
        value_ref: float = np.nan,
) -> Tuple[float, float]:
    """
    Check if initial value should be updated by relaxed/smoothed value.

    Parameters
    ----------
    value_init : float
        Value before the smoothing / relaxation sweep
    value_smoothed : float
        Value after the smoothing / relaxation sweep.
    min_change_v_thr : float
        The minimum required change so the value is updated.
         If abs(value_1 - value_init) < min_change_v_thr, return value_init, 0.
    value_ref : float
        Optional. If finite value, only updates if value_1 is closer to it than value_init.

    Returns
    -------
    Tuple[float, float]
        (i) The choice between value_init and value_smoothed;
        (ii) The difference between update value and the initial value.
    """
    # get absolute change in value
    delta_v = np.abs(value_smoothed - value_init)

    # if change is not significant
    if delta_v < min_change_v_thr:
        return value_init, 0.

    if np.isfinite(value_ref):
        # only update if closer to ref
        if np.abs(value_smoothed - value_ref) <= np.abs(value_init - value_ref):
            return value_smoothed, delta_v
        else:
            return value_init, 0.

    return value_smoothed, delta_v


def _relaxation_check_interchange(
        value_0: float,
        value_1: float,
        b: Literal["dec", "inc", "",] = "",
        b_min_thr: float = SIC4DVarLowCostDefaults().def_float_atol,
        b_max_thr: float = np.inf,
):
    # no behavior for interchange
    if not b:
        return False

    # difference in values is too big
    if abs(value_1 - value_0) > b_max_thr:
        return False

    # difference in values is too small
    if abs(value_1 - value_0) < b_min_thr:
        return False

    # if behavior is set to decreasing
    if b == "dec":
        # if behavior is already in place, no need to interchange
        if value_1 < value_0:
            return False
        return True

    # if increasing and behavior is already in place, no need to interchange
    if value_1 > value_0:
        return False

    return True


def relaxation_core(
        sub_value0_array: np.ndarray | np.ma.MaskedArray,
        sub_base0_array: np.ndarray | np.ma.MaskedArray,
        cor: float | np.ndarray,
        min_change_v_thr: float = SIC4DVarLowCostDefaults().def_float_atol,  # TODO Isa: added option
        inter_behavior: Literal["decrease", "increase", "none", ""] = "",  # TODO Isa: added option
        inter_behavior_min_thr: float = SIC4DVarLowCostDefaults().def_float_atol,  # TODO Isa: added option
        inter_behavior_max_thr: float = np.inf,  # TODO Isa: added option
        inter_only: bool = False,
        check_nan: bool = True,
        debug_mode: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Smoothing based on ?
    Run one forward and one backward iteration.
    Copy sub_value0_array.

    Parameters
    ----------
    sub_value0_array : np.ndarray | np.ma.MaskedArray
        The (masked) array with the values to be smoothed
    sub_base0_array : np.ndarray | np.ma.MaskedArray
        The (masked) array with the values from which the relaxation factor is computed.
    cor : float
        The correlation factor to compute the relaxation factor.
    min_change_v_thr : float
        The minimum required change so the value is updated in each iteration.
    inter_behavior : Literal["decrease", "increase", "none", ""]
        The interchange behavior. In each relaxation sweep, if difference in values is smaller than threshold, and if
         allowed by bounds, interchange the values instead of smoothing them.
         Attention: Interchange does not change the arrays with the bounds.
    inter_behavior_min_thr: float
        The minimum threshold for the interchange behavior. If difference in values is smaller than this threshold,
         interchange is not performed.
    inter_behavior_max_thr: float
        The maximum threshold for the interchange behavior. If difference in values is greater than this threshold,
         interchange is not performed.
    inter_only : bool
        Whether to run only the interchange or interchange + smoothing.
    check_nan : bool
        Whether to check for NaN values before running. Only set to False if it was already checked!
        Otherwise NaN will propagate.
    debug_mode : bool
        Whether to print debug statements.

    Returns
    ----------
    tuple
         (i) the smoothed array
         (ii) and the max change in value
    """
    if "dec" in inter_behavior.lower():
        b = "dec"
    elif "inc" in inter_behavior.lower():
        b = "inc"
    else:
        b = ""
        if inter_only:
            raise TypeError("if inter_only, must specify inter_behavior as decrease or increase")
    p = "interchange" if inter_only else "relaxation"

    # copy arrays
    sub_value_array = masked_array_to_nan_array(sub_value0_array)
    sub_base_array = masked_array_to_nan_array(sub_base0_array)

    arr_tmp_list = [sub_value_array, sub_base_array, ]

    # check n dimensions
    if np.any([arr.ndim != 1 for arr in arr_tmp_list]):
        raise RuntimeError(f"cannot run {p} with more than 1D data")
    # check shape
    if np.any([arr.shape != sub_value_array.shape for arr in arr_tmp_list]):
        raise RuntimeError(f"cannot run {p} with when shape of arrays do not match")
    # check NaN
    if check_nan:
        if np.any([np.any(np.isnan(arr)) for arr in arr_tmp_list[:-1]]):
            raise RuntimeError(f"cannot run {p} with NaN values")

    # get number of digits for printing debug messages
    print_n_dig = int(abs(np.log10(min_change_v_thr)))

    # init max delta v for this iteration
    max_delta_v = 0

    # Forward sweep - skips first point
    for n_0i, n_1i in pairwise(range(len(sub_value_array))):

        print_msg = f"Forward sweep for point {n_1i}, (previous point {n_0i}),"

        # get value of current and previous point
        sub_v0i, sub_v1i = sub_value_array[n_0i], sub_value_array[n_1i]

        new_cor = 0.
        if isinstance(cor,np.ndarray):
            if cor.shape[0] > 1:
                new_cor = (cor[n_0i]+cor[n_1i])/2
            else:
                new_cor = (cor[n_0i])
        elif isinstance(cor, float):
            new_cor = cor
        elif isinstance(cor, int):
            new_cor = cor

        if not inter_only:

            # compute relax factor
            relax = get_relax_factor(sub_base_array[n_0i], sub_base_array[n_1i], new_cor)
            #print("relax factor:", relax)
            # compute value for relaxed point
            sub_v1i_new = sub_v0i * (1. - relax) + sub_v1i * relax

            # check if value should be updated
            sub_v1i_new, delta_v = _relaxation_check_update(
                value_init=sub_v1i,
                value_smoothed=sub_v1i_new,
                min_change_v_thr=min_change_v_thr,
            )

            # update/keep value
            sub_value_array[n_1i] = sub_v1i_new

            # update max delta t
            if delta_v > max_delta_v:
                max_delta_v = delta_v

            # update print message
            print_msg += f" relax factor {relax:.{print_n_dig}f}, change in values {delta_v:.{print_n_dig}f},"

            # get value of current and previous point
            sub_v0i, sub_v1i = sub_value_array[n_0i], sub_value_array[n_1i]

        # check if values should be interchanged
        if _relaxation_check_interchange(
                value_0=sub_v0i, value_1=sub_v1i,
                b=b, b_min_thr=inter_behavior_min_thr, b_max_thr=inter_behavior_max_thr,
        ):
            # interchange values
            sub_value_array[n_0i], sub_value_array[n_1i] = sub_v1i, sub_v0i

            # define delta_v
            delta_v = abs(sub_v1i - sub_v0i)

            # update max delta t
            if delta_v > max_delta_v:
                max_delta_v = delta_v

            # update print message
            print_msg += f" values interchanged to force {b}reasing behavior, change in values" \
                         f" {delta_v:.{print_n_dig}f}: {sub_v1i:.{print_n_dig}f}, {sub_v0i:.{print_n_dig}f}"

        else:
            if inter_only:
                # update print message
                print_msg += f" values not interchanged."

        if debug_mode:
            pass

    # Backward sweep - skips last point
    for n_1i, n_0i in pairwise(reversed(range(len(sub_value_array)))):

        print_msg = f"Backward sweep for point {n_0i}, (forward point {n_1i}),"

        # get value of current and next point
        sub_v0i, sub_v1i = sub_value_array[n_0i], sub_value_array[n_1i]

        new_cor = 0.
        if isinstance(cor,np.ndarray):
            if cor.shape[0] > 1:
                new_cor = (cor[n_0i]+cor[n_1i])/2
            else:
                new_cor = (cor[n_1i])
        elif isinstance(cor, float):
            new_cor = cor
        elif isinstance(cor, int):
            new_cor = cor
        

        if not inter_only:

            # compute relax factor
            relax = get_relax_factor(sub_base_array[n_0i], sub_base_array[n_1i], new_cor)
            #print("relax factor backward:", relax)
            # compute value for relaxed point
            sub_v0i_new = sub_v1i * (1. - relax) + sub_v0i * relax

            # check if value should be updated
            sub_v0i_new, delta_v = _relaxation_check_update(
                value_init=sub_v0i,
                value_smoothed=sub_v0i_new,
                min_change_v_thr=min_change_v_thr,
            )

            # update/keep value
            sub_value_array[n_0i] = sub_v0i_new

            # update max delta t
            if delta_v > max_delta_v:
                max_delta_v = delta_v

            # update max delta t
            if delta_v > max_delta_v:
                max_delta_v = delta_v

            # update print message
            print_msg += f" relax factor {relax:.{print_n_dig}f}, change in values {delta_v:.{print_n_dig}f}, "

            # get value of current and next point
            sub_v0i, sub_v1i = sub_value_array[n_0i], sub_value_array[n_1i]

        # check if values should be interchanged
        if _relaxation_check_interchange(
                value_0=sub_v0i, value_1=sub_v1i,
                b=b, b_min_thr=inter_behavior_min_thr, b_max_thr=inter_behavior_max_thr,
        ):
            # interchange values
            sub_value_array[n_0i], sub_value_array[n_1i] = sub_v1i, sub_v0i

            # define delta_v
            delta_v = abs(sub_v1i - sub_v0i)

            # update max delta t
            if delta_v > max_delta_v:
                max_delta_v = delta_v

            # update print message
            print_msg += f" values interchanged to force {b}reasing behavior, change in values" \
                         f" {delta_v:.{print_n_dig}f}: {sub_v1i:.{print_n_dig}f}, {sub_v0i:.{print_n_dig}f}"

        else:
            if inter_only:
                # update print message
                print_msg += f" values not interchanged."

        if debug_mode:
            pass

    return sub_value_array, max_delta_v

def _arrays_de_in_crease(
        my_array: np.ndarray | Iterable,
        comp_op: operator,
        check: bool,
        force: bool,
        remove_nan: bool = True,
) -> np.ndarray | bool:
    if check and force:
        raise TypeError('only one of force or check can be True')
    if (not check) and (not force):
        raise TypeError('one of force or check must be True')

    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)

    if my_array.ndim != 1:
        raise TypeError('can only check/force increase/decrease with 1D arrays')

    # make a copy of the array
    my_array1 = copy.deepcopy(my_array)

    if remove_nan:
        my_array1 = my_array1[np.isfinite(my_array1)]

    for n_0i, n_1i in pairwise(range(my_array1.shape[0])):

        v_0i, v_1i = my_array1[n_0i], my_array1[n_1i]

        if comp_op(v_1i, v_0i):
            if check:
                return False
            my_array1[n_1i] = v_0i

    if check:
        return True

    return my_array1

def arrays_check_decrease(
        my_array: np.ndarray | Iterable,
        remove_nan: bool = True,
) -> bool:
    return _arrays_de_in_crease(my_array, comp_op=operator.gt, check=True, force=False, remove_nan=remove_nan)


def arrays_check_increase(
        my_array: np.ndarray | Iterable,
        remove_nan: bool = True,
) -> bool:
    return _arrays_de_in_crease(my_array, comp_op=operator.lt, check=True, force=False, remove_nan=remove_nan)


def get_mask_nan_across_arrays(
        *my_arrays: np.ndarray | np.ma.MaskedArray,
) -> np.ndarray:
    expected_shape = None
    sum_array = np.empty(0)

    if not isinstance(my_arrays, tuple):
        my_arrays = (my_arrays,)

    for arr in my_arrays:
        if arr.ndim > 2:
            raise TypeError('only works with 1D or 2D arrays')

        # copy do we do not modify initial values
        arr1 = copy.deepcopy(arr)

        # set masked array to NaN
        arr1 = masked_array_to_nan_array(arr1)

        # if first array in the loop
        if expected_shape is None:
            expected_shape = arr.shape
            sum_array = copy.deepcopy(arr1)
            continue

        if arr.shape != expected_shape:
            raise TypeError('shape of all arrays should be the same')

        # NaN will propagate, sum
        sum_array += arr1

        # divide so values are not above float32/float64 limits
        if np.any(np.isfinite(sum_array)):
            sum_array /= np.nanmin(sum_array)

    # get the boolean of NaNs
    nan_bool_array = np.isnan(sum_array)

    return nan_bool_array



def get_index_valid_data(
        *my_arrays: np.ndarray | np.ma.MaskedArray,
        drop_na_how: Literal["any", "all"] = "any",
        axis: int | tuple = (0, 1),
) -> tuple:
    """
    Parameters
    ----------
    my_arrays : np.ndarray | np.ma.MaskedArray
    drop_na_how : Literal["any", "all"]
        used in in 2D arrays. Pandas dropna argument: determine if row or column is removed, when we have at least one
         NaN or all NaN.
    axis: tuple
        axis (in order) to use pandas dropna in 2D arrays

    Returns
    -------
    tuple
        tuple with the valid indexes.
    """
    if isinstance(axis, int):
        axis = (axis,)

    if not isinstance(my_arrays, tuple):
        my_arrays = (my_arrays,)

    # get the boolean of NaNs
    nan_bool_array = get_mask_nan_across_arrays(*my_arrays)

    expected_shape = nan_bool_array.shape

    tmp_array = np.full_like(nan_bool_array, fill_value=1., dtype=np.float32)
    tmp_array[nan_bool_array] = np.nan

    # if no NaN
    if not np.any(nan_bool_array):
        idx_dim0 = np.array(range(expected_shape[0]))

        if len(expected_shape) == 1:
            return (idx_dim0,)

        return idx_dim0, np.array(range(expected_shape[1]))

    # if all NaN values
    if np.all(nan_bool_array):
        return (np.array([], dtype=np.int32),) * len(expected_shape)

    if len(expected_shape) == 1:
        return ((~nan_bool_array).nonzero()[0],)

    # if some NaN values, use pandas to drop rows/cols that do not have any/all data
    df_valid = pd.DataFrame(copy.deepcopy(tmp_array))

    for ax in axis:
        df_valid = df_valid.dropna(axis=ax, how=drop_na_how)

    if df_valid.empty:
        return (np.array([], dtype=np.int32),) * len(expected_shape)

    return np.array(df_valid.index, dtype=np.int32), np.array(df_valid.columns, dtype=np.int32)



def relaxation_sweeps(
        value0_array: np.ndarray | np.ma.MaskedArray,
        base0_array: np.ndarray | np.ma.MaskedArray,
        max_iter: int,
        cor: float | np.ndarray,
        always_run_first_iter: bool = True,  # TODO Isa: added option
        behavior: Literal["decrease", "increase", "none", ""] = "",  # TODO Isa: added option
        inter_behavior: bool = False,
        inter_behavior_min_thr: float = SIC4DVarLowCostDefaults().def_float_atol,  # TODO Isa: added option
        inter_behavior_max_thr: float = np.inf,  # TODO Isa: added option
        check_behavior: Literal["force", "raise", "warn", "print", "none", ""] = "raise",  # TODO Isa: added option
        min_change_v_thr: float = SIC4DVarLowCostDefaults().def_float_atol,  # TODO Isa: added option
        plot: bool = True,
        plot_title: str = "",
        clean_run: bool = False,
        debug_mode: bool = False,
        time_integration: bool = False,
) -> np.ndarray:
    """
    Smoothing based on ?
    Copy value0_array.

    Parameters
    ----------
    value0_array : np.ndarray | np.ma.MaskedArray
        The (masked) array with the values to be smoothed
    base0_array : np.ndarray | np.ma.MaskedArray
        The (masked) array with the values from which the relaxation factor is computed.
    max_iter : int
        The maximum number of iterations
    cor : float
        The correlation factor to compute the relaxation factor.
    always_run_first_iter : bool
        Whether to always run the first iteration, even if values are already according to desired behavior.
    behavior : Literal[decrease, increase, none, ]
        Expected behavior, stopping criteria for iterations, forcing values, warnings statements...
    inter_behavior : bool
        Whether to allow interchanges inside the relaxation sweeps. If difference in values is smaller than threshold,
         and if allowed by bounds, interchange the values instead of smoothing them.
         Attention: This interchange does not change the arrays with the bounds.
    inter_behavior_min_thr: float
        The minimum threshold for the interchange behavior. If difference in values is smaller than this threshold,
         interchange is not performed.
    inter_behavior_max_thr: float
        The maximum threshold for the interchange behavior. If difference in values is greater than this threshold,
         interchange is not performed.
    check_behavior : Literal[force, force_bounds, raise, warn, print, none, ]
        if behavior was specified, option to:
        - force: the behavior by changing the values (also change the bounds!)
        - raise: raise a RuntimeError in case behavior was not obtained
        - warn: raise a warning in case behavior was not obtained
        - print: print a statement in case behavior was not obtained
        - none or empty: nothing
    min_change_v_thr : float
        The minimum required change so the value is updated in each iteration.
    plot : bool
        Whether to plot the relaxation sweeps.
    plot_title : str
        The name of the plot
    clean_run : bool
        Whether to print statements while running this function
    debug_mode : bool
        Whether to print debug statements.

    Returns
    -------
    np.ndarray
        smoothed value_array
    """
    if debug_mode:
        clean_run = False
    if clean_run:
        debug_mode = False

    # copy array
    value_array = masked_array_to_nan_array(value0_array)
    base_array = masked_array_to_nan_array(base0_array)

    arr_tmp_list = [value_array, base_array]

    # check n dimensions
    if np.any([arr.ndim != 1 for arr in arr_tmp_list]):
        raise RuntimeError(f"cannot run relaxation with more than 1D data")

    # check shape
    if np.any([arr.shape != value_array.shape for arr in arr_tmp_list]):
        raise RuntimeError(f"cannot run relaxation when shape of arrays do not match")

    # TODO Isa: modif
    # stop criteria for iterations, force values, warnings statements, related to the expected behavior
    if "dec" in behavior.lower():
        check_func = arrays_check_decrease
        print_beh = "decreasing"
        inter_behavior = "decrease" if inter_behavior else ""

    elif "inc" in behavior.lower():
        check_func = arrays_check_increase
        print_beh = "increasing"
        inter_behavior = "increase" if inter_behavior else ""

    # no expected behavior, just smoothing
    else:
        behavior = ""
        always_run_first_iter = True  # set to True as the goal is smoothing, not enforcing behavior
        check_func = None
        print_beh = ""
        check_behavior = "none"
        inter_behavior = ""

    if behavior:
        if not any(
                [i_ == check_behavior.lower() for i_ in [
                    "force", "raise", "warn", "print", "none", ""]]):
            raise TypeError("check behavior must be one of force, raise, warn, print, none")

    # get number of digits for printing debug messages
    print_n_dig = int(abs(np.log10(min_change_v_thr)))

    # vector the index of all valid points (both in value and in base array)
    val_index_array = get_index_valid_data(value_array, base_array)[0]

    if not clean_run:
        pass

    plot_ys, plot_labels = [], []

    # if running the first iteration is not mandatory, check if values are already all decreasing/increasing
    if not always_run_first_iter:  # TODO Isa: added option
        # check if values are already all decreasing/increasing
        if check_func(value_array, remove_nan=True):
            if plot:
                pass
            return value_array

    # need at least two values for relaxation sweeps
    if len(val_index_array) < 2:
        if not clean_run:
            pass
        return value_array

    # get array with valid data
    v_val_array = value_array[val_index_array]

    # get mean of valid data
    #v_val_mean_05 = np.mean(v_val_array)

    if time_integration:
        v_val_mean_0 = 0.
        from copy import deepcopy
        base_array2 = deepcopy(base_array[val_index_array])
        time_scaling = 0.
        for t in range(1,len(base_array2)):
            v_val_mean_0 += ((v_val_array[t]+v_val_array[t-1])/2) \
                * (base_array2[t]-base_array2[t-1])
            time_scaling += (base_array2[t]-base_array2[t-1])
        v_val_mean_0 = v_val_mean_0/time_scaling
    else:
        v_val_mean_0 = np.mean(v_val_array)

    if not clean_run:
        pass

    if plot:
        plot_ys.append(copy.deepcopy(value_array[val_index_array]))
        plot_labels.append("Initial")

    # run relaxation (limited to max number of iterations)
    for relax_iter in range(max_iter):
        
        # compute relaxation
        v_val_array, max_delta_v = relaxation_core(
            sub_value0_array=value_array[val_index_array],
            sub_base0_array=base_array[val_index_array],
            cor=cor, min_change_v_thr=min_change_v_thr,
            inter_behavior=inter_behavior, inter_only=False,
            inter_behavior_min_thr=inter_behavior_min_thr, inter_behavior_max_thr=inter_behavior_max_thr,
            check_nan=False, debug_mode=debug_mode,
        )

        if not clean_run:
            logging.debug(f"relaxation iteration {relax_iter + 1}: max change in single value {max_delta_v:.{print_n_dig}f}")

        # If max change in a single value in the relaxation sweeps is smaller than threshold
        if max_delta_v <= min_change_v_thr:  # TODO Isa: modif

            # break the loop
            break_loop = True

        # change of value is bigger than threshold
        else:
            if check_func is not None:
                # check if values in any node is always decreasing/increasing (stop criteria for loop)
                break_loop = True if check_func(v_val_array, remove_nan=False) else False

            else:
                break_loop = False

        if plot:
            plot_ys.append(copy.deepcopy(v_val_array))
            plot_labels.append(f"Iteration {relax_iter + 1}")

        # assign smoothed values to array
        value_array[val_index_array] = v_val_array

        if break_loop:
            break

    # compute mean after smoothing

    if time_integration:
        #v_val_mean_2 = np.mean(v_val_array)
        from copy import deepcopy
        base_array2 = deepcopy(base_array[val_index_array])
        v_val_mean_1 = 0.
        time_scaling = 0.

        for t in range(1,len(base_array2)):
            v_val_mean_1 += ((v_val_array[t]+v_val_array[t-1])/2) \
                * (base_array2[t]-base_array2[t-1])
            time_scaling += (base_array2[t]-base_array2[t-1])

        v_val_mean_1 = v_val_mean_1/time_scaling

    else:
        v_val_mean_1 = np.mean(v_val_array)

    # bias correction
    v_val_array = v_val_array + (v_val_mean_0 - v_val_mean_1)

    if check_func is not None:
        if behavior:
            # TODO Isa: modif
            error_msg = f"values are not always {print_beh}"
            if not check_behavior:
                pass
            elif check_behavior.lower() == "none":
                pass
            elif not check_func(v_val_array, remove_nan=False):
                if check_behavior.lower() == "warn":
                    # error_msg += f": {v_val_array}"
                    warnings.warn(error_msg, RuntimeWarning)
                elif check_behavior.lower() == "print":
                    pass
                elif "force" in check_behavior.lower():
                    v_val_array = np.sort(v_val_array, )
                    if "dec" in behavior.lower():
                        v_val_array = v_val_array[::-1]
                else:
                    # error_msg += f": {v_val_array}"
                    with np.printoptions(precision=4, suppress=True):
                        pass
                    raise RuntimeError(error_msg)

    if not clean_run:
        pass

    if plot:
        plot_ys.append(copy.deepcopy(v_val_array))
        plot_labels.append("Final")

        helper_plot_lines(
            xs=[val_index_array, ] * len(plot_labels),
            ys=plot_ys,
            show=True,
            line_labels=plot_labels,
            title=plot_title,
            x_axis_title="Indexes",
            y_axis_title="Values",
            fig_width=15,
            fig_height=5,
            add_legend=True,
        )

    # set the smoothed values to the valid values
    value_array[val_index_array] = v_val_array

    return value_array

def relaxation_sweeps_in_dim(
        dim: int,
        value0_array: np.ndarray,
        base0_array: np.ndarray,
        max_iter: int,
        cor: float | np.ndarray,
        always_run_first_iter: bool = True,  # TODO Isa: added option
        behavior: Literal["decrease", "increase", "none", ""] = "",  # TODO Isa: added option
        inter_behavior: bool = False,
        inter_behavior_min_thr: float = SIC4DVarLowCostDefaults().def_float_atol,  # TODO Isa: added option
        inter_behavior_max_thr: float = np.inf,  # TODO Isa: added option
        check_behavior: Literal["force", "raise", "warn", "print", "none", ""] = "raise",  # TODO Isa: added option
        min_change_v_thr: float = SIC4DVarLowCostDefaults().def_float_atol,  # TODO Isa: added option
        plot: bool = False,
        plot_title: str = "",
        clean_run: bool = False,
        debug_mode: bool = False,
        time_integration: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smoothing based on ?
    Copy value0_array.

    Parameters
    ----------
    dim : int
        the dimension to run the sweeps on.
    value0_array : np.ndarray | np.ma.MaskedArray
        The (masked) array with the values to be smoothed
    base0_array : np.ndarray | np.ma.MaskedArray
        The (masked) array with the values from which the relaxation factor is computed.
    max_iter : int
        The maximum number of iterations
    cor : float
        The correlation factor to compute the relaxation factor.
    always_run_first_iter : bool
        Whether to always run the first iteration, even if values are already according to desired behavior.
    behavior : Literal["decrease", "increase", "none", ""]
        Expected behavior, stopping criteria for iterations, forcing values, warnings statements...
    inter_behavior : bool
        Whether to allow interchanges inside the relaxation sweeps. If difference in values is smaller than threshold,
         and if allowed by bounds, interchange the values instead of smoothing them.
         Attention: This interchange does not change the arrays with the bounds.
    inter_behavior_min_thr: float
        The minimum threshold for the interchange behavior. If difference in values is smaller than this threshold,
         interchange is not performed.
    inter_behavior_max_thr: float
        The maximum threshold for the interchange behavior. If difference in values is greater than this threshold,
         interchange is not performed.
    check_behavior : Literal["force", "raise", "warn", "print", "none", ""]
        if behavior was specified, option to:
        - force the behavior by changing the values
        - raise a RuntimeError in case behavior was not obtained
        - raise a warning in case behavior was not obtained
        - print a statement in case behavior was not obtained
        - nothing
    min_change_v_thr : float
        The minimum required change so the value is updated in each iteration.
    plot : bool
        Whether to plot the relaxation sweeps.
    plot_title : str
        The name of the plot
    clean_run : bool
        Whether to print statements while running this function (alert: many!)
    debug_mode : bool
        Whether to print debug statements.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (i) smoothed value_array
        (ii) smoothed value_low_bound_array
        (iii) smoothed value_up_bound_array
    """
    if debug_mode:
        clean_run = False
    if clean_run:
        debug_mode = False

    # TODO: make it work for any number of dimensions

    value_array = masked_array_to_nan_array(value0_array)
    base_array = masked_array_to_nan_array(base0_array)

    arr_tmp_list = [value_array, base_array]

    # check n dimensions
    if np.any([arr.ndim > 2 for arr in arr_tmp_list]):
        raise RuntimeError(f"cannot run relaxation with more than 1D data")
    # check shape
    if np.any([arr.shape != value_array.shape for arr in arr_tmp_list[2:]]):
        raise RuntimeError(f"cannot run relaxation when shape of arrays do not match")

    if dim > 1:
        raise NotImplementedError("only works with dim <= 1")

    # copy
    value_array = copy.deepcopy(value0_array)

    index_base = True if value_array.ndim == base_array.ndim else False

    if not clean_run:
        pass

    # loop in specified dimension
    for n_i in range(value_array.shape[dim]):

        if not clean_run:
            pass

        # get the values at that dimension (only works with 2D arrays and dim <= 1)
        value_at_dim_array = value_array[n_i, :] if dim == 0 else value_array[:, n_i]

        # get the base array values
        if index_base:
            # if base array is also 2D (only works with 2D arrays and dim <= 1)
            base_at_dim_array = base_array[n_i, :] if dim == 0 else base_array[:, n_i]
        else:
            base_at_dim_array = base_array

        values_at_dim_array = relaxation_sweeps(
            value0_array=value_at_dim_array,
            base0_array=base_at_dim_array,
            max_iter=max_iter,
            cor=cor,
            always_run_first_iter=always_run_first_iter,
            behavior=behavior,
            inter_behavior=inter_behavior,
            inter_behavior_min_thr=inter_behavior_min_thr,
            inter_behavior_max_thr=inter_behavior_max_thr,
            check_behavior=check_behavior,
            min_change_v_thr=min_change_v_thr,
            plot=plot,
            plot_title=plot_title + f" {n_i}",
            clean_run=clean_run,
            debug_mode=debug_mode,
            time_integration=time_integration,
        )

        # (only works with 2D arrays and dim <= 1)
        if dim == 0:
            value_array[n_i, :] = values_at_dim_array
        else:
            value_array[:, n_i] = values_at_dim_array

    return value_array

def global_large_deviations_removal_experimental(node_x, z, reach_t, times_debug=np.array([])):
    #original method by Igor
    #adapted by Dylan in Python from Fortran

    reverse_order = False

    # node_x_tmp = deepcopy(node_x)
    # times_tmp = times_debug / 3600 / 24
    node_x = abs(node_x - node_x[0])/1000  

    new_z = deepcopy(z)
    a11=0.;a12=0.;a21=0.;a22=0.;b1=0.;b2=0.
    c1=0.0; c2=0.0
    
    total_valid_points = 0.0
    
    time_acc = 0.
    old_b1=0.;old_b2=0.
    number_of_sections=[]
    prev_index_valid = []

    for t in range(0,len(z[0])):
       
        index_valid = []
        
        for n in range(0,len(z)):
            if not check_na(z[n,t]):
                index_valid.append(n)
        
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0,len(z[0]))
        
        time_acc_node = 0.
        for n in range(0,num_valid_pts):
            a11=a11+node_x[index_valid[n]]*node_x[index_valid[n]]
            a12=a12+node_x[index_valid[n]]
            a21=a21+node_x[index_valid[n]]
            a22=a22+1.0
            b1 = b1 + node_x[index_valid[n]] * z[index_valid[n],t]
            b2 = b2 + z[index_valid[n],t]

            # if t > 0:
            #     value = z[index_valid[n],t]
            #     value_prev = z[index_valid[n],t-1]
            #     print("value, value_prev:", value, value_prev)
                
            #     if (not np.ma.is_masked(value) and np.isfinite(value)) and \
            #         (not np.ma.is_masked(value_prev) and np.isfinite(value_prev)):
            #         b1=b1+((node_x[index_valid[n]]*value + node_x[index_valid[n]]*value_prev)/2) \
            #         * (reach_t[t] - reach_t[t-1])
                    
            #         b2=b2+((value + value_prev)/2) * (reach_t[t] - reach_t[t-1]) 
            #         # print("Valid value and value_prev for n=", n, "t=", t, "value:", value, "value_prev:", value_prev)
            #         # print("b1=", b1, "b2=", b2) 
            #         # print("--------------")
            #         time_acc_node += (reach_t[t] - reach_t[t-1])
            #     else:
            #         # print("Invalid value or value_prev for n=", n, "t=", t, "value:", value, "value_prev:", value_prev)
            #         # print("b1=", b1, "b2=", b2)
            #         # print("--------------")

            #         pass
            
        # time_acc_node = time_acc_node / num_valid_pts if num_valid_pts > 0 else 0.
        # print("Time acc node:", time_acc_node, "t=", t)
        # time_acc += time_acc_node

    # print(b1, b2, time_acc)
    # b1= b1 / time_acc
    # b2= b2 / time_acc
    # print(b1, b2)
    # print(old_b1, old_b2)
    # print(bug)

    if total_valid_points >= 1:
        if a12*a21-a22*a11 == 0:
            logging.warning("a12*a21-a22*a11 == 0, cannot compute c1 and c2. Setting c1 and c2 to 0.")
            c1 = 0.0
            c2 = 0.0
        else:
            c2=(b1*a21-b2*a11)/(a12*a21-a22*a11)
            c1=(b1-a12*c2)/a11

    # print(total_valid_points, c1, c2)
    # print("b1, b2:", b1, b2)
    # print("a11, a12, a21, a22:", a11, a12, a21, a22)
    # print("a12*a21", a12*a21, "a22*a11", a22*a11)
    # # print(z)    
    # print(bug)
    
    sigmp=2. #sigmaplus: elevation below regression surface #2.0
    sigmn=2. #sigmaminus: elevation above regression surface #2.0
    
    total_valid_points = 0.0
    error_array = []

    for t in range(0,len(z[0])):
        index_valid = []
        
        for n in range(0,len(z)):
            if not check_na(z[n,t]):
                index_valid.append(n)
              
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0,len(z[0]))
        
        error_array_tmp = np.ones(num_valid_pts) * np.nan
        #if num_valid_pts > 2:

        if num_valid_pts > 0:
            for n in range(0,num_valid_pts):
                dist=c1*node_x[index_valid[n]]+c2-z[index_valid[n],t]
                error_array_tmp[n] = dist
            error_array.append(error_array_tmp)

    error_array_concat = np.concatenate(error_array)
    #print(error_array_concat, len(error_array_concat))

    from scipy.stats import iqr
    iqr_value = iqr(error_array_concat)
    median_value = np.nanmedian(error_array_concat)

    for t in range(0,len(z[0])):
        index_valid = []
        
        for n in range(0,len(z)):
            if not check_na(z[n,t]):
                index_valid.append(n)
              
        num_valid_pts = len(index_valid)
        total_valid_points += num_valid_pts
        num_overpass = np.arange(0,len(z[0]))
        
        #if num_valid_pts > 2:
        debug_dict = []
        debug_dict.append(0.)
        debug_dict.append(0.)
        debug_dict.append(0.)
        for n in range(0,num_valid_pts):
            dist=c1*node_x[index_valid[n]]+c2-z[index_valid[n],t]
            
            if np.abs(debug_dict[2]) < np.abs(dist-median_value):
                debug_dict[0] = n
                debug_dict[1] = t
                debug_dict[2] = np.abs(dist-median_value)

            if np.abs(dist - median_value) > 1 * iqr_value:
                #print("n=", n, "t=", t, "DIST:", dist, "median:", median_value, np.abs(dist-median_value), "iqr:", iqr_value)
                pass

            if np.abs(dist - median_value) > 3 * iqr_value:
                if dist - median_value > 0:
                    new_z[index_valid[n],t]=c1*node_x[index_valid[n]]+c2+(sigmp*iqr_value+median_value)
                else:
                    new_z[index_valid[n],t]=c1*node_x[index_valid[n]]+c2+(sigmn*iqr_value+median_value)

    if c1 > 0.01:
        reverse_order = True

    return new_z, reverse_order, c1, c2


def find_nearest(
        my_array: np.ndarray | Iterable,
        value,
) -> Tuple[int, float]:
    """
    Find the nearest value in a numpy array

    Parameters
    ----------
    my_array : np.ndarray
        The array in which the nearest entry to value will be selected
    value : Any
        The value that will be searched in the array

    Returns
    -------
    Tuple[int, Any]
        A tuple containing the nearest index to value and the value itself
    """
    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)
    i = (np.abs(my_array - value)).argmin()
    v = my_array[i]
    return i, v


def iterable_to_flattened_array(
        my_iterable: Iterable,
) -> np.ndarray:
    """
    Convert to array/masked array and flatten.
    Creates a copy
    """
    my_array1 = copy.deepcopy(my_iterable)

    # convert to arrays
    if (not isinstance(my_array1, np.ndarray)) or not (isinstance(my_array1, np.ma.MaskedArray)):
        my_array1 = np.array(my_array1)
    else:
        my_array1 = my_array1

    # flatten arrays
    if my_array1.ndim != 1:
        my_array1 = my_array1.flatten()

    return my_array1


def masked_array_to_nan_array(
        my_array: np.ma.MaskedArray,
):
    """
    If array is a masked array, set masked values in array to np.nan.
    Might change the data type to float32 / float64
    Creates a copy.
    """

    my_array1 = copy.deepcopy(my_array)

    if my_array1.dtype == np.int32 or my_array1.dtype == np.int64:

        if my_array1.size == 0:
            my_array1 = np.empty(0, dtype=np.float32)
            return my_array1

        if isinstance(my_array1, np.ma.MaskedArray):
            max_value = np.ma.max(my_array1)
        else:
            max_value = np.nanmax(my_array1)

        if my_array1.dtype == np.int32:
            if not np.isnan(max_value):
                error_msg = 'max value of array is bigger than max allowed valued in float 32'
                if max_value > np.finfo(np.float32).max:
                    raise RuntimeError(error_msg)
            my_array1 = my_array1.astype(np.float32)
            return my_array1

        if my_array1.dtype == np.int64:
            if not np.isnan(max_value):
                error_msg = 'max value of array is bigger than max allowed valued in float 64'
                if max_value > np.finfo(np.float64).max:
                    raise RuntimeError(error_msg)
            my_array1 = my_array1.astype(np.float64)
            return my_array1

    if isinstance(my_array1, np.ma.MaskedArray):
        my_array1 = my_array1.filled(np.nan)

    return my_array1


def nan_array_to_masked_array(
        my_array: np.ndarray | np.ma.MaskedArray,
        fill_value: None | float | int = None,
) -> np.ma.MaskedArray:
    """
    If array is not masked array, set NaN values to masked.
    Option to set the fill value of the masked array.
    Creates a copy.
    """
    my_array1 = copy.deepcopy(my_array)

    if not isinstance(my_array1, np.ma.MaskedArray):
        my_array1 = np.ma.masked_invalid(my_array1)
    if fill_value is not None:
        my_array1.set_fill_value(fill_value)
    return my_array1

def linear_interp_get_coef(x_array, y_array):
    """ y = a + bx """
    # noinspection PyTupleAssignmentBalance
    b, a = np.polyfit(x_array, y_array, 1, )
    return a, b

def piecewise_linear_interpolation(
        values_in_array: np.ndarray,
        base_in_array: np.ndarray,
        limits: str | None = None,
        check_nan: bool = True,  # check nan in space array
        float_atol: float = SIC4DVarLowCostDefaults().def_float_atol,
) -> np.ndarray:
    """

    Parameters
    ----------
    values_in_array
    base_in_array
    limits :
        fill, linear or None
    check_nan
    float_atol

    Returns
    -------

    """
    if any([isinstance(arr, np.ma.MaskedArray) for arr in [values_in_array, base_in_array]]):
        raise NotImplementedError("does not work with Masked Arrays")  # TODO: store masked filled values and mask back
    values_out_array, base_out_array = [iterable_to_flattened_array(i_) for i_ in [values_in_array, base_in_array]]

    if check_nan:
        if np.any(np.isnan(base_out_array)):
            raise RuntimeError("cannot run when there NaN values in base_out_array")

    if not limits:
        interp_kwargs = dict(left=np.nan, right=np.nan)
    elif "fill" in limits.lower():
        interp_kwargs = dict()
    elif "lin" in limits.lower():
        interp_kwargs = dict(left=np.nan, right=np.nan)
    else:
        raise TypeError("limits must be one of fill, linear or None")

    # get a boolean array with all the valid points
    v_t_val_bool_array = np.isfinite(values_out_array)

    # get an array with all the valid indexes
    v_t_val_index_array = v_t_val_bool_array.nonzero()[0]

    # get the first and the last valid indexes
    n_t_start_0, n_t_end_1 = v_t_val_index_array[[0, -1]]

    if np.all(v_t_val_bool_array):
        # print("all values are valid, no need to interpolate")
        return values_out_array

    # if less than 2 valid entries
    if np.count_nonzero(v_t_val_bool_array) < 2:
        # print("no valid values, impossible to interpolate")
        return values_out_array

    if limits and ("lin" in limits.lower()):
        interp_kwargs = dict(
            left=values_out_array[v_t_val_bool_array][0],
            right=values_out_array[v_t_val_bool_array][-1],
        )

    # one-dimensional piecewise linear interpolation for non-edge values
    np_interp_out = np.interp(
        x=base_out_array,
        xp=base_out_array[v_t_val_bool_array],
        fp=values_out_array[v_t_val_bool_array],
        **interp_kwargs,
    )

    # replace values in array
    values_out_array = np_interp_out

    # return if limits is not linear
    if not limits or "fil" in limits.lower():
        return values_out_array

    # if no boundary values with NaN
    if n_t_start_0 == 0:
        if n_t_end_1 == v_t_val_bool_array.size - 1:
            return values_out_array

    # set limit values back to NaN
    if n_t_start_0 > 0:
        values_out_array[:n_t_start_0] = np.nan
    if n_t_end_1 < v_t_val_bool_array.size - 1:
        if n_t_end_1 + 1 < v_t_val_bool_array.size - 1:
            values_out_array[n_t_end_1 + 1:] = np.nan
        else:
            values_out_array[n_t_end_1 + 1] = np.nan

    # get a boolean array with all the valid points
    v_t_val_bool_array = np.isfinite(values_out_array)

    # get an array with all the valid indexes
    v_t_val_index_array = v_t_val_bool_array.nonzero()[0]

    # get the first and the last valid indexes
    n_t_start_0, n_t_end_1 = v_t_val_index_array[[0, -1]]

    # interpolate values in the beginning
    if n_t_start_0 > 0:

        # get the following valid value
        n_t_start_1 = v_t_val_index_array[1]

        start_b_array = base_out_array[[n_t_start_0, n_t_start_1]]
        start_v_array = copy.deepcopy(values_out_array[[n_t_start_0, n_t_start_1]])

        # if values are too close
        if np.isclose(start_v_array[0], start_v_array[1], rtol=0.0, atol=float_atol):
            values_out_array[:n_t_start_0] = start_v_array[0]
        else:
            start_v0, b = linear_interp_get_coef(start_b_array, start_v_array)
            values_out_array[:n_t_start_0] = start_v0 + b * base_out_array[:n_t_start_0]

    # interpolate values at the end
    if n_t_end_1 < v_t_val_bool_array.size - 1:
        # get the previous valid value
        n_t_end_0 = v_t_val_index_array[-2]

        end_b_array = base_out_array[[n_t_end_0, n_t_end_1]]
        end_v_array = copy.deepcopy(values_out_array[[n_t_end_0, n_t_end_1]])

        # if values are too close
        if np.isclose(end_v_array[0], end_v_array[1], rtol=0.0, atol=float_atol):
            end_interp_out = end_v_array[0]
        else:
            end_v0, b = linear_interp_get_coef(end_b_array, end_v_array)
            if n_t_end_1 + 1 < v_t_val_bool_array.size - 1:
                end_interp_out = end_v0 + b * base_out_array[n_t_end_1 + 1:]
            else:
                end_interp_out = end_v0 + b * base_out_array[n_t_end_1 + 1]

        if n_t_end_1 + 1 < v_t_val_bool_array.size - 1:
            values_out_array[n_t_end_1 + 1:] = end_interp_out
        else:
            values_out_array[n_t_end_1 + 1] = end_interp_out

    if np.count_nonzero(np.isnan(values_out_array)) > 0:
        raise AssertionError("should have interpolated all values, check code")

    return values_out_array


def __preprocess(
        values0_array: np.ndarray or np.ma.MaskedArray,  # 2D
        space0_array: np.ndarray or np.ma.MaskedArray,  # 1D
        clean_run: bool = False,
):
    # important: this function assumes the following index for the dimensions
    space_dim, time_dim = 0, 1

    if not clean_run:
        pass

    # shape check
    if values0_array.ndim != 2:
        raise TypeError(f'values array must be 2D')
    if (values0_array.shape[space_dim],) != space0_array.shape:
        error_msg = f'space array must be a 1D array with size {values0_array.shape[space_dim]}, ' \
                    f'not an array with shape {space0_array.shape}'
        raise TypeError(error_msg)

    # get the total number of node instances
    total_n_nodes = values0_array.shape[0]

    # copy to store the output values
    values_out_array = copy.deepcopy(values0_array)

    # if arrays are all masked arrays
    try:
        arrays_masked = True

        v_fill_value = values_out_array.get_fill_value()

        # set masked array to NaN
        values_out_array = masked_array_to_nan_array(values_out_array)

    # array are not all masked arrays
    except AttributeError:
        arrays_masked = False
        v_fill_value = None

    # copy to store initial values with NaN instead of masked
    values1_array = copy.deepcopy(values_out_array)
    space1_array = masked_array_to_nan_array(copy.deepcopy(space0_array))

    # get an array with the number of valid observations in each time instance
    n_val_value1_per_t_array = np.count_nonzero(np.isfinite(values1_array), axis=0)

    # get an array with the time indexes that can be interpolated
    miss_t_index_array = np.nonzero(
        # at least one valid value
        (n_val_value1_per_t_array >= 1) & (
            # not all valid values
                n_val_value1_per_t_array < values1_array.shape[0])  # & (
    )[0]

    return values_out_array, values1_array, space1_array, miss_t_index_array, total_n_nodes, arrays_masked, v_fill_value


def __create_aux_near_and_interp_arrays(
        values_out_array: np.ndarray,
        space1_array: np.ndarray,
        dx_max_in: float,
        dx_max_out: float,
        float_atol: float = SIC4DVarLowCostDefaults().def_float_atol,
):
    # array with missing values linearly interpolated in space (used in case missing cond.)
    values2_array = np.full_like(values_out_array, fill_value=np.nan, dtype=np.float32)

    # arrays to store the position of nearest point in each time instance
    near_idx_array = np.full_like(values_out_array, fill_value=np.nan, dtype=np.float32)

    # tmp array with positions
    s_tmp_array = range(values_out_array.shape[0])

    # fill arrays values2_array, near_idx_array
    for t_ist in range(values_out_array.shape[1]):

        # get NaN bool
        nan_bool_t_array = np.isnan(values_out_array[:, t_ist])

        # skip if all NaN
        if np.all(nan_bool_t_array):
            continue

        # skip linear interpolation if all valid
        if not np.any(nan_bool_t_array):
            values2_array[:, t_ist] = copy.deepcopy(values_out_array[:, t_ist])
            continue

        # get the valid space indexes at this time instance -> this is sorted by default
        s_val_idx_at_t_array = np.isfinite(values_out_array[:, t_ist]).nonzero()[0]

        # get the number of valid points at this time instance
        n_s_val_idxs_at_t = len(s_val_idx_at_t_array)

        # sanity check
        if n_s_val_idxs_at_t == 0:
            raise AssertionError("this condition should have been caught earlier, check code")

        # 1 valid element, set all nearest values to this except own
        elif n_s_val_idxs_at_t == 1:
            s_ist = s_val_idx_at_t_array[0]
            near_idx_array[:, t_ist] = s_ist
            near_idx_array[s_ist, t_ist] = np.nan
            continue

        # 2 or more elements, piece wise linear interpolation
        values2_array[:, t_ist] = piecewise_linear_interpolation(
            values_in_array=values_out_array[:, t_ist],
            base_in_array=space1_array,
            limits='linear', check_nan=True, float_atol=float_atol,
        )

        sub_s_val_idx_at_t_array = copy.deepcopy(s_val_idx_at_t_array)

        # loop to find nearest valid points for each (valid or invalid) point
        for s_ist in s_tmp_array:

            # if point is valid
            if s_ist in s_val_idx_at_t_array:
                # assign itself in position matrix
                near_idx_array[s_ist, t_ist] = s_ist
                continue

            # check if value was extrapolated
            if s_ist > s_val_idx_at_t_array[-1]:
                # assign in matrix
                near_idx_array[s_ist, t_ist] = s_val_idx_at_t_array[-1]
                # if point is too far, set extrapolated value to NaN
                if np.abs(space1_array[s_val_idx_at_t_array[-1]] - space1_array[s_ist]) > dx_max_out:
                    values2_array[s_ist, t_ist] = np.nan
                continue

            if s_ist < s_val_idx_at_t_array[0]:
                # assign in matrix
                near_idx_array[s_ist, t_ist] = s_val_idx_at_t_array[0]
                # if point is too far, set extrapolated value to NaN
                if np.abs(space1_array[s_val_idx_at_t_array[0]] - space1_array[s_ist]) > dx_max_out:
                    values2_array[s_ist, t_ist] = np.nan
                continue

            # find closest point that was used in the interpolation
            near_sub_s_id, near_s_id = find_nearest([i_ for i_ in sub_s_val_idx_at_t_array if i_ != s_ist], s_ist)

            # if nearest point is located to the left of the point
            if near_s_id < s_ist:
                sub_s_val_idx_at_t_array = sub_s_val_idx_at_t_array[near_sub_s_id:]

            # assign in matrix
            near_idx_array[s_ist, t_ist] = near_s_id

            # if point is too far, set extrapolated value to NaN
            if np.abs(space1_array[near_s_id] - space1_array[s_ist]) > dx_max_in:
                values2_array[s_ist, t_ist] = np.nan

    # free memory
    del s_tmp_array

    return near_idx_array, values2_array


def spatiotemporal_interpolation_drop(
        values0_array: np.ndarray or np.ma.MaskedArray,  # 2D
        space0_array: np.ndarray or np.ma.MaskedArray,  # 1D
        dx_max_in: float,
        dx_max_out: float,
        dw_min: float,
        float_atol: float = SIC4DVarLowCostDefaults().def_float_atol,
        interp_missing_nodes: bool = False,
        clean_run: bool = False,
        debug_mode: bool = False,
) -> np.ndarray:
    """
    FIll values by weighted interpolation considering the linear drops between valid points at each time instance
    Weight is defined according to absolute difference between nearest values (in space) values at the different time
     instances used in each step.

    Parameters
    ----------
    values0_array: np.ndarray | np.ma.MaskedArray
        2D (masked) array with the values to be filled by interpolation. Rows are the nodes (space distribution) while
         columns are the time instances.
    space0_array: np.ndarray | np.ma.MaskedArray
        1D (masked) array with the distance between points (ditance between rows).
    dx_max_in : float
         Max distance between missing points in other time instances in interpolation case.
    dx_max_out : float
         Max distance between missing points in other time instances in extrapolation case.
    dw_min : float
        minimum weight distance for ensemble averaging (actual value is 1/dw).
    float_atol : float
        The float absolute tolerance for piecewise linear interpolation.
    interp_missing_nodes : bool
        Whether to fill nodes that are missing in all time instances.
    clean_run : bool
        Whether to print statements while running this function
    debug_mode : bool
        Whether to print debug statements.

    Returns
    -------
    np.ndarray
        array with NaN values filled by interpolation
    """
    if debug_mode:
        clean_run = False
    if clean_run:
        debug_mode = False

    # preprocess
    (values_out_array, values1_array, space1_array, miss_t_index_array, total_n_nodes, arrays_masked,
     v_fill_value) = __preprocess(
        values0_array=values0_array, space0_array=space0_array, clean_run=clean_run)

    # return if all values are empty
    if np.all(np.isnan(values1_array)):
        if arrays_masked:
            values_out_array = nan_array_to_masked_array(values_out_array, fill_value=v_fill_value)
        return values_out_array

    # return if all values are filled
    if np.all(np.isfinite(values1_array)):
        if arrays_masked:
            values_out_array = nan_array_to_masked_array(values_out_array, fill_value=v_fill_value)
        return values_out_array

    # get boolean array of nodes that can be interpolated
    if interp_missing_nodes:
        pos_node_bool_array = np.full(values1_array.shape[0], fill_value=True)
    else:
        pos_node_bool_array = np.count_nonzero(np.isfinite(values1_array), axis=1) > 0

    # array with missing values linearly interpolated in space (used in case missing cond.) and array to store the
    # position of nearest point in each time instance

    near_idx_array, values2_array = __create_aux_near_and_interp_arrays(
        values_out_array=values_out_array,
        space1_array=space1_array,
        dx_max_in=dx_max_in,
        dx_max_out=dx_max_out,
        float_atol=float_atol,
    )

    # loop in time instances that have missing values which are possible to interpolate
    for t_ist in miss_t_index_array:

        interp_msg = 'interpolation: trying to fill point {} at time ' + str(t_ist)

        # get boolean vector with the missing points in the space dimension at this time instance
        miss_node_at_t_bool_array = np.isnan(values1_array[:, t_ist])

        # TODO: remove this condition to speed up if we are 100% sure of the code
        # no value to interpolate, move to next time instance
        if np.all(~miss_node_at_t_bool_array):
            # miss_t_index_array says there should be at least one point that needs to be filled
            raise AssertionError("this should have been caught earlier, check code")

        # TODO: remove this condition to speed up if we are 100% sure of the code
        # all values should be interpolated, not possible, move to next time instance
        if np.all(miss_node_at_t_bool_array):
            # miss_t_index_array says there should be at least one valid point
            raise AssertionError("this should have been caught earlier, check code")

        if debug_mode:
            miss_node_at_t_index_array = np.nonzero(miss_node_at_t_bool_array)[0]
            pos_node_at_t_index_array = np.nonzero(pos_node_bool_array)[0]
            for miss_n in miss_node_at_t_index_array:
                if miss_n not in pos_node_at_t_index_array:
                    pass

        # get the common indexes between invalid points and valid weights
        miss_node_at_t_index_array = np.nonzero(miss_node_at_t_bool_array & pos_node_bool_array)[0]

        # loop in space instances that are missing for this time instance
        for miss_node_at_t_id in miss_node_at_t_index_array:

            if debug_mode:
                pass

            # get index of nearest (spatially) valid point at the current time instance
            near_node_at_t_id = int(near_idx_array[miss_node_at_t_id, t_ist])

            # get the value of the nearest (spatially) valid point at the current time instance
            near_node_at_t_value = values1_array[near_node_at_t_id, t_ist]

            # TODO: remove this condition to speed up if we are 100% sure of the code
            # check if near value for this point was available
            if np.isnan(near_node_at_t_value):
                raise AssertionError(
                    "near value is missing, but this should have been caught earlier, check code")

            # get the time indexes of the points for this missing node that have valid:
            poss_t_index_array = np.nonzero(
                # linear interpolated value at missing node
                np.isfinite(values2_array[miss_node_at_t_id, :]
                            # linear interpolated value at other node
                            ) & np.isfinite(values2_array[near_node_at_t_id, :])
            )[0]

            if poss_t_index_array.size == 0:
                if debug_mode:
                    pass
                continue

            # initiate variables for filling
            n_points, drop, ss = 0, 0., 0.

            # loop in other time instances that have valid values
            for other_t_ist in poss_t_index_array:

                # skip if same time instance
                if other_t_ist == t_ist:
                    continue

                # get the (possible interpolated) value of the missing point at other time instance
                miss_node_at_other_t_value = values2_array[miss_node_at_t_id, other_t_ist]

                # get the value of the nearest (spatially) point in other time instance
                near_node_at_other_t_value = values2_array[near_node_at_t_id, other_t_ist]

                # TODO: remove this condition to speed up if we are 100% sure of the code
                # if value could not be linearly interpolated (points were too far)
                if np.isnan(miss_node_at_other_t_value):
                    raise AssertionError(
                        "no interpolated values for missing node, but this should have been caught earlier, check code")

                # TODO: remove this condition to speed up if we are 100% sure of the code
                # if value could not be linearly interpolated (points were too far)
                if np.isnan(near_node_at_other_t_value):
                    raise AssertionError(
                        "no interpolated values for near node, but this should have been caught earlier, check code")

                # compute drop_i: difference between missing and nearest values at the other time inst
                drop_i = near_node_at_other_t_value - miss_node_at_other_t_value

                # compute dist_i: absolute difference between nearest in space values at the different time inst
                dist_i = (near_node_at_other_t_value - near_node_at_t_value)**2
                #dist_i = np.abs(near_node_at_other_t_value - near_node_at_t_value)

                # get the inverse of the dist, TODO Isa: added, correct for low values
                dist_i = 1. / max(dist_i, dw_min)

                # add missing value to global drop
                drop += drop_i * dist_i

                # add to ss
                ss += dist_i

                n_points += 1

            if ss > 0.:

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="invalid value encountered")

                    drop = drop / ss
                    values_out_array[miss_node_at_t_id, t_ist] = near_node_at_t_value - drop

                if debug_mode:
                    pass

            else:
                if debug_mode:
                    pass

    if not clean_run:
        pass

    if arrays_masked:
        values_out_array = nan_array_to_masked_array(values_out_array, fill_value=v_fill_value)

    return values_out_array


def spatiotemporal_interpolation_drop_weighted(
        values0_array: np.ndarray or np.ma.MaskedArray,  # 2D
        space0_array: np.ndarray or np.ma.MaskedArray,  # 1D
        weight0_array: np.ndarray or np.ma.MaskedArray,  # 2D
        dx_max_in: float,
        dx_max_out: float,
        dw_min: float,
        weight_exp_beta: float = 1e-2,#SIC4DVarLowCostDefaults().def_lsm_w_weight_exp_beta,
        float_atol: float = 1e-2, #SIC4DVarLowCostDefaults().def_float_atol,
        clean_run: bool = False,
        debug_mode: bool = False,
) -> np.ndarray:
    """
    FIll values by weighted interpolation considering the linear drops between valid points at each time instance
    Weight is defined according to the values of weight0_array.

    Parameters
    ----------
    values0_array: np.ndarray | np.ma.MaskedArray
        2D (masked) array with the values to be filled by interpolation. Rows are the nodes (space distribution) while
         columns are the time instances.
    space0_array: np.ndarray | np.ma.MaskedArray
        1D (masked) array with the distance between points (ditance between rows).
    weight0_array: np.ndarray | np.ma.MaskedArray
        2D (masked) array with the values of the variable to be used for the weight. Rows are the nodes (space
         distribution) while columns are the time instances.
    dx_max_in : float
         Max distance between missing points in other time instances in interpolation case.
    dx_max_out : float
         Max distance between missing points in other time instances in extrapolation case.
    dw_min : float
        minimal weight distance for ensemble averaging.
    weight_exp_beta : float
        the value of Beta in the exponential function which determines the weight of the observations.
    float_atol : float
        The float absolute tolerance for piecewise linear interpolation.
    clean_run : bool
        Whether to print statements while running this function
    debug_mode : bool
        Whether to print debug statements.

    Returns
    -------
    np.ndarray
        array with NaN values filled by interpolation
    """
    if debug_mode:
        clean_run = False
    if clean_run:
        debug_mode = False

    # shape check
    if values0_array.shape != weight0_array.shape:
        raise TypeError("shape mismatch between values and weights")

    # preprocess
    (values_out_array, values1_array, space1_array, miss_t_index_array, total_n_nodes, arrays_masked,
     v_fill_value) = __preprocess(
        values0_array=values0_array, space0_array=space0_array, clean_run=clean_run)

    # return if all values are empty
    if np.all(np.isnan(values1_array)):
        if arrays_masked:
            values_out_array = nan_array_to_masked_array(values_out_array, fill_value=v_fill_value)
        return values_out_array

    # return if all values are filled
    if np.all(np.isfinite(values1_array)):
        if arrays_masked:
            values_out_array = nan_array_to_masked_array(values_out_array, fill_value=v_fill_value)
        return values_out_array

    # array with missing values linearly interpolated in space (used in case missing cond.)
    _, values2_array = __create_aux_near_and_interp_arrays(
        values_out_array=values_out_array,
        space1_array=space1_array,
        dx_max_in=dx_max_in,
        dx_max_out=dx_max_out,
        float_atol=float_atol,
    )

    # if weight was masked
    weight1_array = masked_array_to_nan_array(weight0_array)

    # set interp values to NaN where weight array is NaN
    values2_array[np.isnan(weight1_array)] = np.nan

    # TODO Isa: added compute normalize constant for the weight
    weight_k = np.nanmax(weight1_array) - np.nanmin(weight1_array)

    # loop in time instances that have missing values which are possible to interpolate
    for t_ist in miss_t_index_array:

        interp_msg = 'interpolation: trying to fill point {} at time ' + str(t_ist)

        # get boolean vector with the missing points in the space dimension at this time instance
        miss_node_at_t_bool_array = np.isnan(values1_array[:, t_ist])

        # TODO: remove this condition to speed up if we are 100% sure of the code
        # no value to interpolate, move to next time instance
        if np.all(~miss_node_at_t_bool_array):
            # miss_t_index_array says there should be at least one point that needs to be filled
            raise AssertionError("this should have been caught earlier, check code")

        # TODO: remove this condition to speed up if we are 100% sure of the code
        # all values should be interpolated, not possible, move to next time instance
        if np.all(miss_node_at_t_bool_array):
            # miss_t_index_array says there should be at least one valid point
            raise AssertionError("this should have been caught earlier, check code")

        # get boolean array with the valid weight points in the space dimension at this time instance
        val_weight_node_at_t_bool_array = np.isfinite(weight1_array[:, t_ist])

        if debug_mode:
            miss_n_at_t_index_array = np.nonzero(miss_node_at_t_bool_array)[0]
            val_weight_n_t_index_array = np.nonzero(val_weight_node_at_t_bool_array)[0]
            for miss_node_at_t_id in miss_n_at_t_index_array:
                if miss_node_at_t_id not in val_weight_n_t_index_array:
                    pass

        # get the common indexes between invalid points and valid weights
        miss_node_at_t_index_array = np.nonzero(miss_node_at_t_bool_array & val_weight_node_at_t_bool_array)[0]

        # loop in space instances that are missing for this time instance
        for miss_node_at_t_id in miss_node_at_t_index_array:

            if debug_mode:
                pass

            # get the weight value of this point in space and time
            miss_node_at_t_weight_value = weight1_array[miss_node_at_t_id, t_ist]

            # TODO: remove this condition to speed up if we are 100% sure of the code
            # check if weight value for this point was available
            if np.isnan(miss_node_at_t_weight_value):
                raise AssertionError(
                    "weight is missing, but this should have been caught earlier, check code")

            # get the time indexes of the points for this missing node that have valid:
            # linear interpolated value and weight.
            # Why checking with only one? because values2 was masked
            poss_t_index_array = np.isfinite(values2_array[miss_node_at_t_id, :]).nonzero()[0]

            if poss_t_index_array.size == 0:
                if debug_mode:
                    pass
                    # raise AssertionError('point not filled, no values available for filling')
                continue

            # initiate variables for filling
            n_points, drop, ss = 0, 0., 0.

            # loop in other time instances that have valid (possible interpolated) values
            for other_t_ist in poss_t_index_array:

                # skip if same time instance
                if other_t_ist == t_ist:
                    continue

                # get the (possibly interpolated) value of the missing point at other time instance
                miss_node_at_other_t_value = values2_array[miss_node_at_t_id, other_t_ist]

                # get the weight value of the missing point at other time instance
                miss_node_at_other_t_weight_value = weight1_array[miss_node_at_t_id, other_t_ist]

                # TODO: remove this condition to speed up if we are 100% sure of the code
                if np.isnan(miss_node_at_other_t_value):
                    raise AssertionError(
                        'interpolated value at other time is missing, but this should have been caught earlier')
                # TODO: remove this condition to speed up if we are 100% sure of the code
                if np.isnan(miss_node_at_other_t_weight_value):
                    raise AssertionError(
                        'weight at other time is missing, but this should have been caught earlier')

                # compute delta_w, the absolute difference in weight between the points in different time inst
                delta_w = np.abs(miss_node_at_t_weight_value - miss_node_at_other_t_weight_value)

                # TODO Isa: added normalize by weight
                delta_w /= weight_k

                # y = exp(-x), when x [0, ∞) -> y [1, 0), the higher x, the lower y
                dist_i = np.exp(-weight_exp_beta * delta_w)  # the closer the weights, the lower higher dist_i

                # TODO Isa: added, correct for low values
                #dist_i = max(dist_i, dw_min)

                # add missing value to global drop
                drop += miss_node_at_other_t_value * dist_i

                # add to ss
                ss += dist_i

                n_points += 1

            if ss > 0.:

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="invalid value encountered")

                    values_out_array[miss_node_at_t_id, t_ist] = drop / ss

                if debug_mode:
                    pass

            else:
                if debug_mode:
                    pass

    if not clean_run:
        pass

    if arrays_masked:
        values_out_array = nan_array_to_masked_array(values_out_array, fill_value=v_fill_value)

    return values_out_array



def igor_method(node_z, corx, node_x, LSMX, eps1, eps2):
    k_T1 = node_z.shape[1]
    lmax = node_z.shape[0]
    rel0 = np.zeros(lmax)
    orig_z = deepcopy(node_z)

    for ist in range(0, k_T1):
        ss=0.
        for n in range(0, lmax):
            ss = ss + node_z[n, ist]
        ZMEAN0 = ss / lmax
        ss0 = 0.
        for n in range(0, lmax):
            ss0 = ss0 + (node_z[n, ist] - ZMEAN0) ** 2
        ss0 = np.sqrt(ss0 / lmax)

        #Exponential stationary correlation function
        for n in range(1, lmax):
            rel0[n] = 1.0 - np.exp(-1.0*((node_x[n]-node_x[n-1])/corx))

        #Relaxation sweeps
        ss1 = 0.
        for it in range(0, LSMX):
            i_reo = 0
            for n in range(lmax-2, 0, -1):
                node_z[n, ist]=node_z[n+1, ist] * (1.0 - rel0[n+1]) + node_z[n, ist] * rel0[n+1]
                #New part: achieving continuous decrease of the WSE
                if it > 0:
                    if (node_z[n,ist] < node_z[n+1,ist] - eps1):
                        ss=node_z[n, ist]
                        node_z[n, ist]=node_z[n+1, ist]
                        node_z[n+1, ist]=ss
                        i_reo = i_reo + 1

            for n in range(1, lmax):
                node_z[n, ist] = node_z[n-1, ist] * (1.0 - rel0[n]) + node_z[n, ist] * rel0[n]
                #New part: achieving continuous decrease of the WSE
                if it > 0:
                    if (node_z[n,ist] > node_z[n-1,ist] + eps1):
                        ss=node_z[n-1, ist]
                        node_z[n-1, ist]=node_z[n, ist]
                        node_z[n, ist]=ss
                        i_reo = i_reo + 1

            ss=0.
            for n in range(0, lmax):
                ss = ss + (node_z[n, ist] - orig_z[n, ist]) ** 2
            ss = np.sqrt(ss / lmax)
            if it > 0:
                if np.abs((ss - ss1)/ss0) > eps2:
                    ss1=ss
                else:
                    break
            
            else:
                ss1=ss

        #ZSTD[ist] = ss1
        #LSMX0 = it

        #Bias correction
        ss=0.
        for n in range(0, lmax):
            ss=ss+node_z[n, ist]
        ZMEAN1 = ss / lmax
        for n in range(0, lmax):
            node_z[n, ist] = node_z[n, ist] + ZMEAN0 - ZMEAN1
        # for n in range(0, lmax):
        #     node_z[n, ist] = node_z[n, ist] * (ZMEAN0/ZMEAN1)

    return node_z

def nan_substitution_elevation(array, cslope, node_x):
    from lib.lib_verif import check_na
    # ivalid = -1
    # nx = node_z.shape[0]
    # for n in range(0, nx):
    #     if not check_na(node_z[n]):
    #         ivalid=n
    #     else:
    #         if ivalid==-1:
    #             for n1 in range(n+1, nx):
    #                 if not check_na(node_z[n1]):
    #                     ivalid=n1

    #         node_z[n] = node_z[ivalid] + mean_elevation_profile[n] - mean_elevation_profile[ivalid]

    i_vl=-1 #valid left
    i_vr2=-1 #valid right
    #i_vr1 = 0
    i_vr = -1

    new_array = deepcopy(array)

    for n in range(0, len(array)):
        if not check_na(array[n]):
            i_vl = n
            new_array[n] = array[n]
        else:
            if n > i_vr:
                i_vr1 = -1
                if i_vr2 != 0:
                    for n1 in range(n+1, len(array)):
                        if not check_na(array[n1]):
                            i_vr1 = n1
                            break
                        else:
                            pass

            i_vr=i_vr1
            if i_vr1 == -1:
                i_vr2 = 0

            #update
            if (i_vl!=-1) and (i_vr!=-1):
                new_array[n] = array[i_vl] + (array[i_vr] - array[i_vl]) * (node_x[n] - node_x[i_vl])/(node_x[i_vr] - node_x[i_vl])
                logging.info(f"Interpolating node {n} between valid nodes {i_vl} and {i_vr}")
            elif (i_vl==-1) and (i_vr!=-1):
                new_array[n] = array[i_vr] + cslope * (node_x[n] - node_x[i_vr])
                logging.info(f"Extrapolating to the left node {n} using valid node {i_vr}")
            elif (i_vl!=-1) and (i_vr==-1):
                new_array[n] = array[i_vl] + cslope * (node_x[n] - node_x[i_vl])
                logging.info(f"Extrapolating to the right node {n} using valid node {i_vl}")
            elif (i_vl==-1) and (i_vr==-1):
                new_array[n] = np.nan
                logging.warning(f"Cannot interpolate or extrapolate node {n} because no valid nodes found.")
        
    
    # for n in range(0, nx):
    #     if check_na(node_z[n]):
    #         node_z[n] = node_z[ivalid] + mean_elevation_profile[n] - mean_elevation_profile[ivalid]

    return new_array


def Extrapolation(
    node_z, node_w, node_x, reach_t, corx, corx_array, cort_wse, cort_width,  \
    gnuplot_saving=False, reach_id=None, output_dir=None, run_type="seq", use_large_deviations=True,
    start_from_downstream=False, run_preprocessing=True, run_extrapolation=True, pooling=False, regularization_profile=False, 
    use_quantiles_for_densification=False, 
    ):
    #sic4dvar_dict, params
    ## Authors   : Hind OUBANAS, Igor GEJADZE, Isadora REZENDE DE OLIVEIRA, Dylan QUITTARD
    ## Date     : November 2023
    
    """
    Interpolates SWOT WSE observations in order to have all nodes
    observed at all SWOT time instants.
    Option 1 : Fortran version Uses weighted drops computed using
    the same pair of unknown and observed nodes from all overpasses.
    The drops (delta h between both nodes of tha same overpass) is
    weighted by the distance in time (delta h between the same node
    obeserved two different time instants).
    NOTE: This version involves a smoothing step before and after
    interpolation.

    Parameters
    ----------
    """

    node_z_ini = deepcopy(node_z)

    optionnal_output_dict = {}
            
    ######## WATER SURFACE ELEVATION ########
    #########################################
        
    optionnal_output_dict['tmp_interp_values'] = []  # save data between interpolation and smoothing for comparison (Z)
    optionnal_output_dict['tmp_interp_values_w'] = []  # same for W

    reverse_order = False
    if use_large_deviations and run_type=="seq":
        
        # node_z, reverse_order = global_large_deviations_removal_relative(sic4dvar_dict['input_data']['node_x'],
        #                                                           sic4dvar_dict['input_data']["node_z_ini"])
        #                                                           #times_debug=reach_t)        
                                                
        # node_z, c1, c2 = global_large_deviations_removal(sic4dvar_dict['input_data']['node_x'],
        #                                                           sic4dvar_dict['input_data']["node_z_ini"],
        #                                                           times_debug=reach_t)
        
        node_z, reverse_order, c1, c2 = global_large_deviations_removal_experimental(
            node_x, node_z, reach_t, times_debug=reach_t
        )

        node_z_devia = deepcopy(node_z)

        optionnal_output_dict['tmp_interp_values'].append(node_z)

    behavior = ""
    if reverse_order:
        if start_from_downstream:
             behavior = "decrease"
        else:
            behavior="increase"
        logging.info("c1 > 0. : reverse order of nodes for interpolation")
    else:
        if start_from_downstream:
             behavior = "increase"
        else:
            behavior = "decrease"
        logging.info("c1 < 0. : normal order of nodes for interpolation")

    optionnal_output_dict['reverse_order'] = reverse_order

    # if pooling:
    #     pooling(optionnal_output_dict, node_z)

    # if not start_from_downstream:
    #     node_z = node_z[::-1, :]
    #     node_w = node_w[::-1, :]
    #     node_x = node_x[::-1]
    #     logging.info("Reversing order of nodes for interpolation and smoothing only")

    if run_preprocessing:
        # Relaxation sweeps in space for WSE / z 1st smoothing
        node_z = relaxation_sweeps_in_dim(
            dim=1,
            value0_array=node_z,
            base0_array=node_x,
            max_iter=1,#LSMX,
            cor=corx_array,
            always_run_first_iter=True,  # TODO Isa: ask Hind/Igor
            #behavior="",  # TODO Isa: added option
            behavior=behavior,  # TODO Isa: added option
            inter_behavior=True,  # TODO Isa: added option
            inter_behavior_min_thr=1e-2,#def_float_atol,  # TODO Isa: added option
            inter_behavior_max_thr=np.inf,#DX_max_in,
            #check_behavior="",  # TODO Isa: added option
            check_behavior="force",  # TODO Isa: added option
            min_change_v_thr=1e-4,
            plot=False,
            plot_title="Relaxation sweep in space 1 WSE without Interchange",
            clean_run=True,
            debug_mode=False,
            time_integration=False
        )
 

    optionnal_output_dict['tmp_interp_values'].append(node_z)

    if regularization_profile:
        if c1 == 0.:
            c1 = -1e-6
        else:
            c1 = c1 * 1e-3

        if not use_quantiles_for_densification:
            logging.info("Using mean elevation profile for densification")
            from sic4dvar_functions.sic4dvar_helper_functions import compute_mean_elevation_profile
            mean_elevation_profile = compute_mean_elevation_profile(node_z, reach_t)
            new_mean_elevation_profile = nan_substitution_elevation(mean_elevation_profile, c1, node_x)

            # new_node_z = np.zeros((node_z.shape[0], node_z.shape[1]+1))
            # new_node_z[:,0:node_z.shape[1]] = node_z
            # new_node_z[:,node_z.shape[1]] = new_mean_elevation_profile

            # new_reach_t = np.zeros(node_z.shape[1]+1)
            # new_reach_t[0:node_z.shape[1]] = reach_t
            # new_reach_t[node_z.shape[1]] = reach_t[-1] + 21.0 * 24.0 * 3600.0  # add 21 days to the last time step

            c1 = 0.
            mean_width_profile = compute_mean_elevation_profile(node_w, reach_t)
            new_mean_width_profile = nan_substitution_elevation(mean_width_profile, c1, node_x)

            node_z[:,-1] = new_mean_elevation_profile
            node_w[:,-1] = new_mean_width_profile

        else:
            logging.info("Using quantiles for densification")
            #Option use quantiles
            quantile_matrix0 = np.zeros((node_z.shape[0], 3)) #np.zeros((3, node_z.shape[1]))
            for n in range(0,node_z.shape[0]):
                quantile_matrix0[n,:] = np.nanquantile(node_z[n,:], [0.33, 0.66, 0.99])

            for nb_quant in range(0,3):
                quantile_matrix0[:,nb_quant] = nan_substitution_elevation(quantile_matrix0[:,nb_quant], c1, node_x)
                node_z[:, -1-nb_quant] = quantile_matrix0[:,nb_quant]

            # from matplotlib import pyplot as plt
            # plt.plot(quantile_matrix0[:,0], label="quantile 0.33" )
            # plt.plot(quantile_matrix0[:,1], label="quantile 0.66" )
            # plt.plot(quantile_matrix0[:,2], label="quantile 0.99" )
            # plt.legend()
            # plt.show()

            # from matplotlib import pyplot as plt
            # plt.plot(node_z[:,-1], label="node_z 0.33" )
            # plt.plot(node_z[:,-2], label="node_z 0.66" )
            # plt.plot(node_z[:,-3], label="node_z 0.99" )
            # plt.legend()
            # plt.show()
            
            # print(bug)
        # new_node_w = np.zeros((node_w.shape[0], node_w.shape[1]+1))
        # new_node_w[:,0:node_w.shape[1]] = node_w
        # new_node_w[:,node_w.shape[1]] = new_mean_width_profile

        # # from matplotlib import pyplot as plt
        # # plt.plot(mean_elevation_profile, label="mean_elevation_profile" )
        # # plt.plot(new_mean_elevation_profile, label="new_mean_elevation_profile", linestyle='dashed')
        # # plt.legend()
        # # plt.show()
        # # print(bug)
        # # 
        # node_z = new_node_z
        # reach_t = new_reach_t 
        # node_w = new_node_w
        

    node_z_in_between = deepcopy(node_z)

    # Interp/Extrapolation for WSE  / z 1st interpolation
    if run_extrapolation:
        node_z = spatiotemporal_interpolation_drop(
            values0_array=node_z, #node_z,
            space0_array=node_x,
            dx_max_in=np.inf,#DX_max_in,
            dx_max_out=500,#DX_max_out,
            dw_min=1e-2,
            clean_run=False,
            debug_mode=False,
            interp_missing_nodes = True
        )

    

    optionnal_output_dict['tmp_interp_values'].append(node_z)


    if run_preprocessing:
        #TO DO:
        for i in range(0,1): 
            # 2nd relaxation sweeps in space with interchange for WSE  / z 2nd smoothing
            node_z = relaxation_sweeps_in_dim(
                dim=1,
                value0_array=node_z,
                base0_array=node_x,
                max_iter=10, #LSMX,
                cor=corx_array,
                always_run_first_iter=True,  # TODO Isa: ask Hind/Igor
                behavior=behavior, #decrease # TODO Isa: added option
                inter_behavior=True,
                inter_behavior_min_thr=1e-2,#def_float_atol,  # TODO Isa: added option
                inter_behavior_max_thr=500,#DX_max_in,
                # inter_behavior_thr=1e-2,#def_float_atol,  # TODO Isa: added option
                check_behavior="force",  # TODO Isa: added option  -> Ask Igor/Hind
                min_change_v_thr=1e-4,  # TODO Isa: added option
                plot=False,
                plot_title="Relaxation sweep in space 2 WSE With Interchange",
                clean_run=True,
                debug_mode=False,
                time_integration=False,
            )

    optionnal_output_dict['tmp_interp_values'].append(node_z)

    # if pooling:
    #     node_z = pooling(sic4dvar_dict, node_z)

    tmp = deepcopy(node_z)
    if run_preprocessing:
        # Relaxation sweeps in time for WSE / z 2nd smoothing
        node_z = relaxation_sweeps_in_dim(
            dim=0,
            value0_array=node_z,
            base0_array=reach_t,
            max_iter=1, #LSMT,
            cor=cort_wse,
            always_run_first_iter=False,  # TODO Isa: ask Hind/Igor
            behavior="",  # TODO Isa: added option
            inter_behavior=False,  # TODO Isa: added option
            inter_behavior_min_thr=1e-2,#def_float_atol,  # TODO Isa: added option
            inter_behavior_max_thr=500,#DX_max_in,
            check_behavior="",  # TODO Isa: added option
            min_change_v_thr=1e-4,
            plot=False,
            plot_title="Relaxation sweep in time WSE without Interchange",
            clean_run=True,
            debug_mode=False,
            time_integration=False,
        )

    optionnal_output_dict['tmp_interp_values'].append(node_z)
    #sic4dvar_dict['tmp_interp_values'].append(test_swot_z_obs2)


    ######## WIDTHS ########
    ########################

    if run_preprocessing:

        # Relaxation sweeps in space for widths / w 1st smoothing
        node_w = relaxation_sweeps_in_dim(
            dim=1,
            value0_array=node_w,
            base0_array=node_x,
            max_iter=1, #LSMX,
            cor=corx,
            always_run_first_iter=True,  # TODO Isa: ask Hind/Igor
            behavior="",  # TODO Isa: added option
            inter_behavior=False,  # TODO Isa: added option
            check_behavior="",  # TODO Isa: added option
            min_change_v_thr=1e-2,
            plot=False,
            plot_title="Relaxation sweep in space 1 W without Interchange",
            clean_run=True,
            debug_mode=False,
            time_integration=False,
        )

    optionnal_output_dict['tmp_interp_values_w'].append(node_w)
    
    if run_extrapolation:
        # Interp/Extrapolation for W / w 2nd interpolation
        node_w = spatiotemporal_interpolation_drop_weighted(  # spatiotemporal_interpolation_weighted(
            values0_array=node_w,
            space0_array=node_x,
            weight0_array=node_z,
            dx_max_in=np.inf,#DX_max_in,
            dx_max_out=500,#DX_max_out,
            dw_min=1e-1,
            clean_run=False,
            debug_mode=False,
        )

    optionnal_output_dict['tmp_interp_values_w'].append(node_w)

    # if mean_width_profile is not None:
    #     logging.info("Filling missing values with mean elevation profile")
    #     from lib.lib_verif import check_na
    #     for t in range(0, len(reach_t)):
    #         for n in range(0, len(node_x)):
    #             if check_na(node_w[n,t]):
    #                 node_w[n,t] = mean_width_profile[n]

    #     if gnuplot_saving:
    #         reach_id = str(reach_id)
    #         reach_id = verify_name_length(reach_id)
    
    #         nodes2 = (node_x - node_x[0]) / 1000
    #         times2 = reach_t / 3600 / 24
    
    #         output_path = Path(output_dir).joinpath("gnuplot_data", str(reach_id))
    #         if not Path(output_path).is_dir():
    #             Path(output_path).mkdir(parents=True, exist_ok=True)
    
    #         output_path = Path(output_dir).joinpath("gnuplot_data", str(reach_id), "after_extrapolation_nan_filled_w")
    #         gnuplot_save(nodes2, times2, node_z, node_w, output_path, np.min(node_z), 2)
    
    if run_preprocessing:

        # 2nd relaxation sweeps in space for W without interchange / w 2nd smoothing - x
        node_w = relaxation_sweeps_in_dim(
            dim=1,
            value0_array=node_w,
            base0_array=node_x,
            max_iter=1, #LSMX,
            cor=corx,
            always_run_first_iter=True,  # TODO Isa: ask Hind/Igor
            behavior="",  # TODO Isa: added option
            inter_behavior=False,  # TODO Isa: added option
            check_behavior="",  # TODO Isa: added option
            min_change_v_thr=1e-2,
            plot=False,
            plot_title="Relaxation sweep in space 2 W without Interchange",
            clean_run=True,
            debug_mode=False,
            time_integration=False,
        )
        
    optionnal_output_dict['tmp_interp_values_w'].append(node_w)
    
    # if pooling:
    #     node_w = pooling(sic4dvar_dict, node_w)

    if run_preprocessing:
        # Relaxation sweeps in time for W / w 2nd smoothing - t
        node_w = relaxation_sweeps_in_dim(
            dim=0,
            value0_array=node_w,
            base0_array=reach_t,
            max_iter=1, #LSMT,
            cor=cort_wse,#cort_width,
            always_run_first_iter=True,  # TODO Isa: ask Hind/Igor
            behavior="",  # TODO Isa: added option
            inter_behavior=False,  # TODO Isa: added option
            check_behavior="",  # TODO Isa: added option
            min_change_v_thr=1e-2,
            plot=False,
            plot_title="Relaxation sweep in time W without Interchange",
            clean_run=True,
            debug_mode=False,
            time_integration=False,
        )

    optionnal_output_dict['tmp_interp_values_w'].append(node_w)

    """ if not start_from_downstream:
        node_z = node_z[::-1, :]
        node_w = node_w[::-1, :]
        node_x = node_x[::-1]
        logging.info("Reversing order of nodes for interpolation and smoothing only") """

    # sic4dvar_dict['input_data']["node_z"] = node_z.filled(np.nan)
    # sic4dvar_dict['input_data']["node_w"] = node_w.filled(np.nan)

    
    return np.ma.filled(node_z, np.nan), np.ma.filled(node_w, np.nan), optionnal_output_dict
    #node_z.filled(np.nan), node_w.filled(np.nan), optionnal_output_dict


def modified_manning_integrated(node_xr, node_yr, node_a, node_p, Wmean, last_node_for_integral, Zb2, \
    ZB_update, Zb, t, node_x, SLOPEM1, ZM=1., KMI=1., option_recompute_area = False, sic4dvar_dict=[]):
    SS1 = 0.

    tmp_R_array = []
    ## Start node loop
    #for n in range(len(node_z)): ## Iterate over nodes 
    for n in range(last_node_for_integral):
        #Zmin, Wmin = node_yr[n].min(), 2*node_xr[n].min()
        #Zmin, Wmin = node_yr[n].min(), node_xr[n].min()
        Zmin, Wmin = np.nanmin(node_yr[n]), np.nanmin(node_xr[n])

        if Wmin == 0:
            Wmin = 1.0 #sorted(set(2*node_xr[n]))[1]
            #D.Q: put the min value as the second minimal value when min = 0 (testing only)

        if t == 0: ## Calc. initial value of Zb for section
            #Zb[n] = node_yr[n][0] + Zb2 + (i2 - 1) * dZb * (Wmean[n]/(2*node_xr[n][0])) ## NOTE corrected equ.
            #Zb[n] = node_yr[n][0] + Zb2 + (i2 - 1) * dZb * (Wmean[n]/(2*Wmin)) ## NOTE corrected equ.
            #modif. D.Q: Wmean is a scalar
            #Zb[n] = node_yr[n][0] + Zb2 + (i2 - 1) * dZb * (Wmean/(2*Wmin)) ## NOTE corrected equ.
            Zb[n] = node_yr[n][0] + Zb2 + ZB_update * (Wmean/(Wmin)) ## NOTE corrected equ.

        #Zmin, Wmin = node_yr[n].min(), 2*node_xr[n].min()

        Z1 = node_yr[n][0] ## From approx_section_v6()
        #W1 = 2*node_xr[n][0] ## From approx_section_v6()
        W1 = node_xr[n][0] ## From approx_section_v6()

        if Zmin != Z1: 
            pass
            #self.logger.error('Z1 must be minimum value of Yr!')
            # assert Zmin == Z1, "Z1 must be minimum value of Yr!"
        if Wmin != W1:
            pass
            #self.logger.error('W1 must be minimum value of 2*Xr!')
            # assert Wmin == W1, "W1 must be minimum value of 2*Xr!"
        if Wmin == 0 or check_na(Wmin):
            pass
        
        #D.Q: modif W1 = 0 -> choose second minimal value (testing only)
        if W1 == 0:
            pass
            #W1 = 1.0 #node_xr[n][1]

        if option_recompute_area:
            # node_a, node_p, \
            # node_r, node_w_simp, _ = call_func_APR(sic4dvar_dict['filtered_data']["node_w"], \
            # sic4dvar_dict['filtered_data']["node_z"], \
            # node_xr, node_yr, params, sic4dvar_dict['param_dict'], coeff_array=[], t=t)
            unpad_node_xr = node_xr[n][~np.isnan(node_xr[n])]
            unpad_node_yr = node_yr[n][~np.isnan(node_yr)[n]]

            node_a[n], node_p[n], _, _ = fnc_APR( [sic4dvar_dict["filtered_data"]["node_z"][n,t] * ZM], unpad_node_xr, unpad_node_yr)
            #node_a[n], node_p[n], _, _ = fnc_APR( [sic4dvar_dict["filtered_data"]["node_z"][n,t] * ZM], node_xr[n], node_yr[n])
            #print(node_a[n][t], ZM, sic4dvar_dict["filtered_data"]["node_z"][n,t] * ZM)
            #print(buggy)

        ## NOTE this formation below for AA & PP, came from POM in email "Re: [prj.swot] Files for Algo 3.1 & 5" @ 14 Avril 2021 16:11
        A1 = node_a[n][t] ## A from fnc_APR()
        P1 = node_p[n][t] ## P from fnc_APR()

        # if check_na(A1) or check_na(P1):
        #     print(n, t, sic4dvar_dict["filtered_data"]["node_z"][n,t], node_xr[n], node_yr[n])
        #     print(A1, P1)
        #     print(node_a, node_p)
        #     print(buggy)
        
        A0 = W1 * (Z1 - Zb[n]) ## Wetted area
        if check_na(A0):
            pass

        P0 = 2*(Z1 - Zb[n]) ## Wetted perimeter # W1 + 2*(Z1 - Zb[n]) # W1 already added to P
        AA = A0 + A1 ## Wetted + dry area
        PP = P0 + P1 ## Wetted + dry perimeter
        if AA < 0:
            pass
            #logging.info("Area (AA) must be > 0 !")
            #self.logger.error('Area (AA) must be +ve!')
        if PP < 0:
            pass
            #logging.info("Perimeter (PP) must be > 0 !")
            #self.logger.error('Perimeter (PP) must be +ve!')
        R = AA/PP ## Hydraulic Radius
        tmp_R_array.append(R)

        ## Original equation: Q = math.sqrt(S)*A*Km*R**(2/3) where S = 1.0 & Km = 1.0.
        #print("A0, W1, Z1, Zb[n], bb:", A0, W1, Z1, Zb[n], bb)
        
        QMi2i3 = AA*math.pow(R, 2.0/3.0) ## Also known as Simplified Mannings-Strickler flow law
        # print("R=", R, "A0=", A0, "A1=", A1, "P0=", P0, "P1=", P1, "node_z=", sic4dvar_dict["filtered_data"]["node_z"][n,t])
        # print("n=", n,"node_xr[n], node_yr[n]=", node_xr[n], node_yr[n])
        #temp_Q_t.append(QMi2i3)
        
        if n == 0:
            QM0 = QMi2i3
        else:
            #SS1 += abs(self.input_data['node_x'][n] - self.input_data['node_x'][n-1])*(math.pow(QM0, -2) + math.pow(QMi2i3, -2))/2.0 ## Intergeral
            # H.O. : Modification. We need to use filtered X

            #D.Q: Ensure QM0 and QMi2i3 > 0:
            if QM0 <= 0 or QMi2i3 <= 0:
                pass
                #print("n, t:", n, t)
                #print(bug)
            
            if (QM0 > 0 and QMi2i3 > 0):# and (not check_na(QM0) and not check_na(QMi2i3)):

                SS1 += abs(node_x[n] - node_x[n-1]) \
                    *(math.pow(QM0, -2) + math.pow(QMi2i3, -2))/2.0 ## Integral    

            QM0 = QMi2i3

    #End Node loop
    tmp_R_array = np.array(tmp_R_array)
    tmp_R_array = np.mean(tmp_R_array)

    #Original code:
    print("t=", t, "SLOPEM1=", SLOPEM1)
    Sl = SLOPEM1[t] * ZM #TODO * ZM

    #if (Sl/abs(self.input_data['node_x'][-1]  - self.input_data['node_x'][0])) < 1e-6: ## NOTE 1e-7 recommended by Igor.
    #    Sl = abs(self.input_data['node_x'][-1] - self.input_data['node_x'][0]) * 1e-6 ## NOTE 1e-7 recommended by Igor.
    # H.O. : Modification. We need to use filtered X

    if (Sl/abs(node_x[-1]  - node_x[0])) < 1e-6: ## NOTE 1e-7 recommended by Igor.
        Sl = abs(node_x[-1] - node_x[0]) * 1e-6 ## NOTE 1e-7 recommended by Igor.

    #temp_ss1_array.append(SS1)
    # print("SS1=", SS1, "KMI=", KMI)
    # print("Sl=", Sl, "QMi2i3=", QMi2i3)
    QMi2i3 = np.sqrt(Sl/SS1) * KMI #TODO * friction

    return QMi2i3, SS1, Zb, tmp_R_array

