"""
Extract discharge paramaters from Confluence

This contains libraries to read from the integrator formatted
files. Adapted from functions in ReadQParams.py

"""
# Standard imports
from glob import glob
from pathlib import Path
import warnings
import os

# Third-party imports
from netCDF4 import Dataset
import numpy as np


def extract_alg(alg_dir, r_id, run_type):
    """Extracts and stores reach-level FLPE algorithm data in alg_dict.
    Parameters
    ----------
    alg_dir: Path
        path to FLPE algorithm directory
    r_id: int
        unique reach identifier
    run_type: str
        constrained or unconstrained data product indicator
    """

    non_run_type = "constrained" if run_type == "unconstrained" \
        else "unconstrained"
    non_run_array = np.array([-9999], dtype=float)
    int_file = os.path.join(alg_dir, f'{int(r_id)}_integrator.nc')
    int_file = str(int_file)
    int_data = Dataset(int_file, 'r', format="NETCDF4")

    # note constrained/unconstrained not implemented yet
    alg_dict = {
        'unconstrained': {
            'MetroMan': {},
            'BUSBOI': {},
            'HiVDI': {},
            'MOMMA': {},
            'SADS': {},
            'SIC4DVar': {}
        },
        'constrained': {
            'MetroMan': {},
            'BUSBOI': {},
            'HiVDI': {},
            'MOMMA': {},
            'SADS': {},
            'SIC4DVar': {}
        }
    }

    # busboi - to be implemented
    alg_dict[run_type]['BUSBOI'] = {
        "n": int_data["busboi"]["n"][:].filled(np.nan),
        "Abar": int_data["busboi"]["a0"][:].filled(np.nan),
        "sbQ_rel": int_data["busboi"]["sbQ_rel"][:].filled(np.nan)
    }
    alg_dict[non_run_type]['BUSBOI'] = {
        "n": non_run_array,
        "Abar": non_run_array,
        "sbQ_rel": int_data["busboi"]["sbQ_rel"][:].filled(np.nan)
    }

    # hivdi
    alg_dict[run_type]['HiVDI'] = {
        "alpha": int_data["hivdi"]["alpha"][:].filled(np.nan),
        "beta": int_data["hivdi"]["beta"][:].filled(np.nan),
        "Abar": np.array(int_data["hivdi"]["Abar"][:].filled(np.nan)),
        "sbQ_rel": int_data["hivdi"]["sbQ_rel"][:].filled(np.nan)

    }
    alg_dict[non_run_type]['HiVDI'] = {
        "alpha": non_run_array,
        "beta": non_run_array,
        "Abar": non_run_array,
        "sbQ_rel": int_data["hivdi"]["sbQ_rel"][:].filled(np.nan)
    }

    # momma
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        alg_dict[run_type]['MOMMA'] = {
            "B": int_data["momma"]["B"][:].filled(np.nan),
            "H": int_data["momma"]["H"][:].filled(np.nan),
            "Save": np.nanmean(int_data["momma"]["Save"][:].filled(np.nan)),
            "sbQ_rel": int_data["momma"]["sbQ_rel"][:].filled(np.nan)
        }
        alg_dict[non_run_type]['MOMMA'] = {
            "B": non_run_array,
            "H": non_run_array,
            "Save": non_run_array,
            "sbQ_rel": int_data["momma"]["sbQ_rel"][:].filled(np.nan)
        }

    # sad
    alg_dict[run_type]['SADS'] = {
        "n": int_data["sad"]["n"][:].filled(np.nan),
        "Abar": np.array(int_data["sad"]["a0"][:].filled(np.nan)),
        "sbQ_rel": int_data["sad"]["sbQ_rel"][:].filled(np.nan)
    }
    alg_dict[non_run_type]['SADS'] = {
        "n": non_run_array,
        "Abar": non_run_array,
        "sbQ_rel": int_data["sad"]["sbQ_rel"][:].filled(np.nan)
    }

    # metroman    
    alg_dict[run_type]['MetroMan'] = {
        "ninf": int_data['metroman']["na"][:].filled(np.nan),
        "p": int_data['metroman']["x1"][:].filled(np.nan),
        "Abar": int_data['metroman']["Abar"][:].filled(np.nan),
        "sbQ_rel": int_data["metroman"]["sbQ_rel"][:].filled(np.nan)
    }
    alg_dict[non_run_type]['MetroMan'] = {
        "ninf": non_run_array,
        "p": non_run_array,
        "Abar": non_run_array,
        "sbQ_rel": int_data["metroman"]["sbQ_rel"][:].filled(np.nan)
    }

    # SIC4DVar
    alg_dict[run_type]['SIC4DVar'] = {
        "n": int_data["sic4dvar"]["n"][:].filled(np.nan),
        "Abar": np.array(int_data["sic4dvar"]["a0"][:].filled(np.nan)),
        "sbQ_rel": int_data["sic4dvar"]["sbQ_rel"][:].filled(np.nan)
    }
    alg_dict[non_run_type]['SIC4DVar'] = {
        "n": non_run_array,
        "Abar": non_run_array,
        "sbQ_rel": int_data["sic4dvar"]["sbQ_rel"][:].filled(np.nan)
    }

    int_data.close()

    return alg_dict
