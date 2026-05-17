"""
Extract discharge paramaters from Confluence
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
    gb_file = os.path.join(alg_dir, 'busboi', f'{r_id}_busboi.nc')
    hv_file = os.path.join(alg_dir, 'hivdi', f'{r_id}_hivdi.nc')
    mo_file = os.path.join(alg_dir, 'momma', f'{r_id}_momma.nc')
    sd_file = os.path.join(alg_dir, 'sad', f'{r_id}_sad.nc')
    sv_file = os.path.join(alg_dir, 'sic4dvar', f'{r_id}_sic4dvar.nc')
    mm_file = glob(str(alg_dir / 'metroman' / f'*{r_id}*_metroman.nc'))
    # temporarily change how to construct paths, below doesn't work for the Python version Rui's using
    #     gb_file = alg_dir / 'geobam' / f'{r_id}_geobam.nc'
    #     hv_file = alg_dir / 'hivdi' / f'{r_id}_hivdi.nc'
    #     mo_file = alg_dir / 'momma' / f'{r_id}_momma.nc'
    #     sd_file = alg_dir / 'sad' / f'{r_id}_sad.nc'
    #     mm_file = glob(str(alg_dir / 'metroman' / f'*{r_id}*_metroman.nc'))
    mm_file = Path(mm_file[0])

    #     if gb_file.exists() and hv_file.exists() and mm_file.exists() and mo_file.exists() and sd_file.exists():
    if os.path.exists(gb_file) and os.path.exists(hv_file) and \
            os.path.exists(mm_file) and os.path.exists(
        mo_file) and os.path.exists(sd_file):
        param_dict = extract_valid(r_id, run_type, gb_file, hv_file, mo_file,
                                   sd_file, sv_file, mm_file)
    else:
        param_dict = indicate_no_data(r_id)

    return param_dict


def extract_valid(r_id, run_type, gb_file, hv_file, mo_file, sd_file,
                  sv_file, mm_file):
    """ Extract valid data from the output of each reach-level FLPE alg.
    Parameters
    ----------
    r_id: str
        Unique reach identifier
    run_type: str
        constrained or unconstrained data product indicator
    gb_file: Path
        Path to geoBAM results file
    hv_file: Path
        Path to HiVDI results file
    mo_file: Path
        Path to MOMMA results file
    sd_file: Path
        Path to SAD results file
    sv_file: Path
        Path to SIC4dDVar results file
    mm_file: Path
        Path to MetroMan results file
    """

    non_run_type = "constrained" if run_type == "unconstrained" else "unconstrained"
    non_run_array = np.array([-9999], dtype=float)

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

    # busboi
    gb = Dataset(gb_file, 'r', format="NETCDF4")
    alg_dict[run_type]['BUSBOI'] = {
        "n": np.array(get_gb_data(gb, "logn", "mean", True)),
        # "Abar": np.array(np.nanmean(np.array(get_gb_data(gb, "A0", False))))
        "Abar": np.array(1)
        # Temp placeholder while work out neoBAM AO calculation
    }
    alg_dict[non_run_type]['BUSBOI'] = {
        "n": non_run_array,
        "Abar": non_run_array
    }
    gb.close()

    # hivdi
    hv = Dataset(hv_file, 'r', format="NETCDF4")
    alg_dict[run_type]['HiVDI'] = {
        "alpha": hv["reach"]["alpha"][:].filled(np.nan),
        "beta": hv["reach"]["beta"][:].filled(np.nan),
        "Abar": np.array(hv["reach"]["A0"][:].filled(np.nan))
    }
    alg_dict[non_run_type]['HiVDI'] = {
        "alpha": non_run_array,
        "beta": non_run_array,
        "Abar": non_run_array
    }
    hv.close()

    # momma
    mo = Dataset(mo_file, 'r', format="NETCDF4")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        alg_dict[run_type]['MOMMA'] = {
            "B": mo["zero_flow_stage"][:].filled(np.nan),
            "H": mo["bankfull_stage"][:].filled(np.nan),
            "Save": np.nanmean(mo["slope"][:].filled(np.nan))
        }
        alg_dict[non_run_type]['MOMMA'] = {
            "B": non_run_array,
            "H": non_run_array,
            "Save": non_run_array
        }
    mo.close()

    # sad
    sd = Dataset(sd_file, 'r', format="NETCDF4")
    alg_dict[run_type]['SADS'] = {
        "n": sd["n"][:].filled(np.nan),
        "Abar": np.array(sd["A0"][:].filled(np.nan))
    }
    alg_dict[non_run_type]['SADS'] = {
        "n": non_run_array,
        "Abar": non_run_array
    }
    sd.close()

    # metroman    
    mm = Dataset(mm_file, 'r', format="NETCDF4")
    index = np.where(mm["reach_id"][:] == int(r_id))
    alg_dict[run_type]['MetroMan'] = {
        "ninf": mm["nahat"][index].filled(np.nan),
        "p": mm["x1hat"][index].filled(np.nan),
        "Abar": mm["A0hat"][index].filled(np.nan)
    }
    alg_dict[non_run_type]['MetroMan'] = {
        "ninf": non_run_array,
        "p": non_run_array,
        "Abar": non_run_array
    }
    mm.close()

    # sic4dvar
    sv = Dataset(sv_file, 'r', format="NETCDF4")
    index = np.where(mm["reach_id"][:] == int(r_id))
    alg_dict[run_type]['SIC4DVar'] = {
        "n": sv["n"][:].filled(np.nan),
        "Abar": np.array(sv["A0"][:].filled(np.nan))
    }
    alg_dict[non_run_type]['SIC4DVar'] = {
        "n": non_run_array,
        "Abar": non_run_array
    }
    sd.close()

    return alg_dict


def indicate_no_data(r_id):
    """Indicate no data is available for the reach.
    TODO: Metroman results
    Parameters
    ----------
    r_id: str
        Unique reach identifier
    """
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

    # busboi
    alg_dict['unconstrained']['BUSBOI'] = {
        "n": np.nan,
        "Abar": np.nan
    }
    alg_dict['constrained']['BUSBOI'] = {
        "n": np.nan,
        "Abar": np.nan
    }

    # hivdi
    alg_dict['unconstrained']['HiVDI'] = {
        "alpha": np.nan,
        "beta": np.nan,
        "Abar": np.nan
    }
    alg_dict['constrained']['HiVDI'] = {
        "alpha": np.nan,
        "beta": np.nan,
        "Abar": np.nan
    }

    # momma
    alg_dict['unconstrained']['MOMMA'] = {
        "B": np.nan,
        "H": np.nan,
        "Save": np.nan
    }
    alg_dict['constrained']['MOMMA'] = {
        "B": np.nan,
        "H": np.nan,
        "Save": np.nan
    }

    # sad
    alg_dict['unconstrained']['SADS'] = {
        "n": np.nan,
        "Abar": np.nan
    }
    alg_dict['constrained']['SADS'] = {
        "n": np.nan,
        "Abar": np.nan
    }

    # MetroMan    
    alg_dict['unconstrained']['MetroMan'] = {
        "ninf": np.nan,
        "p": np.nan,
        "Abar": np.nan
    }
    alg_dict['constrained']['MetroMan'] = {
        "ninf": np.nan,
        "p": np.nan,
        "Abar": np.nan
    }

    # sic4dvar
    alg_dict['unconstrained']['SIC4DVar'] = {
        "n": np.nan,
        "Abar": np.nan
    }
    alg_dict['constrained']['SIC4DVar'] = {
        "n": np.nan,
        "Abar": np.nan
    }

    return alg_dict


def get_gb_data(gb, group, pre, logged):
    """Return geoBAM data as a numpy array.

    Parameters
    ----------
    gb: netCDF4.Dataset
        NetCDF file dataset to extract discharge time series
    group: str
        string name of group to access chains
    pre: str
        string prefix of variable name
    logged: bool
        boolean indicating if result is logged
    """

    # chain1 = gb[group][f"{pre}1"][:].filled(np.nan)
    # chain2 = gb[group][f"{pre}2"][:].filled(np.nan)
    # chain3 = gb[group][f"{pre}3"][:].filled(np.nan)
    # chains = np.vstack((chain1, chain2, chain3))
    q = gb[group][pre][:].filled(np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if logged:
            return np.exp(np.nanmean(q, axis=0))
        else:
            return np.nanmean(q, axis=0)
