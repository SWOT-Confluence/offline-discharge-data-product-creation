# Standard imports
import json
import os
import pdb
from pathlib import Path
import sys
# Third-party imports
import numpy as np
import glob

# Local imports
from offline.ReadOb import Rivertile
from offline.ReadObs import Rivertile
from offline.ReadPRD import ReachDatabase
#from offline.ReadQparams import extract_alg # use with flpe dir
from offline.ReadQparamsIntegrator import extract_alg #  use with moi dir
from offline.discharge import compute, empty_q
from offline.WriteQ import write_q
from offline.WriteQ2Shp import write_q2shp

#Constants constrained
# INPUT = Path("/Users/rwei/Documents/confluence/offline_data_mar/constrained/mnt/input")
INPUT = os.path.join('/mnt', 'data', 'input')
FLPE_DIR = os.path.join('/mnt', 'data', 'moi')
# FLPE_DIR = Path("/Users/rwei/Documents/confluence/offline_data_mar/constrained/mnt/constrained_moi_update")
#FLPE_DIR = Path("/Users/rwei/Documents/confluence/OneDrive_1_9-23-2022/offline_inputs/mnt/flpe")
# OUTPUT = Path("/Users/rwei/Documents/confluence/offline_data_mar/constrained/mnt/constrained_output_apr27")
OUTPUT = os.path.join('/mnt', 'data', 'output')
# SWORD dir for single_pass run
# read in reach json
SWORD = Path("/Users/rwei/Documents/confluence/offline_data_mar/constrained/mnt/input/sword/na_sword_v11_moi.nc")

# Constants unconstrained
# INPUT = Path("/Users/rwei/Documents/confluence/offline_data_mar/unconstrained/mnt/input")
# #FLPE_DIR = Path("/Users/rwei/Documents/confluence/offline_data_mar/unconstrained/mnt/moi")
# FLPE_DIR = Path("/Users/rwei/Documents/confluence/offline_data_mar/unconstrained/mnt/unconstrained_moi_update")
# OUTPUT = Path("/Users/rwei/Documents/confluence/offline_data_mar/unconstrained/mnt/unconstrained_output")
# # SWORD dir for single_pass run
# SWORD = Path("/Users/rwei/Documents/confluence/offline_data_mar/unconstrained/mnt/input/sword/na_sword_v11_moi.nc")

DSCHG_KEYS = [
    'dschg' + a + b + c for a in ['_', '_g']
    for b in ['m', 'b', 'h', 'o', 's', 'i', 'c']
    for c in ['', '_u', '_q', 'sf']]
DSCHG_KEYS += ['dschg_q_b', 'dschg_gq_b', 'd_x_area', 'd_x_area_u']


def get_reach_data(reach_json, index_to_run):
    """Extract and return a dictionary of reach identifier, SoS and SWORD files.
    
    Parameters
    ----------
    reach_json : str
        Path to the file that contains the list of reaches to process
    """

    if index_to_run == -235:
        index = int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX"))
    else:
        index = index_to_run

    with open(reach_json) as json_file:
        data = json.load(json_file)
    return data[index]


def initialize_data_dict(nt, time_steps, reach_id):
    """Create an empty dictionary for holding model discharge values.
    
    Parameters
    ----------
    nt: int
        Number of time steps
    reach_id: int
        Unique reach identifier
    time_steps: np.ndarray
        Array of dates
    """
    data_dict = {}
    # iterating through the keys of key list
    for key in DSCHG_KEYS:
        data_dict[key] = np.repeat(np.nan, nt)
    data_dict['nt'] = nt
    data_dict['reach_id'] = reach_id
    data_dict['time_steps'] = time_steps
    return data_dict


def initialize_data_dict_sp(rch_n):
    """Create an empty dictionary for holding model discharge values.
    Parameters
    ----------
    rch_n: int
        Number of reaches in shapefile
    """
    data_dict = {}
    # iterating through the keys of key list
    for key in DSCHG_KEYS:
        data_dict[key] = np.repeat(np.nan, rch_n)
    data_dict['reach_id'] = np.repeat(np.nan, rch_n)
    return data_dict


def populate_data_array(data_dict, outputs, index):
    """Populate data_dict with outputs data.
    
    Parameters
    ----------
    data_dict: dict
        dictionary of model discharge values over time
    outputs: dict
        dictionary of model discharge values for one time
    index: int
        index value for data selection
    """

    # Insert data
    data_dict["d_x_area"][index] = outputs["d_x_area"]
    data_dict["d_x_area_u"][index] = outputs[
        "d_x_area_u"] if "d_x_area_u" in outputs.keys() else None

    for key in DSCHG_KEYS:
        data_dict[key][index] = outputs[key][0] if type(
            outputs[key]) is np.ndarray else outputs[key]

    # Convert missing values to NaN values
    for k, v in data_dict.items():
        if k != "nt" and k != "reach_id" and k != "time_steps":
            v[np.isclose(v, -1.00000000e+12)] = np.nan


def main(input, output, index_to_run):
    """Main function to execute offline discharge product generation and
    storage.
    Command line arguments:
    run_type: options are `unconstrained` or `constrained` (argument 1)
    input_type: options are 'timeseries' or 'single_pass' (argument 2)
                - timeseries: timer-series swot observations in one NetCDF file,
                              cross-sectional area change (d_x_area) should be
                              already computed and saved in time-series files.
                              discharge params from different discharge models
                              will be extracted from either moi-sword or integrator
                - single-pass: single-pass swot observation shapefiles, this is
                               what the swot data will look like. sword database
                               will be pulled d_x_area and discharge will be computed.
    flp_source: options are 'sword' or 'integrator' (argument 3)
    reach_json: title of JSON file that contains reach data (argument 4)

    Note: if input_type = single_pass, run_type, flp_source and reach_json will not be used.
          Set them to None or any other strings will work
    """
    # Command line arguments
    try:
        run_type = sys.argv[1]
        input_type = sys.argv[2]
        flp_source = sys.argv[3] # options are 'sword' or 'integrator'
        if input_type == 'timeseries':
            reach_json = os.path.join(INPUT, sys.argv[4])
    except IndexError:
        run_type = None
        if input_type == 'timeseries':
            reach_json = os.path.join(INPUT, "reaches.json")

    # Input timeseries data
    if input_type == 'timeseries':
        reach_data = get_reach_data(reach_json, index_to_run)
        obs = Rivertile(os.path.join(input , "swot" , reach_data["swot"]), input_type)
        print(flp_source)
        if flp_source == 'sword':
            priors = ReachDatabase(os.path.join(input , "sword" , reach_data["sword"]),
                                   reach_data["reach_id"])
            # if dA from timeseries, remove 'area_fit' params, so discharge module
            # won't compute dA again, don't need to set up dA source
            del priors['area_fits']
        elif flp_source == 'integrator':
            priors = {"discharge_models": extract_alg(FLPE_DIR,
                                                      reach_data["reach_id"],
                                                      run_type)}
        else:
            sys.exit('Warning: flp source not valid, exiting')

        #constrain width need to make this optional in the future
        area_fit={}
        area_fit['h_break']=obs['h_break'][:]
        area_fit['fit_coeffs']=obs['fit_coeffs'][:] #slope: index 1; intercept: index 0
        nt=len(obs["height"][:])
        hhat,what=ConstrainWidth(obs["height"][:],obs["width"][:],area_fit,nt)
        obs["width"]=what
        # Compute discharge        
        data_dict = initialize_data_dict(obs["nt"], obs["time_steps"],
                                         reach_data["reach_id"])
        for i in range(obs["nt"]):
            outputs = compute(priors, obs["height"][i], obs["wse_u"][i],
                              obs["width"][i], obs["width_u"][i],
                              obs["slope"][i], obs["slope_u"][i],
                              obs["d_x_area"][i], obs["d_x_area_u"][i])
            populate_data_array(data_dict, outputs, i)

        # Output discharge model values
        write_q(output, data_dict)

    # List all the reach .shp files in INPUT directory and process one by one
    if input_type == 'single_pass':
        input_shapefile = input.joinpath('shapefile')
        shapefiles = list(input_shapefile.glob('SWOT_L2_HR_RiverSP_reach*.shp'))

        for shapefile in shapefiles:
            print('SHAPEFILE: ', shapefile)
            obs = Rivertile(shapefile, input_type)
            data_dict = initialize_data_dict_sp(len(obs['reach_id']))
            for j in range(len(obs['reach_id'])):
                priors = ReachDatabase(SWORD, obs['reach_id'][j])
                if obs['height'][j] != -999999999999 \
                        and priors["area_fit"]["h_variance"] != -9999:
                    # sword doesn't have the values below, so made up some values for testing
                    # remove after
                    priors["area_fit"]["w_variance"] = 5
                    priors["area_fit"]["hw_covariance"] = 3
                    priors["area_fit"]["h_break"] = [[[50], [100], [160]]]
                    priors["area_fit"]["w_break"] = [[[200], [400], [600]]]
                    priors["area_fit"]["h_err_stdev"] = 0.01
                    priors["area_fit"]["w_err_stdev"] = 2
                    priors["area_fit"]["h_w_nobs"] = 25
                    priors["area_fit"]["fit_coeffs"] = [[[2], [1], [2]]]
                    priors["area_fit"]["med_flow_area"] = 100
                    # remove above after testing
                    outputs = compute(priors,
                                      obs["height"][j], obs["wse_u"][j],
                                      obs["width"][j], obs["width_u"][j],
                                      obs["slope"][j], obs["slope_u"][j],
                                      obs['d_x_area'][j], obs['d_x_area_u'][j])
                else:
                    outputs = empty_q()
                populate_data_array(data_dict, outputs, j)
            # Output discharge model values
            write_q2shp(shapefile, output, data_dict)

if __name__ == "__main__":
    from datetime import datetime

    start = datetime.now()

    try:
        index_to_run = int(sys.argv[5])  # integer
    except IndexError:
        index_to_run = -235  # AWS

    # print('indx=',index_to_run)

    main(INPUT, OUTPUT, index_to_run)

    end = datetime.now()
    print(f"Exeuction time: {end - start}")
