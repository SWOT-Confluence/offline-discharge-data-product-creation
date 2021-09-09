"""
Read reach data from SWOT rivertile file.
"""
import netCDF4 as nc
import numpy as np

def Rivertile(swot_file):
    
    dataset = nc.Dataset(swot_file, 'r')
    rivertile = {}
    rivertile['reach_id'] = dataset['reach']['reach_id'][:].filled(np.nan)
    rivertile['height'] = dataset['reach']['wse'][:].filled(np.nan)
    rivertile['width'] = dataset['reach']['width'][:].filled(np.nan)
    rivertile['slope'] = dataset['reach']['slope2'][:].filled(np.nan)
    rivertile['nt'] = dataset.dimensions["nt"].size
    rivertile["time_steps"] = dataset["nt"][:]
 
    dataset.close()
    return rivertile