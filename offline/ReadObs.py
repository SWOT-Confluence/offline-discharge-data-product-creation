"""
Read reach data from SWOT rivertile file.
"""
import netCDF4 as nc
import numpy as np
import geopandas as gpd

def Rivertile(rivertile_path, input_type):
    """
    Read in swot observations, inputs can be in .nc or .shp 
    format
    """   
    
    # shapefile inputs, rivertile_path[-3:] == 'shp'
    if input_type == 'single_pass': 
        dataset = gpd.read_file(rivertile_path)
        rivertile = {'reach_id': np.array( \
            dataset['reach_id'][:].replace(-9.999999999990000e+11, np.nan), dtype=float),
            'height': dataset['wse'][:].replace(-9.999999999990000e+11, np.nan),
            'wse_u': dataset['wse_u'][:].replace(-9.999999999990000e+11, np.nan),
            'width': dataset['width'][:].replace(-9.999999999990000e+11, np.nan),
            'width_u': dataset['width_u'].replace(-9.999999999990000e+11, np.nan),
            'slope': dataset['slope'][:].replace(-9.999999999990000e+11, np.nan),
            'slope_u': dataset['slope_u'][:].replace(-9.999999999990000e+11, np.nan),
            'd_x_area': dataset['d_x_area'][:].replace(-9.999999999990000e+11, np.nan),
            'd_x_area_u': dataset['d_x_area_u'][:].replace(-9.999999999990000e+11, np.nan)}
    #         rivertile['nt'] = 1
#         rivertile["time_steps"] = 1
    
    # timeseries inputs, rivertile_path[-2:] == 'nc'
    #elif input_pass == 'timeseries':
    elif input_type == 'timeseries':
        dataset = nc.Dataset(rivertile_path, 'r')
        rivertile = {'reach_id': dataset['reach']['reach_id'][:].filled(np.nan),
                     'height': dataset['reach']['wse'][:].filled(np.nan),
                     'wse_u': dataset['reach']['wse_u'][:].filled(np.nan),
                     'width': dataset['reach']['width'][:].filled(np.nan),
                     'width_u': dataset['reach']['width_u'][:].filled(np.nan),
                     'slope': dataset['reach']['slope2'][:].filled(np.nan),
                     'slope_u': dataset['reach']['slope2_u'][:].filled(np.nan),
                     'd_x_area': dataset['reach']['d_x_area'][:].filled(np.nan),
                     'd_x_area_u': dataset['reach']['d_x_area_u'][:].filled(np.nan),
                     'h_break': dataset['reach']['hwfit']['h_break'][:].filled(np.nan),
                     'fit_coeffs': dataset['reach']['hwfit']['fit_coeffs'][:].filled(np.nan),
                     'nt': dataset.dimensions["nt"].size, "time_steps": dataset["observations"][:]}
        dataset.close()
    else:
        raise NotImplementedError(
            'input format is not supported!')
        
    return rivertile
