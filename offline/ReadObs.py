"""
Read reach data from SWOT rivertile file.
"""
import netCDF4 as nc
import numpy as np
import geopandas as gpd
def build_filter_dic(obs):
    filterdict={}
    filterdict['time']=obs['time']
    filterdict['xtrk_dist']=obs['xtrk_dist']
    filterdict['ice_clim_f']=obs['ice_clim_f']
    filterdict['dark_frac']=obs['dark_frac']
    filterdict['obs_frac_n']=obs['obs_frac_n']
    filterdict['xovr_cal_q']=obs['xovr_cal_q']
    filterdict['n_good_nod']=obs['n_good_nod']
    filterdict['p_width']=obs['p_width']
    filterdict['p_length']=obs['p_length']
    filterdict['reach_q_b']=obs['reach_q_b']
    return filterdict    

def filterNRTdata(rivertile,filterdict=None):
    if filterdict !=None:
        reach_height=[]
        reach_width=[]
        reach_slope=[] 
        for i in range(len(rivertile['time'])):
            badob=False #keep ob unless filter is tripped
            badob=(np.isnan(filterdict['time'][i])) | \
            (np.abs(filterdict['xtrk_dist'][i]) > 60e3) | \
            (np.abs(filterdict['xtrk_dist']) < 10e3) | \
            (filterdict['ice_clim_f'][i] > 1) | \
            (filterdict['dark_frac'][i] > .6) | \
            (filterdict['obs_frac_n'][i] < .4) | \
            (filterdict['xovr_cal_q'][i] > 1) | \
            (filterdict['n_good_nod'][i] < 10) | \
            (filterdict['p_width'][i] < 60)| \
            (filterdict['p_length'][i] < 5000)|\
            (filterdict['reach_q_b'][i] > 507510784)                
            if badob:
                    reach_height.append(np.nan)                    
                    reach_width.append(np.nan)                
                    reach_slope.append(np.nan)
            else:
                 reach_height.append(rivertile['height'][i])                    
                 reach_width.append(rivertile['width'][i])            
                 reach_slope.append(rivertile['slope'][i])
    FRT={'reach_height':reach_height,
         'reach_width':reach_width,
         'reach_slope':reach_slope

    }   
                
    return FRT
def OutlierFilter(rivertile,Tukey_number=1.5):
  
  Wobs=rivertile['reach_width']
  Hobs=rivertile['reach_height']
  Sobs=rivertile['reach_slope']
  
  
  #flag and remove all data that are > n IQRs away from the upper and lower quartile (Tukey method)
  
  #calculate quartiles
  W_IQR = np.quantile(Wobs,[0.25,0.75])
  W_upper_outlier=W_IQR[1] + (Tukey_number* (W_IQR[1]-W_IQR[0]))
  W_lower_outlier=W_IQR[0] - (Tukey_number* (W_IQR[1]-W_IQR[0]))
  
  H_IQR = np.quantile(Hobs,[0.25,0.75])
  H_upper_outlier=H_IQR[1] + (Tukey_number* (H_IQR[1]-H_IQR[0]))
  H_lower_outlier=H_IQR[0] - (Tukey_number* (H_IQR[1]-H_IQR[0]))
  
  S_IQR = np.quantile(Sobs,[0.25,0.75])
  S_upper_outlier=S_IQR[1] + (Tukey_number* (S_IQR[1]-S_IQR[0]))
  S_lower_outlier=S_IQR[0] - (Tukey_number* (S_IQR[1]-S_IQR[0]))
  Tukey_fliter_lims={
      'W_upper_outlier' : W_upper_outlier,
      'W_lower_outlier' : W_lower_outlier,
      'H_upper_outlier' : H_upper_outlier,
      'H_lower_outlier' : H_lower_outlier,
      'S_upper_outlier' : S_upper_outlier,
      'S_lower_outlier' : S_lower_outlier      
  }
  return Tukey_fliter_lims

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
        #will need do Tukey filter here with entire TS. This is something that needs a workaround for single pass

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
                     'nt': dataset.dimensions["nt"].size, "time_steps": dataset["observations"][:],
                     'time':dataset['reach']['time'][:].filled(np.nan),
                     'xtrk_dist':dataset['reach']['xtrk_dist'][:].filled(np.nan),
                     'ice_clim_f':dataset['reach']['ice_clim_f'][:].filled(np.nan),
                     'dark_frac':dataset['reach']['dark_frac'][:].filled(np.nan),
                     'obs_frac_n':dataset['reach']['obs_frac_n'][:].filled(np.nan),
                     'xovr_cal_q':dataset['reach']['xovr_cal_q'][:].filled(np.nan),
                     'n_good_nod':dataset['reach']['n_good_nod'][:],
                     'p_width':dataset['reach']['p_width'][:].filled(np.nan),
                     'p_length':dataset['reach']['p_length'][:].filled(np.nan),
                     'reach_q_b':dataset['reach']['reach_q_b'][:].filled(np.nan)
                         }
        dataset.close()
        #need to filter data prior to calculating Tukey filter values
        filterdict=build_filter_dic(rivertile)
        filtered_rivertile=filterNRTdata(rivertile,filterdict)
        Tukey_fliter_lims=OutlierFilter(filtered_rivertile)
        rivertile['W_upper_outlier']=Tukey_fliter_lims['W_upper_outlier']
        rivertile['W_lower_outlier']=Tukey_fliter_lims['W_lower_outlier']
        rivertile['H_upper_outlier']=Tukey_fliter_lims['H_upper_outlier']
        rivertile['H_lower_outlier']=Tukey_fliter_lims['H_lower_outlier']
        rivertile['S_upper_outlier']=Tukey_fliter_lims['S_upper_outlier']
        rivertile['S_lower_outlier']=Tukey_fliter_lims['S_lower_outlier']
       
                
    else:
        raise NotImplementedError(
            'input format is not supported!')
        
    return rivertile
