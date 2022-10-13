"""
Extract all of the reaches from SWORD overlapping with SWOT rivertile file.
"""
import netCDF4 as nc
import numpy as np


def ReachDatabase(reach_db_path, rch):
    dataset = nc.Dataset(reach_db_path, 'r')

    reaches = dataset['reaches']['reach_id'][:]
    reach_ind = np.where(reaches == rch)
    reach = {'area_fit': {},
             'discharge_models': {'unconstrained': {'MetroMan': {},
                                                    'BAM': {},
                                                    'HiVDI': {},
                                                    'MOMMA': {},
                                                    'SADS': {}},
                                  'constrained': {'MetroMan': {},
                                                  'BAM': {},
                                                  'HiVDI': {},
                                                  'MOMMA': {},
                                                  'SADS': {}}}}

    area_fit_key = ['h_variance', 'w_variance', 'hw_covariance',
                    'med_flow_area', 'h_err_stdev', 'w_err_stdev',
                    'h_w_nobs']
    metroman_key = ['Abar', 'Abar_stdev', 'ninf', 'ninf_stdev',
                    'p', 'p_stdev', 'ninf_p_cor', 'p_Abar_cor',
                    'ninf_Abar_cor']
    bam_key = ['Abar', 'n']
    hivdi_key = ['Abar', 'alpha', 'beta']
    momma_key = ['B', 'H', 'Save']
    sads_key = ['Abar', 'n']

    # area fits
    for key in area_fit_key:
        reach['area_fit'][key] = np.array(
            dataset['reaches']['area_fits'][key][reach_ind])

    reach['area_fit']['fit_coeffs'] = np.reshape(
        np.array(
            dataset['reaches']['area_fits']['fit_coeffs'])[:, :, reach_ind],
        (2, 3, 1))
    reach['area_fit']['h_break'] = np.reshape(
        np.array(
            dataset['reaches']['area_fits']['h_break'])[:, reach_ind], (4, 1))
    reach['area_fit']['w_break'] = np.reshape(
        np.array(
            dataset['reaches']['area_fits']['w_break'])[:, reach_ind], (4, 1))

    # unconstrained
    for key in metroman_key:
        reach['discharge_models']['unconstrained']['MetroMan'][key] \
            = np.array(dataset['reaches']['discharge_models']
                       ['unconstrained']['MetroMan'][key][reach_ind])
    for key in bam_key:
        reach['discharge_models']['unconstrained']['BAM'][key] \
            = np.array(dataset['reaches']['discharge_models']
                       ['unconstrained']['BAM'][key][reach_ind])
    for key in hivdi_key:
        reach['discharge_models']['unconstrained']['HiVDI'][key] \
            = np.array(dataset['reaches']['discharge_models']
                       ['unconstrained']['HiVDI'][key][reach_ind])
    for key in momma_key:
        reach['discharge_models']['unconstrained']['MOMMA'][key] \
            = np.array(dataset['reaches']['discharge_models']
                       ['unconstrained']['MOMMA'][key][reach_ind])
    for key in sads_key:
        reach['discharge_models']['unconstrained']['SADS'][key] \
            = np.array(dataset['reaches']['discharge_models']
                       ['unconstrained']['SADS'][key][reach_ind])
    # constrained
    for key in metroman_key:
        reach['discharge_models']['constrained']['MetroMan'][key] \
            = np.array(dataset['reaches']['discharge_models']
                       ['constrained']['MetroMan'][key][reach_ind])
    for key in bam_key:
        reach['discharge_models']['constrained']['BAM'][key] \
            = np.array(dataset['reaches']['discharge_models']
                       ['constrained']['BAM'][key][reach_ind])
    for key in hivdi_key:
        reach['discharge_models']['constrained']['HiVDI'][key] \
            = np.array(dataset['reaches']['discharge_models']
                       ['constrained']['HiVDI'][key][reach_ind])
    for key in momma_key:
        reach['discharge_models']['constrained']['MOMMA'][key] \
            = np.array(dataset['reaches']['discharge_models']
                       ['constrained']['MOMMA'][key][reach_ind])
    for key in sads_key:
        reach['discharge_models']['constrained']['SADS'][key] \
            = np.array(dataset['reaches']['discharge_models']
                       ['constrained']['SADS'][key][reach_ind])
    dataset.close()

    return reach
