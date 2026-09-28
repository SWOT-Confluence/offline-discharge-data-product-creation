"""Offline pieces that must match MOI's flow-law fitting.

Run with:  python3 -m unittest discover -s tests -v
The full cross-repository check is MOI's tools/check_offline_parity.py.
(geopandas is only needed for single-pass input and is stubbed if absent.)
"""
import os
import sys
import tempfile
import types
import unittest

from netCDF4 import Dataset
import numpy as np

try:
    import geopandas  # noqa: F401
except ImportError:
    sys.modules['geopandas'] = types.ModuleType('geopandas')

from offline import flowlaw_preprocess as fp  # noqa: E402
from offline.ReadObs import Rivertile  # noqa: E402
import run_offline  # noqa: E402

NT = 12
QUALITY = ('xtrk_dist', 'ice_clim_f', 'dark_frac', 'obs_frac_n', 'xovr_cal_q',
           'n_good_nod', 'p_width', 'p_length', 'reach_q_b')


def _write_swot(path, hwfit):
    good = {'xtrk_dist': 30e3, 'ice_clim_f': 0, 'dark_frac': 0, 'obs_frac_n': 1,
            'xovr_cal_q': 0, 'n_good_nod': 50, 'p_width': 100, 'p_length': 1e4,
            'reach_q_b': 0}
    with Dataset(path, 'w') as ds:
        ds.createDimension('nt', NT)
        ds.createVariable('observations', 'i4', ('nt',))[:] = np.arange(NT)
        reach = ds.createGroup('reach')
        reach.createVariable('reach_id', 'i8')[...] = 1
        series = {'wse': np.linspace(10, 11, NT), 'width': np.linspace(100, 110, NT),
                  'slope2': np.full(NT, 1e-4), 'd_x_area': np.linspace(-50, 50, NT),
                  'time': np.arange(NT, dtype=float)}
        series.update({k: np.full(NT, float(v)) for k, v in good.items()})
        for name in ('wse_u', 'width_u', 'slope2_u', 'd_x_area_u'):
            series[name] = np.full(NT, 0.1)
        for name, values in series.items():
            reach.createVariable(name, 'f8', ('nt',), fill_value=-1e12)[:] = values
        if hwfit:
            fit = reach.createGroup('hwfit')
            for dim, size in (('b', 4), ('c', 2), ('s', 3), ('o', 1)):
                fit.createDimension(dim, size)
            fit.createVariable('h_break', 'f8', ('b', 'o'))[:] = np.full((4, 1), np.nan)
            fit.createVariable('fit_coeffs', 'f8', ('c', 's', 'o'))[:] = np.zeros((2, 3, 1))


class SharedPreprocessingTest(unittest.TestCase):
    def test_negative_slope_produces_no_discharge_input(self):
        used, valid, _status = fp.prepare_slope([-1e-5, 1e-6, 1e-4])
        np.testing.assert_array_equal(valid, [False, True, True])
        self.assertTrue(np.isnan(used[0]))
        self.assertEqual(used[1], fp.SLOPE_FLOOR)

    def test_rivertile_tolerates_missing_hwfit(self):
        with tempfile.TemporaryDirectory() as tmp:
            for hwfit in (True, False):
                path = os.path.join(tmp, f'{hwfit}_SWOT.nc')
                _write_swot(path, hwfit)
                obs = Rivertile(path, 'timeseries')
                prep = fp.prepare_flowlaw_inputs(
                    obs['height'], obs['width'], obs['slope'],
                    {'h_break': obs['h_break'], 'fit_coeffs': obs['fit_coeffs']},
                    {name: obs[name] for name in fp.QUALITY_VARIABLES})
                self.assertEqual(prep['width_source'], fp.WIDTH_SOURCE_MEDIAN)
                self.assertTrue(prep['flowlaw_valid'].all())

    def test_preprocess_options_come_from_the_integrator_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            with Dataset(os.path.join(tmp, '7_integrator.nc'), 'w'):
                pass
            self.assertEqual(run_offline.read_preprocess_config(tmp, 7), {
                'observation_filter': fp.DEFAULT_OBSERVATION_FILTER,
                'slope_policy': fp.DEFAULT_SLOPE_POLICY})
            with Dataset(os.path.join(tmp, '8_integrator.nc'), 'w') as ds:
                ds.setncattr(fp.CONFIG_ATTRIBUTES['observation_filter'], 'robust_v2')
                ds.setncattr(fp.CONFIG_ATTRIBUTES['slope_policy'], 'l2_3p4')
            self.assertEqual(run_offline.read_preprocess_config(tmp, 8), {
                'observation_filter': 'robust_v2', 'slope_policy': 'l2_3p4'})


if __name__ == '__main__':
    unittest.main()
