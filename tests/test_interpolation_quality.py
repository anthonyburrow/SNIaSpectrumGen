import numpy as np
from SNIaSpectrumGen.SpectrumGenerator import SpectrumGeneratorWorker
from SpectrumCore import Spectrum


def test_interpolation_residual_stats():
    worker = SpectrumGeneratorWorker()
    base_data = worker._sample_from_KDE()
    spec = Spectrum(base_data.copy())

    wave_min = spec.wave[0] + 200.
    wave_max = spec.wave[-1] - 200.
    wave_dense = np.linspace(wave_min, wave_max, 1200)
    interp_flux = worker._interpolate_spectrum(spec, wave_dense)

    mask = (spec.wave >= wave_min) & (spec.wave <= wave_max)
    orig_wave = spec.wave[mask]
    orig_flux = spec.flux[mask]

    orig_interp_flux = np.interp(orig_wave, wave_dense, interp_flux)
    residuals = orig_flux - orig_interp_flux
    rms = np.sqrt(np.mean(residuals**2))

    assert rms < 0.5
