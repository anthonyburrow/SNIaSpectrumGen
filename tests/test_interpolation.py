import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler

from SpectrumCore import Spectrum
from SNIaSpectrumGen.SpectrumGenerator import SpectrumGeneratorWorker


def test_interpolation_zoom(plots_dir):
    worker = SpectrumGeneratorWorker()
    sample_data = worker._sample_from_KDE()
    sample_spectrum = Spectrum(sample_data.copy())

    wave_min = 6000.
    wave_max = 10000.

    zoom_mask = (sample_spectrum.wave >= wave_min) & (sample_spectrum.wave <= wave_max)
    zoom_wave_original = sample_spectrum.wave[zoom_mask]
    zoom_flux_original = sample_spectrum.flux[zoom_mask]

    wave_interp_dense = np.linspace(wave_min, wave_max, 1000)

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    wave_scaled = scaler_x.fit_transform(sample_spectrum.wave.reshape(-1, 1)).ravel()
    flux_scaled = scaler_y.fit_transform(sample_spectrum.flux.reshape(-1, 1)).ravel()

    s = 1e-2 * len(wave_scaled) * worker.noise_range[1]**2
    spline = UnivariateSpline(wave_scaled, flux_scaled, s=s, k=3)

    wave_interp_scaled = scaler_x.transform(wave_interp_dense.reshape(-1, 1)).ravel()
    flux_interp_scaled = spline(wave_interp_scaled)
    flux_interp_dense = scaler_y.inverse_transform(
        np.asarray(flux_interp_scaled).reshape(-1, 1)
    ).ravel()

    zoom_wave_scaled = scaler_x.transform(zoom_wave_original.reshape(-1, 1)).ravel()
    zoom_flux_interp_scaled = spline(zoom_wave_scaled)
    zoom_flux_interp = scaler_y.inverse_transform(
        np.asarray(zoom_flux_interp_scaled).reshape(-1, 1)
    ).ravel()

    residuals = zoom_flux_original - zoom_flux_interp

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                     gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(wave_interp_dense, flux_interp_dense, 'b-', linewidth=1.5, 
             label='Spline interpolation', alpha=0.8, zorder=6)
    ax1.scatter(zoom_wave_original, zoom_flux_original, s=8, c='red', 
                marker='o', label='Downsampled data', zorder=5, alpha=0.5)
    ax1.set_ylabel('Normalized Flux')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2.scatter(zoom_wave_original, residuals, s=8, c='red', marker='o', alpha=0.5)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Residuals')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = plots_dir / 'test_interpolation_zoom.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSaved interpolation zoom plot: {output_path}")
    print(f"Number of points in zoom: {len(zoom_wave_original)}")

    assert output_path.exists()
    assert np.all(np.isfinite(residuals))

