import numpy as np
from pathlib import Path
from typing import Optional
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import StandardScaler

from SpectrumCore import Spectrum

from .KDE.KDEPipeline import KDEPipeline


_DEFAULT_RANGE_LENGTH: tuple[int, int] = (800, 2000)
_DEFAULT_RANGE_WAVE_MIN: tuple[float, float] = (5000., 5500.)
_DEFAULT_RANGE_WAVE_MAX: tuple[float, float] = (8750., 10000.)
_DEFAULT_NOISE_RANGE: tuple[float, float] = (0.01, 0.05)

_DATA_DIR: Path = Path(__file__).resolve().parent.parent / 'data'
_FILENAME_EIGENVECTORS: Path = _DATA_DIR / 'eigenvectors.dat'
_FILENAME_MEAN: Path = _DATA_DIR / 'mean.dat'
_FILENAME_STD: Path = _DATA_DIR / 'std.dat'
_FILENAME_WAVELENGTHS: Path = _DATA_DIR / 'wavelengths.dat'


class SpectrumGeneratorWorker:

    def __init__(
            self,
            length_range: Optional[tuple[int, int]] = None,
            wave_min_range: Optional[tuple[float, float]] = None,
            wave_max_range: Optional[tuple[float, float]] = None,
            noise_range: Optional[tuple[float, float]] = None,
        ):
        self.length_range: tuple[int, int] = (
            _DEFAULT_RANGE_LENGTH if length_range is None
            else length_range
        )
        self.wave_min_range: tuple[float, float] = (
            _DEFAULT_RANGE_WAVE_MIN if wave_min_range is None
            else wave_min_range
        )
        self.wave_max_range: tuple[float, float] = (
            _DEFAULT_RANGE_WAVE_MAX if wave_max_range is None
            else wave_max_range
        )
        self.noise_range: tuple[float, float] = (
            _DEFAULT_NOISE_RANGE if noise_range is None
            else noise_range
        )

        self.wave_range: tuple[float, float] = \
            (self.wave_min_range[0], self.wave_max_range[-1])

        self.kde: KDEPipeline = KDEPipeline()
        self._load_data()

    def _sample_from_KDE(self) -> np.ndarray:
        N_wave: int = len(self.spec_wave)
        sample_spectrum: np.ndarray = np.zeros((N_wave, 2))

        sample_spectrum[:, 0] = self.spec_wave

        sample_eigenvalues: np.ndarray = self.kde.sample()
        reconstructed = sample_eigenvalues @ self.eigenvectors
        sample_spectrum[:, 1] = (
            reconstructed * self.spec_std + self.spec_mean
        )

        return sample_spectrum

    def _generate_noise(self, base_flux: np.ndarray) -> np.ndarray:
        N_wave = len(base_flux)
        noise_level = np.random.uniform(*self.noise_range)
        noise = np.random.normal(0., noise_level * base_flux, N_wave)

        return noise

    def _get_wavelengths(self) -> np.ndarray:
        N_wave: int = np.random.randint(*self.length_range)
        wave_min: float = np.random.uniform(*self.wave_min_range)
        wave_max: float = np.random.uniform(*self.wave_max_range)
        return np.linspace(wave_min, wave_max, N_wave)

    def _load_data(self):
        self.spec_wave = np.loadtxt(_FILENAME_WAVELENGTHS)
        optical_mask = (
            (self.wave_min_range[0] <= self.spec_wave) &
            (self.spec_wave <= self.wave_max_range[-1])
        )
        self.spec_wave = self.spec_wave[optical_mask]

        self.eigenvectors = np.loadtxt(_FILENAME_EIGENVECTORS)[:, optical_mask]
        self.spec_mean = np.loadtxt(_FILENAME_MEAN)[optical_mask]
        self.spec_std = np.loadtxt(_FILENAME_STD)[optical_mask]

    def _interpolate_spectrum(
            self,
            spectrum: Spectrum,
            wave_interp: np.ndarray,
        ) -> np.ndarray:
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        wave_scaled = scaler_x.fit_transform(spectrum.wave.reshape(-1, 1)).ravel()
        flux_scaled = scaler_y.fit_transform(spectrum.flux.reshape(-1, 1)).ravel()

        s = len(wave_scaled) * self.noise_range[1]**2
        spline = UnivariateSpline(wave_scaled, flux_scaled, s=s, k=3)

        wave_interp_scaled = scaler_x.transform(wave_interp.reshape(-1, 1)).ravel()
        flux_interp_scaled = spline(wave_interp_scaled)

        flux_interp = scaler_y.inverse_transform(
            np.asarray(flux_interp_scaled).reshape(-1, 1)
        ).ravel()

        return flux_interp  # type: ignore[return-value]

    def __call__(self, _=None) -> np.ndarray:
        sample_spectrum: Spectrum = Spectrum(self._sample_from_KDE())

        # Sample a variable-length grid
        new_wave: np.ndarray = self._get_wavelengths()
        N_wave: int = len(new_wave)
        new_data: np.ndarray = np.zeros((N_wave, 2))

        new_data[:, 0] = new_wave
        new_data[:, 1] = self._interpolate_spectrum(
            sample_spectrum,
            new_wave,
        )

        new_spectrum: Spectrum = Spectrum(new_data)

        # Add variable noise
        noise = self._generate_noise(new_spectrum.flux)
        new_spectrum.add_flux(noise)

        # Renormalize after interp/noise
        norm_range = (5500., 6500.)
        new_spectrum.normalize_flux(
            method='mean', wave_range=norm_range
        )
        new_spectrum.normalize_wave(wave_range=self.wave_range)

        return new_spectrum.data


class SpectrumGenerator:

    def __init__(self):
        self.worker: SpectrumGeneratorWorker = SpectrumGeneratorWorker()

    def generate(self) -> np.ndarray:
        return self.worker()

    def generate_batch(self, batch_size: int = 8) -> list[np.ndarray]:
        return [self.generate() for _ in range(batch_size)]

