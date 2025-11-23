import numpy as np
from pathlib import Path
from typing import Optional
from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from SpectrumCore import Spectrum

from .KDE.KDEPipeline import KDEPipeline


_DEFAULT_RANGE_LENGTH: tuple[int, int] = (800, 2000)
_DEFAULT_RANGE_WAVE_MIN: tuple[float, float] = (5000., 5500.)
_DEFAULT_RANGE_WAVE_MAX: tuple[float, float] = (8750., 10000.)
_DEFAULT_NOISE_RANGE: tuple[float, float] = (0.01, 0.05)


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

        root_dir = Path(__file__).resolve().parent
        self.data_dir: Path = root_dir.parent / 'data'

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
        self.spec_wave = np.loadtxt(self.data_dir / 'wavelengths.dat')
        optical_mask = (
            (self.wave_min_range[0] <= self.spec_wave) &
            (self.spec_wave <= self.wave_max_range[-1])
        )
        self.spec_wave = self.spec_wave[optical_mask]

        eigvec_path = self.data_dir / 'eigenvectors.dat'
        self.eigenvectors = np.loadtxt(eigvec_path)[:, optical_mask]
        self.spec_mean = np.loadtxt(self.data_dir / 'mean.dat')[optical_mask]
        self.spec_std = np.loadtxt(self.data_dir / 'std.dat')[optical_mask]

    def _interpolate_spectrum(
            self,
            spectrum: Spectrum,
            wave_interp: np.ndarray,
        ) -> np.ndarray:
        spectrum.downsample(2.)

        pipeline: Pipeline = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=Matern(
                    length_scale=0.35,
                    length_scale_bounds=(0.01, 1.)
                ),
                normalize_y=True,
            )
        )

        pipeline.fit(spectrum.wave.reshape(-1, 1), spectrum.flux)

        pred = pipeline.predict(wave_interp.reshape(-1, 1))
        flux_interp: np.ndarray = (
            pred if isinstance(pred, np.ndarray) else pred[0]
        )

        return flux_interp

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

    def __init__(self, N_workers: int = 1):
        self.N_workers: int = N_workers

        if self.N_workers > 1:
            self.parallel_pool = Parallel(
                n_jobs=N_workers, backend='loky'
            )
            self.workers: list[SpectrumGeneratorWorker] = [
                SpectrumGeneratorWorker()
                for _ in range(self.N_workers)
            ]
        else:
            self.worker: SpectrumGeneratorWorker = SpectrumGeneratorWorker()

    def generate(self) -> np.ndarray:
        return self.worker()

    def generate_batch(self, batch_size: int = 8) -> list[np.ndarray]:
        if self.N_workers <= 1:
            return [self.generate() for _ in range(batch_size)]

        results = self.parallel_pool(
            delayed(self.workers[i % self.N_workers])()
            for i in range(batch_size)
        )
        return list(results)  # type: ignore[return-value]
