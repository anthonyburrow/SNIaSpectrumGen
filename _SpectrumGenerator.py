import numpy as np
import tensorflow as tf
import pickle
from joblib import Parallel, delayed

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from SpectrumCore import Spectrum


class SpectrumWorker:

    def __init__(self, *args, **kwargs):
        self.length_range = (800, 2000)
        self.wave_min_range = (5000., 5500.)
        self.wave_max_range = (8750., 10000.)
        self.wave_range = (self.wave_min_range[0], self.wave_max_range[-1])

        self.noise_range = (0.01, 0.05)

        self.spex_params = {
            'normalize': True,
            'plot': False,
            'verbose': False,
            'log': False,
        }

        self.data_dir = './data'
        self.model_dir = './models'

        self._load_data()

    def generate_from_KDE(self):
        N_wave = len(self.spec_wave)
        sample_spectrum = np.zeros((N_wave, 2))
        sample_spectrum[:, 0] = self.spec_wave

        sample_eigenvalues = self.kde.sample()
        sample_spectrum[:, 1] = \
            (sample_eigenvalues @ self.eigenvectors) * self.spec_std + self.spec_mean

        return sample_spectrum

    def generate_noise(self, base_flux):
        N_wave = len(base_flux)
        noise_level = np.random.uniform(*self.noise_range)
        noise = np.random.normal(0., noise_level * base_flux, N_wave)

        return noise

    def choose_wavelengths(self):
        N_wave = np.random.randint(*self.length_range)
        wave_min = np.random.uniform(*self.wave_min_range)
        wave_max = np.random.uniform(*self.wave_max_range)
        return np.linspace(wave_min, wave_max, N_wave)

    def _load_data(self):
        fn = f'{self.model_dir}/kde.pkl'
        with open(fn, 'rb') as file:
            self.kde = pickle.load(file)

        self.spec_wave = np.loadtxt(f'{self.data_dir}/wavelengths.dat')
        optical_mask = (self.wave_min_range[0] <= self.spec_wave) & (self.spec_wave <= self.wave_max_range[-1])
        self.spec_wave = self.spec_wave[optical_mask]

        self.eigenvectors = np.loadtxt(f'{self.data_dir}/eigenvectors.dat')[:, optical_mask]
        self.spec_mean = np.loadtxt(f'{self.data_dir}/mean.dat')[optical_mask]
        self.spec_std = np.loadtxt(f'{self.data_dir}/std.dat')[optical_mask]

    def _interp_spectrum(self, spectrum, wave_interp):
        spectrum.downsample(2.)

        kernel = Matern(length_scale=0.35, length_scale_bounds=(0.01, 1.))
        pipeline = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        )

        pipeline.fit(spectrum.wave.reshape(-1, 1), spectrum.flux)
        flux_interp = pipeline.predict(wave_interp.reshape(-1, 1))

        return flux_interp

    def __call__(self, _=None):
        '''Generate single random spectrum of random length.'''
        sample_spectrum = self.generate_from_KDE()
        sample_spectrum = Spectrum(sample_spectrum)

        # Sample a variable-length grid
        new_wave = self.choose_wavelengths()
        N_wave = len(new_wave)
        new_spectrum = np.zeros((N_wave, 2))

        new_spectrum[:, 0] = new_wave
        new_spectrum[:, 1] = self._interp_spectrum(
            sample_spectrum, new_spectrum[:, 0]
        )

        new_spectrum = Spectrum(new_spectrum)

        # Add variable noise
        noise = self.generate_noise(new_spectrum.flux)
        new_spectrum.add_flux(noise)

        # Renormalize after interp/noise
        norm_range = (5500., 6500.)
        new_spectrum.normalize_flux(method='mean', wave_range=norm_range)
        new_spectrum.normalize_wave(wave_range=self.wave_range)

        return new_spectrum.data


class SpectrumGenerator(tf.keras.utils.Sequence):

    def __init__(self, mode=None, batch_size=16, steps=100, N_workers=1):
        super().__init__()

        self.batch_size = batch_size
        self.steps = steps
        self.N_workers = N_workers

        self.counter = 1

        self.mode = 'pretrain' if mode is None else mode

        if self.N_workers > 1:
            self.parallel_pool = Parallel(n_jobs=self.N_workers, backend='loky')
        else:
            self.parallel_pool = None

    def generate_batch(self):
        if self.parallel_pool is None:
            worker = SpectrumWorker()
            batch = [worker() for _ in range(self.batch_size)]
        else:
            batch = self.parallel_pool(
                delayed(lambda: SpectrumWorker()())() for _ in range(self.batch_size)
            )

        # print(f'Generated batch {self.counter}', flush=True)
        self.counter += 1

        return batch

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        x_batch = self.generate_batch()
        x_batch = tf.keras.preprocessing.sequence.pad_sequences(
            x_batch, padding='post', dtype='float32', value=0., maxlen=2000
        )

        # For pretraining, the output is the same to minimize MSE
        if self.mode == 'pretrain':
            y_batch = x_batch[:, :, 1:]
        elif self.mode == 'supervised':
            # Spextractor for line ID
            pass
        else:
            raise ValueError('Incorrect mode for SpectrumGenerator.')

        return x_batch, y_batch


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    fig, ax = plt.subplots()

    t0 = time.time()

    offset = 0.
    batch_size = 4
    gen = SpectrumGenerator(batch_size=batch_size, N_workers=4)

    for _ in range(1):
        spectra = gen.generate_batch()

    print(f'Time: {(time.time() - t0) / 64:.3f} sec/spectrum')

    for i in range(batch_size):
        spectrum = spectra[i]
        ax.plot(spectrum[:, 0], spectrum[:, 1] + offset)
        offset += 0.8

    plt.show()
