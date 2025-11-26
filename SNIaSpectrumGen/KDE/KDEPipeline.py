import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


_FILENAME_DATA: str = 'all_params.dat'
_FILENAME_MODEL: str = 'kde.pkl'
_FILENAME_PLOT: str = 'kde.png'


class KDEPipeline:
    def __init__(self) -> None:
        root_dir = Path(__file__).resolve().parent.parent

        self.data_dir: Path = root_dir / 'data'
        self.model_dir: Path = root_dir.parent / 'models'
        self.plot_dir: Path = root_dir.parent / 'plots'

        self.data: np.ndarray | None = None
        self.pc: np.ndarray | None = None
        self.kde: KernelDensity | None = None

        model_path = self.model_dir / _FILENAME_MODEL
        if model_path.exists():
            with model_path.open('rb') as file:
                self.kde = pickle.load(file)
        else:
            self.generate()

    def generate(self) -> None:
        self._load_data()
        self._filter_data()
        self._fit_kde()
        self._save_model()

    def sample(self, n_samples: int = 1) -> np.ndarray:
        if self.kde is None:
            raise RuntimeError(
                'KDE model not created. Call generate() first.'
            )

        return self.kde.sample(n_samples)

    def _load_data(self) -> None:
        data_path: Path = self.data_dir / _FILENAME_DATA
        self.data = np.loadtxt(data_path, usecols=range(1, 24))

        self.pc = self.data[:, 13:]

    def _filter_data(self) -> None:
        if self.data is None:
            raise RuntimeError('Data not loaded. Call _load_data() first.')

        mask = self.data[:, 3] == 0.0
        self.data = self.data[~mask]

        mask = self.data[:, 2] > 5.0
        self.data = self.data[~mask]

        mask = np.isnan(self.data).any(axis=1)
        self.data = self.data[~mask]

        # Removes one outlier
        mask = self.data[:, 13] < -10.0
        self.data = self.data[~mask]

    def _fit_kde(self) -> None:
        if self.data is None:
            raise RuntimeError(
                'Data not loaded. Call _load_data() first.'
            )
        pc = self.data[:, 13:]

        param_grid = {
            'bandwidth': np.linspace(0.1, 5.0, 30),
        }
        grid_search = GridSearchCV(
            KernelDensity(kernel='gaussian'),
            param_grid,
            cv=5,
        )
        grid_search.fit(pc)

        self.kde = KernelDensity(**grid_search.best_params_)
        self.kde.fit(pc)

    def _save_model(self) -> None:
        if self.kde is None:
            raise RuntimeError(
                'KDE not fitted. Call _fit_kde() first.'
            )

        self.model_dir.mkdir(parents=True, exist_ok=True)
        path = self.model_dir / _FILENAME_MODEL

        with path.open('wb') as file:
            pickle.dump(self.kde, file)

    def plot(
        self,
        n_samples: int = 5000,
    ) -> None:
        if self.data is None:
            raise RuntimeError(
                'Data not loaded. Call _load_data() first.'
            )
        if self.kde is None:
            raise RuntimeError(
                'KDE not fitted. Call _fit_kde() first.'
            )
        pc = self.data[:, 13:]

        n_pcs: int = 4

        fig, ax = plt.subplots(
            n_pcs,
            n_pcs,
            figsize=(8, 6),
            sharex=True,
            sharey=True,
            dpi=125,
        )
        ax = ax[list(reversed(range(n_pcs))), :]

        params_data = {
            'c': 'k',
            'marker': 'o',
            's': 5.0,
        }
        params_samples = {
            'c': 'r',
            'marker': 'o',
            's': 5.0,
            'alpha': 0.05,
            'zorder': -1,
        }

        sample_pvs = self.kde.sample(n_samples)

        for row in range(n_pcs):
            for col in range(n_pcs):
                if row == col:
                    continue
                ax[row, col].scatter(
                    pc[:, col], pc[:, row], **params_data
                )
                ax[row, col].scatter(
                    sample_pvs[:, col], sample_pvs[:, row], **params_samples
                )

        for row in range(n_pcs):
            ax[0, row].set_xlabel(f'PC$_{{{row + 1}}}$')
        for col in range(n_pcs):
            ax[col, 0].set_ylabel(f'PC$_{{{col + 1}}}$')

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.10, hspace=0.10)

        self.plot_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.plot_dir / _FILENAME_PLOT

        fig.savefig(out_path)
        plt.close(fig)


