import numpy as np
import pytest

from SNIaSpectrumGen.SpectrumGenerator import SpectrumGenerator


def _validate_spectrum_array(arr: np.ndarray) -> None:
    assert isinstance(arr, np.ndarray), \
        "Batch item must be a numpy array"
    assert arr.ndim == 2 and arr.shape[1] == 2, \
        "Spectrum array must be (N, 2)"
    assert arr.shape[0] > 10, \
        "Spectrum should have >10 wavelength samples"
    assert np.all(np.diff(arr[:, 0]) >= 0), \
        "Wavelength grid must be sorted ascending"
    assert np.isfinite(arr[:, 1]).all(), \
        "Flux values must be finite"


def test_generate_batch_serial():
    gen = SpectrumGenerator(N_workers=1)
    batch = gen.generate_batch(batch_size=3)
    assert len(batch) == 3, "Batch size mismatch"
    for arr in batch:
        _validate_spectrum_array(arr)


def test_generate_batch_parallel():
    gen = SpectrumGenerator(N_workers=2)
    batch = gen.generate_batch(batch_size=4)
    assert len(batch) == 4, "Batch size mismatch (parallel)"
    for arr in batch:
        _validate_spectrum_array(arr)


def test_generate_single_equivalence():
    np.random.seed(42)
    gen = SpectrumGenerator(N_workers=1)
    single = gen.generate()

    np.random.seed(42)
    gen2 = SpectrumGenerator(N_workers=1)
    batch = gen2.generate_batch(batch_size=1)[0]
    
    assert single.shape == batch.shape
