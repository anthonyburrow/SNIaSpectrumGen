import numpy as np

from SNIaSpectrumGen.SpectrumGenerator import SpectrumGenerator
from helpers import validate_spectrum_array


def test_generate_batch_serial():
    gen = SpectrumGenerator(N_workers=1)
    batch = gen.generate_batch(batch_size=3)
    assert len(batch) == 3, "Batch size mismatch"
    for arr in batch:
        validate_spectrum_array(arr)


def test_generate_batch_parallel():
    gen = SpectrumGenerator(N_workers=2)
    batch = gen.generate_batch(batch_size=4)
    assert len(batch) == 4, "Batch size mismatch (parallel)"
    for arr in batch:
        validate_spectrum_array(arr)


def test_generate_single_equivalence():
    np.random.seed(42)
    gen = SpectrumGenerator(N_workers=1)
    single = gen.generate()

    np.random.seed(42)
    gen2 = SpectrumGenerator(N_workers=1)
    batch = gen2.generate_batch(batch_size=1)[0]
    
    assert single.shape == batch.shape
