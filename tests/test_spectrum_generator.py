import numpy as np

from SNIaSpectrumGen.SpectrumGenerator import SpectrumGenerator
from helpers import validate_spectrum_array


def test_generate_single():
    gen = SpectrumGenerator()
    spectrum = gen.generate()
    validate_spectrum_array(spectrum)


def test_generate_batch():
    gen = SpectrumGenerator()
    batch = gen.generate_batch(batch_size=5)
    assert len(batch) == 5
    for arr in batch:
        validate_spectrum_array(arr)

