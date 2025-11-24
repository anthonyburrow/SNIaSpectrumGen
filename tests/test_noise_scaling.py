import numpy as np
from SNIaSpectrumGen.SpectrumGenerator import SpectrumGeneratorWorker


def test_noise_scaling_fixed_level():
    worker = SpectrumGeneratorWorker(noise_range=(0.02, 0.02))
    base_flux = np.ones(500)
    noise = worker._generate_noise(base_flux)

    std = noise.std()
    assert 0.01 < std < 0.03
