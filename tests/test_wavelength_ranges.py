import numpy as np
from SNIaSpectrumGen.SpectrumGenerator import SpectrumGeneratorWorker


def test_custom_wavelength_and_length_ranges():
    length_range = (900, 910)
    wave_min_range = (6000., 6000.)
    wave_max_range = (7000., 7000.)

    worker = SpectrumGeneratorWorker(
        length_range=length_range,
        wave_min_range=wave_min_range,
        wave_max_range=wave_max_range,
    )
    wave = worker._get_wavelengths()

    assert len(wave) >= length_range[0] and len(wave) <= length_range[1]
    assert wave[0] == wave_min_range[0]
    assert wave[-1] == wave_max_range[-1]
    assert np.all(np.diff(wave) > 0)
