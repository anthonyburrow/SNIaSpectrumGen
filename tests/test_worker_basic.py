import numpy as np
from SNIaSpectrumGen.SpectrumGenerator import SpectrumGenerator
from helpers import validate_spectrum_array


def test_generate_single_basic():
    gen = SpectrumGenerator()
    arr = gen.generate()
    validate_spectrum_array(arr)
    # wavelengths strictly increasing
    assert np.all(np.diff(arr[:,0]) > 0)
    # flux finite
    assert np.all(np.isfinite(arr[:,1]))


def test_generate_batch_count_and_independence():
    gen = SpectrumGenerator()
    batch = gen.generate_batch(batch_size=5)
    assert len(batch) == 5
    for arr in batch:
        validate_spectrum_array(arr)
    # independence heuristic: pairwise correlation should not all be ~1
    fluxes = [b[:,1] for b in batch]
    corrs = []
    for i in range(len(fluxes)):
        for j in range(i+1,len(fluxes)):
            # resample shortest length segment
            m = min(len(fluxes[i]), len(fluxes[j]))
            c = np.corrcoef(fluxes[i][:m], fluxes[j][:m])[0,1]
            corrs.append(c)
    assert not all(c > 0.99 for c in corrs)
