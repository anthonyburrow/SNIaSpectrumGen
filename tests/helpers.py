"""Helper functions for testing."""

import numpy as np


def validate_spectrum_array(arr: np.ndarray) -> None:
    """Validate that an array is a properly formatted spectrum.
    
    Args:
        arr: Array to validate, expected shape (N, 2) with wavelengths
             in column 0 and flux in column 1.
    
    Raises:
        AssertionError: If validation fails.
    """
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
