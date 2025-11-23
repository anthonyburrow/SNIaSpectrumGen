import time

from SNIaSpectrumGen.SpectrumGenerator import SpectrumGenerator
from helpers import validate_spectrum_array


def test_parallel_speedup():
    """Integration test comparing serial vs parallel generation performance."""
    total_samples = 80
    batch_size = 16
    n_workers = 4
    n_batches = total_samples // batch_size

    print(f"\n{'='*60}")
    print(f"Performance Comparison: {total_samples} spectra")
    print(f"Batch size: {batch_size}, Batches: {n_batches}")
    print(f"{'='*60}")

    # Serial timing
    print("\nRunning serial generation...")
    gen_serial = SpectrumGenerator(N_workers=1)
    all_batches_serial = []
    t0_serial = time.perf_counter()
    for _ in range(n_batches):
        batch = gen_serial.generate_batch(batch_size=batch_size)
        all_batches_serial.extend(batch)
    t_serial = time.perf_counter() - t0_serial

    # Validate serial results
    assert len(all_batches_serial) == total_samples
    for arr in all_batches_serial:
        validate_spectrum_array(arr)

    print(f"Serial time:   {t_serial:.3f} seconds")

    # Parallel timing
    print(f"\nRunning parallel generation ({n_workers} workers)...")
    gen_parallel = SpectrumGenerator(N_workers=n_workers)
    all_batches_parallel = []
    t0_parallel = time.perf_counter()
    for _ in range(n_batches):
        batch = gen_parallel.generate_batch(batch_size=batch_size)
        all_batches_parallel.extend(batch)
    t_parallel = time.perf_counter() - t0_parallel

    # Validate parallel results
    assert len(all_batches_parallel) == total_samples
    for arr in all_batches_parallel:
        validate_spectrum_array(arr)

    print(f"Parallel time: {t_parallel:.3f} seconds")

    # Calculate speedup
    speedup = t_serial / t_parallel
    expected_speedup = n_workers
    print(f"\nSpeedup factor: {speedup:.2f}x")
    print(f"Expected (ideal): {expected_speedup:.2f}x")
    print(f"Efficiency: {speedup / n_workers * 100:.1f}%")
    print(f"{'='*60}\n")

    # Assert we get some speedup
    assert speedup > 1.5, \
        f"Expected speedup > 1.5x, got {speedup:.2f}x"
