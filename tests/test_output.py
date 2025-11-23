import matplotlib.pyplot as plt

from SNIaSpectrumGen.SpectrumGenerator import SpectrumGenerator
from helpers import validate_spectrum_array


def test_output_serial_single(plots_dir):
    """Visual test: generate and plot a single spectrum (serial)."""
    gen = SpectrumGenerator(N_workers=1)
    spectrum = gen.generate()
    
    validate_spectrum_array(spectrum)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spectrum[:, 0], spectrum[:, 1], 'k-', linewidth=1.0, alpha=0.8)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Serial Single Spectrum')
    ax.grid(True, alpha=0.3)
    
    output_path = plots_dir / 'test_output_serial_single.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved plot: {output_path}")
    assert output_path.exists()


def test_output_serial_batch(plots_dir):
    """Visual test: generate and plot batch of spectra (serial)."""
    batch_size = 4
    gen = SpectrumGenerator(N_workers=1)
    batch = gen.generate_batch(batch_size=batch_size)
    
    assert len(batch) == batch_size
    for spectrum in batch:
        validate_spectrum_array(spectrum)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    offset = 1.5
    
    for i, spectrum in enumerate(batch):
        shifted_flux = spectrum[:, 1] + i * offset
        ax.plot(
            spectrum[:, 0], shifted_flux,
            linewidth=1.0, alpha=0.8,
            label=f'Spectrum {i+1}'
        )
    
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalized Flux (offset)')
    ax.set_title(f'Serial Batch ({batch_size} spectra)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    output_path = plots_dir / 'test_output_serial_batch.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved plot: {output_path}")
    assert output_path.exists()


def test_output_parallel_batch(plots_dir):
    """Visual test: generate and plot batch of spectra (parallel)."""
    batch_size = 4
    n_workers = 4
    gen = SpectrumGenerator(N_workers=n_workers)
    batch = gen.generate_batch(batch_size=batch_size)
    
    assert len(batch) == batch_size
    for spectrum in batch:
        validate_spectrum_array(spectrum)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    offset = 1.5
    
    for i, spectrum in enumerate(batch):
        shifted_flux = spectrum[:, 1] + i * offset
        ax.plot(
            spectrum[:, 0], shifted_flux,
            linewidth=1.0, alpha=0.8,
            label=f'Spectrum {i+1}'
        )
    
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalized Flux (offset)')
    ax.set_title(
        f'Parallel Batch ({batch_size} spectra, {n_workers} workers)'
    )
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    output_path = plots_dir / 'test_output_parallel_batch.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved plot: {output_path}")
    assert output_path.exists()
