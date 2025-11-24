import matplotlib.pyplot as plt

from SNIaSpectrumGen.SpectrumGenerator import SpectrumGenerator
from helpers import validate_spectrum_array


def test_output_single(plots_dir):
    """Visual test: generate and plot a single spectrum."""
    gen = SpectrumGenerator()
    spectrum = gen.generate()
    
    validate_spectrum_array(spectrum)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spectrum[:, 0], spectrum[:, 1], 'k-', linewidth=1.0, alpha=0.8)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Normalized Flux')
    ax.set_title('Single Spectrum')
    ax.grid(True, alpha=0.3)
    
    output_path = plots_dir / 'test_output_single.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved plot: {output_path}")
    assert output_path.exists()


def test_output_batch(plots_dir):
    """Visual test: generate and plot batch of spectra."""
    batch_size = 5
    gen = SpectrumGenerator()
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
    ax.set_title(f'Batch Generation ({batch_size} spectra)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    output_path = plots_dir / 'test_output_batch.pdf'
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\nSaved plot: {output_path}")
    assert output_path.exists()

