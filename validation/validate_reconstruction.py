"""
Validate eigenvector basis reconstruction accuracy.

Tests whether the CPSD eigenvector basis can accurately reconstruct
pressure fields from a validation set not used in the original
eigenvalue computation.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt


class ReconstructionValidator:
    """
    Validates eigenvector basis by testing reconstruction of pressure fields.

    Parameters
    ----------
    eigendata_dir : Path
        Directory containing eigendata_freq{i}.npz files.
    frequencies : ndarray
        Frequency array in Hz.
    all_freqs_eigendata : Path, optional
        Path to all-frequencies eigendata file (eigendata_all_freqs.npz).
        When provided, uses the same eigenvectors for all frequencies
        instead of loading per-frequency eigenvectors.
    """

    def __init__(
        self,
        eigendata_dir: Path,
        frequencies: np.ndarray,
        all_freqs_eigendata: Optional[Path] = None
    ):
        self.eigendata_dir = Path(eigendata_dir)
        self.frequencies = frequencies
        self.nfreqs = len(frequencies)

        # Load all-frequencies eigendata once if provided
        if all_freqs_eigendata is not None:
            data = np.load(all_freqs_eigendata)
            self._all_freqs_eigenvalues = data['eigenvalues']
            self._all_freqs_eigenvectors = data['eigenvectors']
        else:
            self._all_freqs_eigenvalues = None
            self._all_freqs_eigenvectors = None

    def load_eigendata(self, freq_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load eigenvalues and eigenvectors for a frequency index.

        If all-frequencies eigendata was provided at construction,
        returns those (shared) eigenvectors regardless of freq_idx.
        """
        if self._all_freqs_eigenvectors is not None:
            return self._all_freqs_eigenvalues, self._all_freqs_eigenvectors

        filepath = self.eigendata_dir / f"eigendata_freq{freq_idx}.npz"
        data = np.load(filepath)
        return data['eigenvalues'], data['eigenvectors']

    def project_onto_basis(
        self,
        pressure_field: np.ndarray,
        eigenvectors: np.ndarray,
        n_modes: Optional[int] = None
    ) -> np.ndarray:
        """
        Project pressure field onto eigenvector basis.

        Parameters
        ----------
        pressure_field : ndarray
            Pressure field(s) to project, shape (ndof,) or (ndof, n_fields).
        eigenvectors : ndarray
            Eigenvector basis, shape (ndof, n_eigenvectors).
        n_modes : int, optional
            Number of modes to use. If None, uses all available.

        Returns
        -------
        reconstructed : ndarray
            Reconstructed field(s), same shape as pressure_field.
        """
        if n_modes is None:
            n_modes = eigenvectors.shape[1]
        V = eigenvectors[:, :n_modes]

        # Project: coefficients = V^H @ p
        # Reconstruct: p_approx = V @ coefficients = V @ V^H @ p
        if pressure_field.ndim == 1:
            coeffs = V.conj().T @ pressure_field
            return V @ coeffs
        else:
            coeffs = V.conj().T @ pressure_field
            return V @ coeffs

    def compute_reconstruction_error(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray
    ) -> float:
        """
        Compute relative L2 reconstruction error.

        Returns ||original - reconstructed||_2 / ||original||_2
        """
        error = np.linalg.norm(original - reconstructed)
        norm = np.linalg.norm(original)
        return error / norm if norm > 0 else 0.0

    def validate(
        self,
        validation_set: np.ndarray,
        freq_indices: Optional[List[int]] = None,
        n_modes_list: Optional[List[int]] = None
    ) -> Dict:
        """
        Validate reconstruction using a validation set.

        Parameters
        ----------
        validation_set : ndarray
            Pressure fields for validation, shape (ndof, n_fields, nfreqs).
            Each column is a pressure field to reconstruct.
        freq_indices : list of int, optional
            Frequency indices to validate. If None, validates all.
        n_modes_list : list of int, optional
            List of mode counts to test. If None, uses [10, 25, 50, 75, 100].

        Returns
        -------
        results : dict
            Dictionary with validation results including errors per frequency
            and per mode count.
        """
        if freq_indices is None:
            freq_indices = list(range(self.nfreqs))

        n_fields = validation_set.shape[1]

        results = {
            'frequencies': self.frequencies[freq_indices],
            'freq_indices': freq_indices,
            'n_modes_used': [],
            'mean_errors': [],
            'max_errors': [],
            'per_field_errors': []
        }

        for freq_idx in freq_indices:
            eigenvalues, eigenvectors = self.load_eigendata(freq_idx)
            n_modes = eigenvectors.shape[1]

            # Get validation fields at this frequency
            p_validation = validation_set[:, :, freq_idx]

            errors = []
            for field_idx in range(n_fields):
                p_true = p_validation[:, field_idx]
                p_recon = self.project_onto_basis(p_true, eigenvectors, n_modes)
                error = self.compute_reconstruction_error(p_true, p_recon)
                errors.append(error)

            results['n_modes_used'].append(n_modes)
            results['mean_errors'].append(np.mean(errors))
            results['max_errors'].append(np.max(errors))
            results['per_field_errors'].append(errors)

        return results

    def plot_results(
        self,
        results: Dict,
        title: str = "Reconstruction Error vs Frequency",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Plot reconstruction error results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        frequencies = results['frequencies']
        mean_errors = results['mean_errors']
        max_errors = results['max_errors']

        ax1.semilogy(frequencies, mean_errors, 'o-')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Mean Relative L2 Error')
        ax1.set_title('Mean Reconstruction Error')
        ax1.grid(True, alpha=0.3)

        ax2.semilogy(frequencies, max_errors, 'o-')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Max Relative L2 Error')
        ax2.set_title('Max Reconstruction Error')
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_error_histogram(
        self,
        results: Dict,
        n_bins: int = 30,
        title: str = "Reconstruction Error Distribution",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot histogram of all errors aggregated across frequencies.

        Parameters
        ----------
        results : dict
            Results from validate() method.
        n_bins : int
            Number of histogram bins.
        title : str
            Plot title.
        save_path : Path, optional
            Path to save the figure.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        # Flatten all errors across frequencies
        all_errors = []
        for errors in results['per_field_errors']:
            all_errors.extend(errors)
        all_errors = np.array(all_errors)

        fig, ax = plt.subplots(figsize=(8, 5))

        # Use log scale bins
        log_min = np.floor(np.log10(all_errors.min()))
        log_max = np.ceil(np.log10(all_errors.max()))
        bins = np.logspace(log_min, log_max, n_bins + 1)

        ax.hist(all_errors, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_xscale('log')
        ax.set_xlabel('Relative L2 Error')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        # Add 95th percentile annotation
        p95 = np.percentile(all_errors, 95)
        ax.axvline(p95, color='red', linestyle='--', linewidth=2, label=f'95th percentile: {p95:.4f}')
        ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_octave_band_histograms(
        self,
        results: Dict,
        n_bins: int = 30,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot histograms for each octave band in a grid layout.

        Parameters
        ----------
        results : dict
            Results from validate() method.
        n_bins : int
            Number of histogram bins.
        save_path : Path, optional
            Path to save the figure.

        Returns
        -------
        fig : Figure
            Matplotlib figure.
        """
        # Define octave bands (center frequency, lower bound, upper bound)
        octave_bands = [
            (250, 177, 354),
            (500, 354, 707),
            (1000, 707, 1414),
            (2000, 1414, 2828),
            (4000, 2828, 5657),
        ]

        frequencies = results['frequencies']
        per_field_errors = results['per_field_errors']

        # Collect errors for each octave band
        band_errors = {}
        for center, low, high in octave_bands:
            errors = []
            for i, freq in enumerate(frequencies):
                if low <= freq < high:
                    errors.extend(per_field_errors[i])
            if errors:
                band_errors[center] = np.array(errors)

        # Create subplots
        n_bands = len(band_errors)
        if n_bands == 0:
            return None

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        axes = axes.flatten()

        for idx, (center, errors) in enumerate(band_errors.items()):
            ax = axes[idx]

            # Use log scale bins
            log_min = np.floor(np.log10(errors.min()))
            log_max = np.ceil(np.log10(errors.max()))
            bins = np.logspace(log_min, log_max, n_bins + 1)

            ax.hist(errors, bins=bins, edgecolor='black', alpha=0.7)
            ax.set_xscale('log')
            ax.set_xlabel('Relative L2 Error')
            ax.set_ylabel('Count')
            ax.set_title(f'{center} Hz Octave Band')
            ax.grid(True, alpha=0.3)

            # Add 95th percentile
            p95 = np.percentile(errors, 95)
            ax.axvline(p95, color='red', linestyle='--', linewidth=1.5, label=f'95th: {p95:.4f}')
            ax.legend(fontsize=8)

        # Hide unused axes
        for idx in range(len(band_errors), len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('Reconstruction Error by Octave Band', fontsize=12)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def run_validation(
    eigendata_dir: str,
    validation_set_path: str,
    frequencies: np.ndarray,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Run full reconstruction validation.

    Parameters
    ----------
    eigendata_dir : str
        Path to directory with eigendata files.
    validation_set_path : str
        Path to validation set .npy file, shape (ndof, n_fields, nfreqs).
    frequencies : ndarray
        Frequency array in Hz.
    output_dir : str, optional
        Directory to save results and plots.

    Returns
    -------
    results : dict
        Validation results.
    """
    validation_set = np.load(validation_set_path)

    validator = ReconstructionValidator(
        eigendata_dir=Path(eigendata_dir),
        frequencies=frequencies
    )

    results = validator.validate(validation_set)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        validator.plot_results(
            results,
            title="Reconstruction Error for Validation Set",
            save_path=output_path / "reconstruction_error.png"
        )
        np.savez(
            output_path / "reconstruction_results.npz",
            frequencies=results['frequencies'],
            n_modes_list=np.array(results['n_modes_list']),
            mean_errors={str(k): v for k, v in results['mean_errors'].items()},
            max_errors={str(k): v for k, v in results['max_errors'].items()}
        )

    return results


if __name__ == "__main__":
    print("Reconstruction Validator")
    print("=" * 50)
    print("\nUsage:")
    print("  from validate_reconstruction import ReconstructionValidator")
    print("  validator = ReconstructionValidator(eigendata_dir, frequencies)")
    print("  results = validator.validate(validation_set)")
    print("\nValidation set shape: (ndof, n_fields, nfreqs)")
    print("See docstrings for full API details.")
