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
    """

    def __init__(
        self,
        eigendata_dir: Path,
        frequencies: np.ndarray
    ):
        self.eigendata_dir = Path(eigendata_dir)
        self.frequencies = frequencies
        self.nfreqs = len(frequencies)

    def load_eigendata(self, freq_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load eigenvalues and eigenvectors for a frequency index."""
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
