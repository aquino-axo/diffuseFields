"""
Analyze frequency-dependent basis dimension requirements.

Determines the minimum number of eigenvectors needed at each frequency
to achieve target reconstruction accuracy levels.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt


class BasisDimensionAnalyzer:
    """
    Analyzes how basis dimension requirements vary with frequency.

    Parameters
    ----------
    eigendata_dir : Path
        Directory containing eigendata_freq{i}.npz files.
    frequencies : ndarray
        Frequency array in Hz.
    speed_of_sound : float
        Speed of sound in m/s.
    """

    def __init__(
        self,
        eigendata_dir: Path,
        frequencies: np.ndarray,
        speed_of_sound: float = 343.0
    ):
        self.eigendata_dir = Path(eigendata_dir)
        self.frequencies = frequencies
        self.speed_of_sound = speed_of_sound
        self.nfreqs = len(frequencies)
        self.wavenumbers = 2 * np.pi * frequencies / speed_of_sound

    def load_eigendata(self, freq_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load eigenvalues and eigenvectors for a frequency index."""
        filepath = self.eigendata_dir / f"eigendata_freq{freq_idx}.npz"
        data = np.load(filepath)
        return data['eigenvalues'], data['eigenvectors']

    def compute_variance_capture(self, eigenvalues: np.ndarray) -> np.ndarray:
        """Compute cumulative variance captured by eigenvalues."""
        total = np.sum(eigenvalues)
        if total == 0:
            return np.ones(len(eigenvalues))
        return np.cumsum(eigenvalues) / total

    def find_modes_for_variance(
        self,
        eigenvalues: np.ndarray,
        target_variance: float
    ) -> int:
        """Find minimum modes needed to capture target variance fraction."""
        cumvar = self.compute_variance_capture(eigenvalues)
        idx = np.searchsorted(cumvar, target_variance)
        return min(idx + 1, len(eigenvalues))

    def analyze_all_frequencies(
        self,
        target_variances: Optional[List[float]] = None,
        freq_indices: Optional[List[int]] = None
    ) -> Dict:
        """
        Analyze basis dimension requirements across frequencies.

        Parameters
        ----------
        target_variances : list of float, optional
            Target variance fractions. Default [0.90, 0.95, 0.99, 0.999].
        freq_indices : list of int, optional
            Frequency indices to analyze. If None, analyzes all.

        Returns
        -------
        results : dict
            Analysis results including modes required per frequency.
        """
        if target_variances is None:
            target_variances = [0.90, 0.95, 0.99, 0.999]
        if freq_indices is None:
            freq_indices = list(range(self.nfreqs))

        results = {
            'frequencies': self.frequencies[freq_indices],
            'freq_indices': freq_indices,
            'target_variances': target_variances,
            'modes_required': {v: [] for v in target_variances},
            'eigenvalue_spectra': [],
            'cumulative_variance': [],
            'total_eigenvalues': [],
            'condition_numbers': []
        }

        for freq_idx in freq_indices:
            eigenvalues, _ = self.load_eigendata(freq_idx)

            results['eigenvalue_spectra'].append(eigenvalues)
            results['cumulative_variance'].append(
                self.compute_variance_capture(eigenvalues)
            )
            results['total_eigenvalues'].append(len(eigenvalues))

            # Condition number of significant eigenvalues
            sig_eigs = eigenvalues[eigenvalues > 1e-10 * eigenvalues[0]]
            if len(sig_eigs) > 0:
                cond = sig_eigs[0] / sig_eigs[-1]
            else:
                cond = np.inf
            results['condition_numbers'].append(cond)

            for target in target_variances:
                n_modes = self.find_modes_for_variance(eigenvalues, target)
                results['modes_required'][target].append(n_modes)

        return results

    def fit_scaling_law(
        self,
        frequencies: np.ndarray,
        modes_required: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Fit power law: N_modes = A * f^beta.

        Returns (A, beta, r_squared).
        """
        # Use log-log linear regression
        valid = (frequencies > 0) & (modes_required > 0)
        if np.sum(valid) < 2:
            return 0.0, 0.0, 0.0

        log_f = np.log(frequencies[valid])
        log_n = np.log(modes_required[valid])

        coeffs = np.polyfit(log_f, log_n, 1)
        beta = coeffs[0]
        A = np.exp(coeffs[1])

        # Compute R^2
        log_n_pred = np.polyval(coeffs, log_f)
        ss_res = np.sum((log_n - log_n_pred) ** 2)
        ss_tot = np.sum((log_n - np.mean(log_n)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return A, beta, r_squared

    def plot_eigenvalue_decay(
        self,
        results: Dict,
        freq_indices_to_plot: Optional[List[int]] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Plot eigenvalue decay at selected frequencies."""
        if freq_indices_to_plot is None:
            # Select ~5 frequencies spread across range
            n_plot = min(5, len(results['freq_indices']))
            indices = np.linspace(0, len(results['freq_indices']) - 1, n_plot, dtype=int)
            freq_indices_to_plot = [results['freq_indices'][i] for i in indices]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(freq_indices_to_plot)))

        for i, freq_idx in enumerate(freq_indices_to_plot):
            idx_in_results = results['freq_indices'].index(freq_idx)
            eigs = results['eigenvalue_spectra'][idx_in_results]
            cumvar = results['cumulative_variance'][idx_in_results]
            freq = results['frequencies'][idx_in_results]

            mode_indices = np.arange(1, len(eigs) + 1)

            # Normalized eigenvalue decay
            ax1.semilogy(mode_indices, eigs / eigs[0], 'o-',
                        color=colors[i], label=f'{freq:.0f} Hz', markersize=4)

            # Cumulative variance
            ax2.plot(mode_indices, cumvar, 'o-',
                    color=colors[i], label=f'{freq:.0f} Hz', markersize=4)

        ax1.set_xlabel('Mode Index')
        ax1.set_ylabel('Normalized Eigenvalue (λ/λ₁)')
        ax1.set_title('Eigenvalue Decay')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Mode Index')
        ax2.set_ylabel('Cumulative Variance Fraction')
        ax2.set_title('Cumulative Variance Captured')
        ax2.axhline(0.99, color='k', linestyle='--', alpha=0.5, label='99%')
        ax2.axhline(0.95, color='k', linestyle=':', alpha=0.5, label='95%')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def generate_report(self, results: Dict) -> str:
        """Generate text summary of analysis."""
        lines = ["Basis Dimension Analysis Report", "=" * 40, ""]

        frequencies = results['frequencies']
        lines.append(f"Frequency range: {frequencies.min():.0f} - {frequencies.max():.0f} Hz")
        lines.append(f"Number of frequencies analyzed: {len(frequencies)}")
        lines.append("")

        lines.append("Scaling Law Fits (N ~ A * f^beta):")
        lines.append("-" * 40)
        for target in results['target_variances']:
            modes = np.array(results['modes_required'][target])
            A, beta, r2 = self.fit_scaling_law(frequencies, modes)
            lines.append(f"  {100*target:.1f}% variance: beta = {beta:.3f}, R² = {r2:.3f}")
        lines.append("")

        lines.append("Modes Required at Selected Frequencies:")
        lines.append("-" * 40)

        # Select a few frequencies to report
        n_report = min(5, len(frequencies))
        report_indices = np.linspace(0, len(frequencies) - 1, n_report, dtype=int)

        header = f"{'Freq (Hz)':<12}" + "".join(
            f"{100*t:.0f}%".rjust(8) for t in results['target_variances']
        )
        lines.append(header)

        for idx in report_indices:
            freq = frequencies[idx]
            row = f"{freq:<12.0f}"
            for target in results['target_variances']:
                n = results['modes_required'][target][idx]
                row += f"{n}".rjust(8)
            lines.append(row)

        lines.append("")
        lines.append("Theoretical expectation: N_modes ~ f² (beta = 2)")
        lines.append("Higher beta indicates faster complexity growth with frequency.")

        return "\n".join(lines)


def run_analysis(
    eigendata_dir: str,
    frequencies: np.ndarray,
    speed_of_sound: float = 343.0,
    output_dir: Optional[str] = None,
    target_variances: Optional[List[float]] = None
) -> Dict:
    """
    Run full basis dimension analysis.

    Parameters
    ----------
    eigendata_dir : str
        Path to directory with eigendata files.
    frequencies : ndarray
        Frequency array in Hz.
    speed_of_sound : float
        Speed of sound.
    output_dir : str, optional
        Directory to save results and plots.
    target_variances : list of float, optional
        Target variance fractions to analyze.

    Returns
    -------
    results : dict
        Analysis results.
    """
    analyzer = BasisDimensionAnalyzer(
        eigendata_dir=Path(eigendata_dir),
        frequencies=frequencies,
        speed_of_sound=speed_of_sound
    )

    results = analyzer.analyze_all_frequencies(target_variances=target_variances)

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        analyzer.plot_eigenvalue_decay(
            results,
            save_path=output_path / "eigenvalue_decay.png"
        )

        report = analyzer.generate_report(results)
        with open(output_path / "basis_dimension_report.txt", 'w') as f:
            f.write(report)
        print(report)

    return results


if __name__ == "__main__":
    print("Basis Dimension Analyzer")
    print("=" * 50)
    print("\nUsage:")
    print("  from validate_basis_dimension import BasisDimensionAnalyzer")
    print("  analyzer = BasisDimensionAnalyzer(...)")
    print("  results = analyzer.analyze_all_frequencies()")
    print("\nSee docstrings for full API details.")
