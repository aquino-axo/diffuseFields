"""
Run validation of CPSD eigenvector basis.

Orchestrates reconstruction accuracy and basis dimension analysis.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Optional

from validate_reconstruction import ReconstructionValidator
from validate_basis_dimension import BasisDimensionAnalyzer


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path) as f:
        return json.load(f)


def run_basis_dimension_analysis(
    eigendata_dir: Path,
    frequencies: np.ndarray,
    speed_of_sound: float,
    output_dir: Path
) -> dict:
    """Run basis dimension analysis."""
    print("\n" + "=" * 60)
    print("Basis Dimension Analysis")
    print("=" * 60)

    analyzer = BasisDimensionAnalyzer(
        eigendata_dir=eigendata_dir,
        frequencies=frequencies,
        speed_of_sound=speed_of_sound
    )

    results = analyzer.analyze_all_frequencies()

    # Generate plot
    analyzer.plot_eigenvalue_decay(
        results,
        save_path=output_dir / "eigenvalue_decay.png"
    )

    # Print report
    report = analyzer.generate_report(results)
    print(report)

    with open(output_dir / "basis_dimension_report.txt", 'w') as f:
        f.write(report)

    return results


def run_reconstruction_validation(
    eigendata_dir: Path,
    frequencies: np.ndarray,
    validation_set: np.ndarray,
    output_dir: Path,
    all_freqs_eigendata: Optional[Path] = None
) -> dict:
    """Run reconstruction accuracy validation.

    Parameters
    ----------
    eigendata_dir : Path
        Directory containing eigendata files.
    frequencies : ndarray
        Frequency array in Hz.
    validation_set : ndarray
        Validation pressure fields, shape (ndof, n_fields, nfreqs).
    output_dir : Path
        Output directory for results and plots.
    all_freqs_eigendata : Path, optional
        Path to all-frequencies eigendata file. When provided, uses
        shared eigenvectors for all frequencies instead of per-frequency.
    """
    print("\n" + "=" * 60)
    print("Reconstruction Accuracy Validation")
    if all_freqs_eigendata is not None:
        print("  (using all-frequencies eigenvectors)")
    print("=" * 60)

    validator = ReconstructionValidator(
        eigendata_dir=eigendata_dir,
        frequencies=frequencies,
        all_freqs_eigendata=all_freqs_eigendata
    )

    results = validator.validate(validation_set)

    # Generate plot
    validator.plot_results(
        results,
        title="Reconstruction Error for Validation Set",
        save_path=output_dir / "reconstruction_error.png"
    )

    # Generate histograms
    validator.plot_error_histogram(
        results,
        title="Reconstruction Error Distribution (All Frequencies)",
        save_path=output_dir / "error_histogram.png"
    )

    validator.plot_octave_band_histograms(
        results,
        save_path=output_dir / "error_histogram_octave_bands.png"
    )

    # Print summary
    print("\nReconstruction Error (using all stored eigenvectors per frequency):")
    print("-" * 60)
    print(f"{'Freq (Hz)':<12}{'Modes':<8}{'Mean Error':<14}{'Max Error':<14}")
    print("-" * 60)
    for i, freq in enumerate(results['frequencies']):
        n_modes = results['n_modes_used'][i]
        mean_err = results['mean_errors'][i]
        max_err = results['max_errors'][i]
        print(f"{freq:<12.0f}{n_modes:<8}{mean_err:<14.6f}{max_err:<14.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate CPSD eigenvector basis"
    )
    parser.add_argument(
        "--config", "-c",
        default="../config_cone_range.json",
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--eigendata-dir", "-e",
        default="../results_cone",
        help="Directory containing eigendata files"
    )
    parser.add_argument(
        "--validation-set", "-v",
        help="Path to validation set .npy file, shape (ndof, n_fields, nfreqs)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="validation_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--analysis",
        choices=["reconstruction", "dimension", "both"],
        default="both",
        help="Which analysis to run: reconstruction, dimension, or both"
    )
    parser.add_argument(
        "--all-freqs",
        action="store_true",
        default=False,
        help="Use all-frequencies eigenvectors (eigendata_all_freqs.npz) "
             "instead of per-frequency eigenvectors for reconstruction"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Build frequency array
    freq_cfg = config['physics']['frequencies']
    frequencies = np.arange(
        freq_cfg['min'],
        freq_cfg['max'] + freq_cfg['step'],
        freq_cfg['step']
    )

    speed_of_sound = config['physics']['speed_of_sound']

    # Setup paths
    eigendata_dir = Path(args.eigendata_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("CPSD Eigenvector Basis Validation")
    print("=" * 60)
    print(f"Eigendata directory: {eigendata_dir}")
    print(f"Frequency range: {frequencies[0]:.0f} - {frequencies[-1]:.0f} Hz")
    print(f"Output directory: {output_dir}")

    # Run basis dimension analysis
    if args.analysis in ["dimension", "both"]:
        run_basis_dimension_analysis(
            eigendata_dir=eigendata_dir,
            frequencies=frequencies,
            speed_of_sound=speed_of_sound,
            output_dir=output_dir
        )

    # Run reconstruction validation
    if args.analysis in ["reconstruction", "both"]:
        if args.validation_set:
            validation_set = np.load(args.validation_set)

            # Determine eigendata source
            all_freqs_path = None
            if args.all_freqs:
                all_freqs_path = eigendata_dir / "eigendata_all_freqs.npz"
                if not all_freqs_path.exists():
                    print(f"\nError: {all_freqs_path} not found.")
                    print("Run cone analysis with all_freqs_svd=true first.")
                    return

            run_reconstruction_validation(
                eigendata_dir=eigendata_dir,
                frequencies=frequencies,
                validation_set=validation_set,
                output_dir=output_dir,
                all_freqs_eigendata=all_freqs_path
            )
        else:
            print("\nReconstruction validation skipped: requires --validation-set")
            print("Usage: python run_validation.py --analysis reconstruction -v validation_set.npy")

    print("\n" + "=" * 60)
    print(f"Validation complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
