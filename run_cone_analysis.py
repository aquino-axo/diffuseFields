"""
Driver script for cone surface CPSD analysis.

Reads configuration from a JSON file and computes the covariance eigenvalues
of the total pressure field on a cone surface under diffuse field excitation.

The total field is p_t = p_inc + p_scat, where:
- p_inc is the incident plane wave field
- p_scat is the scattered field from the transfer matrix

Usage:
    python run_cone_analysis.py config_cone.json
    python run_cone_analysis.py --config config_cone.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, Union

import numpy as np
import matplotlib.pyplot as plt

from cone_diffuse_field import ConeDiffuseField
from cone_visualizer import ConeVisualizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def parse_frequencies(freq_config: Any) -> list:
    """
    Parse frequency configuration into a list of frequencies.

    Supports two formats:
    1. List of frequencies: [100.0, 200.0, 500.0]
    2. Range specification: {"min": 100.0, "step": 100.0, "max": 500.0}

    Parameters
    ----------
    freq_config : list or dict
        Frequency specification in either format.

    Returns
    -------
    list
        List of frequencies in Hz.

    Raises
    ------
    ValueError
        If the format is invalid or parameters are incorrect.
    """
    if freq_config is None:
        raise ValueError("physics.frequencies is required")

    # Format 1: List of frequencies
    if isinstance(freq_config, list):
        if len(freq_config) == 0:
            raise ValueError("physics.frequencies list cannot be empty")
        for f in freq_config:
            if not isinstance(f, (int, float)) or f <= 0:
                raise ValueError(
                    f"All frequencies must be positive numbers, got {f}"
                )
        return list(freq_config)

    # Format 2: Range specification with min, step, max
    if isinstance(freq_config, dict):
        required_keys = {'min', 'step', 'max'}
        missing = required_keys - set(freq_config.keys())
        if missing:
            raise ValueError(
                f"Frequency range specification missing keys: {missing}. "
                f"Required: min, step, max"
            )

        f_min = freq_config['min']
        f_step = freq_config['step']
        f_max = freq_config['max']

        # Validate values
        if not isinstance(f_min, (int, float)) or f_min <= 0:
            raise ValueError(f"frequencies.min must be positive, got {f_min}")
        if not isinstance(f_step, (int, float)) or f_step <= 0:
            raise ValueError(f"frequencies.step must be positive, got {f_step}")
        if not isinstance(f_max, (int, float)) or f_max <= 0:
            raise ValueError(f"frequencies.max must be positive, got {f_max}")
        if f_max < f_min:
            raise ValueError(
                f"frequencies.max ({f_max}) must be >= frequencies.min ({f_min})"
            )

        # Generate frequency list using numpy arange
        # Add small epsilon to include f_max if it falls exactly on a step
        frequencies = np.arange(f_min, f_max + f_step * 0.001, f_step)
        # Ensure we don't exceed f_max due to floating point
        frequencies = frequencies[frequencies <= f_max + 1e-10]
        return frequencies.tolist()

    raise ValueError(
        f"physics.frequencies must be a list or dict with min/step/max, "
        f"got {type(freq_config).__name__}"
    )


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fill in default values for configuration."""
    defaults = {
        'input': {
            'transfer_matrix_path': None,
            'coordinates_path': None,
            'directions_path': None
        },
        'physics': {
            'frequencies': None,
            'speed_of_sound': 343.0,
            'amplitude': 1.0
        },
        'cone_geometry': {
            'half_angle': None,
            'height': None
        },
        'eigenvalues': {
            'var_ratio': 0.99,
            'n_components': None,
            'solver': 'direct',  # 'direct' or 'randomized'
            'n_oversamples': 10,
            'n_power_iter': 2,
            'freq_indices': None  # None means all frequencies
        },
        'output': {
            'output_dir': 'results_cone',
            'save_figures': True,
            'figure_format': 'png',
            'save_eigenvectors': True,
            'plot_eigenvectors': True,
            'n_vectors_to_plot': 4,
            'plot_component': 'magnitude'
        }
    }

    # Merge defaults with provided config
    for section, section_defaults in defaults.items():
        if section not in config:
            config[section] = section_defaults
        else:
            for key, value in section_defaults.items():
                if key not in config[section]:
                    config[section][key] = value

    # Validate required inputs
    inp = config['input']
    if inp['transfer_matrix_path'] is None:
        raise ValueError("input.transfer_matrix_path is required")
    if inp['coordinates_path'] is None:
        raise ValueError("input.coordinates_path is required")
    if inp['directions_path'] is None:
        raise ValueError("input.directions_path is required")

    # Validate paths exist
    for key in ['transfer_matrix_path', 'coordinates_path', 'directions_path']:
        path = inp[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"{key}: {path} not found")

    # Parse and validate frequencies (supports list or min/step/max format)
    physics = config['physics']
    physics['frequencies'] = parse_frequencies(physics['frequencies'])

    if physics['speed_of_sound'] <= 0:
        raise ValueError("physics.speed_of_sound must be positive")
    if physics['amplitude'] <= 0:
        raise ValueError("physics.amplitude must be positive")

    # Validate cone geometry (only half_angle and height required)
    geom = config['cone_geometry']
    for key in ['half_angle', 'height']:
        if geom.get(key) is None:
            raise ValueError(f"cone_geometry.{key} is required")
        if geom[key] <= 0:
            raise ValueError(f"cone_geometry.{key} must be positive")

    # Validate eigenvalue parameters
    eig = config['eigenvalues']
    if eig['var_ratio'] is not None:
        if not 0 < eig['var_ratio'] <= 1:
            raise ValueError("eigenvalues.var_ratio must be in (0, 1]")
    if eig['n_components'] is not None:
        if eig['n_components'] < 1:
            raise ValueError("eigenvalues.n_components must be at least 1")
    if eig['solver'] not in ['direct', 'randomized']:
        raise ValueError("eigenvalues.solver must be 'direct' or 'randomized'")

    # Validate output parameters
    output = config['output']
    if output['figure_format'] not in ['png', 'pdf', 'svg', 'eps']:
        raise ValueError("output.figure_format must be 'png', 'pdf', 'svg', or 'eps'")
    valid_components = ['magnitude', 'real', 'imag', 'phase']
    if output['plot_component'] not in valid_components:
        raise ValueError(f"output.plot_component must be one of {valid_components}")

    return config


def load_input_data(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Load input arrays from .npy files (single array per file)."""
    inp = config['input']

    # Load transfer matrix
    T = np.load(inp['transfer_matrix_path'])

    # Load coordinates
    coords = np.load(inp['coordinates_path'])

    # Load directions
    directions = np.load(inp['directions_path'])

    # Validate transfer matrix 3rd dimension matches number of frequencies
    frequencies = config['physics']['frequencies']
    nfreqs = len(frequencies)
    if T.ndim != 3:
        raise ValueError(
            f"Transfer matrix must be 3D (ndof, npws, nfreqs), got shape {T.shape}"
        )
    if T.shape[2] != nfreqs:
        raise ValueError(
            f"Transfer matrix 3rd dimension ({T.shape[2]}) does not match "
            f"number of frequencies ({nfreqs})"
        )

    return {
        'transfer_matrix': T,
        'coordinates': coords,
        'directions': directions
    }


def run_analysis(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run the cone CPSD analysis."""
    print("Loading input data...")
    data = load_input_data(config)

    print(f"Transfer matrix shape: {data['transfer_matrix'].shape}")
    print(f"Coordinates shape: {data['coordinates'].shape}")
    print(f"Directions shape: {data['directions'].shape}")

    # Get frequencies from config
    frequencies = np.asarray(config['physics']['frequencies'])
    print(f"Frequencies: {frequencies} Hz")

    # Create ConeDiffuseField instance
    cone = ConeDiffuseField(
        transfer_matrix=data['transfer_matrix'],
        coordinates=data['coordinates'],
        directions=data['directions'],
        frequencies=frequencies,
        speed_of_sound=config['physics']['speed_of_sound'],
        amplitude=config['physics']['amplitude'],
        cone_geometry=config['cone_geometry']
    )

    print(f"Number of DOFs: {cone.ndof}")
    print(f"Number of plane waves: {cone.npws}")
    print(f"Number of frequencies: {cone.nfreqs}")
    print(f"Speed of sound: {cone.speed_of_sound} m/s")

    # Determine which frequencies to analyze
    eig_config = config['eigenvalues']
    if eig_config['freq_indices'] is None:
        freq_indices = list(range(cone.nfreqs))
    else:
        freq_indices = eig_config['freq_indices']

    results = {
        'config': config,
        'freq_indices': freq_indices,
        'frequencies': frequencies,
        'eigenvalues': {},
        'eigenvectors': {},
        'variance_explained': {},
        'n_components_kept': {}
    }

    # Compute eigenvalues for each frequency
    for freq_idx in freq_indices:
        freq = frequencies[freq_idx]
        print(f"\nProcessing frequency {freq:.1f} Hz (index {freq_idx})...")

        eigenvalues, eigenvectors = cone.compute_covariance_eigenvalues(
            freq_idx=freq_idx,
            var_ratio=eig_config['var_ratio'],
            n_components=eig_config['n_components'],
            solver=eig_config['solver'],
            n_oversamples=eig_config['n_oversamples'],
            n_power_iter=eig_config['n_power_iter']
        )

        _, cumulative = cone.get_variance_explained()

        results['eigenvalues'][freq_idx] = eigenvalues
        results['eigenvectors'][freq_idx] = eigenvectors
        results['variance_explained'][freq_idx] = cumulative
        results['n_components_kept'][freq_idx] = cone._n_components_kept

        print(f"  Computed {len(eigenvalues)} eigenvalues, kept {cone._n_components_kept} eigenvectors")
        print(f"  Variance captured by kept components: {cumulative[cone._n_components_kept - 1]:.4f}")

    return results, cone


def save_results(
    results: Dict[str, Any],
    cone: ConeDiffuseField,
    config: Dict[str, Any]
) -> None:
    """Save results to files."""
    output_config = config['output']
    output_dir = Path(output_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save eigenvalues and eigenvectors
    if output_config['save_eigenvectors']:
        for freq_idx in results['freq_indices']:
            freq = results['frequencies'][freq_idx]
            np.savez(
                output_dir / f'eigendata_freq{freq_idx}.npz',
                frequency=freq,
                eigenvalues=results['eigenvalues'][freq_idx],
                eigenvectors=results['eigenvectors'][freq_idx],
                variance_explained=results['variance_explained'][freq_idx]
            )
        print(f"Saved eigenvalue data to {output_dir}")

    # Save summary JSON
    summary = {
        'freq_indices': results['freq_indices'],
        'frequencies': [float(results['frequencies'][i]) for i in results['freq_indices']],
        'n_eigenvalues': {
            str(k): len(v) for k, v in results['eigenvalues'].items()
        },
        'variance_captured': {
            str(k): float(v[-1]) for k, v in results['variance_explained'].items()
        }
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)


def generate_plots(
    results: Dict[str, Any],
    cone: ConeDiffuseField,
    config: Dict[str, Any]
) -> None:
    """Generate and save plots."""
    output_config = config['output']
    output_dir = Path(output_config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    fig_format = output_config['figure_format']

    # Create visualizer from cone
    visualizer = ConeVisualizer(cone.coordinates, cone.cone_geometry)

    for freq_idx in results['freq_indices']:
        freq = results['frequencies'][freq_idx]
        eigenvalues = results['eigenvalues'][freq_idx]
        eigenvectors = results['eigenvectors'][freq_idx]
        n_kept = results['n_components_kept'][freq_idx]

        # Plot variance explained
        fig, ax = plt.subplots(figsize=(10, 6))
        visualizer.plot_variance_explained(
            eigenvalues,
            ax=ax,
            title=f'Variance Explained (f = {freq:.1f} Hz)',
            n_components_kept=n_kept
        )

        if output_config['save_figures']:
            fig.savefig(
                output_dir / f'variance_explained_freq{freq_idx}.{fig_format}',
                dpi=150, bbox_inches='tight'
            )
            plt.close(fig)
        else:
            plt.show()

        # Plot eigenvectors
        if output_config['plot_eigenvectors']:
            n_plot = min(
                output_config['n_vectors_to_plot'],
                eigenvectors.shape[1]
            )

            # Create labels with eigenvalue info
            labels = [
                f'Eigenvector {i + 1} (λ = {eigenvalues[i]:.2e})'
                for i in range(n_plot)
            ]

            fig = visualizer.plot_multiple_fields(
                eigenvectors[:, :n_plot],
                labels=labels,
                component=output_config['plot_component']
            )

            if output_config['save_figures']:
                fig.savefig(
                    output_dir / f'eigenvectors_freq{freq_idx}.{fig_format}',
                    dpi=150, bbox_inches='tight'
                )
                plt.close(fig)
            else:
                plt.show()

    print(f"Plots saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run cone surface CPSD analysis'
    )
    parser.add_argument(
        'config_file',
        nargs='?',
        default='config_cone.json',
        help='Path to configuration JSON file (default: config_cone.json)'
    )
    parser.add_argument(
        '--config', '-c',
        dest='config_alt',
        help='Alternative way to specify config file'
    )

    args = parser.parse_args()

    config_path = args.config_alt if args.config_alt else args.config_file

    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        return 1

    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    config = validate_config(config)

    print("\nRunning cone CPSD analysis...")
    results, cone = run_analysis(config)

    print("\nSaving results...")
    save_results(results, cone, config)

    print("\nGenerating plots...")
    generate_plots(results, cone, config)

    print("\nAnalysis complete!")
    return 0


if __name__ == '__main__':
    exit(main())
