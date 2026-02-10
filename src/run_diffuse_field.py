"""
Driver script for DiffuseField3D simulations.

Reads configuration from a JSON file and runs the diffuse field simulation,
computing correlations and generating plots.

Usage:
    python run_diffuse_field.py config.json
    python run_diffuse_field.py --config config.json
    python run_diffuse_field.py  # Uses default 'config.json' in current directory
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from diffuse_field import DiffuseField3D


def load_config(config_path: str) -> dict:
    """
    Load configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def validate_config(config: dict) -> dict:
    """
    Validate and fill in default values for configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Validated configuration with defaults filled in
    """
    # Default values
    defaults = {
        'frequency': {
            'f_min': 100,
            'f_max': 1000,
            'n_freq': 5
        },
        'physics': {
            'c': 343.0
        },
        'simulation': {
            'n_waves': 200,
            'n_realizations': 500
        },
        'output': {
            'plot_plane': 'Z',
            'save_figures': True,
            'figure_format': 'png',
            'output_dir': 'results'
        },
        'eigenvalues': {
            'compute': False,
            'n_components': 20,
            'n_oversamples': 10,
            'n_power_iter': 2,
            'save_eigenvectors': False,
            'plot_eigenvectors': False,
            'n_vectors_to_plot': 4,
            'plot_component': 'real'
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

    # Validate frequency parameters
    freq = config['frequency']
    if freq['f_min'] <= 0:
        raise ValueError("f_min must be positive")
    if freq['f_max'] <= 0:
        raise ValueError("f_max must be positive")
    if freq['f_min'] > freq['f_max']:
        raise ValueError("f_min must be less than or equal to f_max")
    if freq['n_freq'] < 1:
        raise ValueError("n_freq must be at least 1")
    if freq['f_min'] == freq['f_max'] and freq['n_freq'] > 1:
        raise ValueError("n_freq must be 1 when f_min == f_max")

    # Validate physics parameters
    if config['physics']['c'] <= 0:
        raise ValueError("Speed of sound must be positive")

    # Validate simulation parameters
    sim = config['simulation']
    if sim['n_waves'] < 1:
        raise ValueError("n_waves must be at least 1")
    if sim['n_realizations'] < 1:
        raise ValueError("n_realizations must be at least 1")

    # Validate output parameters
    output = config['output']
    if output['plot_plane'].upper() not in ['X', 'Y', 'Z']:
        raise ValueError("plot_plane must be 'X', 'Y', or 'Z'")
    if output['figure_format'] not in ['png', 'pdf', 'svg', 'eps']:
        raise ValueError("figure_format must be 'png', 'pdf', 'svg', or 'eps'")

    # Validate eigenvalue parameters
    eig = config['eigenvalues']
    if eig['n_components'] < 1:
        raise ValueError("n_components must be at least 1")
    if eig['n_oversamples'] < 0:
        raise ValueError("n_oversamples must be non-negative")
    if eig['n_power_iter'] < 0:
        raise ValueError("n_power_iter must be non-negative")
    if eig['n_vectors_to_plot'] < 1:
        raise ValueError("n_vectors_to_plot must be at least 1")
    valid_components = ['magnitude', 'real', 'imag', 'phase']
    if eig['plot_component'] not in valid_components:
        raise ValueError(f"plot_component must be one of {valid_components}")

    return config


def run_simulation(config: dict) -> dict:
    """
    Run the diffuse field simulation.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    dict
        Results dictionary containing NMSE values and other outputs
    """
    # Extract parameters
    freq = config['frequency']
    physics = config['physics']
    sim = config['simulation']

    # Create diffuse field object
    print("=" * 60)
    print("Creating DiffuseField3D object...")
    print("=" * 60)

    df = DiffuseField3D(
        f_min=freq['f_min'],
        f_max=freq['f_max'],
        n_freq=freq['n_freq'],
        c=physics['c'],
        n_waves=sim['n_waves'],
        n_realizations=sim['n_realizations']
    )

    print(df)
    print()

    # Compute correlations and NMSE for all frequencies
    print("=" * 60)
    print("Computing correlations...")
    print("=" * 60)

    results = {
        'frequencies': df.frequencies.tolist(),
        'nmse': [],
        'config': config
    }

    for i, f in enumerate(df.frequencies):
        print(f"\nFrequency {i+1}/{df.n_freq}: {f:.1f} Hz")
        nmse = df.compute_normalized_mse(i)
        results['nmse'].append(nmse)
        print(f"  NMSE = {nmse:.4e}")

    return results, df


def compute_eigenvalues(df: DiffuseField3D, config: dict, results: dict) -> tuple:
    """
    Compute eigenvalues of the covariance matrix for each frequency.

    Parameters
    ----------
    df : DiffuseField3D
        The diffuse field object
    config : dict
        Configuration dictionary
    results : dict
        Results dictionary to update with eigenvalue data

    Returns
    -------
    tuple
        (updated results dict, dict of eigenvectors keyed by frequency index)
    """
    eig_config = config['eigenvalues']
    n_components = eig_config['n_components']
    n_oversamples = eig_config['n_oversamples']
    n_power_iter = eig_config['n_power_iter']
    save_eigenvectors = eig_config['save_eigenvectors']
    output_dir = config['output']['output_dir']

    print()
    print("=" * 60)
    print("Computing covariance eigenvalues...")
    print("=" * 60)

    results['eigenvalues'] = []
    all_eigenvectors = {}

    for i, f in enumerate(df.frequencies):
        print(f"\nFrequency {i+1}/{df.n_freq}: {f:.1f} Hz")
        print(f"  Computing {n_components} eigenvalues...")

        eigenvalues, eigenvectors = df.compute_covariance_eigenvalues(
            freq_idx=i,
            n_components=n_components,
            n_oversamples=n_oversamples,
            n_power_iter=n_power_iter
        )

        # Store eigenvalues in results (for JSON)
        results['eigenvalues'].append(eigenvalues.tolist())

        # Store eigenvectors for plotting
        all_eigenvectors[i] = {
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'frequency': f
        }

        # Print summary
        print(f"  Top 5 eigenvalues: {eigenvalues[:5]}")
        print(f"  Eigenvalue ratio (1st/last): {eigenvalues[0]/eigenvalues[-1]:.2e}")

        # Save eigenvectors to file if requested
        if save_eigenvectors:
            os.makedirs(output_dir, exist_ok=True)
            filename = f"eigenvectors_f{f:.0f}Hz.npz"
            filepath = os.path.join(output_dir, filename)
            np.savez_compressed(
                filepath,
                eigenvalues=eigenvalues,
                eigenvectors=eigenvectors,
                frequency=f,
                grid_shape=df.grid_shape,
                grid_points=df.grid_points
            )
            print(f"  Saved eigenvectors: {filepath}")

    return results, all_eigenvectors


def generate_plots(df: DiffuseField3D, config: dict, results: dict,
                   eigenvector_data: dict = None):
    """
    Generate and optionally save plots.

    Parameters
    ----------
    df : DiffuseField3D
        The diffuse field object
    config : dict
        Configuration dictionary
    results : dict
        Results dictionary
    eigenvector_data : dict, optional
        Dictionary of eigenvector data keyed by frequency index
    """
    output = config['output']
    plane = output['plot_plane'].upper()
    save_figs = output['save_figures']
    fmt = output['figure_format']
    output_dir = output['output_dir']

    # Create output directory if needed
    if save_figs:
        os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 60)
    print("Generating plots...")
    print("=" * 60)

    # Plot for each frequency
    for i, f in enumerate(df.frequencies):
        print(f"\nPlotting frequency {i+1}/{df.n_freq}: {f:.1f} Hz")

        # Correlation comparison plot
        fig = df.plot_correlation_comparison(i, plane=plane)

        if save_figs:
            filename = f"correlation_comparison_f{f:.0f}Hz.{fmt}"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")
            plt.close(fig)

        # Radial profile plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        df.plot_radial_profile(i, ax=ax2)

        if save_figs:
            filename = f"radial_profile_f{f:.0f}Hz.{fmt}"
            filepath = os.path.join(output_dir, filename)
            fig2.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")
            plt.close(fig2)

    # NMSE vs frequency plot (if multiple frequencies)
    if df.n_freq > 1:
        print("\nPlotting NMSE vs frequency...")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        ax3.semilogy(results['frequencies'], results['nmse'], 'bo-', linewidth=2, markersize=8)
        ax3.set_xlabel('Frequency [Hz]')
        ax3.set_ylabel('Normalized MSE')
        ax3.set_title('NMSE vs Frequency')
        ax3.grid(True, alpha=0.3, which='both')

        if save_figs:
            filename = f"nmse_vs_frequency.{fmt}"
            filepath = os.path.join(output_dir, filename)
            fig3.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved: {filepath}")
            plt.close(fig3)

    # Cumulative variance plots (if eigenvalues were computed)
    if 'eigenvalues' in results and results['eigenvalues']:
        for i, f in enumerate(df.frequencies):
            print(f"\nPlotting cumulative variance for {f:.1f} Hz...")
            eigenvalues = np.array(results['eigenvalues'][i])

            fig4, ax4 = plt.subplots(figsize=(10, 6))
            indices = np.arange(1, len(eigenvalues) + 1)

            # Compute cumulative variance captured
            cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

            ax4.plot(indices, cumulative_variance, 'bo-', linewidth=2, markersize=8)
            ax4.set_xlabel('Number of Modes')
            ax4.set_ylabel('Cumulative Variance Captured')
            ax4.set_title(f'Cumulative Variance Captured (f = {f:.1f} Hz)')
            ax4.set_ylim([0, 1.05])
            ax4.grid(True, alpha=0.3)

            # Add horizontal reference lines at key thresholds
            ax4.axhline(y=0.90, color='g', linestyle='--', alpha=0.7, label='90%')
            ax4.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
            ax4.axhline(y=0.99, color='r', linestyle='--', alpha=0.7, label='99%')
            ax4.legend(loc='lower right')

            if save_figs:
                filename = f"cumulative_variance_f{f:.0f}Hz.{fmt}"
                filepath = os.path.join(output_dir, filename)
                fig4.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"  Saved: {filepath}")
                plt.close(fig4)

    # Eigenvector visualization (if requested and data available)
    eig_config = config['eigenvalues']
    if (eig_config.get('plot_eigenvectors', False) and
            eigenvector_data is not None):
        n_vectors = eig_config['n_vectors_to_plot']
        component = eig_config['plot_component']

        for i, f in enumerate(df.frequencies):
            if i not in eigenvector_data:
                continue

            print(f"\nPlotting {n_vectors} eigenvector(s) for {f:.1f} Hz...")

            eig_data = eigenvector_data[i]
            fig5 = df.plot_eigenvectors(
                eigenvectors=eig_data['eigenvectors'],
                eigenvalues=eig_data['eigenvalues'],
                n_vectors=n_vectors,
                plane=plane,
                component=component
            )

            if save_figs:
                filename = f"eigenvectors_f{f:.0f}Hz.{fmt}"
                filepath = os.path.join(output_dir, filename)
                fig5.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"  Saved: {filepath}")
                plt.close(fig5)

    if not save_figs:
        plt.show()


def save_results(results: dict, config: dict):
    """
    Save results to a JSON file.

    Parameters
    ----------
    results : dict
        Results dictionary
    config : dict
        Configuration dictionary
    """
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, 'results.json')

    # Prepare results for JSON serialization
    output_results = {
        'frequencies': results['frequencies'],
        'nmse': results['nmse'],
        'config': results['config']
    }

    # Include eigenvalues if computed
    if 'eigenvalues' in results:
        output_results['eigenvalues'] = results['eigenvalues']

    with open(filepath, 'w') as f:
        json.dump(output_results, f, indent=2)

    print(f"\nResults saved to: {filepath}")


def main():
    """Main entry point for the driver script."""
    parser = argparse.ArgumentParser(
        description='Run DiffuseField3D simulation from JSON configuration.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example config.json:
{
    "frequency": {
        "f_min": 500,
        "f_max": 1000,
        "n_freq": 3
    },
    "physics": {
        "c": 343.0
    },
    "simulation": {
        "n_waves": 200,
        "n_realizations": 500
    },
    "output": {
        "plot_plane": "Z",
        "save_figures": true,
        "figure_format": "png",
        "output_dir": "results"
    },
    "eigenvalues": {
        "compute": true,
        "n_components": 20,
        "n_oversamples": 10,
        "n_power_iter": 2,
        "save_eigenvectors": true,
        "plot_eigenvectors": true,
        "n_vectors_to_plot": 4,
        "plot_component": "real"
    }
}
        """
    )
    parser.add_argument(
        'config',
        nargs='?',
        default='config.json',
        help='Path to JSON configuration file (default: config.json)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to file'
    )

    args = parser.parse_args()

    # Load and validate configuration
    print(f"Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
        config = validate_config(config)
    except FileNotFoundError:
        print(f"Error: Configuration file '{args.config}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}")
        sys.exit(1)

    print("Configuration loaded successfully.\n")

    # Run simulation
    results, df = run_simulation(config)

    # Compute eigenvalues if requested
    eigenvector_data = None
    if config['eigenvalues']['compute']:
        results, eigenvector_data = compute_eigenvalues(df, config, results)

    # Save results
    if not args.no_save:
        save_results(results, config)

    # Generate plots
    if not args.no_plots:
        generate_plots(df, config, results, eigenvector_data)

    print()
    print("=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
