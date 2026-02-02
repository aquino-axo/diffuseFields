"""
Driver script for pressure field interpolation between cone surfaces.

Reads configuration from a JSON file and interpolates pressure fields
from source to target cone coordinates.

Usage:
    python run_interpolation.py config_interpolation.json
    python run_interpolation.py --config config_interpolation.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from pressure_interpolator import PressureFieldInterpolator
from cone_visualizer import ConeVisualizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and fill in default values for configuration.

    Required fields:
    - input.source_coordinates_path
    - input.target_coordinates_path
    - input.pressure_fields_path
    - input.input_type: "pressure_field" or "eigendata"

    Optional fields with defaults:
    - interpolation.fill_value: {"real": NaN, "imag": NaN}
    - output.output_dir: "results_interpolation"
    - output.save_fields: true
    """
    # Validate input section
    if 'input' not in config:
        raise ValueError("Configuration must have 'input' section")

    input_cfg = config['input']
    required_input = ['source_coordinates_path', 'target_coordinates_path',
                      'pressure_fields_path', 'input_type']
    for key in required_input:
        if key not in input_cfg:
            raise ValueError(f"input.{key} is required")

    # Validate input_type
    valid_types = ['pressure_field', 'eigendata']
    if input_cfg['input_type'] not in valid_types:
        raise ValueError(
            f"input.input_type must be one of {valid_types}, "
            f"got '{input_cfg['input_type']}'"
        )

    # Validate file paths exist
    for key in ['source_coordinates_path', 'target_coordinates_path',
                'pressure_fields_path']:
        if not os.path.exists(input_cfg[key]):
            raise ValueError(f"File not found: {input_cfg[key]}")

    # Set defaults for output section
    if 'output' not in config:
        config['output'] = {}

    output_cfg = config['output']
    output_cfg.setdefault('output_dir', 'results_interpolation')
    output_cfg.setdefault('save_fields', True)

    # Set defaults for visualization section
    if 'visualization' not in config:
        config['visualization'] = {}

    vis_cfg = config['visualization']
    vis_cfg.setdefault('enabled', False)
    vis_cfg.setdefault('n_fields', 3)
    vis_cfg.setdefault('component', 'magnitude')
    vis_cfg.setdefault('save_figures', True)
    vis_cfg.setdefault('figure_format', 'png')

    return config




def load_input_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load input arrays based on input_type.

    Returns dict with:
    - 'source_coords': (n_source, 3)
    - 'target_coords': (n_target, 3)
    - 'pressure_fields': (n_source, n_fields) or (n_source,)
    - 'eigendata_metadata': dict or None (for eigendata input type)
    """
    input_cfg = config['input']
    input_type = input_cfg['input_type']

    source_coords = np.load(input_cfg['source_coordinates_path'])
    target_coords = np.load(input_cfg['target_coordinates_path'])

    print(f"Loaded source coordinates: {source_coords.shape}")
    print(f"Loaded target coordinates: {target_coords.shape}")

    if input_type == 'eigendata':
        # Load eigenvector data from .npz file
        data = np.load(input_cfg['pressure_fields_path'])
        pressure_fields = data['eigenvectors']

        # Handle both per-frequency ('frequency' scalar) and
        # all-frequencies ('frequencies' array) eigendata formats
        if 'frequency' in data:
            freq_info = float(data['frequency'])
            freq_label = f"{freq_info:.1f} Hz"
        else:
            freqs = data['frequencies']
            freq_info = freqs.tolist()
            freq_label = f"{freqs[0]:.0f}-{freqs[-1]:.0f} Hz ({len(freqs)} frequencies)"

        eigendata_metadata = {
            'frequency': freq_info,
            'eigenvalues': data['eigenvalues'].copy(),
            'variance_explained': data['variance_explained'].copy()
        }
        print(f"Loaded eigendata: {pressure_fields.shape} "
              f"(frequency={freq_label}, "
              f"{pressure_fields.shape[1]} eigenvectors)")
    else:
        # Load regular pressure field from .npy file
        pressure_fields = np.load(input_cfg['pressure_fields_path'])
        eigendata_metadata = None
        print(f"Loaded pressure fields: {pressure_fields.shape}")

    return {
        'source_coords': source_coords,
        'target_coords': target_coords,
        'pressure_fields': pressure_fields,
        'eigendata_metadata': eigendata_metadata
    }


def run_interpolation(
    data: Dict[str, np.ndarray],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run the interpolation workflow.

    Returns dict with:
    - 'interpolated_fields': interpolated pressure data
    - 'extrapolation_info': information about extrapolation (always 0 for RBF)
    """
    print(f"\nCreating interpolator...")
    interpolator = PressureFieldInterpolator(
        data['source_coords'],
        data['target_coords']
    )

    print(f"Interpolating {data['pressure_fields'].shape} pressure fields...")
    interpolated = interpolator.interpolate(data['pressure_fields'])

    extrap_info = interpolator.get_extrapolation_info()
    print(f"Extrapolated points: {extrap_info['n_extrapolated']} "
          f"({extrap_info['extrapolation_ratio']*100:.1f}%)")

    return {
        'interpolated_fields': interpolated,
        'extrapolation_info': extrap_info
    }


def save_results(
    results: Dict[str, Any],
    data: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """Save interpolated fields and metadata to output directory."""
    output_cfg = config['output']
    input_type = config['input']['input_type']
    output_dir = Path(output_cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_cfg.get('save_fields', True):
        # Derive output filename from input filename with _interpolated suffix
        input_path = Path(config['input']['pressure_fields_path'])
        output_filename = f"{input_path.stem}_interpolated{input_path.suffix}"

        if input_type == 'eigendata':
            # Save as .npz preserving eigendata format
            eigendata_metadata = data['eigendata_metadata']
            fields_path = output_dir / output_filename
            freq_info = eigendata_metadata['frequency']

            # Preserve the original format: scalar → 'frequency', list → 'frequencies'
            save_kwargs = dict(
                eigenvalues=eigendata_metadata['eigenvalues'],
                eigenvectors=results['interpolated_fields'],
                variance_explained=eigendata_metadata['variance_explained']
            )
            if isinstance(freq_info, list):
                save_kwargs['frequencies'] = np.array(freq_info)
            else:
                save_kwargs['frequency'] = freq_info

            np.savez(fields_path, **save_kwargs)
            print(f"Saved interpolated eigendata to: {fields_path}")
        else:
            # Save as regular .npy
            fields_path = output_dir / output_filename
            np.save(fields_path, results['interpolated_fields'])
            print(f"Saved interpolated fields to: {fields_path}")

    # Save metadata (exclude interpolation section which is unused with RBF)
    config_to_save = {
        'input': config['input'],
        'output': config['output'],
        'visualization': config.get('visualization', {})
    }
    metadata = {
        'input_type': input_type,
        'source_shape': list(results['interpolated_fields'].shape),
        'n_extrapolated': results['extrapolation_info']['n_extrapolated'],
        'extrapolation_ratio': results['extrapolation_info']['extrapolation_ratio'],
        'config': config_to_save
    }
    if input_type == 'eigendata':
        metadata['frequency'] = data['eigendata_metadata']['frequency']

    metadata_path = output_dir / 'interpolation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to: {metadata_path}")


def generate_comparison_plots(
    results: Dict[str, Any],
    data: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """
    Generate side-by-side comparison plots of source and interpolated fields.

    Creates a figure with source fields on the left and corresponding
    interpolated fields on the right for visual verification.
    """
    vis_cfg = config['visualization']
    if not vis_cfg.get('enabled', False):
        return

    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get source and interpolated fields
    source_fields = data['pressure_fields']
    interp_fields = results['interpolated_fields']

    # Ensure 2D shape
    if source_fields.ndim == 1:
        source_fields = source_fields.reshape(-1, 1)
    if interp_fields.ndim == 1:
        interp_fields = interp_fields.reshape(-1, 1)

    n_total = source_fields.shape[1]
    n_to_plot = min(vis_cfg.get('n_fields', 3), n_total)
    component = vis_cfg.get('component', 'magnitude')

    print(f"\nGenerating comparison plots for {n_to_plot} fields...")

    # Get cone geometry from config (optional)
    cone_geometry = vis_cfg.get('cone_geometry')

    # Create visualizers for source and target
    source_vis = ConeVisualizer(data['source_coords'], cone_geometry)
    target_vis = ConeVisualizer(data['target_coords'], cone_geometry)

    # Create comparison figure
    fig = plt.figure(figsize=(12, 5 * n_to_plot))

    for i in range(n_to_plot):
        # Source field (left column)
        ax_source = fig.add_subplot(n_to_plot, 2, 2*i + 1, projection='3d')
        source_vis.plot_pressure_field(
            source_fields[:, i],
            component=component,
            ax=ax_source,
            title=f'Source - Field {i+1}'
        )

        # Interpolated field (right column)
        ax_interp = fig.add_subplot(n_to_plot, 2, 2*i + 2, projection='3d')
        target_vis.plot_pressure_field(
            interp_fields[:, i],
            component=component,
            ax=ax_interp,
            title=f'Interpolated - Field {i+1}'
        )

    plt.tight_layout()

    # Add frequency info to title if eigendata
    if data.get('eigendata_metadata'):
        freq = data['eigendata_metadata']['frequency']
        fig.suptitle(f'Interpolation Comparison (f = {freq:.1f} Hz)', y=1.02)

    # Save figure
    if vis_cfg.get('save_figures', True):
        fmt = vis_cfg.get('figure_format', 'png')
        input_stem = Path(config['input']['pressure_fields_path']).stem
        fig_path = output_dir / f'{input_stem}_interpolated.{fmt}'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {fig_path}")

    plt.close(fig)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Interpolate pressure fields between cone surfaces'
    )
    parser.add_argument(
        'config',
        nargs='?',
        default='config_interpolation.json',
        help='Path to configuration JSON file (default: config_interpolation.json)'
    )
    parser.add_argument(
        '--config', '-c',
        dest='config_flag',
        help='Alternative way to specify config file'
    )

    args = parser.parse_args()
    config_path = args.config_flag if args.config_flag else args.config

    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    config = validate_config(config)

    print("\n" + "=" * 60)
    print("Pressure Field Interpolation")
    print("=" * 60)

    data = load_input_data(config)
    results = run_interpolation(data, config)
    save_results(results, data, config)
    generate_comparison_plots(results, data, config)

    print("\n" + "=" * 60)
    print("Interpolation complete!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
