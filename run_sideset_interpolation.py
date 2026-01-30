"""
Driver script for interpolating pressure fields onto ExodusII sideset faces.

Reads configuration from a JSON file, interpolates source pressure fields
(or eigendata) onto sideset face centroids, and writes the results as
sideset variables in the ExodusII database.

Usage:
    python run_sideset_interpolation.py config_sideset_interpolation.json
    python run_sideset_interpolation.py --config config_sideset_interpolation.json
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np

from pressure_interpolator import PressureFieldInterpolator
from exodus_side_interpolator import ExodusSideInterpolator


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
    - input.pressure_field_path
    - input.input_type: "pressure_field" or "eigendata"
    - input.exodus_file
    - input.sideset_id

    Optional fields with defaults:
    - interpolation.kernel: "thin_plate_spline"
    - interpolation.smoothing: 0.0
    - output.variable_prefix: "pressure"
    - output.time_step: 1
    """
    if 'input' not in config:
        raise ValueError("Configuration must have 'input' section")

    input_cfg = config['input']
    required_input = [
        'source_coordinates_path', 'pressure_field_path',
        'input_type', 'exodus_file', 'sideset_id'
    ]
    for key in required_input:
        if key not in input_cfg:
            raise ValueError(f"input.{key} is required")

    valid_types = ['pressure_field', 'eigendata']
    if input_cfg['input_type'] not in valid_types:
        raise ValueError(
            f"input.input_type must be one of {valid_types}, "
            f"got '{input_cfg['input_type']}'"
        )

    for key in ['source_coordinates_path', 'pressure_field_path', 'exodus_file']:
        if not os.path.exists(input_cfg[key]):
            raise ValueError(f"File not found: {input_cfg[key]}")

    if not isinstance(input_cfg['sideset_id'], int):
        raise ValueError("input.sideset_id must be an integer")

    # Set defaults for interpolation section
    if 'interpolation' not in config:
        config['interpolation'] = {}

    interp_cfg = config['interpolation']
    interp_cfg.setdefault('kernel', 'linear')
    interp_cfg.setdefault('smoothing', 0.0)

    # Set defaults for output section
    if 'output' not in config:
        config['output'] = {}

    output_cfg = config['output']
    output_cfg.setdefault('variable_prefix', 'pressure')
    output_cfg.setdefault('time_step', 1)

    return config


def load_input_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load source coordinates and pressure/eigendata arrays.

    Returns dict with:
    - 'source_coords': (n_source, 3)
    - 'pressure_fields': (n_source, n_fields) or (n_source,)
    - 'eigendata_metadata': dict or None
    """
    input_cfg = config['input']
    input_type = input_cfg['input_type']

    source_coords = np.load(input_cfg['source_coordinates_path'])
    print(f"Loaded source coordinates: {source_coords.shape}")

    if input_type == 'eigendata':
        data = np.load(input_cfg['pressure_field_path'])
        pressure_fields = data['eigenvectors']
        eigendata_metadata = {
            'frequency': float(data['frequency']),
            'eigenvalues': data['eigenvalues'].copy(),
            'variance_explained': data['variance_explained'].copy()
        }
        print(f"Loaded eigendata: {pressure_fields.shape} "
              f"(frequency={eigendata_metadata['frequency']:.1f} Hz, "
              f"{pressure_fields.shape[1]} eigenvectors)")
    else:
        pressure_fields = np.load(input_cfg['pressure_field_path'])
        eigendata_metadata = None
        print(f"Loaded pressure field: {pressure_fields.shape}")

    return {
        'source_coords': source_coords,
        'pressure_fields': pressure_fields,
        'eigendata_metadata': eigendata_metadata
    }


def run_sideset_interpolation(config: Dict[str, Any]) -> None:
    """
    Main workflow: load data, interpolate onto sideset, write to exodus.
    """
    input_cfg = config['input']
    interp_cfg = config['interpolation']
    output_cfg = config['output']
    input_type = input_cfg['input_type']
    prefix = output_cfg['variable_prefix']
    step = output_cfg['time_step']

    # Load source data
    data = load_input_data(config)
    source_coords = data['source_coords']
    pressure_fields = data['pressure_fields']

    # Open exodus file and compute sideset centroids
    exodus_file = input_cfg['exodus_file']
    sideset_id = input_cfg['sideset_id']

    print(f"\nOpening ExodusII file: {exodus_file}")
    with ExodusSideInterpolator(exodus_file, mode='a') as db:
        centroids = db.get_sideset_face_centroids(sideset_id)
        print(f"Sideset {sideset_id}: {centroids.shape[0]} face centroids")

        # Create interpolator
        print(f"Creating RBF interpolator "
              f"(kernel={interp_cfg['kernel']}, "
              f"smoothing={interp_cfg['smoothing']})...")
        interpolator = PressureFieldInterpolator(
            source_coords,
            centroids,
            kernel=interp_cfg['kernel'],
            smoothing=interp_cfg['smoothing']
        )

        # Interpolate
        print(f"Interpolating {pressure_fields.shape} -> "
              f"{centroids.shape[0]} target points...")
        interpolated = interpolator.interpolate(pressure_fields)

        extrap_info = interpolator.get_extrapolation_info()
        print(f"Extrapolated points: {extrap_info['n_extrapolated']} "
              f"({extrap_info['extrapolation_ratio']*100:.1f}%)")

        # Build variable names and pre-register them
        if input_type == 'eigendata':
            # Ensure 2D
            if interpolated.ndim == 1:
                interpolated = interpolated.reshape(-1, 1)

            n_fields = interpolated.shape[1]
            var_names = []
            for i in range(n_fields):
                var_names.append(f"{prefix}_ev{i+1}_real")
                var_names.append(f"{prefix}_ev{i+1}_imag")
        else:
            # Single pressure field
            if interpolated.ndim == 2:
                interpolated = interpolated.squeeze()
            var_names = [f"{prefix}_real", f"{prefix}_imag"]

        db.prepare_sideset_variables(var_names)

        # Write sideset variables
        if input_type == 'eigendata':
            print(f"\nWriting {n_fields} eigenvectors "
                  f"({2 * n_fields} sideset variables)...")

            for i in range(n_fields):
                real_name = f"{prefix}_ev{i+1}_real"
                imag_name = f"{prefix}_ev{i+1}_imag"

                db.write_sideset_variable(
                    sideset_id, real_name,
                    np.real(interpolated[:, i]), step=step
                )
                db.write_sideset_variable(
                    sideset_id, imag_name,
                    np.imag(interpolated[:, i]), step=step
                )
                print(f"  Wrote {real_name}, {imag_name}")
        else:
            real_name = f"{prefix}_real"
            imag_name = f"{prefix}_imag"

            print(f"\nWriting sideset variables: {real_name}, {imag_name}")
            db.write_sideset_variable(
                sideset_id, real_name,
                np.real(interpolated), step=step
            )
            db.write_sideset_variable(
                sideset_id, imag_name,
                np.imag(interpolated), step=step
            )

    print(f"\nSideset variables written to: {exodus_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Interpolate pressure fields onto ExodusII sideset faces'
    )
    parser.add_argument(
        'config',
        nargs='?',
        default='config_sideset_interpolation.json',
        help='Path to configuration JSON file '
             '(default: config_sideset_interpolation.json)'
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
    print("Sideset Pressure Field Interpolation")
    print("=" * 60)

    run_sideset_interpolation(config)

    print("\n" + "=" * 60)
    print("Interpolation complete!")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    exit(main())
