"""
Driver script for computing total pressure fields from scattered fields.

Reads configuration from a JSON file, parses direction and amplitude files,
reads scattered field from ExodusII database, computes total field, and
writes results back to the database.

The total field is P_total = P_incident + P_scattered, where:
- P_incident is the superposition of incident plane waves
- P_scattered is read from the Exodus database

Usage:
    python run_total_field.py config_total_field.json
    python run_total_field.py --config config_total_field.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from directions_parser import DirectionsParser
from amplitudes_parser import AmplitudesParser
from exodus_nodal_interface import ExodusNodalInterface
from total_pressure_field import TotalPressureField


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def parse_frequencies(freq_config: Any) -> Optional[List[float]]:
    """
    Parse frequency configuration into a list of frequencies.

    Supports three modes:
    1. List of frequencies: [100.0, 200.0, 500.0]
    2. Range specification: {"min": 100.0, "step": 100.0, "max": 500.0}
    3. None/omitted: process all time steps in Exodus file

    Parameters
    ----------
    freq_config : list, dict, or None
        Frequency specification.

    Returns
    -------
    list or None
        List of frequencies in Hz, or None to process all time steps.
    """
    if freq_config is None:
        return None  # Will use all time steps

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

        # Generate frequency list
        frequencies = np.arange(f_min, f_max + f_step * 0.001, f_step)
        frequencies = frequencies[frequencies <= f_max + 1e-10]
        return frequencies.tolist()

    raise ValueError(
        f"physics.frequencies must be a list, dict with min/step/max, or null, "
        f"got {type(freq_config).__name__}"
    )


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and fill in default values for configuration."""
    defaults = {
        'input': {
            'exodus_file': None,
            'scattered_field_real': None,
            'scattered_field_imag': None,
            'directions_file': None,
            'amplitudes_file': None,
            'nodeset_id': None
        },
        'physics': {
            'frequencies': None,  # None = use all time steps
            'speed_of_sound': 343.0
        },
        'output': {
            'total_field_real': 'total_pressure_real',
            'total_field_imag': 'total_pressure_imag',
            'incident_field_real': None,
            'incident_field_imag': None
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
    required_fields = [
        'exodus_file',
        'scattered_field_real',
        'scattered_field_imag',
        'directions_file',
        'amplitudes_file'
    ]
    for field in required_fields:
        if inp[field] is None:
            raise ValueError(f"input.{field} is required")

    # Validate file paths exist
    for key in ['exodus_file', 'directions_file', 'amplitudes_file']:
        path = inp[key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"input.{key}: {path} not found")

    # Parse frequencies (optional - None means all time steps)
    physics = config['physics']
    physics['frequencies'] = parse_frequencies(physics.get('frequencies'))

    if physics['speed_of_sound'] <= 0:
        raise ValueError("physics.speed_of_sound must be positive")

    return config


def get_frequency_step_mapping(
    exodus_times: np.ndarray,
    frequencies: Optional[List[float]]
) -> List[tuple]:
    """
    Map frequencies to time steps.

    Parameters
    ----------
    exodus_times : ndarray
        Time values from Exodus file.
    frequencies : list or None
        Requested frequencies, or None for all time steps.

    Returns
    -------
    List of (step, frequency) tuples
        1-based time step index and corresponding frequency.
    """
    if frequencies is None:
        # Use all time steps
        return [(i + 1, t) for i, t in enumerate(exodus_times)]

    # Match requested frequencies to time steps
    mapping = []
    for freq in frequencies:
        # Find closest time step
        idx = np.argmin(np.abs(exodus_times - freq))
        step = idx + 1  # 1-based
        actual_freq = exodus_times[idx]

        # Warn if not exact match
        if abs(actual_freq - freq) > 1e-6:
            print(f"  Warning: Requested frequency {freq} Hz mapped to "
                  f"time step {step} with value {actual_freq} Hz")

        mapping.append((step, actual_freq))

    return mapping


def run_total_field_computation(config: Dict[str, Any]) -> None:
    """
    Main workflow for total field computation.

    Parameters
    ----------
    config : dict
        Validated configuration dictionary.
    """
    inp = config['input']
    physics = config['physics']
    output = config['output']

    print("Parsing input files...")

    # Parse directions and amplitudes
    dir_parser = DirectionsParser(inp['directions_file'])
    amp_parser = AmplitudesParser(inp['amplitudes_file'])

    directions = dir_parser.get_directions()
    amplitudes = amp_parser.get_complex_amplitudes()

    print(f"  Loaded {len(directions)} plane wave directions")
    print(f"  Loaded {len(amplitudes)} complex amplitudes")

    if len(directions) != len(amplitudes):
        raise ValueError(
            f"Mismatch: {len(directions)} directions vs "
            f"{len(amplitudes)} amplitudes"
        )

    # Open Exodus database
    print(f"\nOpening Exodus file: {inp['exodus_file']}")

    with ExodusNodalInterface(inp['exodus_file'], mode='a') as db:
        # Get coordinates
        if inp['nodeset_id'] is not None:
            print(f"  Using nodeset {inp['nodeset_id']}")
            coords = db.get_nodeset_coords(inp['nodeset_id'])
            node_indices = db.get_nodeset_nodes(inp['nodeset_id'])
        else:
            print("  Using all nodes")
            coords = db.get_coords()
            node_indices = None

        print(f"  Number of nodes: {len(coords)}")

        # Get time steps
        exodus_times = db.get_times()
        print(f"  Number of time steps: {len(exodus_times)}")

        # Determine which frequencies/steps to process
        freq_step_map = get_frequency_step_mapping(
            exodus_times,
            physics['frequencies']
        )

        print(f"\nProcessing {len(freq_step_map)} frequencies...")

        # Prepare output variables
        output_vars = [output['total_field_real'], output['total_field_imag']]
        if output['incident_field_real']:
            output_vars.append(output['incident_field_real'])
        if output['incident_field_imag']:
            output_vars.append(output['incident_field_imag'])

        db.prepare_nodal_variables(output_vars)

        # Create field calculator (frequency will be updated per step)
        field = TotalPressureField(
            coordinates=coords,
            directions=directions,
            amplitudes=amplitudes,
            frequency=1.0,  # placeholder
            speed_of_sound=physics['speed_of_sound']
        )

        # Process each frequency
        for step, freq in freq_step_map:
            print(f"  Step {step}: frequency = {freq:.2f} Hz")

            # Update frequency
            field.set_frequency(freq)

            # Read scattered field
            if node_indices is not None:
                scat_real = db.get_nodal_variable_on_nodeset(
                    inp['scattered_field_real'],
                    inp['nodeset_id'],
                    step
                )
                scat_imag = db.get_nodal_variable_on_nodeset(
                    inp['scattered_field_imag'],
                    inp['nodeset_id'],
                    step
                )
            else:
                scat_real = db.get_nodal_variable(
                    inp['scattered_field_real'],
                    step
                )
                scat_imag = db.get_nodal_variable(
                    inp['scattered_field_imag'],
                    step
                )

            # Compute incident field
            P_inc = field.compute_incident_field()

            # Compute total field
            P_total = field.compute_total_field(scat_real, scat_imag)

            # Write results
            if node_indices is not None:
                db.write_nodal_variable_on_nodeset(
                    output['total_field_real'],
                    P_total.real,
                    inp['nodeset_id'],
                    step
                )
                db.write_nodal_variable_on_nodeset(
                    output['total_field_imag'],
                    P_total.imag,
                    inp['nodeset_id'],
                    step
                )
                if output['incident_field_real']:
                    db.write_nodal_variable_on_nodeset(
                        output['incident_field_real'],
                        P_inc.real,
                        inp['nodeset_id'],
                        step
                    )
                if output['incident_field_imag']:
                    db.write_nodal_variable_on_nodeset(
                        output['incident_field_imag'],
                        P_inc.imag,
                        inp['nodeset_id'],
                        step
                    )
            else:
                db.write_nodal_variable(
                    output['total_field_real'],
                    P_total.real,
                    step
                )
                db.write_nodal_variable(
                    output['total_field_imag'],
                    P_total.imag,
                    step
                )
                if output['incident_field_real']:
                    db.write_nodal_variable(
                        output['incident_field_real'],
                        P_inc.real,
                        step
                    )
                if output['incident_field_imag']:
                    db.write_nodal_variable(
                        output['incident_field_imag'],
                        P_inc.imag,
                        step
                    )

    print("\nTotal field computation complete!")
    print(f"Results written to: {inp['exodus_file']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compute total pressure field from scattered field'
    )
    parser.add_argument(
        'config_file',
        nargs='?',
        default='config_total_field.json',
        help='Path to JSON configuration file (default: config_total_field.json)'
    )
    parser.add_argument(
        '--config', '-c',
        dest='config_alt',
        help='Alternative way to specify config file'
    )

    args = parser.parse_args()
    config_path = args.config_alt if args.config_alt else args.config_file

    print(f"Loading configuration from: {config_path}")

    try:
        config = load_config(config_path)
        config = validate_config(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid configuration: {e}")
        sys.exit(1)

    print("Configuration loaded successfully.\n")

    try:
        run_total_field_computation(config)
    except Exception as e:
        print(f"Error during computation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
