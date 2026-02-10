# Total Pressure Field Computation from Scattered Fields

## Context

This implementation adds the capability to compute total pressure fields from scattered fields stored in an Exodus database. The total field is the sum of incident plane waves and a scattered field from a simulation. This is needed for post-processing acoustic scattering results where the incident field must be added to the computed scattered field.

**Physics**: `P_total = P_incident + P_scattered`

Where the incident field is:
```
P_inc(x) = Σ_j A_j * exp(i * k * d_j · x)
```
- `A_j` = complex amplitude for plane wave j
- `k` = wavenumber = 2πf/c
- `d_j` = unit direction vector for plane wave j
- `x` = position vector

## Input File Formats

### Directions File (functions.txt)
```
Function 1
    type plane_wave_freq
    Direction -0.855707 0.319087 0.407369
    ...
END
Function 2
    type iplane_wave_freq
    Direction -0.855707 0.319087 0.407369
    ...
END
```
- Odd functions (1,3,5...) = real component (`plane_wave_freq`)
- Even functions (2,4,6...) = imaginary component (`iplane_wave_freq`)
- Paired functions share the same direction

### Amplitudes File (loads.txt)
```
sideset 3
 acoustic_vel = -0.984113
 scale = {vscale_a}
 function = 1

sideset 3
 iacoustic_vel = 0.177546
 scale = {vscale_a}
 function = 2
```
- `acoustic_vel` (odd function) = real amplitude
- `iacoustic_vel` (even function) = imaginary amplitude
- Complex amplitude: `A = acoustic_vel + i*iacoustic_vel`

## Implementation Plan

### 1. Create DirectionsParser (`directions_parser.py`)

**Responsibility**: Parse functions.txt to extract plane wave directions.

**Key methods**:
- `parse()` → List of FunctionEntry dataclasses
- `get_directions()` → ndarray (n_pws, 3) of unique directions
- `get_function_to_pw_map()` → Dict mapping function ID to plane wave index

**Pattern**: Each plane wave has two functions (real/imaginary) sharing the same direction. Extract unique directions from odd-numbered functions.

### 2. Create AmplitudesParser (`amplitudes_parser.py`)

**Responsibility**: Parse loads.txt to extract complex amplitudes.

**Key methods**:
- `parse()` → List of LoadEntry dataclasses
- `get_complex_amplitudes()` → ndarray (n_pws,) complex
- `get_sideset_id()` → int

**Pairing logic**: Group by consecutive function pairs (1,2), (3,4), etc. Combine acoustic_vel (real) + i*iacoustic_vel (imaginary).

### 3. Create ExodusNodalInterface (`exodus_nodal_interface.py`)

**Responsibility**: Read/write nodal fields and access nodesets in Exodus databases.

**Key methods**:
- `get_coords()` → ndarray (n_nodes, 3)
- `num_nodes()` → int
- `get_nodeset_ids()` → List[int]
- `get_nodeset_nodes(nodeset_id)` → ndarray (1-based indices)
- `get_nodeset_coords(nodeset_id)` → ndarray (n_set_nodes, 3)
- `get_nodal_variable_names()` → List[str]
- `get_nodal_variable(name, step)` → ndarray (n_nodes,)
- `prepare_nodal_variables(names)` → Register variable names
- `write_nodal_variable(name, values, step)` → Write nodal field

**Reference**: Follow patterns from `exodus_side_interpolator.py` for exodusii workarounds.

### 4. Create TotalPressureField (`total_pressure_field.py`)

**Responsibility**: Compute incident and total pressure fields.

**Key methods**:
- `compute_incident_field()` → ndarray (n_nodes,) complex
- `compute_total_field(scattered_real, scattered_imag)` → ndarray complex

**Implementation** (following `cone_diffuse_field.py` lines 176-182):
```python
def compute_incident_field(self) -> np.ndarray:
    k = 2 * np.pi * self.frequency / self.speed_of_sound
    dot_products = self.coordinates @ self.directions.T  # (n_nodes, n_pws)
    phase = np.exp(1j * k * dot_products)
    return phase @ self.amplitudes  # Sum over plane waves
```

### 5. Create Driver Script (`run_total_field.py`)

**Workflow**:
1. Load and validate JSON configuration
2. Parse directions and amplitudes files
3. Open Exodus database (append mode)
4. Read node coordinates (optionally filtered by nodeset)
5. Determine frequencies to process:
   - If `frequencies` specified as list: use those frequencies
   - If `frequencies` specified as range (min/step/max): generate list
   - If `frequencies` is null/omitted: get all time values from Exodus
6. For each frequency/time step:
   a. Read scattered field (real + imaginary) at that time step
   b. Compute incident field at that frequency
   c. Compute total field = incident + scattered
   d. Write total field (real + imaginary) to same time step
   e. Optionally write incident field

### 6. Configuration Schema (`config_total_field.json`)

The configuration supports multi-frequency computation. Each time step in the Exodus file corresponds to a frequency. The frequencies can be specified as:
- A list of frequencies: `[100.0, 200.0, 500.0]`
- A range with min/step/max: `{"min": 100.0, "step": 100.0, "max": 1000.0}`
- Omitted or null: process all time steps in the Exodus file

**Example with separate source and target files:**
```json
{
    "input": {
        "exodus_file": "data/source.e",
        "scattered_field_real": "scattered_pressure_real",
        "scattered_field_imag": "scattered_pressure_imag",
        "directions_file": "data/functions.txt",
        "amplitudes_file": "data/loads.txt",
        "nodeset_id": null
    },
    "physics": {
        "frequencies": {
            "min": 300.0,
            "step": 100.0,
            "max": 4000.0
        },
        "speed_of_sound": 343.0
    },
    "output": {
        "exodus_file": "data/target.e",
        "total_field_real": "total_pressure_real",
        "total_field_imag": "total_pressure_imag",
        "incident_field_real": null,
        "incident_field_imag": null
    }
}
```

**Example with frequency list:**
```json
{
    "input": { ... },
    "physics": {
        "frequencies": [100.0, 500.0, 1000.0],
        "speed_of_sound": 343.0
    },
    "output": { ... }
}
```

**Example processing all time steps (frequencies omitted):**
```json
{
    "input": { ... },
    "physics": {
        "speed_of_sound": 343.0
    },
    "output": { ... }
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| input.exodus_file | Yes | - | Path to source Exodus database (with scattered field) |
| input.scattered_field_real | Yes | - | Name of real scattered pressure variable |
| input.scattered_field_imag | Yes | - | Name of imaginary scattered pressure variable |
| input.directions_file | Yes | - | Path to functions.txt |
| input.amplitudes_file | Yes | - | Path to loads.txt |
| input.nodeset_id | No | null | Nodeset ID (null = all nodes) |
| physics.frequencies | No | null | List, range (min/step/max), or null for all steps |
| physics.speed_of_sound | No | 343.0 | Speed of sound in m/s |
| output.exodus_file | No | null | Path to target Exodus file (null = write to source) |
| output.total_field_real | No | "total_pressure_real" | Output real field name |
| output.total_field_imag | No | "total_pressure_imag" | Output imaginary field name |
| output.incident_field_real | No | null | If set, write incident real field |
| output.incident_field_imag | No | null | If set, write incident imaginary field |

**Note on source and target files:**

- Source file (input.exodus_file): Contains scattered field from simulation
- Target file (output.exodus_file): Clean geometry file where output is written
- If output.exodus_file is null/omitted, output is written to the source file
- Target file must have the same mesh geometry as source file

**Note on time steps and frequencies:**
- Each time step in Exodus corresponds to a frequency value stored as the time value
- When `frequencies` is specified, the driver finds matching time steps
- When `frequencies` is null/omitted, all time steps are processed
- Output is written to the same time steps as input

## File Organization

```
diffuseFields/
├── directions_parser.py        # NEW
├── amplitudes_parser.py        # NEW
├── exodus_nodal_interface.py   # NEW
├── total_pressure_field.py     # NEW
├── run_total_field.py          # NEW
├── config_total_field.json     # NEW (example)
└── tests/
    ├── test_directions_parser.py      # NEW
    ├── test_amplitudes_parser.py      # NEW
    ├── test_exodus_nodal_interface.py # NEW
    └── test_total_pressure_field.py   # NEW
```

## Critical Files to Reference

- `exodus_side_interpolator.py` - Exodus I/O patterns and workarounds
- `cone_diffuse_field.py` - Incident field computation pattern
- `run_cone_analysis.py` - Driver script and config validation patterns
- `data/functions.txt` - Reference input for directions
- `data/loads.txt` - Reference input for amplitudes

## Proposed Unit Tests

### test_directions_parser.py
1. Parse single function block - verify Direction extraction
2. Parse multiple functions - verify correct pairing of real/imaginary
3. Get unique directions - verify n_pws = n_functions / 2

### test_amplitudes_parser.py
1. Parse load entries - verify amplitude and function extraction
2. Complex amplitude pairing - verify A = real + i*imag
3. Sideset consistency - verify all entries use same sideset

### test_exodus_nodal_interface.py
1. Read nodal coordinates - verify against known mesh
2. Read nodal variable - verify correct values
3. Write and read-back nodal variable - round-trip test
4. Nodeset node extraction - verify subset selection

### test_total_pressure_field.py
1. Single plane wave - verify P_inc = A*exp(ikd·x) analytically
2. Wavenumber computation - verify k = 2πf/c
3. Total field = incident + scattered - verify addition

## Verification

1. Run unit tests: `python -m pytest tests/test_directions_parser.py tests/test_amplitudes_parser.py tests/test_exodus_nodal_interface.py tests/test_total_pressure_field.py -v`
2. Integration test with sample data:
   - Create test exodus file with known scattered field
   - Run `python run_total_field.py config_total_field.json`
   - Verify output fields written to exodus file
3. Validate physics: For a single plane wave, verify P_total = A*exp(ikd·x) + P_scat at sample points
