"""
ExodusII nodal interface for reading/writing nodal fields and nodesets.

Provides access to nodal variables and nodesets in ExodusII databases,
designed for use with pressure field computation workflows.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional

from exodusii import ExodusIIFile


class ExodusNodalInterface:
    """
    Interface to an ExodusII database for nodal field operations.

    Provides reading/writing of nodal variables and access to nodesets.
    Designed for use with pressure field workflows.

    Parameters
    ----------
    filename : str
        Path to an ExodusII (.e, .exo) file.
    mode : str, optional
        File access mode: 'r' for read-only (default), 'a' for append
        (required for writing nodal variables).

    Examples
    --------
    >>> with ExodusNodalInterface("mesh.e") as db:
    ...     coords = db.get_coords()
    ...     pressure = db.get_nodal_variable("pressure_real", step=1)
    """

    def __init__(self, filename: str, mode: str = 'r'):
        self._filename = filename
        self._mode = mode
        self._validate_file(filename)
        self._exo = ExodusIIFile(filename, mode=mode)
        self._coords: Optional[np.ndarray] = None
        self._nodal_var_names: Optional[List[str]] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self) -> None:
        """Close the ExodusII database."""
        if self._exo is not None:
            self._exo.close()
            self._exo = None

    # ---- Coordinate access ----

    def get_coords(self) -> np.ndarray:
        """
        Return node coordinates.

        Returns
        -------
        coords : ndarray, shape (n_nodes, 3)
        """
        if self._coords is None:
            fh = self._exo.fh
            if 'coord' in fh.variables:
                # Single array with shape (n_dims, n_nodes)
                coord = np.array(fh.variables['coord'][:])
                self._coords = coord.T
            else:
                # Separate coordx, coordy, coordz arrays
                x = np.array(fh.variables['coordx'][:])
                y = np.array(fh.variables['coordy'][:])
                z = np.array(fh.variables['coordz'][:])
                self._coords = np.column_stack([x, y, z])
        return self._coords

    def num_nodes(self) -> int:
        """Return total number of nodes in mesh."""
        return len(self.get_coords())

    def get_node_num_map(self) -> np.ndarray:
        """
        Get the node number map (internal index to global node ID).

        Returns
        -------
        node_map : ndarray, shape (n_nodes,)
            Global node IDs for each internal node index.
            If no map exists, returns 1-based sequential indices.
        """
        fh = self._exo.fh
        if 'node_num_map' in fh.variables:
            return np.array(fh.variables['node_num_map'][:])
        # Default: identity map (1-based)
        return np.arange(1, self.num_nodes() + 1)

    def build_global_to_internal_map(self) -> dict:
        """
        Build a mapping from global node IDs to internal (0-based) indices.

        Returns
        -------
        mapping : dict
            Dictionary mapping global node ID to internal 0-based index.
        """
        node_map = self.get_node_num_map()
        return {global_id: idx for idx, global_id in enumerate(node_map)}

    # ---- Nodeset operations ----

    def get_nodeset_ids(self) -> List[int]:
        """Return all nodeset IDs in the database."""
        return list(self._exo.get_node_set_ids())

    def get_nodeset_nodes(self, nodeset_id: int) -> np.ndarray:
        """
        Get node indices in a nodeset.

        Parameters
        ----------
        nodeset_id : int
            The nodeset ID.

        Returns
        -------
        node_indices : ndarray
            1-based node indices in the nodeset.
        """
        self._validate_nodeset_id(nodeset_id)
        return np.array(self._exo.get_node_set_nodes(nodeset_id))

    def get_nodeset_coords(self, nodeset_id: int) -> np.ndarray:
        """
        Get coordinates of nodes in a nodeset.

        Parameters
        ----------
        nodeset_id : int
            The nodeset ID.

        Returns
        -------
        coords : ndarray, shape (n_nodes_in_set, 3)
        """
        node_indices = self.get_nodeset_nodes(nodeset_id)
        all_coords = self.get_coords()
        # Convert to 0-based indexing
        return all_coords[node_indices - 1]

    # ---- Time step operations ----

    def get_times(self) -> np.ndarray:
        """
        Get all time values in the database.

        Returns
        -------
        times : ndarray
            Time values for each time step.
        """
        return np.array(self._exo.get_times())

    def num_time_steps(self) -> int:
        """Return total number of time steps."""
        return len(self.get_times())

    # ---- Nodal variable read operations ----

    def get_nodal_variable_names(self) -> List[str]:
        """
        Return list of nodal variable names.

        Returns
        -------
        names : List[str]
            List of nodal variable names in the database.
        """
        if self._nodal_var_names is not None:
            return self._nodal_var_names

        from exodusii import exodus_h as ex
        fh = self._exo.fh

        # Check for nodal variable names
        name_var = 'name_nod_var'
        if name_var not in fh.variables:
            return []

        raw = fh.variables[name_var][:]
        self._nodal_var_names = [
            b"".join(row).decode().rstrip('\x00').strip() for row in raw
        ]
        return self._nodal_var_names

    def get_nodal_variable(
        self,
        variable_name: str,
        step: int = 1
    ) -> np.ndarray:
        """
        Read nodal variable values at a time step.

        Parameters
        ----------
        variable_name : str
            Name of the nodal variable.
        step : int, optional
            Time step index (1-based). Default is 1.

        Returns
        -------
        values : ndarray, shape (n_nodes,)
            Variable values at all nodes.
        """
        names = self.get_nodal_variable_names()
        if variable_name not in names:
            raise ValueError(
                f"Nodal variable '{variable_name}' not found. "
                f"Available variables: {names}"
            )

        var_index = names.index(variable_name) + 1
        fh = self._exo.fh

        # Variable key: vals_nod_var{index} or vals_nod_var{index}_{step}
        # The format varies by exodus version
        key = f'vals_nod_var{var_index}'
        if key not in fh.variables:
            raise RuntimeError(
                f"Cannot find nodal variable data for '{variable_name}'"
            )

        # Shape is (n_time_steps, n_nodes)
        return np.array(fh.variables[key][step - 1, :])

    def get_nodal_variable_on_nodeset(
        self,
        variable_name: str,
        nodeset_id: int,
        step: int = 1
    ) -> np.ndarray:
        """
        Read nodal variable restricted to nodeset nodes.

        Parameters
        ----------
        variable_name : str
            Name of the nodal variable.
        nodeset_id : int
            The nodeset ID.
        step : int, optional
            Time step index (1-based). Default is 1.

        Returns
        -------
        values : ndarray, shape (n_nodes_in_set,)
            Variable values at nodeset nodes.
        """
        all_values = self.get_nodal_variable(variable_name, step)
        node_indices = self.get_nodeset_nodes(nodeset_id)
        return all_values[node_indices - 1]

    # ---- Nodal variable write operations ----

    def prepare_nodal_variables(self, variable_names: List[str]) -> None:
        """
        Pre-register nodal variable names and create netCDF storage.

        Must be called before write_nodal_variable() when writing
        multiple variables, since the netCDF dimension for the number
        of nodal variables is fixed at creation time.

        Parameters
        ----------
        variable_names : List[str]
            All nodal variable names that will be written.
        """
        if self._mode == 'r':
            raise RuntimeError(
                "Cannot prepare nodal variables in read-only mode. "
                "Open the database with mode='a'."
            )

        from exodusii import exodus_h as ex
        fh = self._exo.fh

        # Read existing variable names (if any)
        existing_names = self.get_nodal_variable_names()

        # Merge with new names, preserving order
        all_names = list(existing_names)
        for name in variable_names:
            if name not in all_names:
                all_names.append(name)

        if len(all_names) == len(existing_names):
            return  # nothing new to register

        num_vars = len(all_names)

        # Create or validate the num_nod_var dimension
        dim_name = 'num_nod_var'
        if dim_name in fh.dimensions:
            existing_size = fh.dimensions[dim_name].size
            if existing_size < num_vars:
                raise RuntimeError(
                    f"Cannot add {num_vars} nodal variables: the exodus "
                    f"file already has a fixed dimension '{dim_name}' of "
                    f"size {existing_size}. Start from a fresh copy of the "
                    f"exodus file, or ensure all variable names are "
                    f"registered in a single call to prepare_nodal_variables()."
                )
        else:
            fh.createDimension(dim_name, num_vars)

        # Create the variable name array
        str_dim = 'len_string'
        name_var = 'name_nod_var'
        if name_var not in fh.variables:
            fh.createVariable(name_var, 'S1', (dim_name, str_dim))

        # Fill variable names
        str_len = fh.dimensions[str_dim].size
        for i, nm in enumerate(all_names):
            padded = nm.ljust(str_len)[:str_len]
            fh.variables[name_var][i] = np.array(
                list(padded), dtype='S1'
            )

        # Create data variables for each nodal variable
        time_dim = 'time_step'
        num_nodes = self.num_nodes()

        # Ensure num_nodes dimension exists
        node_dim = 'num_nodes'
        if node_dim not in fh.dimensions:
            fh.createDimension(node_dim, num_nodes)

        for vi in range(1, num_vars + 1):
            key = f'vals_nod_var{vi}'
            if key not in fh.variables:
                fh.createVariable(
                    key, 'f8', (time_dim, node_dim),
                    fill_value=0.0
                )

        self._nodal_var_names = all_names

    def write_nodal_variable(
        self,
        variable_name: str,
        values: np.ndarray,
        step: int = 1
    ) -> None:
        """
        Write nodal variable values at a time step.

        For writing multiple variables, call prepare_nodal_variables()
        first with all variable names. For a single variable, this method
        handles preparation automatically.

        Parameters
        ----------
        variable_name : str
            Name of the nodal variable.
        values : ndarray, shape (n_nodes,)
            Variable values at all nodes.
        step : int, optional
            Time step index (1-based). Default is 1.
        """
        if self._mode == 'r':
            raise RuntimeError(
                "Cannot write nodal variable in read-only mode. "
                "Open the database with mode='a'."
            )

        n_nodes = self.num_nodes()
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (n_nodes,):
            raise ValueError(
                f"values must have shape ({n_nodes},), got {values.shape}"
            )

        # Auto-prepare if not already done
        existing_names = self.get_nodal_variable_names()
        if variable_name not in existing_names:
            self.prepare_nodal_variables([variable_name])
            existing_names = self.get_nodal_variable_names()

        # Ensure time step exists
        times = self._exo.get_times()
        if len(times) < step:
            self._exo.put_time(step, float(step))

        # Write values
        var_index = existing_names.index(variable_name) + 1
        fh = self._exo.fh
        key = f'vals_nod_var{var_index}'
        fh.variables[key][step - 1, :] = values

    def write_nodal_variable_on_nodeset(
        self,
        variable_name: str,
        values: np.ndarray,
        nodeset_id: int,
        step: int = 1
    ) -> None:
        """
        Write nodal variable only for nodes in a nodeset.

        Other nodes will have zero values.

        Parameters
        ----------
        variable_name : str
            Name of the nodal variable.
        values : ndarray, shape (n_nodes_in_set,)
            Variable values at nodeset nodes.
        nodeset_id : int
            The nodeset ID.
        step : int, optional
            Time step index (1-based). Default is 1.
        """
        self._validate_nodeset_id(nodeset_id)

        node_indices = self.get_nodeset_nodes(nodeset_id)
        values = np.asarray(values, dtype=np.float64)

        if values.shape != (len(node_indices),):
            raise ValueError(
                f"values must have shape ({len(node_indices)},), "
                f"got {values.shape}"
            )

        # Create full array and fill subset
        full_values = np.zeros(self.num_nodes(), dtype=np.float64)
        full_values[node_indices - 1] = values

        self.write_nodal_variable(variable_name, full_values, step)

    # ---- Private helpers ----

    @staticmethod
    def _validate_file(filename: str) -> None:
        """Validate that the file exists."""
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"ExodusII file not found: {filename}")

    def _validate_nodeset_id(self, nodeset_id: int) -> None:
        """Validate that the nodeset ID exists in the database."""
        nodeset_ids = self.get_nodeset_ids()
        if nodeset_id not in nodeset_ids:
            raise ValueError(
                f"Nodeset ID {nodeset_id} not found. "
                f"Available nodesets: {nodeset_ids}"
            )
