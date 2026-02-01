"""
ExodusII sideset interface for face centroid extraction and variable writing.

Reads an ExodusII database to extract sideset face geometry (area-weighted
centroids, areas) and writes sideset variables (e.g., interpolated pressure).
"""

import numpy as np
from pathlib import Path

from exodusii import ExodusIIFile


class ExodusSideInterpolator:
    """
    Interface to an ExodusII database for sideset face operations.

    Provides extraction of area-weighted face centroids from sidesets and
    writing of sideset variables. Designed for use with pressure field
    interpolation workflows.

    Parameters
    ----------
    filename : str
        Path to an ExodusII (.e, .exo) file.
    mode : str, optional
        File access mode: 'r' for read-only (default), 'a' for append
        (required for writing sideset variables).

    Examples
    --------
    >>> with ExodusSideInterpolator("mesh.e") as db:
    ...     centroids = db.get_sideset_face_centroids(1)
    """

    FACE_NODE_MAP = {
        "HEX8": {
            1: [0, 1, 5, 4], 2: [1, 2, 6, 5], 3: [2, 3, 7, 6],
            4: [0, 4, 7, 3], 5: [0, 3, 2, 1], 6: [4, 5, 6, 7],
        },
        "HEX": {
            1: [0, 1, 5, 4], 2: [1, 2, 6, 5], 3: [2, 3, 7, 6],
            4: [0, 4, 7, 3], 5: [0, 3, 2, 1], 6: [4, 5, 6, 7],
        },
        "TET4": {
            1: [0, 1, 3], 2: [1, 2, 3], 3: [0, 3, 2], 4: [0, 2, 1],
        },
        "TET": {
            1: [0, 1, 3], 2: [1, 2, 3], 3: [0, 3, 2], 4: [0, 2, 1],
        },
        "TETRA10": {
            1: [0, 1, 3], 2: [1, 2, 3], 3: [0, 3, 2], 4: [0, 2, 1],
        },
        "TETRA": {
            1: [0, 1, 3], 2: [1, 2, 3], 3: [0, 3, 2], 4: [0, 2, 1],
        },
        "TRI3": {
            1: [0, 1], 2: [1, 2], 3: [2, 0],
        },
        "QUAD4": {
            1: [0, 1], 2: [1, 2], 3: [2, 3], 4: [3, 0],
        },
    }

    def __init__(self, filename: str, mode: str = 'r'):
        self._filename = filename
        self._mode = mode
        self._validate_file(filename)
        self._exo = ExodusIIFile(filename, mode=mode)
        self._coords = None
        self._elem_block_map = None
        self._block_connectivity = {}
        self._block_elem_type = {}

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

    def get_sideset_ids(self) -> list:
        """Return all sideset IDs in the database."""
        return list(self._exo.get_side_set_ids())

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

    def get_sideset_face_centroids(self, sideset_id: int) -> np.ndarray:
        """
        Compute area-weighted centroids of all faces in a sideset.

        Parameters
        ----------
        sideset_id : int
            The sideset ID.

        Returns
        -------
        centroids : ndarray, shape (n_faces, 3)
        """
        self._validate_sideset_id(sideset_id)
        coords = self.get_coords()
        element_ids = self._exo.get_side_set_elems(sideset_id)
        side_ids = self._exo.get_side_set_sides(sideset_id)

        n_faces = len(element_ids)
        centroids = np.empty((n_faces, 3))

        for i in range(n_faces):
            global_node_ids = self._get_face_global_nodes(
                element_ids[i], side_ids[i]
            )
            face_coords = coords[global_node_ids - 1]
            centroid, _ = self._compute_face_centroid_and_area(face_coords)
            centroids[i] = centroid

        return centroids

    def get_sideset_face_areas(self, sideset_id: int) -> np.ndarray:
        """
        Compute areas of all faces in a sideset.

        Parameters
        ----------
        sideset_id : int
            The sideset ID.

        Returns
        -------
        areas : ndarray, shape (n_faces,)
        """
        self._validate_sideset_id(sideset_id)
        coords = self.get_coords()
        element_ids = self._exo.get_side_set_elems(sideset_id)
        side_ids = self._exo.get_side_set_sides(sideset_id)

        n_faces = len(element_ids)
        areas = np.empty(n_faces)

        for i in range(n_faces):
            global_node_ids = self._get_face_global_nodes(
                element_ids[i], side_ids[i]
            )
            face_coords = coords[global_node_ids - 1]
            _, area = self._compute_face_centroid_and_area(face_coords)
            areas[i] = area

        return areas

    def prepare_sideset_variables(self, variable_names: list) -> None:
        """
        Pre-register sideset variable names and create netCDF storage.

        Must be called before write_sideset_variable() when writing
        multiple variables, since the netCDF dimension for the number
        of sideset variables is fixed at creation time.

        Parameters
        ----------
        variable_names : list of str
            All sideset variable names that will be written.
        """
        if self._mode == 'r':
            raise RuntimeError(
                "Cannot prepare sideset variables in read-only mode. "
                "Open the database with mode='a'."
            )
        from exodusii import exodus_h as ex
        fh = self._exo.fh

        # Read existing variable names (if any)
        existing_names = self._get_sideset_variable_names()

        # Merge with new names, preserving order
        all_names = list(existing_names)
        for name in variable_names:
            if name not in all_names:
                all_names.append(name)

        if len(all_names) == len(existing_names):
            return  # nothing new to register

        num_vars = len(all_names)

        # Create the num_sset_var dimension with the total count
        dim_name = ex.DIM_NUM_SIDE_SET_VAR  # 'num_sset_var'
        if dim_name not in fh.dimensions:
            fh.createDimension(dim_name, num_vars)

        # Create the variable name array
        str_dim = ex.DIM_STR  # 'len_string'
        name_var = ex.VAR_NAME_SIDE_SET_VAR  # 'name_sset_var'
        if name_var not in fh.variables:
            fh.createVariable(name_var, 'S1', (dim_name, str_dim))

        # Fill variable names
        str_len = fh.dimensions[str_dim].size
        for i, nm in enumerate(all_names):
            padded = nm.ljust(str_len)[:str_len]
            fh.variables[name_var][i] = np.array(
                list(padded), dtype='S1'
            )

        # Create data variables for each (var_index, sideset) pair,
        # initialized to zero (netCDF default fill value is 9.97e+36)
        time_dim = ex.DIM_TIME  # 'time_step'
        ss_ids = self._exo.get_side_set_ids()
        for ss_id in ss_ids:
            ss_iid = self._exo.get_side_set_iid(ss_id)
            side_dim = f"num_side_ss{ss_iid}"
            for vi in range(1, num_vars + 1):
                key = ex.VAR_SIDE_SET_VAR(vi, ss_iid)
                if key not in fh.variables:
                    fh.createVariable(
                        key, 'f8', (time_dim, side_dim),
                        fill_value=0.0
                    )

        self._sideset_var_names = all_names

    def write_sideset_variable(
        self,
        sideset_id: int,
        variable_name: str,
        values: np.ndarray,
        step: int = 1,
    ) -> None:
        """
        Write a variable on a sideset.

        For writing multiple variables, call prepare_sideset_variables()
        first with all variable names. For a single variable, this method
        handles preparation automatically.

        Parameters
        ----------
        sideset_id : int
            The sideset ID.
        variable_name : str
            Name of the sideset variable.
        values : ndarray, shape (n_faces,)
            Variable values, one per face in the sideset.
        step : int, optional
            Time step index (1-based). Default is 1.
        """
        if self._mode == 'r':
            raise RuntimeError(
                "Cannot write sideset variable in read-only mode. "
                "Open the database with mode='a'."
            )
        self._validate_sideset_id(sideset_id)

        params = self._exo.get_side_set_params(sideset_id)
        n_sides = params.num_sides
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (n_sides,):
            raise ValueError(
                f"values must have shape ({n_sides},), got {values.shape}"
            )

        # Auto-prepare if not already done
        existing_names = self._get_sideset_variable_names()
        if variable_name not in existing_names:
            self.prepare_sideset_variables([variable_name])
            existing_names = self._get_sideset_variable_names()

        # Ensure time step exists
        from exodusii import exodus_h as ex
        times = self._exo.get_times()
        if len(times) < step:
            self._exo.put_time(step, float(step))

        # Write values
        var_index = existing_names.index(variable_name) + 1
        set_iid = self._exo.get_side_set_iid(sideset_id)
        key = ex.VAR_SIDE_SET_VAR(var_index, set_iid)
        self._exo.fh.variables[key][step - 1, :n_sides] = values

    def _get_sideset_variable_names(self) -> list:
        """Read existing sideset variable names from the netCDF handle."""
        if hasattr(self, '_sideset_var_names'):
            return self._sideset_var_names

        from exodusii import exodus_h as ex
        fh = self._exo.fh
        if ex.VAR_NAME_SIDE_SET_VAR not in fh.variables:
            return []
        raw = fh.variables[ex.VAR_NAME_SIDE_SET_VAR][:]
        return [b"".join(row).decode().strip() for row in raw]

    # ---- Private helpers ----

    @staticmethod
    def _validate_file(filename: str) -> None:
        """Validate that the file exists."""
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"ExodusII file not found: {filename}")

    def _validate_sideset_id(self, sideset_id: int) -> None:
        """Validate that the sideset ID exists in the database."""
        sideset_ids = self.get_sideset_ids()
        if sideset_id not in sideset_ids:
            raise ValueError(
                f"Sideset ID {sideset_id} not found. "
                f"Available sidesets: {sideset_ids}"
            )

    def _build_element_block_map(self) -> None:
        """Build mapping from global element ID to (block_id, local_index)."""
        elem_id_map = self._exo.get_element_id_map()
        block_ids = self._exo.get_element_block_ids()

        self._elem_block_map = {}
        global_offset = 0

        for block_id in block_ids:
            elem_type = self._exo.get_element_block_elem_type(block_id)
            num_elems = self._exo.num_elems_in_blk(block_id)
            self._block_elem_type[block_id] = elem_type.upper().strip()

            for local_idx in range(num_elems):
                global_id = elem_id_map[global_offset + local_idx]
                self._elem_block_map[global_id] = (block_id, local_idx)

            global_offset += num_elems

    def _get_face_global_nodes(
        self, element_id: int, side_id: int
    ) -> np.ndarray:
        """
        Get global node IDs for a face of an element.

        Parameters
        ----------
        element_id : int
            Global element ID (1-based).
        side_id : int
            Side number (1-based).

        Returns
        -------
        node_ids : ndarray
            Global node IDs (1-based) of the face vertices.
        """
        if self._elem_block_map is None:
            self._build_element_block_map()

        block_id, local_idx = self._elem_block_map[element_id]

        if block_id not in self._block_connectivity:
            conn = self._exo.get_element_conn(block_id)
            self._block_connectivity[block_id] = np.array(conn)

        connectivity = self._block_connectivity[block_id]
        elem_nodes = connectivity[local_idx]

        elem_type = self._block_elem_type[block_id]
        local_face_nodes = self._get_face_node_indices(elem_type, side_id)

        return elem_nodes[local_face_nodes]

    @classmethod
    def _get_face_node_indices(
        cls, element_type: str, side_number: int
    ) -> list:
        """
        Look up local node indices for a face.

        Parameters
        ----------
        element_type : str
            Element type string (e.g., 'HEX8', 'TET4').
        side_number : int
            Side number (1-based).

        Returns
        -------
        indices : list of int
            0-based local node indices for the face.
        """
        elem_type = element_type.upper().strip()

        if elem_type not in cls.FACE_NODE_MAP:
            raise ValueError(
                f"Unsupported element type: '{element_type}'. "
                f"Supported types: {list(cls.FACE_NODE_MAP.keys())}"
            )

        face_map = cls.FACE_NODE_MAP[elem_type]

        if side_number not in face_map:
            raise ValueError(
                f"Invalid side number {side_number} for element type "
                f"'{elem_type}'. Valid sides: {list(face_map.keys())}"
            )

        return face_map[side_number]

    @staticmethod
    def _compute_face_centroid_and_area(
        node_coords: np.ndarray,
    ) -> tuple:
        """
        Compute area-weighted centroid of a face via triangle fan decomposition.

        Parameters
        ----------
        node_coords : ndarray, shape (n_nodes, 3)
            Coordinates of the face vertices in order.

        Returns
        -------
        centroid : ndarray, shape (3,)
            Area-weighted centroid of the face.
        area : float
            Total face area.
        """
        n = node_coords.shape[0]
        if n < 3:
            raise ValueError(f"Face must have at least 3 nodes, got {n}")

        v0 = node_coords[0]
        total_area = 0.0
        weighted_centroid = np.zeros(3)

        for i in range(1, n - 1):
            v1 = node_coords[i]
            v2 = node_coords[i + 1]

            tri_centroid = (v0 + v1 + v2) / 3.0
            tri_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

            weighted_centroid += tri_centroid * tri_area
            total_area += tri_area

        if total_area < 1e-30:
            return np.mean(node_coords, axis=0), 0.0

        return weighted_centroid / total_area, total_area
