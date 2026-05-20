"""
Simplified data reader for HDF5 files in raw directory.
"""

import base64
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
from labrad import types as T
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PUBLIC_RAW_DATA_DIR = Path(
    "/data/public/logical/repetition_code_exp_data-main/data/raw"
)


def default_raw_data_dir() -> Path:
    """Raw HDF5 directory: ``REPETITION_CODE_RAW_DIR``, then public dataset, then repo ``data/raw``."""
    env = os.environ.get("REPETITION_CODE_RAW_DIR")
    if env:
        return Path(env)
    if PUBLIC_RAW_DATA_DIR.is_dir():
        return PUBLIC_RAW_DATA_DIR
    return RAW_DATA_DIR

DATA_URL_PREFIX = "data:application/labrad;base64,"

_encodings = [
    ("%", "%p"),  # this one MUST be first for encode/decode to work properly
    ("/", "%f"),
    ("\\", "%b"),
    (":", "%c"),
    ("*", "%a"),
    ("?", "%q"),
    ('"', "%r"),
    ("<", "%l"),
    (">", "%g"),
    ("|", "%v"),
]


def filename_decode(name):
    """Decode a string that has been encoded using filename_encode"""
    for char, code in _encodings[1:] + _encodings[0:1]:
        name = name.replace(code, char)
    return name


def labrad_urldecode(data_url):
    if data_url.startswith(DATA_URL_PREFIX):
        # decode parameter data from dataurl
        all_bytes = base64.urlsafe_b64decode(data_url[len(DATA_URL_PREFIX) :])
        t, data_bytes = T.unflatten(all_bytes, "sy")
        data = T.unflatten(data_bytes, t)
        return data
    else:
        raise ValueError(
            "Trying to labrad_urldecode data that doesn't start with prefix: {}".format(
                DATA_URL_PREFIX
            )
        )


def load_hdf5(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load data as LocalDataset from .hdf5 files."""
    with h5py.File(filepath, "r", swmr=True) as hdf5_file:
        datavault = hdf5_file["DataVault"]
        data = np.array(
            [datavault["f" + str(idx)] for idx in range(len(datavault.dtype))]
        ).T
        attrs = dict(datavault.attrs)
        if "parameters" in hdf5_file:  # parameters from dataset
            keys = hdf5_file["parameters"]["keys"]
            values = hdf5_file["parameters"]["values"]
            parameters_from_dataset = [
                (k.decode("utf-8"), labrad_urldecode(v.decode("utf-8")))
                for k, v in zip(keys, values)
            ]
        else:
            parameters_from_dataset = []
    name = attrs.get("Title", "")
    # ------ get indeps/deps/paras/comments -------
    indeps_dict, deps_dict = {}, {}
    parameters = []
    for key, value in attrs.items():
        if key.startswith("Dependent"):
            idx, raw_key = key[len("Dependent") :].split(".", maxsplit=1)
            deps_dict[idx] = deps_dict.get(idx, {})
            deps_dict[idx][raw_key] = value
        elif key.startswith("Independent"):
            idx, raw_key = key[len("Independent") :].split(".", maxsplit=1)
            indeps_dict[idx] = indeps_dict.get(idx, {})
            indeps_dict[idx][raw_key] = value
        elif key.startswith("Param."):
            parameters.append((key[len("Param.") :], labrad_urldecode(value)))
    parameters = parameters + parameters_from_dataset

    return {
        "data": data,
        "inds": [
            {
                "name": indeps_dict[str(idx)]["label"],
                "unit": indeps_dict[str(idx)]["unit"],
            }
            for idx in range(len(indeps_dict))
        ],
        "deps": [
            {
                "name": deps_dict[str(idx)]["label"],
                "legend": deps_dict[str(idx)]["legend"],
                "unit": deps_dict[str(idx)]["unit"],
            }
            for idx in range(len(deps_dict))
        ],
        "parameters": dict(parameters),
        "filename": name,
    }


class GridData:
    """
    Handle grid data efficiently.
    """

    def __init__(self, data: NDArray, shape: List[int]):
        self.shape = list(shape)
        self.dim = len(shape)
        self._data = np.reshape(data, self.shape + [-1])
        self.Is = [
            self._data[(0,) * i + (slice(None),) + (0,) * (self.dim - i - 1) + (i,)]
            for i in range(self.dim)
        ]
        sorted_idx = [np.argsort(_I) for _I in self.Is]
        self.Is = [_I[_sorted_idx] for _sorted_idx, _I in zip(sorted_idx, self.Is)]
        for _idx, _sorted_idx in enumerate(sorted_idx):
            self._data = self._data[(slice(None),) * _idx + (_sorted_idx, ...)]

    def get_matrix(self, dependent=-1):
        return self._data[
            ..., np.asarray(dependent) % (self._data.shape[-1] - self.dim) + self.dim
        ]


class DataLab:
    """
    Data processing class for local HDF5 files.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir is not None else default_raw_data_dir()
        self._clear_data()

    def _clear_data(self):
        self.data: Optional[NDArray] = None
        self.inds: List[Dict] = []
        self.deps: List[Dict] = []
        self.parameters: Dict = {}
        self.dim: int = 0
        self.grid_data: Optional[GridData] = None
        self.dataset_num: Optional[int] = None
        self.dataset_name: Optional[str] = None

    def _list_files(self) -> List[Path]:
        """List all HDF5 files in data directory."""
        if not self.data_dir.exists():
            return []
        return list(self.data_dir.glob("*.hdf5"))

    def _find_file_by_number(self, number: int) -> Path:
        """Find HDF5 file by dataset number in filename."""
        files = self._list_files()
        if not files:
            raise FileNotFoundError(f"No HDF5 files found in {self.data_dir}")

        pattern = re.compile(rf"^{number:05d}\D|^0*{number}\D")

        for filepath in files:
            if pattern.match(filepath.stem):
                return filepath

        for filepath in files:
            stem = filepath.stem
            num_part = ""
            for char in stem:
                if char.isdigit():
                    num_part += char
                else:
                    break

            if num_part and int(num_part) == number:
                return filepath

        raise FileNotFoundError(
            f"No file found with dataset number {number} in {self.data_dir}"
        )

    def load_dataset(self, dataset: int, noisy: bool = True):
        """
        Load dataset by integer number (from filename).

        Parameters
        ----------
        dataset : int
            Dataset number in filename (e.g., 1 for 00001-xxx.hdf5).
        noisy : bool, default True
            Print loading information.
        """
        filepath = self._find_file_by_number(dataset)
        self._load_from_path(filepath, dataset, noisy)

    def _load_from_path(self, filepath: Path, dataset_num: int, noisy: bool):
        """Internal method to load data from path and set all attributes."""
        self._clear_data()

        result = load_hdf5(filepath)
        self.data = result["data"]
        self.inds = result["inds"]
        self.deps = result["deps"]
        self.parameters = result["parameters"]
        self.dim = len(self.inds)

        self.dataset_num = dataset_num
        self.dataset_name = result["filename"]

        if "shape" in self.parameters:
            self.grid_data = GridData(self.data, self.parameters["shape"])

        if noisy:
            print("#" * 40)
            print(dataset_num)
            print(self.dataset_name)
            print("#" * 40)

    def _name2idx(self, name: str, var_list: List[Dict]) -> int:
        """Find index by name in variable list."""
        for i, var in enumerate(var_list):
            if var["name"] == name:
                return i
        return -1

    def get_matrix(self, dep_index: int = 0) -> NDArray:
        """
        Get 2D matrix for grid data.

        Parameters
        ----------
        dep_index : int
            Dependent variable index.

        Returns
        -------
        NDArray
            2D matrix.
        """
        if self.grid_data is None:
            raise ValueError("Grid data not available")
        return self.grid_data.get_matrix(dep_index)

    def get_data(
        self,
        ind_index: Optional[int] = None,
        dep_index: Optional[int] = None,
        return_xy: bool = True,
    ):
        """
        Get data interface.

        Parameters
        ----------
        ind_index : int, optional
            Independent variable index.
        dep_index : int, optional
            Dependent variable index.
        default : float, default np.nan
            Default value for missing data.
        return_xy : bool, default True
            If True, return independent variables and matrix for dependent data.

        Returns
        -------
        NDArray or tuple
            Column data, or (independent_vars, matrix) for grid data.
        """
        if self.data is None:
            raise ValueError("No data loaded")

        # Only one parameter can be specified
        params_specified = sum(p is not None for p in [ind_index, dep_index])
        if params_specified != 1:
            raise ValueError(
                "Must specify exactly one of: data_name, ind_index, or dep_index"
            )

        if ind_index is not None:
            return self.data[:, ind_index]

        if dep_index is not None:
            if self.dim == 1:
                return self.data[:, dep_index + self.dim]
            else:
                if self.grid_data is None:
                    raise ValueError("Grid data not available")
                mat = self.grid_data.get_matrix(dep_index)
                if return_xy:
                    return (*self.grid_data.Is, mat)
                return mat

    def __repr__(self):
        _repr = "DataLab"
        if self.data is not None:
            _repr += f"\nfile: {self.dataset_name}"
        return _repr
