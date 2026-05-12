import pandas as pd
import pyvista as pv
from pathlib import Path

def load_convergence_csv(file_path: Path) -> pd.DataFrame:
    """Loads a convergence CSV file into a Pandas DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing CSV file: {file_path}")
    return pd.read_csv(file_path)

def load_thermal_vtk(file_path: Path) -> pv.DataSet:
    """Loads a VTK file into a PyVista mesh dataset."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing VTK file: {file_path}")
    return pv.read(str(file_path))
