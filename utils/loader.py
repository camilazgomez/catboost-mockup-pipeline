import pandas as pd
from pathlib import Path
from typing import Union

# Carga un archivo .json y lo convierte en un DataFrame de pandas.

def cargar_json(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise ValueError(f"Se esperaba archivo .json, pero se recibió: {path.suffix}")
    return pd.read_json(path)