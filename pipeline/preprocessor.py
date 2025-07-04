import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

CAT_COLS = ["imc_ord"]                  
BIN_COLS = [
    "antecedenteQx_bin",
    "antecedentesAbusoAlcohol_bin",
    "antecedentesHabitoTabaquico_bin",
]
NUM_COLS = [
    "apache2Ingreso_ord",
    "funcionalidad_ord",
    "fragilidad_ord",
    "AntecedentesConsumoDrogas_ord",
    "dias_hospital_hasta_intub",
    "dias_uci_hasta_intub",
]
ALL_COLS = CAT_COLS + BIN_COLS + NUM_COLS

def _first_value(lst):
    """Extrae .value de la lista [{'value': X, ...}] o NaN."""
    if isinstance(lst, list) and lst and isinstance(lst[0], dict):
        return lst[0].get("value", np.nan)
    return np.nan

def categorizar_apache_numerico(puntaje):
    if pd.isna(puntaje):
        return 0
    for lim, cat in zip([4, 9, 14, 19, 24, 29, 34], range(1, 8)):
        if puntaje <= lim:
            return cat
    return 8

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["_id"] = df["_id"]

    out["apache2Ingreso_ord"] = (
        pd.to_numeric(df["apache2Ingreso"], errors="coerce")
        .apply(categorizar_apache_numerico)
        .astype(float)
    )
    out["funcionalidad_ord"] = df["funcionalidad"].apply(_first_value).astype(float)
    out["fragilidad_ord"] = df["fragilidad"].apply(_first_value).astype(float)
    out["AntecedentesConsumoDrogas_ord"] = (
        df["AntecedentesConsumoDrogas"].apply(_first_value).astype(float)
    )

    # binarias
    out["antecedenteQx_bin"] = df["antecedenteQx"].apply(lambda x: int(bool(x)))
    out["antecedentesAbusoAlcohol_bin"] = df["antecedentesAbusoAlcohol"].apply(
        lambda x: 0 if _first_value(x) == 2 else 1
    )
    out["antecedentesHabitoTabaquico_bin"] = df["antecedentesHabitoTabaquico"].apply(
        lambda x: 0 if _first_value(x) == 2 else 1
    )

    # IMC → ordinal categórica (0,1,2) convertida a string
    imc_vals = pd.to_numeric(df["imc"], errors="coerce")
    out["imc_ord"] = (
        pd.cut(imc_vals, [-np.inf, 18.5, 25, np.inf], labels=[0, 1, 2])
        .astype("int")
        .astype(str)
    )

    # días hasta intubación
    to_dt = lambda s: pd.to_datetime(s, dayfirst=True, errors="coerce")
    out["dias_hospital_hasta_intub"] = (
        to_dt(df["fechaIntubacion"]) - to_dt(df["fechaIngresoHospital"])
    ).dt.days.astype(float)
    out["dias_uci_hasta_intub"] = (
        to_dt(df["fechaIntubacion"]) - to_dt(df["fechaIngresoAUci"])
    ).dt.days.astype(float)

    return out

class TabularPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.medians_ = X[NUM_COLS].median()
        return self

    def transform(self, X):
        X = X.copy()
        X[NUM_COLS] = X[NUM_COLS].fillna(self.medians_)

        # tipos finales
        X[CAT_COLS] = X[CAT_COLS].astype(str)
        X[BIN_COLS] = X[BIN_COLS].astype(int)
        X[NUM_COLS] = X[NUM_COLS].astype(float)

        return X[ALL_COLS]
