import sys
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
from pipeline.preprocessor import TabularPreprocessor
from pipeline.preprocessor import prepare_features, ALL_COLS, CAT_COLS, BIN_COLS

class Predictor:
    def __init__(self, bundle_path):
        sys.modules["__main__"].TabularPreprocessor = TabularPreprocessor
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.preproc = bundle["pp"]

    def predict_proba(self, df_raw: pd.DataFrame) -> list:
        df_feat = prepare_features(df_raw)
        X = self.preproc.transform(df_feat)
        cat_features = [X.columns.get_loc(c) for c in CAT_COLS + BIN_COLS]
        pool = Pool(X, cat_features=cat_features)
        return self.model.predict_proba(pool)[:, 1]

    

