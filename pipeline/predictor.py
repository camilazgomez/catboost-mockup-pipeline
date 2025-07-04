import pandas as pd
from catboost import CatBoostClassifier, Pool
from pipeline.preprocessor import prepare_features, TabularPreprocessor, ALL_COLS, CAT_COLS, BIN_COLS

class Predictor:
    def __init__(self, model_path):
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self.preproc = TabularPreprocessor()

    def predict_proba(self, df_raw: pd.DataFrame) -> list:
        df_feat = prepare_features(df_raw)   
        self.preproc.fit(df_feat)
        X = self.preproc.transform(df_feat)
        cat_features = [X.columns.get_loc(c) for c in CAT_COLS + BIN_COLS]
        pool = Pool(X, cat_features=cat_features)

        return self.model.predict_proba(pool)[:, 1]

    

