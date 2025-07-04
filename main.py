import argparse
import pandas as pd
from pipeline.predictor import Predictor
from utils.loader import cargar_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predice prob. de fracaso en pacientes nuevos.")
    parser.add_argument("--input", required=True, help="Archivo JSON con pacientes crudos")
    parser.add_argument("--output", default="predicciones.csv", help="Archivo CSV de salida")
    parser.add_argument("--model", default="models/catboost_model.cbm", help="Ruta al modelo CatBoost entrenado")

    args = parser.parse_args()
    pred = Predictor(model_path=args.model)
    df_raw = cargar_json(args.input)
    probas = pred.predict_proba(df_raw)

    df_out = pd.DataFrame({"_id": df_raw["_id"], "prob_fracaso": probas})
    df_out.to_csv(args.output, index=False)
    print(f" Predicciones guardadas en {args.output}")
