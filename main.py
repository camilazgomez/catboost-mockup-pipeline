import argparse
import pandas as pd
from pipeline.predictor import Predictor
from utils.loader import cargar_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predice prob. de fracaso en pacientes nuevos.")
    parser.add_argument("--input", required=True, help="Archivo JSON con pacientes crudos")
    parser.add_argument("--output", default="predicciones.csv", help="Archivo CSV de salida")
    parser.add_argument("--bundle", default="models/catboost_pipeline.pkl", help="Ruta al modelo + preprocesador")

    args = parser.parse_args()
    pred = Predictor(bundle_path=args.bundle)
    df_raw = cargar_json(args.input)
    probas = pred.predict_proba(df_raw)

    df_out = pd.DataFrame({"_id": df_raw["_id"], "prob_fracaso": probas})
    df_out["pred_clase"] = (df_out["prob_fracaso"] >= 0.5).astype(int)
    df_out.to_csv(args.output, index=False)
    print(f"Predicciones guardadas en {args.output}")