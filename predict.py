"""
predict.py
Загружает обученную модель и сохраняет прогноз в CSV.

Использование:
  python predict.py
  python predict.py --model models/hybrid_forecaster.pkl --test data/test_team_track.parquet --output submission.csv
"""
import argparse
import logging
import pandas as pd
import joblib
import sys
sys.path.insert(0, "forecasting")
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Прогноз отгрузок")
    parser.add_argument("--model", default="models/hybrid_forecaster.pkl", help="Путь к обученной модели")
    parser.add_argument("--train", default="data/train_team_track.parquet", help="Путь к train данным")
    parser.add_argument("--test",  default="data/test_team_track.parquet",  help="Путь к test данным")
    parser.add_argument("--output", default="submission_ensemble.csv", help="Путь для сохранения CSV")
    args = parser.parse_args()

    # Загрузка модели
    logger.info(f"Загрузка модели: {args.model}")
    forecaster = joblib.load(args.model)

    # Загрузка данных
    logger.info(f"Загрузка train: {args.train}")
    train_df = pd.read_parquet(args.train)
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

    logger.info(f"Загрузка test: {args.test}")
    test_df = pd.read_parquet(args.test)
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    logger.info(f"Train: {len(train_df):,} строк | Test: {len(test_df):,} строк")

    # Прогноз
    logger.info("Запуск прогнозирования...")
    predictions = forecaster.predict_hybrid(test_df, train_df)

    # Сохранение
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    logger.info(f"✅ CSV сохранён: {args.output}")
    logger.info(f"   Строк: {len(predictions):,}")
    logger.info(f"   Min:    {predictions['y_pred'].min():.2f}")
    logger.info(f"   Max:    {predictions['y_pred'].max():.2f}")
    logger.info(f"   Mean:   {predictions['y_pred'].mean():.2f}")
    logger.info(f"   Нулевых: {(predictions['y_pred'] == 0).sum()}")

    predictions.to_csv('submission.csv', index=False)
if __name__ == "__main__":
    main()
