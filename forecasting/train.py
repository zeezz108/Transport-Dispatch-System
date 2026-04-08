"""
forecasting/train.py
Скрипт обучения и сохранения модели HybridForecaster.

Использование:
  python forecasting/train.py \
      --train data/train_team_track.parquet \
      --output models/hybrid_forecaster.pkl
"""
import argparse
import logging
import pandas as pd
from pathlib import Path
from model import HybridForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Обучение HybridForecaster")
    parser.add_argument("--train", required=True, help="Путь к train parquet")
    parser.add_argument("--output", default="models/hybrid_forecaster.pkl", help="Путь для сохранения модели")
    parser.add_argument("--forecast-points", type=int, default=10)
    args = parser.parse_args()

    logger.info(f"Загрузка данных: {args.train}")
    train_df = pd.read_parquet(args.train)
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
    logger.info(f"Размер: {len(train_df):,} строк, {train_df['route_id'].nunique()} маршрутов")

    forecaster = HybridForecaster(forecast_points=args.forecast_points, use_status=True)
    forecaster.train_with_baseline_comparison(train_df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(args.output)
    print("\n" + "=" * 60)
    print("СОЗДАНИЕ ПРОГНОЗА НА ТЕСТОВЫХ ДАННЫХ")
    print("=" * 60)

    # Загружаем тестовые данные
    test_df = pd.read_parquet("data/test_team_track.parquet")
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

    # Делаем прогноз
    predictions = forecaster.predict_hybrid(test_df, train_df)

    print("\n✅ Готово! CSV файл с прогнозом создан!")
    logger.info("✅ Готово!")


if __name__ == "__main__":
    main()