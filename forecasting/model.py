"""
forecasting/model.py
HybridForecaster — ансамблевая прогнозная модель объёмов отгрузок.

Модель обучена на данных train_team_track.parquet.
Предсказывает target_2h (объём отгрузок за 2-часовой период)
на 10 шагов вперёд по каждому route_id.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
import warnings
import time
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class HybridForecaster:
    """
    Гибридный прогнозист — финальная версия.

    Архитектура:
    - LightGBM Model 1 (консервативная, вес 0.6)
    - LightGBM Model 2 (агрессивная, вес 0.4)
    - Ансамбль: 0.6 × m1 + 0.4 × m2
    - Fallback: скользящее среднее за 24ч (если ансамбль не бьёт baseline)

    Для каждого из 10 шагов прогноза обучается отдельная пара моделей.
    """

    def __init__(self, forecast_points=10, use_status=True):
        self.forecast_points = forecast_points
        self.use_status = use_status
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.has_status_in_train = False

    # ── Feature Engineering ───────────────────────────────────────────────────

    def create_enhanced_features(self, df, is_train=True):
        """Создание расширенных признаков"""
        df = df.copy()

        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
            df['hour'] = timestamps.astype('datetime64[h]').astype(int) % 24
            df['day_of_week'] = timestamps.astype('datetime64[D]').view('int64') % 7
            df['day'] = timestamps.astype('datetime64[D]').astype(int) % 31
            df['month'] = timestamps.astype('datetime64[M]').astype(int) % 12 + 1
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.int8)

            hour_rad = 2 * np.pi * df['hour'] / 24
            df['hour_sin'] = np.sin(hour_rad)
            df['hour_cos'] = np.cos(hour_rad)

            dow_rad = 2 * np.pi * df['day_of_week'] / 7
            df['dow_sin'] = np.sin(dow_rad)
            df['dow_cos'] = np.cos(dow_rad)

            df['is_payday'] = (((df['day'] >= 5) & (df['day'] <= 10)) |
                               ((df['day'] >= 20) & (df['day'] <= 25))).astype(np.int8)
            df['is_month_start'] = (df['day'] <= 5).astype(np.int8)
            df['is_month_end'] = (df['day'] >= 25).astype(np.int8)
            df['is_weekend_start'] = (df['day_of_week'] == 5).astype(np.int8)
            df['is_weekend_end'] = (df['day_of_week'] == 6).astype(np.int8)

        if is_train and self.use_status:
            status_cols = [f'status_{i}' for i in range(1, 9)]
            existing_status = [col for col in status_cols if col in df.columns]
            if existing_status:
                self.has_status_in_train = True
                status_data = df[existing_status].values
                df['total_in_process'] = status_data.sum(axis=1)
                weights = np.array([1, 2, 3, 4, 5, 6, 7, 8])[:len(existing_status)]
                df['weighted_status'] = (status_data @ weights) / (df['total_in_process'] + 1)
                df['has_high_status'] = (status_data[:, 4:].sum(axis=1) > 0).astype(np.int8)
            else:
                df['total_in_process'] = df['weighted_status'] = df['has_high_status'] = 0
        else:
            df['total_in_process'] = df['weighted_status'] = df['has_high_status'] = 0

        return df

    def create_lag_features_fast(self, df, target_col='target_2h'):
        """Создание лаговых и скользящих признаков"""
        df = df.sort_values(['route_id', 'timestamp']).reset_index(drop=True)
        grouped = df.groupby('route_id')[target_col]

        for lag in [1, 2, 3, 4, 6, 8, 12, 24, 48]:
            df[f'lag_{lag}'] = grouped.shift(lag)

        for window in [4, 8, 12, 24, 48]:
            df[f'rolling_mean_{window}'] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean())
            df[f'rolling_std_{window}'] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0))

        df['trend_7d'] = (df['rolling_mean_8'] - df['rolling_mean_24']).fillna(0)
        df['ratio_to_mean'] = df[target_col] / (df['rolling_mean_24'] + 1)
        return df

    def get_enhanced_features(self):
        features = [
            'hour', 'day_of_week', 'day', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_payday', 'is_month_start', 'is_month_end',
            'is_weekend_start', 'is_weekend_end',
            'office_from_id', 'route_id',
        ]
        if self.has_status_in_train:
            features.extend(['total_in_process', 'weighted_status', 'has_high_status'])
        for lag in [1, 2, 3, 4, 6, 8, 12, 24, 48]:
            features.append(f'lag_{lag}')
        for window in [4, 8, 12, 24, 48]:
            features += [f'rolling_mean_{window}', f'rolling_std_{window}']
        features.extend(['trend_7d', 'ratio_to_mean'])
        return features

    # ── Обучение ──────────────────────────────────────────────────────────────

    def prepare_hybrid_data(self, train_df):
        logger.info("Подготовка данных...")
        train_df = self.create_enhanced_features(train_df, is_train=True)
        train_df = self.create_lag_features_fast(train_df)

        route_group = train_df.groupby('route_id', sort=False)['target_2h']
        for step in range(1, self.forecast_points + 1):
            train_df[f'target_step_{step}'] = route_group.shift(-step)

        target_cols = [f'target_step_{step}' for step in range(1, self.forecast_points + 1)]
        train_df = train_df.dropna(subset=target_cols)
        return train_df

    def train_with_baseline_comparison(self, train_df):
        logger.info("Начало обучения...")
        prepared_df = self.prepare_hybrid_data(train_df)
        self.feature_cols = self.get_enhanced_features()

        for step in range(1, self.forecast_points + 1):
            target = f'target_step_{step}'
            X = prepared_df[self.feature_cols].copy()
            y = prepared_df[target].copy()

            mask = X.notna().all(axis=1)
            X, y = X[mask].fillna(0), y[mask]

            baseline_pred = prepared_df.loc[mask].groupby('route_id')[target].transform(
                lambda x: x.shift(1).rolling(24, min_periods=1).mean()
            ).fillna(y.mean())

            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])

            model1 = lgb.LGBMRegressor(
                n_estimators=120, learning_rate=0.05, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
            )
            model2 = lgb.LGBMRegressor(
                n_estimators=150, learning_rate=0.08, num_leaves=63,
                min_child_samples=10, subsample=0.9, colsample_bytree=0.9,
                reg_alpha=0.05, reg_lambda=0.05, random_state=123, verbose=-1,
            )

            tscv = TimeSeriesSplit(n_splits=3)
            ensemble_maes, baseline_maes = [], []

            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                bl_val = baseline_pred.iloc[val_idx]

                model1.fit(X_tr, y_tr)
                model2.fit(X_tr, y_tr)
                pred_ens = 0.6 * model1.predict(X_val) + 0.4 * model2.predict(X_val)

                ensemble_maes.append(np.mean(np.abs(y_val - pred_ens)))
                baseline_maes.append(np.mean(np.abs(y_val - bl_val)))

            if np.mean(ensemble_maes) < np.mean(baseline_maes):
                model1.fit(X, y)
                model2.fit(X, y)
                self.models[step] = {
                    'type': 'ensemble', 'model1': model1, 'model2': model2,
                    'weight1': 0.6, 'weight2': 0.4, 'mae': np.mean(ensemble_maes),
                }
                logger.info(f"Step {step}: ensemble (MAE={np.mean(ensemble_maes):.2f})")
            else:
                self.models[step] = {'type': 'baseline', 'mae': np.mean(baseline_maes)}
                logger.info(f"Step {step}: baseline (MAE={np.mean(baseline_maes):.2f})")

        return self.models

    # ── Прогнозирование ───────────────────────────────────────────────────────

    def predict_hybrid(self, test_df, train_df):
        logger.info("Запуск прогнозирования...")
        last_train = train_df.groupby('route_id').tail(48).copy()
        last_train = self.create_enhanced_features(last_train, is_train=True)
        last_train = self.create_lag_features_fast(last_train)

        predictions = []
        mean_target = train_df['target_2h'].mean()
        test_with_features = self.create_enhanced_features(test_df, is_train=False)

        route_stats = {}
        for route_id in test_df['route_id'].unique():
            rd = train_df[train_df['route_id'] == route_id]['target_2h']
            route_stats[route_id] = {
                'mean': rd.mean() if len(rd) > 0 else mean_target,
                'std': rd.std() if len(rd) > 0 else train_df['target_2h'].std(),
                'median': rd.median() if len(rd) > 0 else train_df['target_2h'].median(),
            }

        for route_id in test_df['route_id'].unique():
            route_test = test_with_features[test_with_features['route_id'] == route_id]
            route_history = last_train[last_train['route_id'] == route_id].copy()

            if len(route_history) == 0:
                for _, row in route_test.iterrows():
                    predictions.append({'id': row['id'], 'y_pred': max(0, route_stats[route_id]['median'])})
                continue

            route_history = route_history.sort_values('timestamp')
            route_mean = route_stats[route_id]['mean']
            route_std = route_stats[route_id]['std']

            for step_idx, (_, row) in enumerate(route_test.iterrows()):
                current_row = pd.DataFrame([row])
                current_row = self.create_enhanced_features(current_row, is_train=False)

                for lag in [1, 2, 3, 4, 6, 8, 12, 24, 48]:
                    current_row[f'lag_{lag}'] = (
                        route_history.iloc[-lag]['target_2h']
                        if len(route_history) >= lag else route_mean
                    )
                for window in [4, 8, 12, 24, 48]:
                    if len(route_history) >= window:
                        current_row[f'rolling_mean_{window}'] = route_history.tail(window)['target_2h'].mean()
                        current_row[f'rolling_std_{window}'] = route_history.tail(window)['target_2h'].std()
                    else:
                        current_row[f'rolling_mean_{window}'] = route_mean
                        current_row[f'rolling_std_{window}'] = route_std

                current_row['trend_7d'] = current_row['rolling_mean_8'].iloc[0] - current_row['rolling_mean_24'].iloc[0]
                current_row['ratio_to_mean'] = current_row['lag_1'].iloc[0] / (current_row['rolling_mean_24'].iloc[0] + 1)
                if 'office_from_id' in route_history.columns:
                    current_row['office_from_id'] = route_history.iloc[-1]['office_from_id']
                current_row['route_id'] = route_id

                for col in self.feature_cols:
                    if col not in current_row.columns:
                        current_row[col] = 0

                X_pred = current_row[self.feature_cols].fillna(0)
                X_pred[X_pred.select_dtypes(include=[np.number]).columns] = \
                    self.scaler.transform(X_pred.select_dtypes(include=[np.number]))

                step = min(step_idx + 1, self.forecast_points)
                model_info = self.models[step]

                if model_info['type'] == 'ensemble':
                    pred = (model_info['weight1'] * model_info['model1'].predict(X_pred)[0]
                            + model_info['weight2'] * model_info['model2'].predict(X_pred)[0])
                    pred = np.clip(pred, 0, route_mean + 3 * route_std)
                else:
                    pred = route_history.tail(24)['target_2h'].mean() if len(route_history) >= 24 else route_mean

                predictions.append({'id': row['id'], 'y_pred': max(0, pred)})

                new_row = route_history.iloc[-1:].copy()
                new_row['timestamp'] = row['timestamp']
                new_row['target_2h'] = pred
                route_history = pd.concat([route_history, new_row], ignore_index=True)

        return pd.DataFrame(predictions).sort_values('id').reset_index(drop=True)

    # ── Сохранение/загрузка ───────────────────────────────────────────────────

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"Модель сохранена: {path}")

    @classmethod
    def load(cls, path: str) -> "HybridForecaster":
        model = joblib.load(path)
        logger.info(f"Модель загружена: {path}")
        return model
