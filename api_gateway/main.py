"""
AutoDispatch API Gateway
Система автоматического вызова транспорта на склады
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'forecasting'))
import io
import csv
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
import math
import logging
import joblib
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="АвтоДиспетч API",
    description="Система автоматического вызова транспорта на основе прогноза отгрузок",
    version="1.0.0",
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))

PERIOD_HOURS = 2
LEAD_TIME_PERIODS = 2
MIN_VOLUME_THRESHOLD = 1.0

MODEL_PATH = os.environ.get("MODEL_PATH", "../models/hybrid_forecaster.pkl")
TRAIN_PATH = os.environ.get("TRAIN_PATH", "../data/train_team_track.parquet")

forecaster = None
train_df = None

@app.on_event("startup")
def load_model():
    global forecaster, train_df
    try:
        logger.info(f"Загрузка модели: {MODEL_PATH}")
        forecaster = joblib.load(MODEL_PATH)
        logger.info("Модель загружена")
        logger.info(f"Загрузка train данных: {TRAIN_PATH}")
        train_df = pd.read_parquet(TRAIN_PATH)
        train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
        logger.info(f"Train данные загружены: {len(train_df):,} строк")
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
        logger.warning("API запущен без модели")


class DispatchRequest(BaseModel):
    route_ids: List[int] = Field(..., description="Список route_id для прогнозирования")
    horizon_steps: int = Field(10, ge=1, le=10, description="Горизонт прогноза (шагов)")
    min_volume_threshold: float = Field(MIN_VOLUME_THRESHOLD, description="Минимальный объём для заявки")

class ForecastStep(BaseModel):
    step: int
    time: datetime
    volume: float
    n_trucks: int
    confidence: str

class DispatchOrder(BaseModel):
    route_id: int
    office_from_id: Optional[int]
    dispatch_time: datetime
    arrival_time: datetime
    n_trucks: int
    forecast_volume: float
    priority: str
    confidence: str

class DispatchResponse(BaseModel):
    generated_at: datetime
    dispatch_orders: List[DispatchOrder]
    total_trucks: int
    routes_processed: int

class ForecastResponse(BaseModel):
    route_id: int
    generated_at: datetime
    forecast: List[ForecastStep]


def get_priority(step):
    if step <= 2: return "URGENT"
    elif step <= 5: return "PLANNED"
    return "RESERVED"

def get_confidence(step):
    if step <= 3: return "HIGH"
    elif step <= 6: return "MEDIUM"
    return "LOW"

def volume_to_trucks(volume):
    return math.ceil(max(0, volume))


def get_forecast_for_routes(route_ids, horizon):
    if forecaster is None or train_df is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Запустите сервер из папки api_gateway.")

    route_data = train_df[train_df['route_id'].isin(route_ids)]
    if len(route_data) == 0:
        raise HTTPException(status_code=404, detail=f"Маршруты {route_ids} не найдены в данных")

    test_rows = []
    counter = 0
    for route_id in route_ids:
        route_rows = route_data[route_data['route_id'] == route_id].tail(horizon).copy()
        if len(route_rows) == 0:
            continue
        route_rows = route_rows.reset_index(drop=True)
        route_rows['id'] = range(counter, counter + len(route_rows))
        counter += len(route_rows)
        test_rows.append(route_rows)

    if not test_rows:
        raise HTTPException(status_code=404, detail="Нет данных для прогноза")

    test_df_local = pd.concat(test_rows, ignore_index=True)
    predictions = forecaster.predict_hybrid(test_df_local, train_df)

    result = {}
    idx = 0
    for route_id in route_ids:
        route_rows = route_data[route_data['route_id'] == route_id]
        if len(route_rows) == 0:
            continue
        office_id = int(route_rows['office_from_id'].iloc[-1]) if 'office_from_id' in route_rows.columns else None
        n = min(horizon, len(route_rows))
        steps = []
        for step in range(1, n + 1):
            volume = max(0.0, float(predictions['y_pred'].iloc[idx])) if idx < len(predictions) else 0.0
            steps.append({"step": step, "volume": volume})
            idx += 1
        result[route_id] = {"office_from_id": office_id, "steps": steps}

    return result


def build_dispatch_orders(route_id, office_from_id, forecast_steps, now, min_threshold):
    orders = []
    for step_data in forecast_steps:
        volume = step_data["volume"]
        if volume < min_threshold:
            continue
        step = step_data["step"]
        arrival_time = now + timedelta(hours=step * PERIOD_HOURS)
        dispatch_time = arrival_time - timedelta(hours=LEAD_TIME_PERIODS * PERIOD_HOURS)
        if dispatch_time < now:
            dispatch_time = now
        orders.append(DispatchOrder(
            route_id=route_id,
            office_from_id=office_from_id,
            dispatch_time=dispatch_time,
            arrival_time=arrival_time,
            n_trucks=volume_to_trucks(volume),
            forecast_volume=round(volume, 2),
            priority=get_priority(step),
            confidence=get_confidence(step),
        ))
    return orders


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "model_loaded": forecaster is not None,
        "train_data_loaded": train_df is not None,
        "train_rows": len(train_df) if train_df is not None else 0,
        "time": datetime.utcnow(),
    }


@app.get("/api/v1/forecast/{route_id}", response_model=ForecastResponse)
def get_route_forecast(route_id: int, horizon_steps: int = 10):
    now = datetime.utcnow()
    raw = get_forecast_for_routes([route_id], horizon_steps)
    if route_id not in raw:
        raise HTTPException(status_code=404, detail=f"route_id {route_id} не найден")
    route_data = raw[route_id]
    forecast_steps = [
        ForecastStep(
            step=s["step"],
            time=now + timedelta(hours=s["step"] * PERIOD_HOURS),
            volume=round(s["volume"], 2),
            n_trucks=volume_to_trucks(s["volume"]),
            confidence=get_confidence(s["step"]),
        )
        for s in route_data["steps"]
    ]
    return ForecastResponse(route_id=route_id, generated_at=now, forecast=forecast_steps)


@app.post("/api/v1/dispatch", response_model=DispatchResponse)
def create_dispatch_plan(request: DispatchRequest):
    now = datetime.utcnow()
    logger.info(f"Dispatch request: {len(request.route_ids)} routes")
    raw_forecasts = get_forecast_for_routes(request.route_ids, request.horizon_steps)
    all_orders = []
    for route_id, route_data in raw_forecasts.items():
        orders = build_dispatch_orders(
            route_id=route_id,
            office_from_id=route_data.get("office_from_id"),
            forecast_steps=route_data["steps"],
            now=now,
            min_threshold=request.min_volume_threshold,
        )
        all_orders.extend(orders)
    priority_order = {"URGENT": 0, "PLANNED": 1, "RESERVED": 2}
    all_orders.sort(key=lambda o: (o.dispatch_time, priority_order[o.priority]))
    total_trucks = sum(o.n_trucks for o in all_orders)
    return DispatchResponse(
        generated_at=now,
        dispatch_orders=all_orders,
        total_trucks=total_trucks,
        routes_processed=len(raw_forecasts),
    )


@app.get("/api/v1/dispatch/plan")
def get_dispatch_plan():
    return {
        "message": "В production-версии здесь возвращается актуальный план из БД",
        "hint": "Используйте POST /api/v1/dispatch для генерации плана"
    }


@app.post("/api/v1/dispatch/csv")
def create_dispatch_plan_csv(request: DispatchRequest):
    """
    Генерирует CSV файл с планом вызова транспорта
    """
    now = datetime.utcnow()
    logger.info(f"CSV export request: {len(request.route_ids)} routes")

    raw_forecasts = get_forecast_for_routes(request.route_ids, request.horizon_steps)
    all_orders = []

    for route_id, route_data in raw_forecasts.items():
        orders = build_dispatch_orders(
            route_id=route_id,
            office_from_id=route_data.get("office_from_id"),
            forecast_steps=route_data["steps"],
            now=now,
            min_threshold=request.min_volume_threshold,
        )
        all_orders.extend(orders)

    # Сортируем
    priority_order = {"URGENT": 0, "PLANNED": 1, "RESERVED": 2}
    all_orders.sort(key=lambda o: (o.dispatch_time, priority_order[o.priority]))

    # Создаём CSV в памяти
    output = io.StringIO()
    writer = csv.writer(output, delimiter=';')

    # Заголовки
    writer.writerow([
        "route_id",
        "office_from_id",
        "dispatch_time",
        "arrival_time",
        "n_trucks",
        "forecast_volume",
        "priority",
        "confidence"
    ])

    # Данные
    for order in all_orders:
        writer.writerow([
            order.route_id,
            order.office_from_id or "",
            order.dispatch_time.strftime("%Y-%m-%d %H:%M:%S"),
            order.arrival_time.strftime("%Y-%m-%d %H:%M:%S"),
            order.n_trucks,
            order.forecast_volume,
            order.priority,
            order.confidence
        ])

    # Возвращаем как файл
    output.seek(0)
    filename = f"dispatch_plan_{now.strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([output.getvalue().encode('utf-8-sig')]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )