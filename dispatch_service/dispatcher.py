"""
dispatch_service/dispatcher.py
Сервис диспетчеризации: перевод прогноза отгрузок в заявки на транспорт
"""
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

PERIOD_HOURS = 2
LEAD_TIME_HOURS = 4       # минимальный лид-тайм вызова транспорта
TRUCK_CAPACITY = 1.0      # 1 ед. target_2h = 1 грузовик (допущение MVP)


@dataclass
class DispatchOrder:
    order_id: str
    route_id: int
    office_from_id: Optional[int]
    created_at: datetime
    dispatch_time: datetime   # когда вызываем транспорт
    arrival_time: datetime    # когда нужен на складе
    n_trucks: int
    forecast_volume: float
    priority: str             # URGENT | PLANNED | RESERVED
    confidence: str           # HIGH | MEDIUM | LOW
    status: str = "PENDING"   # PENDING | SENT | CONFIRMED | CANCELLED


@dataclass
class ForecastResult:
    route_id: int
    office_from_id: Optional[int]
    generated_at: datetime
    steps: List[Dict]         # [{"step": 1, "volume": 2.5}, ...]


class DispatchService:
    """
    Преобразует прогнозы объёмов отгрузок в конкретные заявки на транспорт.

    Алгоритм:
      1. Для каждого шага прогноза рассчитать кол-во машин: ceil(volume / TRUCK_CAPACITY)
      2. Если volume >= порог → создать DispatchOrder
      3. Вызов за LEAD_TIME_HOURS до момента отгрузки
      4. Приоритет: шаги 1-2 → URGENT, 3-5 → PLANNED, 6-10 → RESERVED
    """

    def __init__(
        self,
        min_volume_threshold: float = 1.0,
        truck_capacity: float = TRUCK_CAPACITY,
        lead_time_hours: int = LEAD_TIME_HOURS,
    ):
        self.min_volume_threshold = min_volume_threshold
        self.truck_capacity = truck_capacity
        self.lead_time_hours = lead_time_hours
        self._order_registry: Dict[str, DispatchOrder] = {}

    # ── Публичный интерфейс ────────────────────────────────────────────────────

    def process_forecasts(
        self, forecasts: List[ForecastResult], now: Optional[datetime] = None
    ) -> List[DispatchOrder]:
        """
        Принимает список прогнозов, возвращает список заявок.
        Дедуплицирует по ключу (route_id, arrival_time).
        """
        now = now or datetime.utcnow()
        new_orders = []

        for forecast in forecasts:
            for step_data in forecast.steps:
                order = self._build_order(forecast, step_data, now)
                if order is None:
                    continue

                key = self._order_key(order)
                if key in self._order_registry:
                    # Обновляем существующую заявку
                    existing = self._order_registry[key]
                    if existing.status == "PENDING":
                        existing.n_trucks = order.n_trucks
                        existing.forecast_volume = order.forecast_volume
                        logger.debug(f"Updated order {key}: {order.n_trucks} trucks")
                else:
                    self._order_registry[key] = order
                    new_orders.append(order)
                    logger.info(
                        f"New {order.priority} order: route={order.route_id} "
                        f"trucks={order.n_trucks} arrival={order.arrival_time.isoformat()}"
                    )

        return new_orders

    def get_urgent_orders(self, now: Optional[datetime] = None) -> List[DispatchOrder]:
        """Заявки, которые нужно отправить немедленно (dispatch_time <= now + 15 мин)"""
        now = now or datetime.utcnow()
        cutoff = now + timedelta(minutes=15)
        return [
            o for o in self._order_registry.values()
            if o.status == "PENDING" and o.dispatch_time <= cutoff
        ]

    def get_plan(self, hours_ahead: int = 20) -> List[DispatchOrder]:
        """Все заявки на ближайшие N часов, отсортированные по времени"""
        now = datetime.utcnow()
        horizon = now + timedelta(hours=hours_ahead)
        orders = [
            o for o in self._order_registry.values()
            if o.arrival_time <= horizon and o.status in ("PENDING", "SENT")
        ]
        orders.sort(key=lambda o: (o.dispatch_time, {"URGENT": 0, "PLANNED": 1, "RESERVED": 2}[o.priority]))
        return orders

    def mark_sent(self, order_id: str):
        if order_id in self._order_registry:
            self._order_registry[order_id].status = "SENT"

    def mark_confirmed(self, order_id: str):
        if order_id in self._order_registry:
            self._order_registry[order_id].status = "CONFIRMED"

    # ── Внутренняя логика ─────────────────────────────────────────────────────

    def _build_order(
        self, forecast: ForecastResult, step_data: dict, now: datetime
    ) -> Optional[DispatchOrder]:
        volume = step_data["volume"]
        step = step_data["step"]

        if volume < self.min_volume_threshold:
            return None

        n_trucks = math.ceil(volume / self.truck_capacity)
        arrival_time = now + timedelta(hours=step * PERIOD_HOURS)
        dispatch_time = arrival_time - timedelta(hours=self.lead_time_hours)

        # Если dispatch_time уже прошёл — вызываем немедленно
        if dispatch_time < now:
            dispatch_time = now

        return DispatchOrder(
            order_id=f"{forecast.route_id}_{arrival_time.strftime('%Y%m%d%H%M')}",
            route_id=forecast.route_id,
            office_from_id=forecast.office_from_id,
            created_at=now,
            dispatch_time=dispatch_time,
            arrival_time=arrival_time,
            n_trucks=n_trucks,
            forecast_volume=round(volume, 2),
            priority=self._get_priority(step),
            confidence=self._get_confidence(step),
        )

    @staticmethod
    def _get_priority(step: int) -> str:
        if step <= 2:
            return "URGENT"
        elif step <= 5:
            return "PLANNED"
        return "RESERVED"

    @staticmethod
    def _get_confidence(step: int) -> str:
        if step <= 3:
            return "HIGH"
        elif step <= 6:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _order_key(order: DispatchOrder) -> str:
        return f"{order.route_id}_{order.arrival_time.strftime('%Y%m%d%H%M')}"
