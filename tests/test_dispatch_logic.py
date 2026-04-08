"""
tests/test_dispatch_logic.py
Unit-тесты бизнес-логики диспетчеризации
"""
import pytest
from datetime import datetime, timedelta
import sys
sys.path.insert(0, "../dispatch_service")

from dispatcher import DispatchService, ForecastResult


@pytest.fixture
def service():
    return DispatchService(min_volume_threshold=1.0, lead_time_hours=4)


@pytest.fixture
def now():
    return datetime(2024, 11, 1, 8, 0, 0)


def make_forecast(route_id=101, steps=None):
    if steps is None:
        steps = [{"step": i, "volume": float(i)} for i in range(1, 11)]
    return ForecastResult(
        route_id=route_id,
        office_from_id=5,
        generated_at=datetime.utcnow(),
        steps=steps,
    )


class TestDispatchPriority:
    def test_step1_is_urgent(self, service, now):
        forecast = make_forecast(steps=[{"step": 1, "volume": 3.0}])
        orders = service.process_forecasts([forecast], now=now)
        assert orders[0].priority == "URGENT"

    def test_step2_is_urgent(self, service, now):
        forecast = make_forecast(steps=[{"step": 2, "volume": 2.0}])
        orders = service.process_forecasts([forecast], now=now)
        assert orders[0].priority == "URGENT"

    def test_step3_is_planned(self, service, now):
        forecast = make_forecast(steps=[{"step": 3, "volume": 2.0}])
        orders = service.process_forecasts([forecast], now=now)
        assert orders[0].priority == "PLANNED"

    def test_step6_is_reserved(self, service, now):
        forecast = make_forecast(steps=[{"step": 6, "volume": 2.0}])
        orders = service.process_forecasts([forecast], now=now)
        assert orders[0].priority == "RESERVED"


class TestTruckCalculation:
    def test_ceil_rounding(self, service, now):
        forecast = make_forecast(steps=[{"step": 1, "volume": 2.1}])
        orders = service.process_forecasts([forecast], now=now)
        assert orders[0].n_trucks == 3  # ceil(2.1) = 3

    def test_exact_volume(self, service, now):
        forecast = make_forecast(steps=[{"step": 1, "volume": 3.0}])
        orders = service.process_forecasts([forecast], now=now)
        assert orders[0].n_trucks == 3

    def test_below_threshold_no_order(self, service, now):
        forecast = make_forecast(steps=[{"step": 1, "volume": 0.5}])
        orders = service.process_forecasts([forecast], now=now)
        assert len(orders) == 0


class TestDispatchTiming:
    def test_arrival_time_correct(self, service, now):
        forecast = make_forecast(steps=[{"step": 2, "volume": 2.0}])
        orders = service.process_forecasts([forecast], now=now)
        expected_arrival = now + timedelta(hours=4)  # step=2, period=2h
        assert orders[0].arrival_time == expected_arrival

    def test_dispatch_before_arrival(self, service, now):
        forecast = make_forecast(steps=[{"step": 3, "volume": 2.0}])
        orders = service.process_forecasts([forecast], now=now)
        assert orders[0].dispatch_time < orders[0].arrival_time

    def test_dispatch_lead_time(self, service, now):
        forecast = make_forecast(steps=[{"step": 5, "volume": 2.0}])
        orders = service.process_forecasts([forecast], now=now)
        diff = orders[0].arrival_time - orders[0].dispatch_time
        assert diff.total_seconds() == 4 * 3600  # 4 часа лид-тайм


class TestDeduplication:
    def test_no_duplicate_orders(self, service, now):
        forecast = make_forecast(steps=[{"step": 1, "volume": 2.0}])
        orders1 = service.process_forecasts([forecast], now=now)
        orders2 = service.process_forecasts([forecast], now=now)
        assert len(orders1) == 1
        assert len(orders2) == 0  # дубликат не добавляется
