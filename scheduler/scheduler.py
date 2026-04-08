"""
scheduler/scheduler.py
Планировщик: каждые 2 часа запускает цикл прогноз → диспетчеризация
"""
import logging
import time
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_forecast_cycle():
    """
    Основной цикл прогнозирования и диспетчеризации.

    Шаги:
      1. Загрузить актуальные данные
      2. Обновить признаки (лаги, скользящие)
      3. Запустить HybridForecaster.predict_hybrid()
      4. Передать результат в DispatchService
      5. Отправить срочные заявки во внешнюю систему
      6. Залогировать результат
    """
    cycle_start = datetime.utcnow()
    logger.info(f"=== Запуск цикла прогнозирования: {cycle_start.isoformat()} ===")

    try:
        # ── Шаг 1: Загрузка данных ────────────────────────────────────────────
        logger.info("Шаг 1/5: Загрузка актуальных данных...")
        # В реальной системе: data = DataLoader.fetch_latest(hours=48)
        # Здесь заглушка для демонстрации
        logger.info("  ✓ Данные загружены")

        # ── Шаг 2: Обновление признаков ───────────────────────────────────────
        logger.info("Шаг 2/5: Обновление признаков...")
        # forecaster.create_lag_features_fast(data)
        logger.info("  ✓ Признаки обновлены")

        # ── Шаг 3: Прогнозирование ────────────────────────────────────────────
        logger.info("Шаг 3/5: Запуск прогнозной модели...")
        # predictions = forecaster.predict_hybrid(test_df, train_df)
        # Структура: [{"route_id": X, "steps": [{"step": 1, "volume": 2.5}, ...]}]
        logger.info("  ✓ Прогноз получен")

        # ── Шаг 4: Диспетчеризация ────────────────────────────────────────────
        logger.info("Шаг 4/5: Формирование заявок на транспорт...")
        # new_orders = dispatch_service.process_forecasts(forecasts)
        # urgent = dispatch_service.get_urgent_orders()
        logger.info("  ✓ Заявки сформированы")

        # ── Шаг 5: Отправка срочных заявок ───────────────────────────────────
        logger.info("Шаг 5/5: Отправка срочных заявок во внешнюю систему...")
        # for order in urgent:
        #     tms_client.send_order(order)
        #     dispatch_service.mark_sent(order.order_id)
        logger.info("  ✓ Срочные заявки отправлены")

        elapsed = (datetime.utcnow() - cycle_start).total_seconds()
        logger.info(f"=== Цикл завершён за {elapsed:.1f} сек ===\n")

    except Exception as e:
        logger.error(f"Ошибка в цикле прогнозирования: {e}", exc_info=True)


def start():
    scheduler = BlockingScheduler(timezone="UTC")

    # Запускать каждые 2 часа (совпадает с временным шагом данных)
    scheduler.add_job(
        run_forecast_cycle,
        trigger=IntervalTrigger(hours=2),
        id="forecast_cycle",
        name="Forecast & Dispatch Cycle",
        replace_existing=True,
        next_run_time=datetime.utcnow(),  # запустить сразу при старте
    )

    logger.info("Планировщик запущен. Интервал: каждые 2 часа.")
    logger.info("Нажмите Ctrl+C для остановки.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Планировщик остановлен.")


if __name__ == "__main__":
    start()
