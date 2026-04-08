-- Инициализация БД для AutoDispatch

CREATE TABLE IF NOT EXISTS dispatch_orders (
    order_id        VARCHAR(64) PRIMARY KEY,
    route_id        INTEGER NOT NULL,
    office_from_id  INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    dispatch_time   TIMESTAMPTZ NOT NULL,
    arrival_time    TIMESTAMPTZ NOT NULL,
    n_trucks        INTEGER NOT NULL,
    forecast_volume NUMERIC(10, 2) NOT NULL,
    priority        VARCHAR(10) NOT NULL,  -- URGENT | PLANNED | RESERVED
    confidence      VARCHAR(10) NOT NULL,  -- HIGH | MEDIUM | LOW
    status          VARCHAR(16) NOT NULL DEFAULT 'PENDING'
);

CREATE INDEX idx_dispatch_orders_route ON dispatch_orders(route_id);
CREATE INDEX idx_dispatch_orders_arrival ON dispatch_orders(arrival_time);
CREATE INDEX idx_dispatch_orders_status ON dispatch_orders(status);

CREATE TABLE IF NOT EXISTS forecast_log (
    id              SERIAL PRIMARY KEY,
    run_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    routes_count    INTEGER NOT NULL,
    orders_created  INTEGER NOT NULL,
    total_trucks    INTEGER NOT NULL,
    duration_sec    NUMERIC(8, 2)
);
