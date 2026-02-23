"""Incident analysis tasks — root cause analysis from logs and metrics.

All factories are seedable. Seed selects from 6 structurally distinct
incident scenarios, each with unique log patterns, metric signatures,
red herrings, and resolution events.
"""

import random as _random

from ..dataset import RubricDatapoint, RubricCategory, BinaryRubricCategory
from tinker_cookbook.rl.envs import tools


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

INCIDENT_SCENARIOS = [
    {
        "key": "conn_pool",
        "service": "payment-gateway",
        "root_cause": "database connection pool exhaustion",
        "description": "payment processing failed for a significant number of transactions",
        "resolution_summary": "pool size increased from 20 to 50",
    },
    {
        "key": "memory_leak",
        "service": "order-processor",
        "root_cause": "memory leak in order serialization causing OOM kill",
        "description": "the order processing pipeline stopped responding",
        "resolution_summary": "service restarted and leak patched in serializer",
    },
    {
        "key": "disk_full",
        "service": "logging-aggregator",
        "root_cause": "disk full on /var/log partition due to unrotated debug logs",
        "description": "multiple services experienced intermittent failures and data loss",
        "resolution_summary": "old logs purged and log rotation configured",
    },
    {
        "key": "dns_failure",
        "service": "api-gateway",
        "root_cause": "internal DNS resolver failure causing upstream lookup timeouts",
        "description": "multiple internal service calls began failing",
        "resolution_summary": "DNS resolver restarted and secondary resolver failover configured",
    },
    {
        "key": "cert_expiry",
        "service": "auth-service",
        "root_cause": "TLS certificate expired on internal service mesh",
        "description": "authentication requests began failing across all services",
        "resolution_summary": "certificate renewed and auto-renewal configured",
    },
    {
        "key": "deadlock",
        "service": "inventory-service",
        "root_cause": "database deadlock from concurrent inventory updates without proper lock ordering",
        "description": "inventory updates began timing out and returning errors",
        "resolution_summary": "deadlocked queries killed and lock ordering fixed in application code",
    },
]


def _gen_ts(minute: int) -> str:
    """Generate a timestamp string for a given minute offset."""
    h = 14 + minute // 60
    m = minute % 60
    return f"2024-03-15T{h:02d}:{m:02d}:00.000Z"


def _gen_ts_short(minute: int) -> str:
    h = 14 + minute // 60
    m = minute % 60
    return f"2024-03-15T{h:02d}:{m:02d}"


# ---------------------------------------------------------------------------
# Per-scenario log generators
# ---------------------------------------------------------------------------


def _logs_conn_pool(rng: _random.Random, svc: str) -> tuple[list[str], list[str]]:
    """Connection pool exhaustion logs."""
    app, lb = [], []
    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 30:
            if rng.random() < 0.3:
                app.append(f"{ts} INFO  [{svc}] Processed txn_{rng.randint(10000,99999)} in {rng.randint(50,200)}ms")
            if rng.random() < 0.1:
                app.append(f"{ts} DEBUG [{svc}] Connection pool: active=8/20, idle=12")
        elif minute < 45:
            if rng.random() < 0.5:
                app.append(f"{ts} INFO  [{svc}] Processed txn_{rng.randint(10000,99999)} in {rng.randint(150,500)}ms")
            if rng.random() < 0.3:
                active = min(20, 8 + (minute - 30))
                app.append(f"{ts} WARN  [{svc}] Connection pool: active={active}/20, idle={20 - active}")
            if minute == 40:
                app.append(f"{ts} WARN  [{svc}] Connection pool utilization >90%: active=19/20")
        elif minute < 60:
            if rng.random() < 0.6:
                app.append(f"{ts} ERROR [{svc}] Failed to acquire connection: pool exhausted (active=20/20, wait_timeout=5000ms)")
            if rng.random() < 0.4:
                app.append(f"{ts} ERROR [{svc}] Payment processing failed for txn_{rng.randint(10000,99999)}: ConnectionPoolTimeoutException")
            if rng.random() < 0.2:
                app.append(f"{ts} WARN  [{svc}] Circuit breaker OPEN for database-primary after 10 consecutive failures")
            if minute == 50:
                # Red herring
                app.append(f"{ts} WARN  [jvm] GC pause: {rng.randint(180,280)}ms (young generation)")
        else:
            if minute == 60:
                app.append(f"{ts} INFO  [{svc}] Configuration updated: max_connections changed from 20 to 50")
                app.append(f"{ts} INFO  [{svc}] Connection pool reinitialized")
            if rng.random() < 0.3:
                app.append(f"{ts} INFO  [{svc}] Processed txn_{rng.randint(10000,99999)} in {rng.randint(60,250)}ms")

    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 45:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(50,200)}ms")
        elif minute < 60:
            if rng.random() < 0.5:
                lb.append(f"{ts} haproxy: {svc}/server1 502 0ms (backend timeout)")
            if rng.random() < 0.3:
                lb.append(f"{ts} haproxy: {svc} health check FAILED")
        else:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(60,250)}ms")
    return app, lb


def _logs_memory_leak(rng: _random.Random, svc: str) -> tuple[list[str], list[str]]:
    """Memory leak → OOM kill logs."""
    app, lb = [], []
    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 30:
            if rng.random() < 0.3:
                app.append(f"{ts} INFO  [{svc}] Processed order ord_{rng.randint(10000,99999)} in {rng.randint(40,150)}ms")
            if rng.random() < 0.1:
                heap_mb = 512 + minute * 8
                app.append(f"{ts} DEBUG [jvm] Heap usage: {heap_mb}MB / 2048MB")
        elif minute < 50:
            if rng.random() < 0.4:
                latency = rng.randint(100, 300) + (minute - 30) * 20
                app.append(f"{ts} INFO  [{svc}] Processed order ord_{rng.randint(10000,99999)} in {latency}ms")
            if rng.random() < 0.3:
                heap_mb = 512 + minute * 30 + rng.randint(-50, 50)
                app.append(f"{ts} WARN  [jvm] Heap usage: {min(heap_mb, 2000)}MB / 2048MB")
            if rng.random() < 0.15:
                app.append(f"{ts} WARN  [jvm] GC pause: {rng.randint(300, 800)}ms (full GC)")
            if minute == 45:
                app.append(f"{ts} WARN  [{svc}] Response time SLA breach: p99 > 1000ms")
                # Red herring
                app.append(f"{ts} INFO  [network] TCP retransmit rate: 0.02%")
        elif minute < 60:
            if rng.random() < 0.5:
                app.append(f"{ts} ERROR [jvm] java.lang.OutOfMemoryError: Java heap space")
            if rng.random() < 0.4:
                app.append(f"{ts} ERROR [{svc}] Failed to serialize order: OutOfMemoryError")
            if minute == 55:
                app.append(f"{ts} ERROR [kernel] Out of memory: Killed process {rng.randint(10000,60000)} ({svc})")
                app.append(f"{ts} INFO  [systemd] {svc}.service: Main process exited, code=killed, status=9/KILL")
        else:
            if minute == 60:
                app.append(f"{ts} INFO  [systemd] {svc}.service: Started")
                app.append(f"{ts} INFO  [{svc}] Application initialized, heap: 450MB / 2048MB")
            if rng.random() < 0.3:
                app.append(f"{ts} INFO  [{svc}] Processed order ord_{rng.randint(10000,99999)} in {rng.randint(40,180)}ms")
            if minute == 70:
                app.append(f"{ts} INFO  [{svc}] Deployed hotfix: serializer buffer pool leak patched")

    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 50:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(50,300)}ms")
        elif minute < 60:
            if rng.random() < 0.5:
                lb.append(f"{ts} haproxy: {svc}/server1 503 0ms (connection refused)")
            if rng.random() < 0.3:
                lb.append(f"{ts} haproxy: {svc} health check FAILED")
        else:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(40,200)}ms")
    return app, lb


def _logs_disk_full(rng: _random.Random, svc: str) -> tuple[list[str], list[str]]:
    """Disk full logs."""
    app, lb = [], []
    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 35:
            if rng.random() < 0.3:
                app.append(f"{ts} INFO  [{svc}] Ingested log batch: {rng.randint(200,800)} events")
            if rng.random() < 0.1:
                pct = 70 + minute
                app.append(f"{ts} DEBUG [storage] /var/log partition: {min(pct, 99)}% used")
        elif minute < 50:
            if rng.random() < 0.3:
                pct = min(88 + (minute - 35), 99)
                app.append(f"{ts} WARN  [storage] /var/log partition: {pct}% used")
            if rng.random() < 0.2:
                app.append(f"{ts} WARN  [{svc}] Write latency elevated: {rng.randint(500, 2000)}ms for log batch")
            if minute == 45:
                # Red herring
                app.append(f"{ts} INFO  [network] NIC eth0: RX errors=0, TX errors=0, dropped=2")
        elif minute < 65:
            if rng.random() < 0.6:
                app.append(f"{ts} ERROR [storage] Write failed: No space left on device (/var/log)")
            if rng.random() < 0.4:
                app.append(f"{ts} ERROR [{svc}] Failed to write audit log: IOError — ENOSPC")
            if rng.random() < 0.3:
                svc2 = rng.choice(["auth-service", "payment-gateway", "user-service"])
                app.append(f"{ts} ERROR [{svc2}] Log shipping failed: cannot write to local buffer")
            if minute == 50:
                app.append(f"{ts} WARN  [storage] /var/log partition: 100% used (0 bytes free)")
        else:
            if minute == 65:
                app.append(f"{ts} INFO  [ops] Running: rm -f /var/log/debug-*.log.gz (freed 42GB)")
                app.append(f"{ts} INFO  [ops] Running: logrotate --force /etc/logrotate.d/{svc}")
            if rng.random() < 0.3:
                app.append(f"{ts} INFO  [{svc}] Ingested log batch: {rng.randint(200,800)} events")
            if rng.random() < 0.1:
                app.append(f"{ts} DEBUG [storage] /var/log partition: {rng.randint(25, 40)}% used")

    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 50:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(30,150)}ms")
        elif minute < 65:
            if rng.random() < 0.4:
                lb.append(f"{ts} haproxy: {svc}/server1 500 {rng.randint(10,50)}ms (internal error)")
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(100,500)}ms")
        else:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(30,150)}ms")
    return app, lb


def _logs_dns_failure(rng: _random.Random, svc: str) -> tuple[list[str], list[str]]:
    """DNS resolver failure logs."""
    app, lb = [], []
    upstreams = ["user-service", "inventory-service", "notification-service", "payment-gateway"]
    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 40:
            if rng.random() < 0.3:
                upstream = rng.choice(upstreams)
                app.append(f"{ts} INFO  [{svc}] Routed request to {upstream}: 200 in {rng.randint(20,100)}ms")
            if rng.random() < 0.05:
                app.append(f"{ts} DEBUG [dns] Resolved user-service.internal → 10.0.{rng.randint(1,5)}.{rng.randint(10,200)}")
        elif minute < 55:
            if rng.random() < 0.5:
                upstream = rng.choice(upstreams)
                app.append(f"{ts} ERROR [{svc}] DNS lookup failed for {upstream}.internal: SERVFAIL")
            if rng.random() < 0.3:
                upstream = rng.choice(upstreams)
                app.append(f"{ts} ERROR [{svc}] Connection to {upstream} timed out after 5000ms")
            if rng.random() < 0.2:
                app.append(f"{ts} WARN  [dns] Resolver 10.0.0.2 not responding (attempt {rng.randint(1,5)})")
            if minute == 45:
                # Red herring
                app.append(f"{ts} WARN  [jvm] GC pause: 150ms (young generation)")
            if minute == 48:
                app.append(f"{ts} WARN  [{svc}] Circuit breaker OPEN for user-service after 15 consecutive timeouts")
        else:
            if minute == 55:
                app.append(f"{ts} INFO  [dns] Resolver 10.0.0.2 restarted (coredns)")
                app.append(f"{ts} INFO  [dns] Added failover resolver 10.0.0.3")
            if rng.random() < 0.3:
                upstream = rng.choice(upstreams)
                app.append(f"{ts} INFO  [{svc}] Routed request to {upstream}: 200 in {rng.randint(20,120)}ms")
            if rng.random() < 0.05:
                app.append(f"{ts} DEBUG [dns] Resolved {rng.choice(upstreams)}.internal → 10.0.{rng.randint(1,5)}.{rng.randint(10,200)}")

    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 40:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(20,100)}ms")
        elif minute < 55:
            if rng.random() < 0.5:
                lb.append(f"{ts} haproxy: {svc}/server1 504 5001ms (gateway timeout)")
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc} health check FAILED")
        else:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(20,120)}ms")
    return app, lb


def _logs_cert_expiry(rng: _random.Random, svc: str) -> tuple[list[str], list[str]]:
    """TLS certificate expiry logs."""
    app, lb = [], []
    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 38:
            if rng.random() < 0.3:
                app.append(f"{ts} INFO  [{svc}] Authenticated user_{rng.randint(1000,9999)} via OAuth2 in {rng.randint(15,80)}ms")
            if minute == 20:
                # Foreshadowing buried in debug
                app.append(f"{ts} DEBUG [tls] Certificate for *.internal.corp expires in 18 minutes")
        elif minute < 55:
            if minute == 38:
                app.append(f"{ts} ERROR [tls] Handshake failed: certificate has expired (not after: 2024-03-15T14:38:00Z)")
            if rng.random() < 0.5:
                app.append(f"{ts} ERROR [{svc}] TLS handshake failed connecting to token-validator.internal: CERTIFICATE_EXPIRED")
            if rng.random() < 0.4:
                app.append(f"{ts} ERROR [{svc}] Authentication failed for user_{rng.randint(1000,9999)}: upstream TLS error")
            if rng.random() < 0.15:
                # Red herring
                app.append(f"{ts} WARN  [{svc}] Auth cache miss rate elevated: {rng.randint(30,60)}%")
            if minute == 45:
                app.append(f"{ts} WARN  [jvm] GC pause: 120ms (young generation)")
        else:
            if minute == 55:
                app.append(f"{ts} INFO  [tls] Certificate renewed for *.internal.corp (new expiry: 2025-03-15)")
                app.append(f"{ts} INFO  [{svc}] TLS connections restored")
            if rng.random() < 0.3:
                app.append(f"{ts} INFO  [{svc}] Authenticated user_{rng.randint(1000,9999)} via OAuth2 in {rng.randint(15,80)}ms")

    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 38:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(15,80)}ms")
        elif minute < 55:
            if rng.random() < 0.5:
                lb.append(f"{ts} haproxy: {svc}/server1 503 0ms (SSL handshake failure)")
            if rng.random() < 0.3:
                lb.append(f"{ts} haproxy: {svc} health check FAILED")
        else:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(15,100)}ms")
    return app, lb


def _logs_deadlock(rng: _random.Random, svc: str) -> tuple[list[str], list[str]]:
    """Database deadlock logs."""
    app, lb = [], []
    tables = ["inventory_items", "stock_reservations"]
    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 35:
            if rng.random() < 0.3:
                app.append(f"{ts} INFO  [{svc}] Updated stock for SKU-{rng.randint(1000,9999)} in {rng.randint(10,80)}ms")
            if rng.random() < 0.1:
                app.append(f"{ts} DEBUG [db] Active transactions: {rng.randint(3,10)}, lock waits: 0")
        elif minute < 50:
            if rng.random() < 0.4:
                app.append(f"{ts} WARN  [db] Lock wait timeout: transaction on {rng.choice(tables)} waiting {rng.randint(5000,15000)}ms")
            if rng.random() < 0.3:
                app.append(f"{ts} ERROR [{svc}] Inventory update failed for SKU-{rng.randint(1000,9999)}: lock wait timeout exceeded")
            if rng.random() < 0.2:
                app.append(f"{ts} ERROR [db] Deadlock detected: txn_{rng.randint(100,999)} holds lock on {tables[0]}, waiting for {tables[1]}; txn_{rng.randint(100,999)} holds lock on {tables[1]}, waiting for {tables[0]}")
            if minute == 42:
                # Red herring
                app.append(f"{ts} WARN  [network] Connection to replica-2 latency: {rng.randint(50,100)}ms (elevated)")
            if minute == 45:
                app.append(f"{ts} WARN  [db] Active transactions: {rng.randint(40,80)}, lock waits: {rng.randint(15,30)}")
        elif minute < 60:
            if rng.random() < 0.5:
                app.append(f"{ts} ERROR [db] Deadlock detected: circular lock dependency on {tables[0]} ↔ {tables[1]}")
            if rng.random() < 0.4:
                app.append(f"{ts} ERROR [{svc}] Inventory update failed: DeadlockException — transaction rolled back")
            if minute == 55:
                app.append(f"{ts} WARN  [{svc}] Error rate >50%: circuit breaker OPEN for inventory-db")
        else:
            if minute == 60:
                app.append(f"{ts} INFO  [dba] Killed {rng.randint(12,25)} deadlocked sessions")
                app.append(f"{ts} INFO  [{svc}] Deployed hotfix: lock acquisition order changed to {tables[0]} → {tables[1]} (consistent ordering)")
            if rng.random() < 0.3:
                app.append(f"{ts} INFO  [{svc}] Updated stock for SKU-{rng.randint(1000,9999)} in {rng.randint(10,80)}ms")
            if rng.random() < 0.1:
                app.append(f"{ts} DEBUG [db] Active transactions: {rng.randint(3,12)}, lock waits: 0")

    for minute in range(0, 90):
        ts = _gen_ts(minute)
        if minute < 35:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(10,80)}ms")
        elif minute < 60:
            if rng.random() < 0.4:
                lb.append(f"{ts} haproxy: {svc}/server1 500 {rng.randint(5000,15000)}ms (backend error)")
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(100,500)}ms")
        else:
            if rng.random() < 0.2:
                lb.append(f"{ts} haproxy: {svc}/server1 200 {rng.randint(10,80)}ms")
    return app, lb


# ---------------------------------------------------------------------------
# Per-scenario metric generators
# ---------------------------------------------------------------------------

_METRICS_HEADER = "timestamp,latency_p99_ms,error_rate_pct,cpu_pct,memory_pct,disk_pct,dns_resolve_ms"


def _metrics_conn_pool(rng: _random.Random) -> list[str]:
    rows = [_METRICS_HEADER]
    for minute in range(0, 90):
        ts = _gen_ts_short(minute)
        if minute < 30:
            rows.append(f"{ts},{rng.randint(80,150)},{round(rng.random()*0.5,1)},{rng.randint(25,40)},{rng.randint(55,65)},{rng.randint(40,55)},{rng.randint(1,3)}")
        elif minute < 45:
            rows.append(f"{ts},{rng.randint(200,600)},{round(rng.random()*5,1)},{rng.randint(45,70)},{rng.randint(60,72)},{rng.randint(40,55)},{rng.randint(1,3)}")
        elif minute < 60:
            rows.append(f"{ts},{rng.randint(5000,10000)},{round(30+rng.random()*40,1)},{rng.randint(30,50)},{rng.randint(65,75)},{rng.randint(40,55)},{rng.randint(1,3)}")
        else:
            rows.append(f"{ts},{rng.randint(80,200)},{round(rng.random(),1)},{rng.randint(30,50)},{rng.randint(60,70)},{rng.randint(40,55)},{rng.randint(1,3)}")
    return rows


def _metrics_memory_leak(rng: _random.Random) -> list[str]:
    rows = [_METRICS_HEADER]
    for minute in range(0, 90):
        ts = _gen_ts_short(minute)
        mem = min(25 + minute, 98) if minute < 55 else rng.randint(22, 35)
        if minute < 30:
            rows.append(f"{ts},{rng.randint(40,150)},{round(rng.random()*0.5,1)},{rng.randint(20,35)},{mem},{rng.randint(40,50)},{rng.randint(1,3)}")
        elif minute < 50:
            rows.append(f"{ts},{rng.randint(150,1200)},{round(rng.random()*8,1)},{rng.randint(40,65)},{mem},{rng.randint(40,50)},{rng.randint(1,3)}")
        elif minute < 60:
            rows.append(f"{ts},{rng.randint(5000,15000)},{round(40+rng.random()*50,1)},{rng.randint(10,25)},{mem},{rng.randint(40,50)},{rng.randint(1,3)}")
        else:
            rows.append(f"{ts},{rng.randint(40,180)},{round(rng.random(),1)},{rng.randint(20,35)},{mem},{rng.randint(40,50)},{rng.randint(1,3)}")
    return rows


def _metrics_disk_full(rng: _random.Random) -> list[str]:
    rows = [_METRICS_HEADER]
    for minute in range(0, 90):
        ts = _gen_ts_short(minute)
        disk = min(70 + minute, 100) if minute < 65 else rng.randint(25, 40)
        if minute < 35:
            rows.append(f"{ts},{rng.randint(30,100)},{round(rng.random()*0.3,1)},{rng.randint(15,30)},{rng.randint(45,55)},{disk},{rng.randint(1,3)}")
        elif minute < 50:
            rows.append(f"{ts},{rng.randint(100,500)},{round(rng.random()*5,1)},{rng.randint(15,35)},{rng.randint(45,55)},{disk},{rng.randint(1,3)}")
        elif minute < 65:
            rows.append(f"{ts},{rng.randint(500,3000)},{round(20+rng.random()*30,1)},{rng.randint(15,35)},{rng.randint(45,55)},{disk},{rng.randint(1,3)}")
        else:
            rows.append(f"{ts},{rng.randint(30,120)},{round(rng.random(),1)},{rng.randint(15,30)},{rng.randint(45,55)},{disk},{rng.randint(1,3)}")
    return rows


def _metrics_dns_failure(rng: _random.Random) -> list[str]:
    rows = [_METRICS_HEADER]
    for minute in range(0, 90):
        ts = _gen_ts_short(minute)
        dns = rng.randint(1, 5) if minute < 40 or minute >= 55 else rng.randint(4000, 5001)
        if minute < 40:
            rows.append(f"{ts},{rng.randint(20,100)},{round(rng.random()*0.3,1)},{rng.randint(15,30)},{rng.randint(50,60)},{rng.randint(40,50)},{dns}")
        elif minute < 55:
            rows.append(f"{ts},{rng.randint(5000,10000)},{round(40+rng.random()*40,1)},{rng.randint(15,30)},{rng.randint(50,60)},{rng.randint(40,50)},{dns}")
        else:
            rows.append(f"{ts},{rng.randint(20,120)},{round(rng.random(),1)},{rng.randint(15,30)},{rng.randint(50,60)},{rng.randint(40,50)},{dns}")
    return rows


def _metrics_cert_expiry(rng: _random.Random) -> list[str]:
    rows = [_METRICS_HEADER]
    for minute in range(0, 90):
        ts = _gen_ts_short(minute)
        if minute < 38:
            rows.append(f"{ts},{rng.randint(15,80)},{round(rng.random()*0.2,1)},{rng.randint(10,25)},{rng.randint(40,55)},{rng.randint(30,45)},{rng.randint(1,3)}")
        elif minute < 55:
            rows.append(f"{ts},{rng.randint(50,200)},{round(60+rng.random()*35,1)},{rng.randint(10,25)},{rng.randint(40,55)},{rng.randint(30,45)},{rng.randint(1,3)}")
        else:
            rows.append(f"{ts},{rng.randint(15,100)},{round(rng.random(),1)},{rng.randint(10,25)},{rng.randint(40,55)},{rng.randint(30,45)},{rng.randint(1,3)}")
    return rows


def _metrics_deadlock(rng: _random.Random) -> list[str]:
    rows = [_METRICS_HEADER]
    for minute in range(0, 90):
        ts = _gen_ts_short(minute)
        if minute < 35:
            rows.append(f"{ts},{rng.randint(10,80)},{round(rng.random()*0.3,1)},{rng.randint(20,35)},{rng.randint(50,60)},{rng.randint(40,50)},{rng.randint(1,3)}")
        elif minute < 50:
            rows.append(f"{ts},{rng.randint(500,5000)},{round(5+rng.random()*20,1)},{rng.randint(25,45)},{rng.randint(50,62)},{rng.randint(40,50)},{rng.randint(1,3)}")
        elif minute < 60:
            rows.append(f"{ts},{rng.randint(5000,15000)},{round(30+rng.random()*40,1)},{rng.randint(25,45)},{rng.randint(50,62)},{rng.randint(40,50)},{rng.randint(1,3)}")
        else:
            rows.append(f"{ts},{rng.randint(10,80)},{round(rng.random(),1)},{rng.randint(20,35)},{rng.randint(50,60)},{rng.randint(40,50)},{rng.randint(1,3)}")
    return rows


# ---------------------------------------------------------------------------
# Per-scenario rubric builders
# ---------------------------------------------------------------------------

def _rubric_conn_pool() -> list[BinaryRubricCategory | RubricCategory]:
    return [
        BinaryRubricCategory(
            name="identifies_root_cause",
            question="Does the analysis correctly identify connection pool exhaustion as the root cause (not CPU, memory, GC, or network)?",
            points=3,
        ),
        BinaryRubricCategory(
            name="mentions_pool_size",
            question="Does the analysis mention the specific pool size limit (max_connections=20) as being insufficient?",
            points=3,
        ),
        BinaryRubricCategory(
            name="timeline_includes_onset",
            question="Does the timeline identify the approximate onset of errors (around minute 45)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_resolution",
            question="Does the timeline identify the resolution (pool size increased to 50)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_warning",
            question="Does the timeline note the early warning (pool utilization >90%)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="cites_specific_error",
            question="Does the analysis cite a specific error message (e.g., 'ConnectionPoolTimeoutException' or 'pool exhausted')?",
            points=2,
        ),
        BinaryRubricCategory(
            name="does_not_blame_gc",
            question="Does the analysis correctly NOT identify the GC pause as the root cause?",
            points=2,
        ),
    ]


def _rubric_memory_leak() -> list[BinaryRubricCategory | RubricCategory]:
    return [
        BinaryRubricCategory(
            name="identifies_root_cause",
            question="Does the analysis correctly identify a memory leak (leading to OOM kill) as the root cause (not GC, CPU, or network)?",
            points=3,
        ),
        BinaryRubricCategory(
            name="identifies_oom_kill",
            question="Does the analysis mention the OOM kill / kernel kill event?",
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_heap_growth",
            question="Does the analysis note the steadily increasing heap/memory usage pattern before the crash?",
            points=3,
        ),
        BinaryRubricCategory(
            name="timeline_includes_onset",
            question="Does the timeline identify the approximate onset of degradation (around minute 30-40)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_crash",
            question="Does the timeline identify the OOM crash event (around minute 55)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_recovery",
            question="Does the timeline note both the service restart and the hotfix deployment?",
            points=2,
        ),
        BinaryRubricCategory(
            name="does_not_blame_network",
            question="Does the analysis correctly NOT blame the TCP retransmit note as the root cause?",
            points=2,
        ),
    ]


def _rubric_disk_full() -> list[BinaryRubricCategory | RubricCategory]:
    return [
        BinaryRubricCategory(
            name="identifies_root_cause",
            question="Does the analysis correctly identify disk full (/var/log partition) as the root cause (not network, CPU, or memory)?",
            points=3,
        ),
        BinaryRubricCategory(
            name="identifies_partition",
            question="Does the analysis specifically identify /var/log as the affected partition?",
            points=2,
        ),
        BinaryRubricCategory(
            name="identifies_unrotated_logs",
            question="Does the analysis connect the disk full condition to unrotated or excessive debug logs?",
            points=3,
        ),
        BinaryRubricCategory(
            name="timeline_includes_growth",
            question="Does the timeline note the steady disk usage growth before the failure?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_full",
            question="Does the timeline identify when disk reached 100% (around minute 50)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_cleanup",
            question="Does the timeline note the cleanup/recovery actions (log purge, logrotate)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="does_not_blame_network",
            question="Does the analysis correctly NOT blame the NIC dropped packets as the root cause?",
            points=2,
        ),
    ]


def _rubric_dns_failure() -> list[BinaryRubricCategory | RubricCategory]:
    return [
        BinaryRubricCategory(
            name="identifies_root_cause",
            question="Does the analysis correctly identify DNS resolver failure as the root cause (not application bugs, CPU, memory, or network hardware)?",
            points=3,
        ),
        BinaryRubricCategory(
            name="identifies_resolver",
            question="Does the analysis mention the specific DNS resolver (10.0.0.2 / coredns) as the failing component?",
            points=3,
        ),
        BinaryRubricCategory(
            name="notes_all_services_affected",
            question="Does the analysis note that multiple upstream services were affected (not just one)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_onset",
            question="Does the timeline identify the approximate onset (around minute 40)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_resolution",
            question="Does the timeline identify the resolution (resolver restart + failover added)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="cites_dns_metric",
            question="Does the analysis cite the elevated DNS resolution time from the metrics?",
            points=2,
        ),
        BinaryRubricCategory(
            name="does_not_blame_gc",
            question="Does the analysis correctly NOT blame the GC pause as the root cause?",
            points=2,
        ),
    ]


def _rubric_cert_expiry() -> list[BinaryRubricCategory | RubricCategory]:
    return [
        BinaryRubricCategory(
            name="identifies_root_cause",
            question="Does the analysis correctly identify TLS certificate expiry as the root cause (not application bugs, DNS, or memory)?",
            points=3,
        ),
        BinaryRubricCategory(
            name="identifies_cert_detail",
            question="Does the analysis mention the expired certificate for *.internal.corp or the specific expiry time?",
            points=3,
        ),
        BinaryRubricCategory(
            name="notes_early_warning",
            question="Does the analysis note the debug log warning about imminent certificate expiry?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_onset",
            question="Does the timeline identify the exact onset (around minute 38 when the cert expired)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_resolution",
            question="Does the timeline identify the resolution (certificate renewed)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="does_not_blame_cache",
            question="Does the analysis correctly NOT blame the auth cache miss rate as the root cause (it was a symptom)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="does_not_blame_gc",
            question="Does the analysis correctly NOT blame the GC pause as the root cause?",
            points=2,
        ),
    ]


def _rubric_deadlock() -> list[BinaryRubricCategory | RubricCategory]:
    return [
        BinaryRubricCategory(
            name="identifies_root_cause",
            question="Does the analysis correctly identify database deadlock (circular lock dependency) as the root cause (not network latency, CPU, or memory)?",
            points=3,
        ),
        BinaryRubricCategory(
            name="identifies_tables",
            question="Does the analysis identify the specific tables involved (inventory_items and stock_reservations)?",
            points=3,
        ),
        BinaryRubricCategory(
            name="identifies_lock_ordering",
            question="Does the analysis mention inconsistent lock ordering or the lock ordering fix?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_onset",
            question="Does the timeline identify the approximate onset of lock waits (around minute 35)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="timeline_includes_resolution",
            question="Does the timeline identify the resolution (deadlocked sessions killed + lock ordering fix)?",
            points=2,
        ),
        BinaryRubricCategory(
            name="cites_deadlock_message",
            question="Does the analysis cite a specific deadlock detection message from the logs?",
            points=2,
        ),
        BinaryRubricCategory(
            name="does_not_blame_replica",
            question="Does the analysis correctly NOT blame the replica connection latency as the root cause?",
            points=2,
        ),
    ]


# ---------------------------------------------------------------------------
# Dispatch tables
# ---------------------------------------------------------------------------

_LOG_GENERATORS = {
    "conn_pool": _logs_conn_pool,
    "memory_leak": _logs_memory_leak,
    "disk_full": _logs_disk_full,
    "dns_failure": _logs_dns_failure,
    "cert_expiry": _logs_cert_expiry,
    "deadlock": _logs_deadlock,
}

_METRIC_GENERATORS = {
    "conn_pool": _metrics_conn_pool,
    "memory_leak": _metrics_memory_leak,
    "disk_full": _metrics_disk_full,
    "dns_failure": _metrics_dns_failure,
    "cert_expiry": _metrics_cert_expiry,
    "deadlock": _metrics_deadlock,
}

_RUBRIC_BUILDERS = {
    "conn_pool": _rubric_conn_pool,
    "memory_leak": _rubric_memory_leak,
    "disk_full": _rubric_disk_full,
    "dns_failure": _rubric_dns_failure,
    "cert_expiry": _rubric_cert_expiry,
    "deadlock": _rubric_deadlock,
}


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------


def make_incident_root_cause(rand_seed: int = 77) -> RubricDatapoint:
    """Incident Root Cause Analysis from logs and metrics.

    Seed selects from 6 structurally distinct scenarios, each with unique
    log patterns, metric signatures, red herrings, and resolution events.
    """
    rng = _random.Random(rand_seed)

    # Select scenario
    scenario = rng.choice(INCIDENT_SCENARIOS)
    key = scenario["key"]
    svc = scenario["service"]

    # Generate logs
    app_lines, lb_lines = _LOG_GENERATORS[key](rng, svc)
    app_log = "\n".join(app_lines) + "\n"
    lb_log = "\n".join(lb_lines) + "\n"

    # Generate metrics
    metrics_rows = _METRIC_GENERATORS[key](rng)
    metrics_csv = "\n".join(metrics_rows) + "\n"

    # Build rubric
    rubric_items: list[BinaryRubricCategory | RubricCategory] = [
        BinaryRubricCategory(
            name="file_exists",
            question="Does /testbed/rca.txt exist with substantial content?",
            points=1,
        ),
    ]
    rubric_items.extend(_RUBRIC_BUILDERS[key]())
    rubric_items.extend([
        BinaryRubricCategory(
            name="cites_metric_value",
            question="Does the analysis cite at least one specific metric value from the CSV (e.g., error rate %, latency p99, or resource usage during the incident)?",
            points=2,
        ),
        RubricCategory(
            name="cause_vs_symptom_distinction",
            description="Does the analysis distinguish the root cause from symptoms?",
            failure="Conflates symptoms (HTTP errors, high latency) with the root cause",
            minor_failure="Mentions both but doesn't clearly separate them",
            minor_success="Clearly identifies symptoms vs cause but explanation could be sharper",
            success="Crisp distinction between root cause, symptoms, trigger, and contributing factors",
            points=3,
        ),
        RubricCategory(
            name="recommendations_quality",
            description="Are the recommendations specific and actionable?",
            failure="No recommendations or only 'fix it'",
            minor_failure="Generic recommendations ('add monitoring')",
            minor_success="Specific recommendations but missing some obvious preventive measures",
            success="Specific, prioritized recommendations addressing both immediate fix and long-term prevention",
            points=3,
        ),
    ])

    return RubricDatapoint(
        problem_statement=f"""# Incident Root Cause Analysis

A production incident occurred on 2024-03-15 affecting the {svc}
service. During the incident, {scenario['description']}.

You have access to:
- /testbed/logs/application.log — Application-level logs
- /testbed/logs/loadbalancer.log — Load balancer access logs
- /testbed/monitoring/metrics.csv — Time-series metrics (1-min intervals)

Write a root cause analysis to /testbed/rca.txt that includes:

1. TIMELINE: Key events with timestamps
2. ROOT CAUSE: What specifically caused the failure
3. CONTRIBUTING FACTORS: What made it worse or delayed recovery
4. EVIDENCE: Specific log lines and metric values that support your diagnosis
5. RECOMMENDATIONS: How to prevent recurrence

Analyze the data carefully — distinguish root cause from symptoms.""",
        rubric=tuple(rubric_items),
        submission_instructions="Write your root cause analysis to /testbed/rca.txt",
        available_tools=(tools.BASH_TOOL, tools.CREATE_FILE_TOOL, tools.EDIT_TOOL, tools.LIST_DIRECTORY_CONTENTS_TOOL, tools.FINISH_TOOL),
        necessary_files={
            "/testbed/logs/application.log": app_log,
            "/testbed/logs/loadbalancer.log": lb_log,
            "/testbed/monitoring/metrics.csv": metrics_csv,
        },
        problem_type="incident_analysis",
    )
