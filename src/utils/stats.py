import json
import threading
from datetime import datetime
from pathlib import Path

STATS_FILE = Path(__file__).resolve().parent.parent.parent / "data" / "stats.json"
_lock = threading.Lock()


def _load() -> dict:
    if STATS_FILE.exists():
        try:
            return json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "total_queries": 0,
        "web_queries": 0,
        "telegram_queries": 0,
        "total_visits": 0,
        "unique_daily_visits": {},
        "recent_hits": [],
    }


def _save(data: dict) -> None:
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def record_query(source: str = "web") -> None:
    with _lock:
        data = _load()
        data["total_queries"] = data.get("total_queries", 0) + 1
        if source == "telegram":
            data["telegram_queries"] = data.get("telegram_queries", 0) + 1
        else:
            data["web_queries"] = data.get("web_queries", 0) + 1
        _save(data)


def _prune_old_hits(data: dict) -> None:
    cutoff = datetime.utcnow().timestamp() - 60
    data["recent_hits"] = [
        ts for ts in data.get("recent_hits", [])
        if isinstance(ts, (int, float)) and ts > cutoff
    ]


def record_visit(client_ip: str | None = None) -> None:
    with _lock:
        data = _load()
        data["total_visits"] = data.get("total_visits", 0) + 1

        today = datetime.utcnow().strftime("%Y-%m-%d")
        daily = data.get("unique_daily_visits", {})
        if today not in daily:
            daily[today] = []
        if client_ip and client_ip not in daily[today]:
            daily[today].append(client_ip)
        data["unique_daily_visits"] = daily

        data.setdefault("recent_hits", []).append(datetime.utcnow().timestamp())
        _prune_old_hits(data)
        _save(data)


def get_stats() -> dict:
    with _lock:
        data = _load()
        daily = data.get("unique_daily_visits", {})
        total_unique_days = len(daily)
        total_unique_ips = sum(len(ips) for ips in daily.values())
        _prune_old_hits(data)
        _save(data)
        return {
            "total_queries": data.get("total_queries", 0),
            "web_queries": data.get("web_queries", 0),
            "telegram_queries": data.get("telegram_queries", 0),
            "total_visits": data.get("total_visits", 0),
            "unique_days_tracked": total_unique_days,
            "total_unique_ips": total_unique_ips,
            "online_now": len(data.get("recent_hits", [])),
        }
