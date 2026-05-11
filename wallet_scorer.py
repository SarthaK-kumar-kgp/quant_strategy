import math
from collections import defaultdict
from datetime import datetime

from config import EDGE_Z_THRESHOLD, MIN_RESOLVED_TRADES


class WalletScorer:
    """
    Scores a wallet based on resolved trade history.
    All methods are static — no state, pure computation.
    """

    @staticmethod
    def score(address: str, trades: list[dict]) -> dict:
        resolved_buys = _resolved_buys(trades)
        edge, se      = _edge_and_se(resolved_buys)
        z             = edge / se if se > 0 else 0.0

        return {
            "address":     address,
            "pnl":         round(_pnl(trades), 2),
            "edge":        round(edge, 4),
            "se":          round(se, 4),
            "z_score":     round(z, 4),
            "trade_count": len(resolved_buys),
            "qualified":   _qualified(len(resolved_buys), z),
            "clusters":    _cluster_scores(resolved_buys),
            "updated_at":  datetime.utcnow().isoformat(),
        }

    @staticmethod
    def qualified_clusters(score: dict) -> list[str]:
        return [
            cluster
            for cluster, info in score.get("clusters", {}).items()
            if info.get("qualified")
        ]


# ── Private computation helpers ────────────────────────────────────────────────

def _resolved_buys(trades: list[dict]) -> list[dict]:
    return [
        t for t in trades
        if t.get("type", "").upper() == "BUY"
        and t.get("resolved")
        and t.get("winner") is not None
    ]


def _pnl(trades: list[dict]) -> float:
    total = 0.0
    for t in trades:
        if not t.get("resolved") or t.get("winner") is None:
            continue
        cost       = float(t.get("amount_usd", 0) or 0)
        size       = float(t.get("size", 0) or 0)
        redemption = size if t["winner"] else 0.0
        total     += redemption - cost
    return total


def _edge_and_se(resolved_buys: list[dict]) -> tuple[float, float]:
    if not resolved_buys:
        return 0.0, float("inf")

    n      = len(resolved_buys)
    prices = [float(t.get("price", 0.5) or 0.5) for t in resolved_buys]
    won    = [1.0 if t["winner"] else 0.0 for t in resolved_buys]

    edge = sum(w - p for w, p in zip(won, prices)) / n
    # SE from the document: sqrt( mean(p*(1-p)) / n )
    se   = math.sqrt(sum(p * (1 - p) for p in prices) / n / n)
    return edge, se


def _qualified(n: int, z: float) -> bool:
    return n >= MIN_RESOLVED_TRADES and z >= EDGE_Z_THRESHOLD


def _cluster_scores(resolved_buys: list[dict]) -> dict:
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for t in resolved_buys:
        for tag in (t.get("tags") or ["Unknown"]):
            by_cluster[tag].append(t)

    result = {}
    for cluster, ctrades in by_cluster.items():
        edge, se = _edge_and_se(ctrades)
        z        = edge / se if se > 0 else 0.0
        n        = len(ctrades)
        result[cluster] = {
            "edge":        round(edge, 4),
            "se":          round(se, 4),
            "z_score":     round(z, 4),
            "trade_count": n,
            "qualified":   _qualified(n, z),
        }
    return result
