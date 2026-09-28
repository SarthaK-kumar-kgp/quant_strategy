import json
from pathlib import Path

from config import REGISTRY_PATH


class WhaleRegistry:
    """
    Persistent dictionary of wallets we actively track.
    Backed by tracked_wallets.json — survives restarts.
    Updated whenever a wallet is promoted (leaderboard seed or organic discovery).
    """

    def __init__(self, path: str = REGISTRY_PATH):
        self._path = Path(path)
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            with open(self._path) as f:
                self._data = json.load(f)
            print(f"[Registry] Loaded {len(self._data)} tracked wallets from {self._path}")

    def _save(self):
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    # ── Mutations ──────────────────────────────────────────────────────────────

    def add(self, address: str, score: dict, source: str = "organic"):
        self._data[address] = {**score, "source": source}
        self._save()
        print(
            f"[Registry] + {address[:10]}...  "
            f"z={score['z_score']:.2f}  pnl=${score['pnl']:,.0f}  "
            f"trades={score['trade_count']}  source={source}"
        )

    def remove(self, address: str):
        if address in self._data:
            del self._data[address]
            self._save()
            print(f"[Registry] - {address[:10]}... removed")

    def refresh(self, address: str, score: dict):
        if address in self._data:
            source = self._data[address].get("source", "organic")
            self._data[address] = {**score, "source": source}
            self._save()

    # ── Reads ──────────────────────────────────────────────────────────────────

    def is_tracked(self, address: str) -> bool:
        return address in self._data

    def get(self, address: str) -> dict:
        return self._data.get(address, {})

    def get_all(self) -> list[dict]:
        return list(self._data.values())

    def qualified_clusters(self, address: str) -> list[str]:
        wallet = self._data.get(address, {})
        return [
            cluster
            for cluster, info in wallet.get("clusters", {}).items()
            if info.get("qualified")
        ]

    def summary(self) -> str:
        total  = len(self._data)
        qual   = sum(1 for w in self._data.values() if w.get("qualified"))
        return f"{total} tracked  |  {qual} fully qualified (z > {2.5}, N >= 50)"
