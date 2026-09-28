# ── Polymarket API endpoints ───────────────────────────────────────────────────
CLOB_API  = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API  = "https://data-api.polymarket.com"

# ── CLOB L2 credentials ────────────────────────────────────────────────────────
CLOB_API_KEY    = "287cb227-d29f-02e5-a657-68e567998a37"
CLOB_SECRET     = "dXPeciyRbgaohGsLu-gwHvWb08SDhIkleJowIPc8Gkw="
CLOB_PASSPHRASE = "f63e919cce5e723d1bf6982d1f9d252488735e000eb765f1b2eb246fe99b27d6"
CLOB_WALLET     = "0x4f8960d35078728dcd31154bba94771d4fb3a8d2"

# ── HTTP client ────────────────────────────────────────────────────────────────
MAX_CONCURRENT_REQUESTS = 8
REQUEST_TIMEOUT_S       = 15
PAGE_SIZE               = 500          # trades per page when paginating

# ── Wallet scoring thresholds ──────────────────────────────────────────────────
MIN_RESOLVED_TRADES = 50               # N >= 50 required to qualify
EDGE_Z_THRESHOLD    = 2.5              # edge / SE must exceed this

# ── Signal detection ───────────────────────────────────────────────────────────
MIN_WHALE_TRADE_USD    = 5_000         # minimum trade size to trigger a signal
SIGNAL_POLL_INTERVAL_S = 5            # seconds between polls

# ── Organic discovery ──────────────────────────────────────────────────────────
ORGANIC_POLL_INTERVAL_S = 10
ORGANIC_MIN_TRADE_USD   = 1_000       # track wallets trading at least this much

# ── File paths ─────────────────────────────────────────────────────────────────
REGISTRY_PATH = "tracked_wallets.json"
SIGNALS_CSV   = "signals.csv"
DATA_DIR      = "data"

# ── Insider strategy ───────────────────────────────────────────────────────────
IS_MIN_THRESHOLD            = 8.0    # composite insider score trigger
SIZE_ZSCORE_TRIGGER         = 4.0    # size z-score must exceed this before computing IS
BASELINE_LOOKBACK_DAYS      = 7      # days of history to warm baselines on startup
OFI_WINDOW_MINUTES          = 60     # rolling window for order flow imbalance

# confirmation gate
CONFIRMATION_WINDOW_S       = 300    # 5-minute window to find a confirming trade
CONFIRMATION_MIN_SIZE_RATIO = 0.10   # confirming trade must be >= 10% of original size

# wallet novelty thresholds (each adds 1 to novelty score, max = 3)
NOVELTY_WALLET_AGE_DAYS     = 30     # wallet first seen < 30 days ago = "new"
NOVELTY_FUNDING_DAYS        = 7      # first trade < 7 days ago = "recently funded"

# WebSocket
WS_URL                      = "wss://ws-subscriptions-clob.polymarket.com/ws/"
WS_POLL_FALLBACK_INTERVAL_S = 2      # fast-poll interval if WebSocket unavailable

INSIDER_SIGNALS_CSV         = "insider_signals.csv"
