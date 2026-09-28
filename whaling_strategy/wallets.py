import requests

def get_top_wallets(
    order_by="PNL",        # "PNL" or "VOL"
    time_period="DAY",     # "DAY", "WEEK", "MONTH", "ALL"
    category="OVERALL",    # "OVERALL", "POLITICS", "SPORTS", "CRYPTO", etc.
    limit=25               # max 50
):
    url = "https://data-api.polymarket.com/v1/leaderboard"
    params = {
        "orderBy": order_by,
        "timePeriod": time_period,
        "category": category,
        "limit": limit,
        "offset": 0
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    traders = r.json()

    print(f"\n🏆 Top {len(traders)} wallets by {order_by} ({time_period}) — {category}")
    print("-" * 80)
    for t in traders:
        rank     = t.get("rank", "?")
        wallet   = t.get("proxyWallet", "")
        username = t.get("userName") or t.get("xUsername") or "anon"
        pnl      = t.get("pnl", 0)
        vol      = t.get("vol", 0)
        print(f"#{rank:<4} {username:<20} PnL: ${pnl:>12,.2f}   Vol: ${vol:>12,.2f}   {wallet}")

    return traders

# --- Run it ---
if __name__ == "__main__":
    # Top 25 by profit today
    traders = get_top_wallets(order_by="PNL", time_period="DAY")

    # Uncomment for other views:
    # get_top_wallets(order_by="VOL", time_period="WEEK")
    # get_top_wallets(order_by="PNL", time_period="ALL", limit=50)
    # get_top_wallets(order_by="PNL", time_period="MONTH", category="POLITICS")