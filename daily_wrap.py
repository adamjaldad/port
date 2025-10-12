#!/usr/bin/env python3
# portfolio_daily_email.py

import os
import io
import ssl
import time
import math
import smtplib
import requests
import pandas as pd
from datetime import datetime, timedelta
from email.message import EmailMessage
from email.utils import formatdate
from zoneinfo import ZoneInfo

# ============================== CONFIG ==============================
PORTFOLIO = pd.DataFrame({
    "Ticker" : ["HOOD","UNH","GLD","RKLB","NVDA","HIMS","SOFI","ASTS","AAPL","NUKZ","TSLA","NVTS","INTC","DUOL","PM","OKLO","META","UPS","GOOGL","GRAB","RDDT","OSCR","SPY","QQQ"],
    "Shares" : [100,20,10,50,15,50,100,30,10,35,5,200,45,5,10,10,2,15,5,200,5,50,1,1],
}).set_index("Ticker")

BENCH = ["SPY","QQQ", "IWM"]

POLYGON_API_KEY = "zzlQUn41Goxb1mQR0lA1odVKLQ6pblFZ"
EMAIL_USER      = "adamjaldad@gmail.com"
EMAIL_PASS      = "tiql gusr ffnm holi"  # Gmail App Password
EMAIL_TO        = os.getenv("EMAIL_TO", "adamjaldad@gmail.com")
EMAIL_FROM      = EMAIL_USER 

POLY_BASE = "https://api.polygon.io"

# Rate limits (Polygon free tier is tight; be gentle)
REQ_SLEEP_SEC = 1.0
RETRY_ON_429_SEC = 65

TZ = ZoneInfo("America/Chicago")

# =========================== UTIL FORMATTING =========================
def fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"{x:.{digits}f}%"

def fmt_usd(x, digits=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    return f"${x:,.{digits}f}"

# =========================== POLYGON HELPERS =========================
def _poly_get(session: requests.Session, path: str, params: dict | None = None):
    """GET helper that appends apiKey, retries on 429, and returns JSON or None for 404."""
    p = dict(params or {})
    p["apiKey"] = POLYGON_API_KEY
    url = f"{POLY_BASE}{path}"
    while True:
        r = session.get(url, params=p, timeout=20)
        if r.status_code == 429:
            time.sleep(RETRY_ON_429_SEC)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

def _last_two_daily_bars(ticker: str, session: requests.Session):
    """
    Return the last two trading-day bars (adjusted). Works on weekends/holidays by
    querying a 14-day window and taking the most recent two bars.
    """
    end = datetime.now(tz=TZ).date()
    start = end - timedelta(days=14)
    path = f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    data = _poly_get(session, path, {"adjusted": "true", "limit": 250})
    time.sleep(REQ_SLEEP_SEC)
    results = (data or {}).get("results") or []
    if len(results) < 2:
        return []
    return results[-2:]  # [prev, last]

def robust_price_table_nearest_market_day(tickers: list[str]) -> pd.DataFrame:
    """
    Price = close of most recent trading day; PrevClose = close of day before.
    DayChgPct = close/close move for the most recent market day.
    Adds df.attrs['session_as_of'] = 'YYYY-MM-DD'.
    """
    rows = {}
    session_ts_max = 0  # ms epoch
    with requests.Session() as s:
        for t in tickers:
            try:
                bars2 = _last_two_daily_bars(t, s)  # [prev, last]
                if len(bars2) == 2:
                    prev_bar, last_bar = bars2[0], bars2[1]
                    price = float(last_bar["c"])
                    prevc = float(prev_bar["c"])
                    session_ts_max = max(session_ts_max, int(last_bar.get("t", 0)))
                else:
                    price, prevc = (None, None)
            except Exception:
                price, prevc = (None, None)
            rows[t] = {"Price": price, "PrevClose": prevc}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["DayChgPct"] = (df["Price"] / df["PrevClose"] - 1.0) * 100.0
    df.attrs["session_as_of"] = (
        datetime.fromtimestamp(session_ts_max/1000, tz=TZ).strftime("%Y-%m-%d")
        if session_ts_max else None
    )
    return df

# ============================ REPORTING ==============================
def build_summary(port: pd.DataFrame, px: pd.DataFrame, session_as_of: str | None) -> dict:
    """Compute allocations, P&L, movers, contributors, and benchmarks."""
    df = port.join(px, how="left")

    # Compute only on rows with both closes
    valid = df["Price"].notna() & df["PrevClose"].notna()
    df.loc[valid, "MarketValue"] = df.loc[valid, "Shares"] * df.loc[valid, "Price"]
    df.loc[valid, "PrevValue"]   = df.loc[valid, "Shares"] * df.loc[valid, "PrevClose"]
    df.loc[valid, "DayPnL"]      = df.loc[valid, "MarketValue"] - df.loc[valid, "PrevValue"]
    df.loc[~valid, ["MarketValue","PrevValue","DayPnL"]] = float("nan")

    total_mv = float(df["MarketValue"].sum(skipna=True))
    total_pv = float(df["PrevValue"].sum(skipna=True))
    day_pnl  = total_mv - total_pv
    day_pct  = (day_pnl / total_pv * 100.0) if total_pv else float("nan")

    df["Weight"] = df["MarketValue"] / total_mv if total_mv else float("nan")

    # Movers (by %)
    movers = df[df["DayChgPct"].notna()].copy()
    top_up = movers.sort_values("DayChgPct", ascending=False).head(5)
    top_dn = movers.sort_values("DayChgPct", ascending=True).head(5)

    # Contributors (by P&L)
    contrib = df[df["DayPnL"].notna()].copy().sort_values("DayPnL", ascending=False)
    top_contrib = contrib.head(5)
    worst_contrib = contrib.tail(5)

    # Benchmarks on the same “nearest market day” logic
    bench_px = robust_price_table_nearest_market_day(BENCH)
    bench_px["DayChgPct"] = (bench_px["Price"] / bench_px["PrevClose"] - 1.0) * 100.0
    bench = {t: float(bench_px.loc[t, "DayChgPct"]) for t in BENCH if t in bench_px.index}

    # Flags
    flags = []
    if abs(day_pct) > 2.0:
        flags.append(f"Portfolio move is large: {fmt_pct(day_pct)}.")
    if (df["DayChgPct"].abs() > 5.0).sum() >= max(1, len(df)//6):
        flags.append("Many names moved >5% today.")
    if df["Price"].isna().any():
        flags.append("Some symbols missing quotes (API limits or tickers?).")

    as_of_dt = (
        datetime.strptime(session_as_of, "%Y-%m-%d").replace(tzinfo=TZ)
        if session_as_of else datetime.now(TZ)
    )

    return {
        "as_of": as_of_dt,
        "detail": df,
        "totals": {
            "market_value": total_mv,
            "day_pnl": day_pnl,
            "day_pct": day_pct,
        },
        "top_up": top_up,
        "top_dn": top_dn,
        "top_contrib": top_contrib,
        "worst_contrib": worst_contrib,
        "bench": bench,
        "flags": flags,
    }

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.reset_index().to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

def make_html_email(report: dict) -> str:
    as_of = report["as_of"].strftime("%Y-%m-%d %I:%M %p %Z")
    totals = report["totals"]
    bench = report["bench"]

    def table_html(df: pd.DataFrame, columns: list[tuple[str,str]]):
        rows = []
        rows.append("<thead><tr>" + "".join(
            f"<th style='text-align:left;padding:6px 10px;border-bottom:1px solid #ddd'>{h}</th>" for _,h in columns
        ) + "</tr></thead>")
        rows.append("<tbody>")
        for _, r in df.iterrows():
            cells = []
            for c, _ in columns:
                val = r[c]
                if c in ("DayChgPct","Weight"):
                    cells.append(fmt_pct(val))
                elif c in ("MarketValue","PrevValue","DayPnL","Price","PrevClose"):
                    cells.append(fmt_usd(val))
                else:
                    cells.append(str(val) if not isinstance(val, float) or not math.isnan(val) else "—")
            rows.append("<tr>" + "".join(f"<td style='padding:6px 10px'>{v}</td>" for v in cells) + "</tr>")
        rows.append("</tbody>")
        return "<table style='border-collapse:collapse;font-family:system-ui,Segoe UI,Arial,sans-serif;font-size:13px'>" + "".join(rows) + "</table>"

    cols_core = [("Shares","Shares"),("Price","Price"),("PrevClose","Prev Close"),
                 ("DayChgPct","Day %"),("MarketValue","Mkt Value"),("DayPnL","Day P&L"),("Weight","Weight")]

    top_up = report["top_up"].reset_index().rename(columns={"index":"Ticker"})
    top_dn = report["top_dn"].reset_index().rename(columns={"index":"Ticker"})
    top_contrib = report["top_contrib"].reset_index().rename(columns={"index":"Ticker"})
    worst_contrib = report["worst_contrib"].reset_index().rename(columns={"index":"Ticker"})

    html_top_up  = table_html(top_up[["Ticker"] + [c for c,_ in cols_core]],  [("Ticker","Ticker")] + cols_core) if not top_up.empty else "<p>—</p>"
    html_top_dn  = table_html(top_dn[["Ticker"] + [c for c,_ in cols_core]],  [("Ticker","Ticker")] + cols_core) if not top_dn.empty else "<p>—</p>"
    html_contrib = table_html(top_contrib[["Ticker","Shares","Price","DayPnL","MarketValue","Weight"]],
                              [("Ticker","Ticker"),("Shares","Shares"),("Price","Price"),("DayPnL","Day P&L"),
                               ("MarketValue","Mkt Value"),("Weight","Weight")]) if not top_contrib.empty else "<p>—</p>"
    html_worst   = table_html(worst_contrib[["Ticker","Shares","Price","DayPnL","MarketValue","Weight"]],
                              [("Ticker","Ticker"),("Shares","Shares"),("Price","Price"),("DayPnL","Day P&L"),
                               ("MarketValue","Mkt Value"),("Weight","Weight")]) if not worst_contrib.empty else "<p>—</p>"

    bench_line = " | ".join(f"{b}: {fmt_pct(v)}" for b, v in bench.items()) if bench else "—"
    flags_html = "".join(f"<li>{f}</li>" for f in report["flags"]) if report["flags"] else "<li>No alerts.</li>"

    return f"""
<html>
  <body style="font-family:system-ui,Segoe UI,Arial,sans-serif;color:#111">
    <div style="max-width:900px;margin:auto">
      <h2 style="margin-bottom:0.2em">Daily Portfolio Summary</h2>
      <div style="color:#666;margin-bottom:12px">As of {as_of}</div>

      <div style="padding:10px;border:1px solid #eee;border-radius:8px;margin-bottom:14px">
        <b>Total Market Value:</b> {fmt_usd(totals['market_value'])} &nbsp;•&nbsp;
        <b>Day P&L:</b> {fmt_usd(totals['day_pnl'])} &nbsp;•&nbsp;
        <b>Day %:</b> {fmt_pct(totals['day_pct'])} &nbsp;•&nbsp;
        <b>Benchmarks:</b> {bench_line}
      </div>

      <h3 style="margin:18px 0 6px">Top Movers (by % up)</h3>
      {html_top_up}

      <h3 style="margin:18px 0 6px">Top Movers (by % down)</h3>
      {html_top_dn}

      <h3 style="margin:18px 0 6px">Top Contributors (P&L)</h3>
      {html_contrib}

      <h3 style="margin:18px 0 6px">Worst Contributors (P&L)</h3>
      {html_worst}

      <h3 style="margin:18px 0 6px">Flags</h3>
      <ul style="margin-top:4px">{flags_html}</ul>

      <div style="margin-top:16px;color:#666;font-size:12px">
        Source: Polygon.io • Session stats are close→close for the latest market day. CSV with full detail attached.
      </div>
    </div>
  </body>
</html>
"""

# ============================== EMAIL ===============================
def send_email(subject: str, html_body: str, csv_bytes: bytes, filename: str):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Date"] = formatdate(localtime=True)
    msg.set_content("Your daily portfolio summary is attached. (Open the HTML version for tables.)")
    msg.add_alternative(html_body, subtype="html")

    if csv_bytes:
        msg.add_attachment(csv_bytes, maintype="text", subtype="csv", filename=filename)

    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=60) as server:
        server.starttls(context=context)
        server.login(EMAIL_USER, EMAIL_PASS)
        server.send_message(msg)

# =============================== MAIN ===============================
def main():
    if not POLYGON_API_KEY:
        raise RuntimeError("Set POLYGON_API_KEY.")
    if not EMAIL_USER or not EMAIL_PASS:
        raise RuntimeError("Set EMAIL_USER and EMAIL_PASS (Gmail app password).")

    tickers = list(PORTFOLIO.index.unique())

    # Weekend/holiday-proof prices
    prices = robust_price_table_nearest_market_day(tickers)
    session_as_of = prices.attrs.get("session_as_of")

    report = build_summary(PORTFOLIO, prices, session_as_of)

    # CSV attachment (full detail)
    csv_bytes = dataframe_to_csv_bytes(report["detail"])

    # Subject
    ts = report["as_of"].strftime("%Y-%m-%d")
    mv = fmt_usd(report["totals"]["market_value"])
    dp = fmt_usd(report["totals"]["day_pnl"])
    dpp = fmt_pct(report["totals"]["day_pct"])
    subject = f"Daily Portfolio: MV {mv} | Day {dp} ({dpp}) • {ts}"

    html = make_html_email(report)
    send_email(subject, html, csv_bytes, filename=f"portfolio_daily_{ts}.csv")

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.reset_index().to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback, sys
        print("ERROR:", e, file=sys.stderr)
        traceback.print_exc()
        raise
