"""
Trade Picker — Pranav's interactive NSE stock analysis app.

Run locally:    streamlit run app.py
Deploy free:    push this repo to GitHub, then deploy on
                https://share.streamlit.io (Streamlit Community Cloud)

This app pulls LIVE data (yfinance for prices/fundamentals, Yahoo Finance
news headlines for sentiment) — it needs real internet access, which is why
it runs on Streamlit Cloud's servers rather than inside a chat sandbox.

This is an analysis and screening tool, not financial advice from a
licensed advisor. It applies a fixed, transparent rule set to public data.
You are still the one deciding what to do with the output.
"""

import datetime as dt
import io

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Trade Picker", layout="wide")
analyzer = SentimentIntensityAnalyzer()

# Fallback universe used if the live Nifty 500 constituent list can't be fetched
NIFTY50_FALLBACK = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", "HINDUNILVR.NS",
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
    "TATASTEEL.NS", "TATAMOTORS.NS", "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS",
    "HCLTECH.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "GRASIM.NS", "CIPLA.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "BRITANNIA.NS", "DIVISLAB.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "BPCL.NS",
    "TECHM.NS", "SBILIFE.NS", "HDFCLIFE.NS", "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "UPL.NS",
    "SHRIRAMFIN.NS", "LTIM.NS",
]

NIFTY_CSV_URLS = {
    "Nifty 50": "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "Nifty 500": "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
}

TIMEFRAMES = {
    "Daily": {"period": "1y", "interval": "1d"},
    "Weekly": {"period": "5y", "interval": "1wk"},
    "Monthly": {"period": "10y", "interval": "1mo"},
}


# ---------------------------------------------------------------- data ----
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_universe(index_name):
    """Pull live Nifty 50 / Nifty 500 constituents from NSE. Falls back to a
    hardcoded Nifty 50 list if the request fails (NSE sometimes blocks
    non-browser requests)."""
    url = NIFTY_CSV_URLS.get(index_name)
    if not url:
        return NIFTY50_FALLBACK, False
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        from io import StringIO
        csv_df = pd.read_csv(StringIO(resp.text))
        symbols = [f"{s.strip()}.NS" for s in csv_df["Symbol"].dropna().tolist()]
        if symbols:
            return symbols, True
    except Exception:
        pass
    return NIFTY50_FALLBACK, False


@st.cache_data(ttl=900, show_spinner=False)
def fetch_data(ticker, period="1y", interval="1d"):
    df = yf.Ticker(ticker).history(period=period, interval=interval)
    if df.empty:
        return None
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_news(ticker):
    try:
        return yf.Ticker(ticker).news[:8]
    except Exception:
        return []


def extract_news_link(item):
    """yfinance news items vary in shape across versions — handle both."""
    link = item.get("link")
    if link:
        return link
    content = item.get("content", {})
    canonical = content.get("canonicalUrl", {})
    if isinstance(canonical, dict) and canonical.get("url"):
        return canonical["url"]
    click = content.get("clickThroughUrl", {})
    if isinstance(click, dict) and click.get("url"):
        return click["url"]
    return None


# --------------------------------------------------------- indicators -----
def add_indicators(df):
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    df["RSI14"] = rsi

    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14).mean()

    df["VolAvg20"] = df["Volume"].rolling(20).mean()

    # ADX (trend strength)
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_smooth = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).sum() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).sum() / tr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX14"] = dx.rolling(14).mean()

    return df


def find_levels(df, window=10, tolerance=0.015):
    highs = df["High"].rolling(window, center=True).max()
    lows = df["Low"].rolling(window, center=True).min()
    swing_high = df["High"][df["High"] == highs]
    swing_low = df["Low"][df["Low"] == lows]

    def cluster(levels, tol):
        levels = sorted(levels.dropna().unique())
        clusters = []
        for lvl in levels:
            placed = False
            for c in clusters:
                if abs(lvl - c["price"]) / c["price"] < tol:
                    c["touches"] += 1
                    c["price"] = (c["price"] * (c["touches"] - 1) + lvl) / c["touches"]
                    placed = True
                    break
            if not placed:
                clusters.append({"price": lvl, "touches": 1})
        return clusters

    res = sorted(cluster(swing_high, tolerance), key=lambda c: -c["touches"])
    sup = sorted(cluster(swing_low, tolerance), key=lambda c: -c["touches"])
    return sup, res


def detect_spring_upthrust(df, support_price, resistance_price, lookback=15):
    recent = df.tail(lookback)
    upthrust = ((recent["High"] > resistance_price) & (recent["Close"] < resistance_price)).any()
    spring = ((recent["Low"] < support_price) & (recent["Close"] > support_price)).any()
    return spring, upthrust


# ------------------------------------------------------------- scoring ----
def swing_score(df):
    latest = df.iloc[-1]
    bd, score = {}, 0

    if latest["Close"] > latest["SMA50"] > latest["SMA200"]:
        bd["Trend (price > SMA50 > SMA200)"] = "Pass (+3)"; score += 3
    elif latest["Close"] > latest["SMA50"]:
        bd["Trend (price > SMA50 > SMA200)"] = "Partial, above SMA50 only (+1)"; score += 1
    else:
        bd["Trend (price > SMA50 > SMA200)"] = "Fail (0)"

    rsi = latest["RSI14"]
    if 50 <= rsi <= 70:
        bd["RSI momentum"] = f"Pass, {rsi:.1f} healthy uptrend zone (+2)"; score += 2
    elif 30 <= rsi < 45:
        bd["RSI momentum"] = f"Watch, {rsi:.1f} oversold bounce zone (+1)"; score += 1
    else:
        bd["RSI momentum"] = f"Neutral/overbought, {rsi:.1f} (0)"

    if latest["MACD_hist"] > 0 and df["MACD_hist"].iloc[-2] <= 0:
        bd["MACD"] = "Pass, just crossed bullish (+3)"; score += 3
    elif latest["MACD_hist"] > 0:
        bd["MACD"] = "Pass, already bullish (+2)"; score += 2
    else:
        bd["MACD"] = "Fail, bearish (0)"

    if latest["Volume"] > latest["VolAvg20"]:
        bd["Volume confirmation"] = "Pass, above 20d avg (+2)"; score += 2
    else:
        bd["Volume confirmation"] = "Fail, below 20d avg (0)"

    adx = latest.get("ADX14", np.nan)
    if pd.notna(adx):
        bd["ADX (trend strength)"] = f"{adx:.1f} — {'trending' if adx >= 25 else 'weak/ranging'}"

    return min(score, 10), bd


def range_score(df):
    bd, score = {}, 0
    latest = df.iloc[-1]
    sup, res = find_levels(df)
    if not sup or not res:
        return None, {"Range detection": "Not enough swing points — likely trending, not range-bound"}, sup, res

    nearest_sup = min(sup, key=lambda c: abs(c["price"] - latest["Close"]))
    nearest_res = min(res, key=lambda c: abs(c["price"] - latest["Close"]))
    dist_sup = abs(latest["Close"] - nearest_sup["price"]) / latest["Close"]
    dist_res = abs(latest["Close"] - nearest_res["price"]) / latest["Close"]
    near_support = dist_sup < dist_res
    level = nearest_sup if near_support else nearest_res
    label = "support" if near_support else "resistance"

    if min(dist_sup, dist_res) < 0.02:
        bd[f"Proximity to {label} ({level['price']:.2f})"] = "Pass, within 2% (+3)"; score += 3
    elif min(dist_sup, dist_res) < 0.04:
        bd[f"Proximity to {label} ({level['price']:.2f})"] = "Watch, within 4% (+1)"; score += 1
    else:
        bd[f"Nearest level ({label} @ {level['price']:.2f})"] = "Too far right now (0)"

    touch_pts = min(level["touches"], 4)
    bd[f"Level reliability ({level['touches']} touches)"] = f"+{touch_pts}"; score += touch_pts

    rsi = latest["RSI14"]
    if near_support and rsi < 35:
        bd["RSI confluence"] = f"Pass, oversold at support ({rsi:.1f}) (+2)"; score += 2
    elif (not near_support) and rsi > 65:
        bd["RSI confluence"] = f"Pass, overbought at resistance ({rsi:.1f}) (+2)"; score += 2
    else:
        bd["RSI confluence"] = f"No confluence ({rsi:.1f}) (0)"

    spring, upthrust = detect_spring_upthrust(df, nearest_sup["price"], nearest_res["price"])
    if (near_support and spring) or ((not near_support) and upthrust):
        bd["Fake-out check"] = "WARNING: spring/upthrust detected — level may be compromised (-3)"; score -= 3
    else:
        bd["Fake-out check"] = "Clean, no recent fake-out at this level"

    return max(min(score, 10), 0), bd, sup, res


def fundamental_score(info):
    bd, score = {}, 0
    pe = info.get("trailingPE")
    if pe is not None and 0 < pe < 30:
        bd["P/E"] = f"{pe:.1f} — reasonable (+3)"; score += 3
    elif pe is not None:
        bd["P/E"] = f"{pe:.1f} — rich or negative (0)"
    else:
        bd["P/E"] = "N/A"

    rg = info.get("revenueGrowth")
    if rg is not None and rg > 0.1:
        bd["Revenue growth"] = f"{rg*100:.1f}% YoY — strong (+3)"; score += 3
    elif rg is not None and rg > 0:
        bd["Revenue growth"] = f"{rg*100:.1f}% YoY — modest (+1)"; score += 1
    else:
        bd["Revenue growth"] = "N/A or negative (0)"

    d2e = info.get("debtToEquity")
    if d2e is not None and d2e < 100:
        bd["Debt/Equity"] = f"{d2e:.1f} — healthy (+2)"; score += 2
    elif d2e is not None:
        bd["Debt/Equity"] = f"{d2e:.1f} — leveraged (0)"
    else:
        bd["Debt/Equity"] = "N/A"

    rec = info.get("recommendationKey")
    if rec in ("strong_buy", "buy"):
        bd["Analyst consensus"] = f"{rec} (+2)"; score += 2
    elif rec == "hold":
        bd["Analyst consensus"] = f"{rec} (+1)"; score += 1
    else:
        bd["Analyst consensus"] = f"{rec} (0)"

    return min(score, 10), bd


def sentiment_score(news):
    if not news:
        return 5, {"News sentiment": "No recent headlines — neutral score assigned"}, []
    scores, headlines = [], []
    for item in news:
        title = item.get("title") or item.get("content", {}).get("title", "")
        if not title:
            continue
        vs = analyzer.polarity_scores(title)
        scores.append(vs["compound"])
        headlines.append((title, vs["compound"], extract_news_link(item)))
    if not scores:
        return 5, {"News sentiment": "No usable headlines — neutral score assigned"}, []
    avg = sum(scores) / len(scores)
    scaled = round((avg + 1) / 2 * 10, 1)
    bd = {"News sentiment (avg)": f"{avg:.2f} -> {scaled}/10"}
    return scaled, bd, headlines


def technical_trend_score(df):
    latest = df.iloc[-1]
    bd, score = {}, 0
    if latest["Close"] > latest["SMA200"]:
        bd["Long-term trend (price vs SMA200)"] = "Above (+5)"; score += 5
    else:
        bd["Long-term trend (price vs SMA200)"] = "Below (0)"
    weekly = df["Close"].resample("W").last().dropna()
    if len(weekly) >= 10:
        up = weekly.iloc[-1] > weekly.iloc[-10]
        bd["10-week price direction"] = "Higher (+5)" if up else "Lower (0)"; score += 5 if up else 0
    return min(score, 10), bd


# ---------------------------------------------------------- trade plan ----
def build_trade_plan(df, sup, res, sw, rg, timeframe_label):
    """Compose a plain-English next-steps plan from the computed levels."""
    latest = df.iloc[-1]
    price = latest["Close"]
    atr = latest["ATR14"]
    atr_stop = price - 1.5 * atr

    plan = {}

    # Bias
    if sw >= 7:
        bias = "Bullish continuation setup — trend, momentum and volume are mostly aligned."
    elif sw >= 4:
        bias = "Mixed signals — some elements support a long, others don't confirm yet."
    else:
        bias = "Weak setup right now — the swing case isn't there on this timeframe."
    plan["Overall read"] = bias

    # Entry
    if sup:
        s = sup[0]["price"]
        if price <= s * 1.02:
            plan["Where to enter"] = (
                f"Near current support (~₹{s:,.0f}). Don't buy the touch — wait for a confirming "
                f"candle (a close back above ₹{s:,.0f}) before entering, since a fresh test can still fail."
            )
        else:
            plan["Where to enter"] = (
                f"Price is away from the nearest support (~₹{s:,.0f}). Either wait for a pullback "
                f"toward that zone, or if momentum is strong, a break-and-retest of the nearest "
                f"resistance (~₹{res[0]['price']:,.0f}) with volume is the more common continuation entry."
                if res else
                f"Price is away from the nearest support (~₹{s:,.0f}). Consider waiting for a pullback "
                f"toward that zone rather than chasing here."
            )
    else:
        plan["Where to enter"] = "No clean support level detected on this timeframe — wait for a pullback to the rising moving average (SMA50) instead of chasing strength."

    # Stop-loss
    if sup:
        struct_stop = sup[0]["price"] * 0.985
        plan["Stop-loss"] = (
            f"~₹{struct_stop:,.0f} (just under the support cluster) or ~₹{atr_stop:,.0f} "
            f"(1.5x ATR from current price) — use whichever is tighter for your risk tolerance. "
            f"A close below this invalidates the setup; exit rather than average down."
        )
    else:
        plan["Stop-loss"] = f"~₹{atr_stop:,.0f} (1.5x ATR from current price), since no clean support level exists to anchor a structural stop."

    # Targets
    if res:
        t1 = res[0]["price"]
        plan["Target 1"] = f"~₹{t1:,.0f} — the nearest tested resistance. A sensible place to book partial profits."
        if len(res) > 1:
            t2 = sorted([r["price"] for r in res if r["price"] > t1])
            if t2:
                plan["Stretch target"] = f"~₹{t2[0]:,.0f} — only relevant if Target 1 breaks with volume."
    else:
        rr_target = price + 2 * (price - atr_stop)
        plan["Target"] = f"~₹{rr_target:,.0f} (roughly a 2:1 reward-to-risk from entry vs. the ATR stop) — no fixed resistance overhead, so trail your stop as price advances instead of fixing one number."

    # What to watch
    watch = []
    adx = latest.get("ADX14", np.nan)
    if pd.notna(adx) and adx < 20:
        watch.append("ADX is low — this stock isn't trending strongly, so breakouts here are more likely to fail. Size smaller.")
    if latest["RSI14"] > 70:
        watch.append("RSI is overbought — a short-term pullback or consolidation is more likely before further upside.")
    if latest["RSI14"] < 30:
        watch.append("RSI is oversold — watch for a reversal candle before entering; don't catch the falling knife.")
    if latest["Volume"] < latest["VolAvg20"]:
        watch.append("Recent volume is below average — wait for a volume pickup to confirm any move, low-volume breakouts tend to fail.")
    if res and sup:
        spring, upthrust = detect_spring_upthrust(df, sup[0]["price"], res[0]["price"])
        if upthrust:
            watch.append("A false breakout (upthrust) was recently detected at resistance — treat any fresh test of that level with extra caution.")
        if spring:
            watch.append("A false breakdown (spring) was recently detected at support — the level may be more reliable than it looks after shaking out weak hands.")
    if not watch:
        watch.append("No major red flags in the indicators right now — the main risk is broader market direction, not this stock's own setup.")
    plan["What to watch in the next few days"] = " ".join(watch)

    plan["Timeframe context"] = f"This plan is based on {timeframe_label.lower()} candles — re-check it if you're trading a different holding period than what {timeframe_label.lower()} data implies (daily ≈ days-to-weeks, weekly ≈ weeks-to-months, monthly ≈ months+)."

    return plan


# ------------------------------------------------------------ charting ----
def make_tradingview_style_chart(df, sup, res, atr_stop, ticker, bars=150):
    d = df.tail(bars)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.15, 0.23],
        vertical_spacing=0.02,
    )

    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name=ticker,
        increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["SMA20"], name="SMA20",
                              line=dict(width=1, color="#42a5f5")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"], name="SMA50",
                              line=dict(width=1, color="#ffa726")), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"], name="SMA200",
                              line=dict(width=1, color="#ab47bc")), row=1, col=1)

    if sup:
        s = sup[0]["price"]
        fig.add_hline(y=s, line_dash="dash", line_color="#26a69a", row=1, col=1,
                      annotation_text=f"Support {s:,.0f}", annotation_position="right")
    if res:
        r = res[0]["price"]
        fig.add_hline(y=r, line_dash="dash", line_color="#ef5350", row=1, col=1,
                      annotation_text=f"Resistance {r:,.0f}", annotation_position="right")
    fig.add_hline(y=atr_stop, line_dash="dot", line_color="#e91e63", row=1, col=1,
                  annotation_text=f"Stop ~{atr_stop:,.0f}", annotation_position="right")

    vol_colors = np.where(d["Close"] >= d["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volume",
                          marker_color=vol_colors, showlegend=False), row=2, col=1)

    # RSI panel: the 30 / 50 / 70 lines are the zones explained in "Understanding RSI" below the chart.
    fig.add_trace(go.Scatter(x=d.index, y=d["RSI14"], name="RSI (14)",
                              line=dict(width=1.5, color="#ce93d8")), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", line_width=1, row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#787b86", line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", line_width=1, row=3, col=1)
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=3, col=1)

    fig.update_layout(
        height=740,
        template="plotly_dark",
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc"),
        margin=dict(l=10, r=60, t=70, b=10),
        # Legend sits below the range-selector buttons row so the two never overlap.
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        dragmode="pan",
        hovermode="x unified",
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#1e222d",
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#1e222d", activecolor="#2962ff", font=dict(color="#d1d4dc"),
            # Pushed above the legend row so buttons and legend text never collide.
            y=1.16, yanchor="top", x=0, xanchor="left",
        ),
        row=1, col=1,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e222d", row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#1e222d", row=1, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#1e222d", row=3, col=1)
    fig.update_yaxes(showgrid=False, row=3, col=1)
    return fig



def classify_regime(nifty_close, stock_closes):
    """Market regime from data up to the latest bar. Same two rules as the backtest:
    1) Nifty 50 close vs its own 200-day average; 2) share of stocks above their own 200-day average."""
    out = {"nifty_close": None, "nifty_sma200": None, "nifty_gap_pct": None, "nifty_up": None,
           "breadth_pct": None, "breadth_up": None, "n_stocks": 0}
    nc = nifty_close.dropna()
    if len(nc) >= 200:
        sma = nc.rolling(200).mean().iloc[-1]
        out.update(nifty_close=float(nc.iloc[-1]), nifty_sma200=float(sma),
                   nifty_gap_pct=float((nc.iloc[-1] / sma - 1) * 100), nifty_up=bool(nc.iloc[-1] > sma))
    if stock_closes is not None and not stock_closes.empty:
        sma = stock_closes.rolling(200, min_periods=200).mean()
        last_c, last_s = stock_closes.iloc[-1], sma.iloc[-1]
        valid = last_c.notna() & last_s.notna()
        if valid.sum() >= 10:
            share = float((last_c[valid] > last_s[valid]).sum() / valid.sum() * 100)
            out.update(breadth_pct=share, breadth_up=share > 50, n_stocks=int(valid.sum()))
    ups = [v for v in (out["nifty_up"], out["breadth_up"]) if v is not None]
    if not ups:
        out["label"] = "unknown"
    elif all(ups):
        out["label"] = "healthy"
    elif not any(ups):
        out["label"] = "weak"
    else:
        out["label"] = "mixed"
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def get_market_regime():
    """Fetch the Nifty 50 index and the Nifty 50 stocks, then classify. Returns None if data is unavailable."""
    try:
        nifty = yf.Ticker("^NSEI").history(period="2y", interval="1d")["Close"]
    except Exception:
        return None
    closes = None
    try:
        tickers, _ = fetch_universe("Nifty 50")
        raw = yf.download(list(tickers), period="2y", interval="1d", auto_adjust=True,
                          group_by="ticker", threads=False, progress=False)
        cols = {t: raw[t]["Close"] for t in tickers if t in raw.columns.get_level_values(0)}
        closes = pd.DataFrame(cols) if cols else None
    except Exception:
        closes = None
    return classify_regime(nifty, closes)


REGIME_TEXT = {
    "healthy": ("🟢", "Healthy market",
                "Historically (2019–2026 backtest), the swing score ranked stocks best in this kind of market: "
                "higher-scoring stocks beat lower-scoring ones. But that edge was weak in 2024 onward, so don't treat it as proven."),
    "weak": ("🔴", "Weak market",
             "Historically, the swing score worked backwards in weak markets: beaten-down, low-scoring stocks "
             "bounced more than high-scoring ones. Treat swing scores with extra caution; range/bounce setups "
             "may suit better, but this is a pattern from few episodes, not a rule."),
    "mixed": ("🟡", "Mixed market",
              "The index and the broad market disagree (one is above its 200-day average, the other below). "
              "The backtest has no clean pattern for this case, so treat the scores with caution."),
    "unknown": ("⚪", "Market regime unavailable", "Couldn't load enough index data right now."),
}


def render_regime_banner():
    reg = get_market_regime()
    if reg is None:
        st.info("⚪ Market regime unavailable right now (couldn't reach live index data).")
        return
    icon, title, msg = REGIME_TEXT[reg["label"]]
    parts = []
    if reg["nifty_up"] is not None:
        parts.append(f"Nifty 50 at {reg['nifty_close']:,.0f} is {abs(reg['nifty_gap_pct']):.1f}% "
                     f"{'above' if reg['nifty_up'] else 'below'} its 200-day average ({reg['nifty_sma200']:,.0f})")
    if reg["breadth_pct"] is not None:
        parts.append(f"{reg['breadth_pct']:.0f}% of the {reg['n_stocks']} Nifty 50 stocks are above their own 200-day averages")
    box = {"healthy": st.success, "weak": st.error, "mixed": st.warning}.get(reg["label"], st.info)
    box(f"**{icon} {title}.** " + "; ".join(parts) + ".  \n" + msg)
    with st.expander("What the regime backtest actually found (and its limits)"):
        st.markdown(
            "Tested on ~470 Nifty 500 stocks, 2019–2026, 20-day holds, after 0.30% costs, vs the Nifty:\n"
            "- **Nifty above its 200-day average:** score 7–10 beat score 0–3 by +0.62% on average (t=2.2). "
            "Earlier years +0.97%; **2024 onward: about 0%**.\n"
            "- **Nifty below its 200-day average:** the order reversed (−0.67%, not statistically significant).\n"
            "- **Limits:** only ~20 independent weak-market samples (9 since 2024), partly the 2020 crash and rebound; "
            "list uses today's index members (flatters results); breadth here uses the Nifty 50 stocks for speed "
            "(the backtest used ~470 stocks). The regime is shown as context. It does not filter or change any score."
        )



# ---------------------------------------------------- relative strength ---
def _naive_daily_index(series):
    s = series.dropna().copy()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep="last")]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark_close():
    """Nifty 50 index daily closes (2 years), used as the yardstick for relative strength."""
    try:
        h = yf.Ticker("^NSEI").history(period="2y", interval="1d")
        return _naive_daily_index(h["Close"]) if not h.empty else None
    except Exception:
        return None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_rs_close(ticker):
    """Two years of daily closes for one stock, independent of the chart timeframe."""
    try:
        h = yf.Ticker(ticker).history(period="2y", interval="1d")
        return _naive_daily_index(h["Close"]) if not h.empty else None
    except Exception:
        return None


def compute_relative_strength(stock_close, bench_close):
    """Relative strength vs the Nifty 50, from daily closes (63 / 126 / 252 trading days = ~3 / 6 / 12 months).
    Returns None if there is not enough overlapping history."""
    if stock_close is None or bench_close is None:
        return None
    a, b = _naive_daily_index(stock_close), _naive_daily_index(bench_close)
    both = pd.concat([a.rename("s"), b.rename("b")], axis=1, join="inner").dropna()
    if len(both) < 70:
        return None
    out = {"n": len(both)}
    for name, days in (("3m", 63), ("6m", 126), ("12m", 252)):
        if len(both) > days:
            sr = both["s"].iloc[-1] / both["s"].iloc[-1 - days] - 1
            br = both["b"].iloc[-1] / both["b"].iloc[-1 - days] - 1
            out[f"stock_{name}"], out[f"bench_{name}"] = sr * 100, br * 100
            out[f"excess_{name}"] = (sr - br) * 100
        else:
            out[f"stock_{name}"] = out[f"bench_{name}"] = out[f"excess_{name}"] = None
    # 12-month return skipping the most recent month (the classic "12-1" momentum measure)
    if len(both) > 252:
        out["mom_12_1"] = (both["s"].iloc[-22] / both["s"].iloc[-1 - 252] - 1) * 100
    else:
        out["mom_12_1"] = None
    line = both["s"] / both["b"]
    out["rs_line"] = line / line.iloc[0] * 100
    out["rs_line_sma50"] = out["rs_line"].rolling(50).mean()
    out["rs_line_up"] = bool(out["rs_line"].iloc[-1] > out["rs_line_sma50"].iloc[-1]) \
        if pd.notna(out["rs_line_sma50"].iloc[-1]) else None
    window = out["rs_line"].tail(252)
    out["rs_near_high"] = bool(out["rs_line"].iloc[-1] >= window.max() * 0.98)
    e3, e6 = out.get("excess_3m"), out.get("excess_6m")
    if e3 is None or e6 is None:
        out["label"] = "unknown"
    elif e3 > 0 and e6 > 0:
        out["label"] = "leader"
    elif e3 > 0 >= e6:
        out["label"] = "improving"
    elif e6 > 0 >= e3:
        out["label"] = "fading"
    else:
        out["label"] = "laggard"
    return out


RS_TEXT = {
    "leader": ("🟢", "Leader",
               "The stock has beaten the Nifty over both the last 3 and 6 months. Money has been flowing into it faster than into the index.",
               "Leaders are where trend-following setups have the best odds. Prefer pullback entries here over weak stocks, "
               "and be more patient with it than with a laggard. A leader can still fall with the market."),
    "improving": ("🟡", "Improving",
                  "Beating the Nifty over 3 months, but still behind over 6 months: it may be turning from a laggard into a leader.",
                  "Early turns are less reliable. Wait for the relative strength line to hold above its 50-day average "
                  "and for the stock's own trend to confirm before sizing up."),
    "fading": ("🟠", "Fading",
               "Ahead of the Nifty over 6 months but behind over the last 3: a former leader that is losing steam.",
               "Be careful with fresh breakout entries. If you hold it, tighten the stop and watch whether the relative "
               "strength line breaks below its 50-day average."),
    "laggard": ("🔴", "Laggard",
                "Behind the Nifty over both 3 and 6 months. Even if the price chart looks fine, the market has preferred other stocks.",
                "Poor ground for trend-following longs. A bounce from support is a trade (not a trend), so keep targets "
                "modest and stops tight, as with your range-bound trades."),
    "unknown": ("⚪", "Unknown", "Not enough overlapping history with the index to measure.", ""),
}


def render_relative_strength(ticker):
    rs = compute_relative_strength(fetch_rs_close(ticker), fetch_benchmark_close())
    st.subheader("Relative strength vs Nifty 50")
    if rs is None:
        st.info("Relative strength unavailable (couldn't load enough index or stock history).")
        return None
    icon, title, what, how = RS_TEXT[rs["label"]]
    cols = st.columns(3)
    for col, key, lbl in zip(cols, ("3m", "6m", "12m"), ("3 months", "6 months", "12 months")):
        if rs[f"excess_{key}"] is None:
            col.metric(f"{lbl} vs Nifty", "n/a")
        else:
            col.metric(f"{lbl} vs Nifty", f"{rs[f'excess_{key}']:+.1f} pts",
                       help=f"Stock {rs[f'stock_{key}']:+.1f}% vs Nifty {rs[f'bench_{key}']:+.1f}% over {lbl}. "
                            "The number shown is the difference in percentage points.")
    st.markdown(f"**{icon} {title}.** {what}")
    notes = []
    if rs["rs_line_up"] is not None:
        notes.append("relative strength line is " + ("above" if rs["rs_line_up"] else "below") + " its 50-day average")
    if rs["rs_near_high"]:
        notes.append("the line is within 2% of its 1-year high (leaders often make new relative highs before price does)")
    if rs["mom_12_1"] is not None:
        notes.append(f"12-month return excluding the latest month: {rs['mom_12_1']:+.0f}%")
    if notes:
        st.caption("; ".join(notes).capitalize() + ".")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs["rs_line"].index, y=rs["rs_line"], name="Stock ÷ Nifty (start = 100)",
                             line=dict(color="#42a5f5", width=2)))
    fig.add_trace(go.Scatter(x=rs["rs_line_sma50"].index, y=rs["rs_line_sma50"], name="50-day average",
                             line=dict(color="#ffa726", width=1, dash="dot")))
    fig.update_layout(height=260, template="plotly_dark", plot_bgcolor="#131722", paper_bgcolor="#131722",
                      margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text="Relative strength line (rising = stock outperforming the Nifty)", font=dict(size=13)),
                      legend=dict(orientation="h", y=1.15, x=0))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**How to use it:** {how}")
    with st.expander("About relative strength (and what's not proven)"):
        st.markdown(
            "- **What it is:** how the stock has performed compared with the Nifty 50, not in isolation. "
            "The line above is the stock's price divided by the index. When it rises, the stock is beating the market.\n"
            "- **Why traders use it:** research on medium-term momentum (roughly 3 to 12 months of past outperformance) "
            "found that winners often kept outperforming for a while in many markets. As far as I know, such effects "
            "have also tended to weaken after they became widely known.\n"
            "- **Not yet in the scores:** relative strength is shown as context only. It is not part of the swing, range or "
            "long-term scores because we haven't tested it on Indian data yet. The earlier backtest covered only the "
            "trend, RSI, MACD and volume rules."
        )
    return rs


# ------------------------------------------------------------ RSI guide ----
def rsi_guide(df, timeframe_label="Daily"):
    """Plain-English reading of RSI for this stock right now: what it means and how to use it."""
    latest = df.iloc[-1]
    rsi = float(latest["RSI14"])
    prev = float(df["RSI14"].iloc[-6]) if len(df) > 6 and pd.notna(df["RSI14"].iloc[-6]) else rsi
    change = rsi - prev
    direction = "rising" if change > 2 else ("falling" if change < -2 else "flat")
    price, s50, s200 = latest["Close"], latest["SMA50"], latest["SMA200"]
    if pd.notna(s50) and pd.notna(s200) and price > s50 > s200:
        trend = "uptrend"
    elif pd.notna(s50) and pd.notna(s200) and price < s50 < s200:
        trend = "downtrend"
    else:
        trend = "mixed"

    if rsi >= 70:
        zone = "Overbought (70+)"
        meaning = ("Recent gains have been much larger than recent losses. That means strong momentum, "
                   "not automatically a reason to sell.")
    elif rsi >= 60:
        zone, meaning = "Strong (60–70)", "Buyers are firmly in control; this is the zone strong uptrends usually live in."
    elif rsi >= 50:
        zone, meaning = "Positive (50–60)", "Momentum leans bullish: gains are slightly outweighing losses."
    elif rsi >= 40:
        zone, meaning = "Soft (40–50)", "Momentum leans weak, but it's not stretched. In an uptrend this is a normal pullback zone."
    elif rsi >= 30:
        zone, meaning = "Weak (30–40)", "Sellers have the upper hand. Momentum is poor, but not yet extreme."
    else:
        zone = "Oversold (below 30)"
        meaning = ("Recent losses have been much larger than recent gains. That means heavy selling, "
                   "not automatically a reason to buy.")

    tips = []
    if trend == "uptrend":
        if rsi >= 70:
            tips.append("In an uptrend, RSI above 70 often stays there for weeks. Don't short it just because RSI is high. "
                        "If you hold, consider trailing your stop or taking partial profit. If you don't hold, avoid chasing here, "
                        "and wait for RSI to cool toward 50–60 and turn up.")
        elif rsi >= 55:
            tips.append("Healthy momentum in an uptrend. Hold, and treat dips toward the 20/50-day average as places to add, not exit.")
        else:
            tips.append("Classic pullback zone: in uptrends RSI often dips to roughly 40–50 and then turns up. "
                        "A bullish candle at support or the 50-day average, with RSI turning up, is the entry to look for. "
                        "Put the stop just below the swing low.")
            if direction == "falling":
                tips.append("RSI is still falling, so the pullback may not be finished. Waiting for it to turn up is cheaper than guessing the bottom.")
    elif trend == "downtrend":
        if rsi < 30:
            tips.append("In a downtrend, RSI can stay below 30 for a long time. Oversold alone is not a buy signal. "
                        "Only consider a bounce trade if price is at a clear support level, a reversal candle appears and volume picks up. "
                        "Keep the target modest (the 20- or 50-day average) and the stop tight.")
        elif rsi >= 50:
            tips.append("Rallies in downtrends often stall when RSI reaches about 50–60. That is a poor place to buy and a "
                        "common place for sellers to return.")
        else:
            tips.append("Momentum is weak and the trend is down. The path of least resistance is lower; wait for the trend to repair before buying.")
    else:
        if rsi < 35:
            tips.append("In a range-bound or unclear trend, RSI works best as a timing tool. RSI near or below 30–35 while price sits at support "
                        "is the setup you trade: buy the support with a stop just below it.")
        elif rsi > 65:
            tips.append("RSI near or above 65–70 with price at resistance is where range traders sell or take profits. "
                        "Don't buy new longs into resistance.")
        else:
            tips.append("RSI is mid-range with no clear trend, so it offers little edge now. Wait for price to reach support or "
                        "resistance and check RSI there.")
    if direction != "flat":
        tips.append(f"RSI is {direction} (about {change:+.0f} points over the last 5 candles). The direction matters as much as the level: "
                    "a turn up from below 50 or a turn down from above 60 is often the first sign the move is changing.")

    return {"rsi": rsi, "zone": zone, "meaning": meaning, "trend": trend, "direction": direction, "tips": tips,
            "tf_note": ("" if timeframe_label == "Daily" else
                        f"Note: on {timeframe_label.lower()} candles, each RSI period is one "
                        f"{'week' if timeframe_label == 'Weekly' else 'month'}, so the reading describes a much longer-term picture.")}


def render_rsi_guide(df, timeframe_label):
    g = rsi_guide(df, timeframe_label)
    st.subheader("Understanding RSI")
    c1, c2, c3 = st.columns(3)
    c1.metric("RSI (14)", f"{g['rsi']:.1f}")
    c2.metric("Zone", g["zone"])
    c3.metric("Trend context", g["trend"].capitalize(), help="Uptrend: price > 50-day > 200-day average. Downtrend: the reverse. Otherwise mixed.")
    st.markdown(f"**What it says about this stock:** {g['meaning']}")
    for t in g["tips"]:
        st.markdown(f"- {t}")
    if g["tf_note"]:
        st.caption(g["tf_note"])
    with st.expander("RSI basics: what it is, zones and rules of thumb"):
        st.markdown(
            "**What it is.** The Relative Strength Index (RSI) compares the size of recent up-moves with recent down-moves over 14 candles "
            "and squeezes the result into a 0 to 100 scale. It measures the *speed and balance* of recent price moves (momentum), "
            "not whether a stock is cheap or expensive.\n\n"
            "**Zones:** above 70 = strong momentum / stretched; 50–70 = bullish side; 30–50 = bearish side; below 30 = heavy selling / stretched.\n\n"
            "**Rules of thumb:**\n"
            "1. **Trend first, RSI second.** In uptrends RSI tends to oscillate roughly between 40 and 80, and in downtrends roughly between 20 and 60 "
            "(an idea popularised, as far as I know, by Andrew Cardwell and Constance Brown). So 'overbought' in an uptrend and "
            "'oversold' in a downtrend are normal, not signals.\n"
            "2. **Overbought and oversold are not buy/sell signals.** They say a move is stretched. Stretched moves can stretch further.\n"
            "3. **Best use for your style:** timing entries. In ranges, buy near support when RSI is low and sell near resistance when RSI is high. "
            "In uptrends, buy pullbacks when RSI dips to the 40–50 area and turns up.\n"
            "4. **Divergence:** if price makes a new low but RSI makes a higher low (or price makes a new high and RSI a lower high), momentum "
            "is fading. It's an early warning, and works only with a confirming candle or level.\n"
            "5. **Never alone:** combine it with trend, support/resistance and volume, which is what the scores and chart above do."
        )
        st.caption(
            "Limits: in the 7-year backtest on Nifty 500 stocks, the RSI rule inside the swing score helped before 2024 but not clearly since, "
            "so treat RSI as a timing aid, not a predictor. This app uses a simple 14-candle average, so its RSI can differ by a few points "
            "from TradingView, which uses Wilder's smoothing."
        )


# ---------------------------------------------------------------- UI ------
st.title("Trade picker")
st.caption("Live NSE analysis — swing, range and long-term scoring, in plain English.")

render_regime_banner()

tab1, tab2 = st.tabs(["Analyze a stock", "Find new ideas"])

with tab1:
    col1, col2, col3 = st.columns([2.5, 1, 1])
    with col1:
        ticker_input = st.text_input("NSE ticker (e.g. RELIANCE, TCS, DRREDDY)", value="DRREDDY")
    with col2:
        mode = st.radio("Horizon", ["Short-term (technical)", "Long-term (fundamental)"])
    with col3:
        timeframe_label = st.selectbox("Candle timeframe", list(TIMEFRAMES.keys()), index=0)

    ticker = ticker_input.strip().upper()
    if ticker and not ticker.endswith(".NS"):
        ticker = ticker + ".NS"

    tf = TIMEFRAMES[timeframe_label]

    if st.button("Analyze", type="primary") or ticker:
        with st.spinner(f"Pulling live {timeframe_label.lower()} data for {ticker}..."):
            df = fetch_data(ticker, period=tf["period"], interval=tf["interval"])

        min_bars = 60 if timeframe_label != "Monthly" else 24
        if df is None or len(df) < min_bars:
            st.error(f"Couldn't fetch enough {timeframe_label.lower()} data for {ticker}. Check the ticker symbol, "
                     f"or try a different timeframe (a young stock may not have enough monthly history).")
        else:
            df = add_indicators(df)
            latest = df.iloc[-1]
            price = latest["Close"]

            st.metric(f"{ticker} — last close ({timeframe_label.lower()} candles)", f"₹{price:,.2f}")

            if mode.startswith("Short"):
                # ---- SHORT-TERM: technical-led ----
                sw, sw_bd = swing_score(df)
                rg, rg_bd, sup, res = range_score(df)
                atr_stop = price - 1.5 * latest["ATR14"]

                c1, c2 = st.columns(2)
                _reg = get_market_regime()
                if _reg and _reg["label"] in ("weak", "mixed"):
                    st.caption(f"⚠️ Market regime is **{_reg['label']}** right now — the swing score has been less reliable in such markets (see banner above).")
                c1.metric("Swing score (trend-following)", f"{sw}/10")
                c2.metric("Range score (support/resistance)", f"{rg}/10" if rg is not None else "N/A")

                fig = make_tradingview_style_chart(df, sup, res, atr_stop, ticker)
                st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

                st.subheader("What to do next")
                plan = build_trade_plan(df, sup, res, sw, rg, timeframe_label)
                for k, v in plan.items():
                    st.markdown(f"**{k}:** {v}")

                render_rsi_guide(df, timeframe_label)
                render_relative_strength(ticker)

                st.subheader("Swing-trade reasoning")
                for k, v in sw_bd.items():
                    st.write(f"- **{k}**: {v}")

                st.subheader("Range-trade reasoning")
                for k, v in rg_bd.items():
                    st.write(f"- **{k}**: {v}")

            else:
                # ---- LONG-TERM: fundamental-led ----
                info = fetch_info(ticker)
                news = fetch_news(ticker)
                tech, tech_bd = technical_trend_score(df)
                fund, fund_bd = fundamental_score(info)
                sent, sent_bd, headlines = sentiment_score(news)
                composite = round(tech * 0.40 + fund * 0.35 + sent * 0.25, 1)

                st.metric("Long-term composite score", f"{composite}/10",
                          help="Technical 40% + Fundamental 35% + Sentiment 25%")

                c1, c2, c3 = st.columns(3)
                c1.metric("Fundamentals", f"{fund}/10")
                c2.metric("Technical trend", f"{tech}/10")
                c3.metric("News sentiment", f"{sent}/10")

                render_relative_strength(ticker)

                st.subheader("Fundamental reasoning")
                for k, v in fund_bd.items():
                    st.write(f"- **{k}**: {v}")

                st.subheader("Technical trend reasoning")
                for k, v in tech_bd.items():
                    st.write(f"- **{k}**: {v}")

                st.subheader("Recent news & sentiment")
                if headlines:
                    for title, comp, link in headlines[:6]:
                        tag = "🟢" if comp > 0.2 else ("🔴" if comp < -0.2 else "⚪")
                        if link:
                            st.markdown(f"{tag} [{title}]({link})  _(sentiment {comp:+.2f})_")
                        else:
                            st.write(f"{tag} {title}  _(sentiment {comp:+.2f}, no link available)_")
                else:
                    st.write("No recent headlines found.")

                fig = go.Figure(data=[go.Scatter(x=df.index, y=df["Close"], name="Close")])
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA200"], name="SMA200", line=dict(color="purple")))
                fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("Scan a wider universe of NSE stocks for new candidates.")
    universe_choice = st.radio("Universe", ["Nifty 50", "Nifty 500"], horizontal=True)
    universe, is_live = fetch_universe(universe_choice)
    if not is_live:
        st.info(f"Using the built-in Nifty 50 fallback list — live {universe_choice} constituents couldn't be fetched right now.")
    else:
        st.caption(f"Loaded {len(universe)} live constituents for {universe_choice}.")
    if universe_choice == "Nifty 500" and is_live:
        st.warning("Scanning 500 stocks pulls live data for each one — this can take a few minutes. Consider narrowing thresholds first on Nifty 50.")

    c1, c2, c3, c4 = st.columns(4)
    min_swing = c1.slider("Min swing score", 0, 10, 7)
    min_range = c2.slider("Min range score", 0, 10, 6)
    min_lt = c3.slider("Min long-term score", 0, 10, 6)
    top_n = c4.slider("Max results", 5, 50, 10)

    match_mode = st.radio(
        "Match",
        ["Must clear ALL three thresholds (stricter)", "Clears ANY one threshold (broader shortlist)"],
        index=0,
        help="ALL means a stock only appears if its Swing, Range AND Long-term scores each meet your minimum. "
             "ANY means it appears if it clears just one of the three — useful for casting a wider net.",
    )
    require_all = match_mode.startswith("Must clear ALL")
    rs_only = st.checkbox(
        "Only show stocks beating the Nifty 50 over the last 6 months (relative strength filter)",
        value=False,
        help="Adds an extra requirement on top of the thresholds above. Relative strength is not part of the scores "
             "and hasn't been backtested on Indian data yet, so use this as a judgement filter.",
    )

    if st.button("Scan now"):
        results = []
        bench_close = fetch_benchmark_close()
        progress = st.progress(0.0, text="Scanning...")
        for i, tkr in enumerate(universe):
            progress.progress((i + 1) / len(universe), text=f"Scanning {tkr} ({i+1}/{len(universe)})...")
            df = fetch_data(tkr)
            if df is None or len(df) < 60:
                continue
            df = add_indicators(df)
            sw, _ = swing_score(df)
            rg, _, _, _ = range_score(df)
            info = fetch_info(tkr)
            news = fetch_news(tkr)
            tech, _ = technical_trend_score(df)
            fund, _ = fundamental_score(info)
            sent, _, _ = sentiment_score(news)
            lt = round(tech * 0.40 + fund * 0.35 + sent * 0.25, 1)
            rg_check = rg if rg is not None else -1

            passes_swing = sw >= min_swing
            passes_range = rg_check >= min_range
            passes_lt = lt >= min_lt
            is_match = (passes_swing and passes_range and passes_lt) if require_all else (passes_swing or passes_range or passes_lt)

            rs_scan = compute_relative_strength(df["Close"], bench_close)
            ex3 = round(rs_scan["excess_3m"], 1) if rs_scan and rs_scan.get("excess_3m") is not None else None
            ex6 = round(rs_scan["excess_6m"], 1) if rs_scan and rs_scan.get("excess_6m") is not None else None
            if rs_only and not (ex6 is not None and ex6 > 0):
                is_match = False

            if is_match:
                why = []
                if passes_swing: why.append("Swing")
                if passes_range: why.append("Range")
                if passes_lt: why.append("LongTerm")
                results.append({"Ticker": tkr, "Price": round(df["Close"].iloc[-1], 2),
                                 "Swing": sw, "Range": rg, "LongTerm": lt,
                                 "RS 3m vs Nifty (pts)": ex3, "RS 6m vs Nifty (pts)": ex6,
                                 "Why": "+".join(why)})
        progress.empty()

        if results:
            out = (pd.DataFrame(results)
                   .sort_values(["Swing", "LongTerm"], ascending=False)
                   .reset_index(drop=True))
            out.index = out.index + 1
            out = out.head(top_n)
            # Store in session_state so the table survives switching tabs and coming back.
            st.session_state["scan_results"] = out
            st.session_state["scan_meta"] = {
                "universe": universe_choice, "match_mode": match_mode,
                "min_swing": min_swing, "min_range": min_range, "min_lt": min_lt,
                "rs_only": rs_only,
            }
        else:
            st.session_state["scan_results"] = None
            st.warning("Nothing cleared the thresholds — try lowering them or switching to ANY match.")

    # Re-display the last scan's results even if this tab was left and revisited
    # without clicking "Scan now" again.
    if st.session_state.get("scan_results") is not None:
        meta = st.session_state.get("scan_meta", {})
        st.caption(
            f"Showing results from the last scan — {meta.get('universe', '')}, "
            f"{meta.get('match_mode', '')}, thresholds: swing≥{meta.get('min_swing')}, "
            f"range≥{meta.get('min_range')}, long-term≥{meta.get('min_lt')}"
            f"{', beating the Nifty over 6 months' if meta.get('rs_only') else ''}. "
            "RS columns = stock return minus Nifty return in percentage points (positive = outperforming)."
        )
        result_df = st.session_state["scan_results"]
        st.dataframe(result_df, use_container_width=True)

        excel_buf = io.BytesIO()
        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
            result_df.to_excel(writer, index=True, index_label="#", sheet_name="Scan results")
        st.download_button(
            "Download as Excel",
            data=excel_buf.getvalue(),
            file_name="trade_picker_scan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.divider()
st.caption(
    "Not financial advice. This applies a fixed, transparent rule set to public data "
    "(yfinance / Yahoo Finance) — you decide what to do with it. Markets carry real risk of loss."
)
