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
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
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

    fig.update_layout(
        height=560,
        template="plotly_dark",
        plot_bgcolor="#131722",
        paper_bgcolor="#131722",
        font=dict(color="#d1d4dc"),
        margin=dict(l=10, r=60, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
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
        ),
        row=1, col=1,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e222d", row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#1e222d", row=1, col=1)
    fig.update_yaxes(showgrid=False, row=2, col=1)
    return fig


# ---------------------------------------------------------------- UI ------
st.title("Trade picker")
st.caption("Live NSE analysis — swing, range and long-term scoring, in plain English.")

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
                c1.metric("Swing score (trend-following)", f"{sw}/10")
                c2.metric("Range score (support/resistance)", f"{rg}/10" if rg is not None else "N/A")

                fig = make_tradingview_style_chart(df, sup, res, atr_stop, ticker)
                st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

                st.subheader("What to do next")
                plan = build_trade_plan(df, sup, res, sw, rg, timeframe_label)
                for k, v in plan.items():
                    st.markdown(f"**{k}:** {v}")

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

    if st.button("Scan now"):
        results = []
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

            if sw >= min_swing or rg_check >= min_range or lt >= min_lt:
                why = []
                if sw >= min_swing: why.append("Swing")
                if rg_check >= min_range: why.append("Range")
                if lt >= min_lt: why.append("LongTerm")
                results.append({"Ticker": tkr, "Price": round(df["Close"].iloc[-1], 2),
                                 "Swing": sw, "Range": rg, "LongTerm": lt, "Why": "+".join(why)})
        progress.empty()

        if results:
            out = pd.DataFrame(results).sort_values(["Swing", "LongTerm"], ascending=False).head(top_n)
            st.dataframe(out, use_container_width=True)
        else:
            st.warning("Nothing cleared the thresholds — try lowering them.")

st.divider()
st.caption(
    "Not financial advice. This applies a fixed, transparent rule set to public data "
    "(yfinance / Yahoo Finance) — you decide what to do with it. Markets carry real risk of loss."
)
