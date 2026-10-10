"""
Market Risk & VaR Platform — 9-phase Streamlit dashboard
Place this file beside: Main_market_risk (5).ipynb

Run in Command Prompt from this folder:
    python -m streamlit run "app (1).py"

If you save this file as app.py instead, run:
    python -m streamlit run app.py
"""
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import warnings
import re

import nbformat
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Market Risk & VaR Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NOTEBOOK_PATH = Path(__file__).with_name("Main_market_risk (5).ipynb")

st.markdown("""
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background:rgba(120,120,120,.07);border:1px solid rgba(128,128,128,.18);padding:12px 15px;border-radius:12px;}
div[data-testid="stSidebar"] {border-right:1px solid rgba(128,128,128,.2);}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def execute_notebook(path, modified_time):
    """Run the notebook in order and return its variables for display."""
    import matplotlib.pyplot as plt
    import plotly.basedatatypes

    old_plt_show = plt.show
    old_plotly_show = plotly.basedatatypes.BaseFigure.show
    plt.show = lambda *args, **kwargs: None
    plotly.basedatatypes.BaseFigure.show = lambda *args, **kwargs: None
    ns = {"__name__": "__market_risk_notebook__"}
    log = StringIO()
    try:
        notebook = nbformat.read(path, as_version=4)
        with warnings.catch_warnings(), redirect_stdout(log), redirect_stderr(log):
            warnings.simplefilter("ignore")
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type != "code":
                    continue
                source = cell.source.strip()
                if not source:
                    continue
                # Skip the commented/legacy dashboard code in the final notebook cell.
                if "ADVANCED EQUITY PORTFOLIO RISK MANAGEMENT DASHBOARD" in source:
                    continue
                try:
                    exec(compile(source, f"{path}:cell_{i+1}", "exec"), ns)
                except Exception as exc:
                    raise RuntimeError(f"Notebook code cell {i+1} failed: {exc}") from exc
    finally:
        plt.show = old_plt_show
        plotly.basedatatypes.BaseFigure.show = old_plotly_show
    return ns, log.getvalue()


def first_value(ns, *names, default=None):
    for name in names:
        if name in ns and ns[name] is not None:
            return ns[name]
    return default


def series_of(obj, col=None):
    if isinstance(obj, pd.Series):
        return obj.dropna()
    if isinstance(obj, pd.DataFrame):
        if col and col in obj.columns:
            return obj[col].dropna()
        if obj.shape[1] == 1:
            return obj.iloc[:, 0].dropna()
    return None


def fmt_money(x, currency="₹"):
    try:
        return f"{currency}{float(x):,.0f}"
    except (TypeError, ValueError):
        return "Not available"


def fmt_pct(x, digits=2):
    try:
        return f"{float(x):.{digits}%}"
    except (TypeError, ValueError):
        return "Not available"


def show_metric(label, value, help_text, delta=None):
    st.metric(label, value, delta=delta, help=help_text)


CHART_LINE_COUNTER = 0


def chart_line(obj, title, y_label, explanation, col=None):
    s = series_of(obj, col)
    if s is None or s.empty:
        st.info(f"No series available for **{title}** in the executed notebook.")
        return
    df = s.rename(y_label).to_frame()
    fig = px.line(df, x=df.index, y=y_label, title=title, template="plotly_dark")
    fig.update_layout(height=390, margin=dict(l=20, r=20, t=55, b=20), hovermode="x unified")
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title=y_label)
    global CHART_LINE_COUNTER
    CHART_LINE_COUNTER += 1
    chart_key = "line_" + str(CHART_LINE_COUNTER) + "_" + re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
    st.plotly_chart(fig, use_container_width=True, key=chart_key)
    st.caption(explanation)


def show_table(obj, title, explanation=None):
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        st.subheader(title, help=explanation)
        st.dataframe(obj, use_container_width=True)
        if explanation:
            st.caption(explanation)
    else:
        st.info(f"{title}: no matching DataFrame was found in the notebook outputs.")


def explain(title, meaning, interpretation, caveat=None):
    with st.expander(f"How to interpret: {title}", expanded=False):
        st.markdown(f"**What it measures:** {meaning}")
        st.markdown(f"**How to interpret it:** {interpretation}")
        if caveat:
            st.markdown(f"**Important caveat:** {caveat}")


def display_dict_table(d, title, description):
    if not isinstance(d, dict) or not d:
        st.info(f"{title} was not found under the expected notebook variable names.")
        return
    rows = []
    for key, value in d.items():
        if np.isscalar(value):
            rows.append({"Metric": str(key), "Value": value})
        elif isinstance(value, (list, tuple, np.ndarray)):
            rows.append({"Metric": str(key), "Value": str(value)})
    if rows:
        st.subheader(title, help=description)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(description)


st.title("📊 Market Risk & VaR Platform")
st.caption("Portfolio construction • Statistical diagnostics • VaR / CVaR • Volatility models • Backtesting • Stress testing")

if not NOTEBOOK_PATH.exists():
    st.error(f"Notebook not found: {NOTEBOOK_PATH.name}")
    st.write("Keep this dashboard Python file in the same folder as your notebook, or update NOTEBOOK_PATH in this script.")
    st.stop()

with st.spinner("Executing your notebook from top to bottom. The first run can take several minutes while data is downloaded and models are fitted…"):
    try:
        ns, notebook_log = execute_notebook(str(NOTEBOOK_PATH), NOTEBOOK_PATH.stat().st_mtime)
    except Exception as exc:
        st.error("The dashboard stopped because a notebook cell raised an error.")
        st.exception(exc)
        st.info("Open the notebook in Jupyter, run it from top to bottom, and fix the first failing cell. Then restart this dashboard.")
        st.stop()

# Actual names taken from the supplied notebook, with fallbacks for earlier naming.
raw_data = first_value(ns, "raw_data")
log_returns = first_value(ns, "log_return")
portfolio = first_value(ns, "dollar_exposure_portfolio")
portfolio_returns = series_of(first_value(ns, "dollar_exposure_portfolio_ret"))
pnl = series_of(portfolio, "Daily_mtm")
portfolio_value = series_of(portfolio, "Portfolio_daily_value")
if portfolio_returns is None and isinstance(portfolio, pd.DataFrame) and "Portfolio_daily_value" in portfolio:
    portfolio_returns = np.log(portfolio["Portfolio_daily_value"]).diff().dropna()

investment = first_value(ns, "Investment_amt", default=1_000_000)
tickers = first_value(ns, "ALL_TICKERS", "TICKERS", default=[])
benchmark = first_value(ns, "Bench_mark", default="Market benchmark")

# Actual VaR result dictionaries from the notebook
historical_var = first_value(ns, "hist_VAR", default={})
historical_cvar = first_value(ns, "hist_C_VAR", default={})
parametric_var = first_value(ns, "parametirc_var_result", default={})
monte_carlo_var = first_value(ns, "monete_carlo_var_result", default={})
monte_carlo_cvar = first_value(ns, "monete_carlo_C_var_result", default={})
var_compare = first_value(ns, "var_compare")
backtest_df = first_value(ns, "backtest_df")
wf_var_series = first_value(ns, "wf_var_series", default={})
scenario_df = first_value(ns, "scenario_df", "stress_scenario_df")

# Volatility model series and forecasts
ewma_vol = series_of(first_value(ns, "ewma_volatility", "ewma_vol"))
garch_vol = series_of(first_value(ns, "garch_conditional_vol"))
skew_garch_vol = series_of(first_value(ns, "skwed_garch_conditional_vol", "skewed_garch_conditional_vol"))
gjr_vol = series_of(first_value(ns, "gjr_garch_conditional", "gjr_garch_vol"))

tabs = st.tabs([
    "Executive Summary",
    "Phase 1 · Data & Portfolio",
    "Phase 2 · Diagnostics",
    "Phase 3 · P&L & Risk",
    "Phase 4 · VaR / CVaR",
    "Phase 5 · EWMA",
    "Phase 6 · GARCH",
    "Phase 7 · Skew-t & GJR",
    "Phase 8 · Backtesting",
    "Phase 9 · Stress Tests",
])

# ---------------- EXECUTIVE SUMMARY ----------------
with tabs[0]:
    st.header("Executive Summary")
    st.markdown("""
    <div style="padding:26px 28px;border-radius:18px;background:linear-gradient(120deg,#101827 0%,#173b5e 55%,#226b75 100%);color:white;margin:8px 0 20px 0;">
      <div style="font-size:12px;letter-spacing:2px;text-transform:uppercase;opacity:.8;">Quantitative Risk Analytics</div>
      <div style="font-size:30px;font-weight:750;line-height:1.2;margin:8px 0 10px 0;">Portfolio Market Risk Command Center</div>
      <div style="font-size:15px;line-height:1.6;max-width:900px;opacity:.92;">
        An end-to-end framework for measuring downside risk, modelling changing volatility, validating VaR forecasts,
        and exploring severe market scenarios — using the calculations and data produced by your research notebook.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("#### At a glance")
    st.markdown("""
    The project follows a nine-phase workflow: **market data → statistical diagnostics → portfolio P&L → VaR/CVaR →
    EWMA → GARCH → skewed-t/GJR-GARCH → VaR backtesting → stress testing.** Each phase includes outputs and
    interpretation notes so a reviewer can understand not just *what the number is*, but *what it means*.
    """)
    st.subheader("Project phases at a glance")
    phase_overview = pd.DataFrame([
        ["1", "Market data & portfolio", "Prices, log returns, weights, asset exposures, benchmark"],
        ["2", "Data cleaning & diagnostics", "Missing data, return distribution, normality, autocorrelation and ARCH effects"],
        ["3", "Portfolio P&L & risk foundation", "Portfolio value, daily mark-to-market P&L and realized losses"],
        ["4", "VaR & CVaR estimation", "Historical, parametric and Monte Carlo loss quantiles / tail loss estimates"],
        ["5", "EWMA volatility", "Exponentially weighted volatility that responds to recent market moves"],
        ["6", "GARCH volatility", "Conditional volatility from a GARCH model and standardized-residual diagnostics"],
        ["7", "Skew-t / GJR-GARCH", "Flexible tail/asymmetry modelling and leverage-effect analysis"],
        ["8", "VaR backtesting", "Walk-forward exceptions, Kupiec, Christoffersen and Basel traffic-light checks"],
        ["9", "Stress testing", "Worst observed daily / multi-day losses and scenario analysis"],
    ], columns=["Phase", "Area", "Purpose"])
    st.dataframe(phase_overview, use_container_width=True, hide_index=True)
    st.markdown("""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 22px 0;">
      <span style="padding:8px 12px;border-radius:9px;background:#e8f1fb;color:#16416b;font-weight:650;">01 Data</span>
      <span style="padding:8px 12px;border-radius:9px;background:#e8f1fb;color:#16416b;font-weight:650;">02 Diagnose</span>
      <span style="padding:8px 12px;border-radius:9px;background:#e8f1fb;color:#16416b;font-weight:650;">03 Measure P&amp;L</span>
      <span style="padding:8px 12px;border-radius:9px;background:#e5f5ef;color:#155b45;font-weight:650;">04 VaR / ES</span>
      <span style="padding:8px 12px;border-radius:9px;background:#e5f5ef;color:#155b45;font-weight:650;">05–07 Volatility</span>
      <span style="padding:8px 12px;border-radius:9px;background:#fff1db;color:#7a4a05;font-weight:650;">08 Validate</span>
      <span style="padding:8px 12px;border-radius:9px;background:#fbe8e8;color:#8b2828;font-weight:650;">09 Stress</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    show_metric("Initial investment", fmt_money(investment), "Starting capital used to construct the portfolio in the notebook.")
    show_metric("Portfolio assets", str(len(raw_data.columns) if isinstance(raw_data, pd.DataFrame) else len(tickers)),
                "Count of assets in the downloaded portfolio price data; the benchmark is intended to be separate.")
    show_metric("Price observations", f"{len(raw_data):,}" if isinstance(raw_data, pd.DataFrame) else "Not available",
                "Number of dated rows in the raw price DataFrame.")
    show_metric("P&L observations", f"{len(pnl):,}" if pnl is not None else "Not available",
                "Number of available daily mark-to-market P&L observations.")

    st.subheader("Portfolio behaviour")
    left, right = st.columns(2)
    with left:
        chart_line(portfolio_value, "Portfolio Value", "Portfolio value (₹)",
                   "A rising line means the marked portfolio value increased over the period; drawdowns show declines from prior peaks.")
    with right:
        chart_line(pnl, "Daily Mark-to-Market P&L", "Daily P&L (₹)",
                   "Positive values are gains and negative values are losses, assuming the notebook's P&L sign convention.")
    st.subheader("Risk concepts used in this project")
    concept_cols = st.columns(3)
    with concept_cols[0]:
        st.markdown("**VaR — Value at Risk**")
        st.write("A loss threshold for a chosen confidence level and horizon. For example, 99% one-day VaR is intended to be exceeded by losses on about 1% of comparable days, if the model is calibrated.")
    with concept_cols[1]:
        st.markdown("**CVaR / Expected Shortfall**")
        st.write("The average loss in the tail beyond the VaR threshold. It describes how severe losses may be when VaR is exceeded.")
    with concept_cols[2]:
        st.markdown("**Volatility forecasting**")
        st.write("EWMA and GARCH-family models estimate changing return variability. Higher forecast volatility generally implies a wider potential loss distribution.")
    explain("VaR confidence level", "The probability mass below the loss quantile boundary.", "A higher confidence level (99% vs 95%) generally produces a larger loss estimate.", "VaR is not a maximum-loss guarantee and does not tell you the size of losses beyond the threshold.")
    explain("CVaR / Expected Shortfall", "The average of observations in the loss tail beyond VaR.", "CVaR should usually be at least as severe as VaR when both are represented as positive loss magnitudes.", "Check sign conventions and ensure VaR and CVaR use the same horizon and P&L series.")
    explain("Backtesting exceptions", "Days when realized loss exceeds the forecast VaR threshold.", "Too many exceptions or clustered exceptions can indicate underestimation or poor responsiveness.", "Backtesting results depend on sample length, confidence level, and the forecast design.")
    st.info("The values and tables in this dashboard are taken from your notebook's executed variables where available. A missing output is labelled as unavailable rather than fabricated.")

# ---------------- PHASE 1 ----------------
with tabs[1]:
    st.header("Phase 1 — Market Data & Portfolio Construction")
    st.write("This phase downloads adjusted closing prices, computes daily log returns, separates the benchmark from portfolio assets, and documents the investment exposure and weighting assumptions.")
    c1, c2, c3 = st.columns(3)
    show_metric("Initial investment", fmt_money(investment), "Capital allocated across portfolio holdings at the start of the sample.")
    show_metric("Portfolio instruments", str(raw_data.shape[1]) if isinstance(raw_data, pd.DataFrame) else "Not available",
                "Number of portfolio instruments in raw_data.")
    show_metric("Benchmark", str(benchmark), "Reference market index used for comparison; it should not be treated as a portfolio holding unless explicitly included.")
    if isinstance(raw_data, pd.DataFrame):
        st.subheader("Adjusted close prices")
        selected_assets = st.multiselect("Assets to display", list(raw_data.columns),
                                         default=list(raw_data.columns[:min(6, len(raw_data.columns))]),
                                         key="p1_assets")
        if selected_assets:
            fig = px.line(raw_data[selected_assets], title="Adjusted Close Prices", template="plotly_dark")
            fig.update_layout(height=430, xaxis_title="Date", yaxis_title="Adjusted close")
            st.plotly_chart(fig, use_container_width=True, key="plot_1")
        st.subheader("Price-data coverage")
        coverage = pd.DataFrame({
            "First valid date": raw_data.apply(lambda s: s.first_valid_index()),
            "Last valid date": raw_data.apply(lambda s: s.last_valid_index()),
            "Missing observations": raw_data.isna().sum(),
        })
        st.dataframe(coverage, use_container_width=True)
    if isinstance(log_returns, pd.DataFrame):
        show_table(log_returns.describe().T, "Log-return descriptive statistics",
                   "Mean and standard deviation describe central tendency and variability; skewness and kurtosis help describe asymmetry and tail heaviness when included.")
    for name in ["invested_in_each_stock", "noof_share_owned", "portfolio_weights", "weights"]:
        if name in ns:
            st.subheader(f"Portfolio allocation — {name}")
            obj = ns[name]
            if isinstance(obj, (pd.Series, pd.DataFrame)):
                st.dataframe(obj, use_container_width=True)
            else:
                st.write(obj)
    explain("Log return", "The natural logarithm of today's price divided by the previous price.", "A positive value means the price rose; a negative value means it fell. Log returns are additive through time.", "Portfolio log returns and daily dollar P&L are different quantities and should not be mixed without a conversion.")
    explain("Dollar exposure", "The amount of initial capital assigned to each holding.", "Under equal allocation, each asset receives approximately the same starting investment.", "Buy-and-hold quantities can cause portfolio weights to drift as asset prices move.")
    explain("Benchmark", "A market index used as a comparison reference.", "Comparing portfolio and benchmark returns helps put performance and market moves in context.", "A benchmark should not be counted as a holding unless the portfolio construction explicitly includes it.")

# ---------------- PHASE 2 ----------------
with tabs[2]:
    st.header("Phase 2 — Data Cleaning & Statistical Diagnostics")
    st.write("This phase examines data quality and whether portfolio returns behave like independent, normally distributed observations. These checks inform which risk models and distributions may be appropriate.")
    if isinstance(raw_data, pd.DataFrame):
        c1, c2, c3 = st.columns(3)
        show_metric("Missing prices", f"{int(raw_data.isna().sum().sum()):,}", "Total missing cells across the raw price matrix. Missing values can affect return calculations and model fitting.")
        show_metric("Duplicate dates", f"{int(raw_data.index.duplicated().sum()):,}", "Repeated timestamps can create duplicate observations and should be investigated.")
        show_metric("Price assets", str(raw_data.shape[1]), "Number of price columns used for portfolio assets.")
        st.dataframe(pd.DataFrame({"Missing": raw_data.isna().sum(), "Missing %": raw_data.isna().mean() * 100}),
                     use_container_width=True)
    left, right = st.columns(2)
    with left:
        chart_line(portfolio_returns, "Portfolio Daily Returns", "Daily return",
                   "Look for volatility clustering: periods of large moves tend to be followed by other large moves.")
    with right:
        if portfolio_returns is not None and not portfolio_returns.empty:
            fig = px.histogram(x=portfolio_returns, nbins=70, marginal="box",
                               title="Return Distribution", template="plotly_dark",
                               labels={"x": "Daily portfolio return"})
            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True, key="plot_2")
            st.caption("A long tail or outliers indicate that extreme returns may be more common than a normal distribution suggests.")
    diagnostic_names = [
        "jb_result", "jb_test", "jb_pvalue", "jarque_bera_result",
        "ljung_box_result", "lb_test", "lb_pvalue",
        "arch_lm_result", "lm_test", "lm_pvalue",
        "adf_result", "adf_pvalue", "kpss_result", "kpss_pvalue",
    ]
    for name in diagnostic_names:
        if name in ns:
            obj = ns[name]
            st.markdown(f"**{name}**")
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                st.dataframe(obj, use_container_width=True)
            else:
                st.write(obj)
    st.subheader("Interpretation guide")
    explain("Jarque–Bera normality test", "Whether sample skewness and kurtosis are consistent with a normal distribution.", "A p-value below 0.05 is evidence against normality; fat tails make Gaussian VaR potentially optimistic.", "Rejecting normality does not identify a single best alternative distribution.")
    explain("Ljung–Box test", "Whether autocorrelations across a set of lags are jointly zero.", "A p-value below 0.05 suggests serial dependence in the tested series.", "Test results depend on the number of lags and whether raw returns or squared returns were tested.")
    explain("ARCH-LM test", "Whether return residuals show conditional heteroskedasticity / ARCH effects.", "A low p-value supports using a model that allows volatility to change over time.", "The test should be applied to the series and lags intended by the model specification.")
    explain("ADF / KPSS", "Complementary tests about stationarity, with different null hypotheses.", "ADF's null is a unit root; KPSS's null is stationarity. Read them together rather than interpreting the p-values identically.", "Results depend on deterministic terms and lag choices.")

# ---------------- PHASE 3 ----------------
with tabs[3]:
    st.header("Phase 3 — Portfolio Value, Realized P&L & Risk Foundation")
    st.write("This phase converts the holdings into a portfolio market value and daily mark-to-market P&L. P&L is the loss/gain series used by the subsequent monetary VaR and CVaR calculations.")
    c1, c2, c3 = st.columns(3)
    show_metric("Latest portfolio value", fmt_money(portfolio_value.iloc[-1]) if portfolio_value is not None and len(portfolio_value) else "Not available",
                "Last available marked portfolio value in the sample.")
    show_metric("Worst daily P&L", fmt_money(pnl.min()) if pnl is not None and len(pnl) else "Not available",
                "Most negative observed daily mark-to-market P&L in the sample; a historical observation, not a maximum possible loss.")
    show_metric("Best daily P&L", fmt_money(pnl.max()) if pnl is not None and len(pnl) else "Not available",
                "Largest observed positive daily mark-to-market P&L in the sample.")
    left, right = st.columns(2)
    with left:
        chart_line(portfolio_value, "Portfolio Market Value", "Portfolio value (₹)",
                   "This line shows the market value of the holdings over time.")
    with right:
        chart_line(pnl, "Daily Mark-to-Market P&L", "Daily P&L (₹)",
                   "Negative observations represent daily losses; the left tail is central to VaR and CVaR.")
    if pnl is not None and not pnl.empty:
        st.subheader("P&L descriptive statistics")
        st.dataframe(pnl.describe().to_frame("Daily P&L"), use_container_width=True)
    explain("Daily mark-to-market P&L", "The day-to-day change in the marked portfolio value.", "Negative P&L is a loss and positive P&L is a gain under the notebook's convention.", "It is denominated in currency, while returns and volatility may be expressed as proportions or percentages.")
    explain("Portfolio value", "The combined market value of all holdings at each date.", "Declines from a previous peak indicate a drawdown; value changes can be affected by price changes and portfolio construction assumptions.", "This dashboard does not assume additional deposits, withdrawals, fees or taxes unless they are modelled in the notebook.")

# ---------------- PHASE 4 ----------------
with tabs[4]:
    st.header("Phase 4 — VaR & CVaR Estimation")
    st.write("This phase estimates downside risk using historical simulation, a parametric distributional approach, and Monte Carlo simulation where outputs are available.")
    st.markdown("""
    - **Historical VaR:** empirical lower-tail quantile of observed P&L.
    - **Parametric VaR:** estimates the tail using a distributional assumption and sample parameters.
    - **Monte Carlo VaR:** estimates the tail from simulated outcomes.
    - **CVaR / Expected Shortfall:** average P&L/loss in the tail beyond the VaR threshold.
    """)
    level = st.select_slider("Confidence level", options=[0.90, 0.95, 0.99], value=0.95,
                             format_func=lambda x: f"{x:.0%}", key="p4_conf")
    result_rows = []
    for method, data in [
        ("Historical VaR", historical_var),
        ("Parametric VaR", parametric_var),
        ("Monte Carlo VaR", monte_carlo_var),
        ("Historical CVaR / ES", historical_cvar),
        ("Monte Carlo CVaR / ES", monte_carlo_cvar),
    ]:
        if isinstance(data, dict):
            candidates = [level, f"historical_var_{level}", f"historical_C_var_{level}",
                          f"parametirc_var_{level}", f"monete_carlo_var_result_{level}",
                          f"monete_carlo_C_var_result_{level}"]
            found = None
            for k in candidates:
                if k in data and np.isscalar(data[k]):
                    found = data[k]
                    break
            if found is None:
                # Accommodate dictionaries whose keys are strings.
                for k, v in data.items():
                    if str(level) in str(k) and np.isscalar(v):
                        found = v
                        break
            if found is not None:
                result_rows.append({"Method": method, "Stored result": float(found),
                                    "Loss magnitude (absolute value)": abs(float(found))})
    if result_rows:
        st.dataframe(pd.DataFrame(result_rows).style.format({"Stored result": "{:,.4f}", "Loss magnitude (absolute value)": "₹{:,.2f}"}),
                     use_container_width=True, hide_index=True)
    else:
        st.warning("The notebook's VaR dictionaries were not matched to the expected confidence-level keys. The variables are shown below for inspection.")
        display_dict_table(historical_var, "Historical VaR outputs", "Raw outputs from the notebook.")
        display_dict_table(parametric_var, "Parametric VaR outputs", "Raw outputs from the notebook.")
        display_dict_table(monte_carlo_var, "Monte Carlo VaR outputs", "Raw outputs from the notebook.")
    if isinstance(var_compare, pd.DataFrame):
        show_table(var_compare, "Notebook VaR comparison table",
                   "The table below is shown directly from the notebook. Check its units and sign convention before comparing magnitudes.")
    if pnl is not None and not pnl.empty:
        fig = px.histogram(x=pnl, nbins=80, title="Daily P&L Distribution", template="plotly_dark",
                           labels={"x": "Daily P&L (₹)"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="plot_3")
        st.caption("The left tail contains the largest losses. VaR marks a quantile threshold; CVaR describes the average tail outcome.")
    explain("Historical VaR", "The empirical lower quantile of realized P&L at alpha = 1 − confidence.", "At 95% confidence, approximately 5% of comparable observations are expected to fall below the threshold if the historical sample is representative.", "It assumes the past sample is informative about future risk and can miss events not present in the sample.")
    explain("Parametric VaR", "A quantile computed from estimated parameters and an assumed distribution.", "It is fast and transparent, but its result depends on the distribution and volatility assumptions.", "A Gaussian assumption can understate extreme risk when returns have fat tails or skewness.")
    explain("Monte Carlo VaR", "A quantile of simulated P&L scenarios.", "More simulations reduce Monte Carlo sampling noise but do not fix a misspecified model.", "Interpret it together with simulation assumptions, seed, horizon, and scaling.")
    explain("CVaR / Expected Shortfall", "Average P&L among outcomes beyond the VaR quantile, or equivalently the average tail loss after sign conversion.", "A more severe CVaR indicates worse average losses conditional on breaching VaR.", "Confirm whether the notebook stores tail P&L as negative values or loss magnitudes as positive values.")

# ---------------- PHASE 5 ----------------
with tabs[5]:
    st.header("Phase 5 — EWMA Volatility")
    st.write("EWMA gives more weight to recent squared returns, allowing estimated volatility to respond to changing market conditions.")
    lam = first_value(ns, "lambda_", "EWMA_LAMBDA", default=0.94)
    show_metric("EWMA decay factor λ", f"{float(lam):.3f}" if np.isscalar(lam) else str(lam),
                "A λ close to 1 makes volatility smoother and gives older observations more persistent weight. A lower λ responds more quickly to recent moves.")
    chart_line(ewma_vol, "EWMA Volatility Through Time", "Daily volatility",
               "Spikes indicate periods with larger recent return variability. Check whether the series is stored as a proportion or percentage before interpreting its scale.")
    if ewma_vol is not None and not ewma_vol.empty:
        latest = float(ewma_vol.iloc[-1])
        peak = float(ewma_vol.max())
        peak_date = str(ewma_vol.idxmax())
        c1, c2, c3 = st.columns(3)
        show_metric("Latest EWMA volatility", fmt_pct(latest), "Latest volatility estimate. This is a measure of typical return variability, not a prediction that the portfolio will move by exactly this amount.")
        show_metric("Peak EWMA volatility", fmt_pct(peak), "Largest estimated volatility in the sample, showing the most turbulent period under this EWMA estimate.")
        show_metric("Peak date", peak_date, "Date on which the maximum EWMA volatility estimate occurred.")
    explain("EWMA volatility", "An exponentially weighted estimate of the standard deviation of returns.", "A rise means recent returns have become more variable; a fall means recent variability has eased.", "Volatility is not itself a loss forecast. Monetary VaR also depends on the portfolio value, confidence level, distribution and horizon.")
    explain("Decay factor λ", "The persistence parameter in EWMA variance updating.", "With λ = 0.94, older information decays gradually; the newest squared return receives weight 1 − λ.", "Do not confuse a volatility estimate of 0.02 (2%) with 2.0 (200%); inspect how returns were scaled.")

# ---------------- PHASE 6 ----------------
with tabs[6]:
    st.header("Phase 6 — GARCH(1,1) Volatility")
    st.write("GARCH models conditional variance using a constant, lagged squared shocks, and lagged variance. It can capture volatility clustering.")
    chart_line(garch_vol, "GARCH Conditional Volatility", "Daily volatility",
               "The conditional volatility estimate changes over time in response to shocks and persistent variance.")
    fit = first_value(ns, "best_model_GARCH_fit")
    if fit is not None and hasattr(fit, "params"):
        st.subheader("Estimated GARCH parameters")
        st.dataframe(pd.DataFrame({"Estimate": fit.params}), use_container_width=True)
    forecast = first_value(ns, "garch_forecast_vol", "garch_forecast_variance")
    if forecast is not None:
        st.subheader("Forecast output")
        st.write(forecast)
    explain("GARCH conditional volatility", "The square root of the model's conditional variance at each date.", "High values indicate higher estimated near-term return variability; this is commonly used to scale parametric or simulation-based VaR.", "If the model was fitted to percentage-scaled returns, dividing the resulting volatility by 100 may be required to return to decimal-return units.")
    explain("GARCH alpha", "Sensitivity of variance to the latest squared shock.", "A larger alpha means recent shocks have a stronger immediate effect on conditional variance.", "Interpret alongside beta and the model's parameterization.")
    explain("GARCH beta", "Persistence of the previous conditional variance.", "A large beta means volatility shocks decay more slowly.", "Stationarity constraints depend on the model specification and distribution; do not judge the model from one coefficient alone.")
    explain("Standardized residuals", "Model residuals divided by the fitted conditional standard deviation.", "Ideally, they should show little remaining autocorrelation or volatility clustering.", "Residual diagnostics help evaluate the model; a good volatility fit does not guarantee accurate VaR.")

# ---------------- PHASE 7 ----------------
with tabs[7]:
    st.header("Phase 7 — Skewed-t GARCH & GJR-GARCH")
    st.write("This phase extends basic volatility modelling to allow for heavier tails and asymmetric responses to negative shocks.")
    left, right = st.columns(2)
    with left:
        chart_line(skew_garch_vol, "Skewed-t GARCH Conditional Volatility", "Daily volatility",
                   "A skewed Student-t distribution allows asymmetry and heavier tails than a Gaussian distribution, subject to fitted parameters.")
    with right:
        chart_line(gjr_vol, "GJR-GARCH Conditional Volatility", "Daily volatility",
                   "GJR-GARCH includes an indicator term that can allow negative shocks to affect variance differently from positive shocks.")
    skew_fit = first_value(ns, "best_skwed_model_GARCH_fit", "best_skewed_model_GARCH_fit")
    gjr_fit = first_value(ns, "gjr_fit", "gjr_garch_fit")
    if skew_fit is not None and hasattr(skew_fit, "params"):
        st.subheader("Skewed-t GARCH parameter estimates")
        st.dataframe(pd.DataFrame({"Estimate": skew_fit.params}), use_container_width=True)
    if gjr_fit is not None and hasattr(gjr_fit, "params"):
        st.subheader("GJR-GARCH parameter estimates")
        st.dataframe(pd.DataFrame({"Estimate": gjr_fit.params}), use_container_width=True)
    for name in ["gjr_stability", "gjr_alpha", "gjr_gamma", "gjr_beta", "gjr_omega"]:
        if name in ns:
            st.metric(name, f"{float(ns[name]):.6f}" if np.isscalar(ns[name]) else str(ns[name]),
                      help="Model parameter from the notebook. Interpret in the context of the fitted GJR-GARCH specification.")
    explain("Skewed Student-t distribution", "A heavy-tailed distribution that can also represent asymmetry.", "It may fit financial returns better when negative and positive extremes occur at different rates.", "A flexible distribution is not automatically better; compare residual diagnostics and out-of-sample backtesting.")
    explain("GJR-GARCH leverage effect", "An extra term allows negative shocks to affect conditional variance differently from positive shocks.", "A positive gamma under the usual specification indicates additional variance response to negative shocks.", "Confirm the exact parameterization used by the arch library and inspect fitted coefficients.")
    explain("GJR-GARCH stability", "A persistence condition involving the fitted variance parameters.", "A condition below one is commonly used as a stationarity/persistence check in the standard specification.", "The exact condition depends on the model and distribution; use the correct formula for the chosen implementation.")
    st.subheader("Model comparison")
    if isinstance(var_compare, pd.DataFrame):
        st.dataframe(var_compare, use_container_width=True)
        st.caption("This comparison is copied from the notebook output; confirm all columns use the same confidence level, horizon, and loss sign convention.")

# ---------------- PHASE 8 ----------------
with tabs[8]:
    st.header("Phase 8 — Walk-forward VaR Backtesting")
    st.write("Backtesting evaluates forecasts against later realized P&L. Walk-forward estimates use information available before the forecast date to avoid look-ahead bias.")
    if isinstance(backtest_df, pd.DataFrame) and not backtest_df.empty:
        st.subheader("Backtesting results")
        st.dataframe(backtest_df, use_container_width=True, hide_index=True)
        if "N_Exceptions" in backtest_df.columns and "Method" in backtest_df.columns:
            color = "Confidence" if "Confidence" in backtest_df.columns else None
            fig = px.bar(backtest_df, x="Method", y="N_Exceptions", color=color, barmode="group",
                         title="Observed VaR Exceptions by Method", template="plotly_dark")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="plot_4")
            st.caption("An exception occurs when realized loss exceeds the VaR threshold. Compare observed exceptions with the number expected at the selected confidence level.")
    else:
        st.warning("The variable `backtest_df` was not found or is empty after notebook execution.")
    if isinstance(wf_var_series, dict) and wf_var_series and pnl is not None:
        st.subheader("Choose a walk-forward VaR series")
        method = st.selectbox("Method", list(wf_var_series.keys()), key="bt_method")
        confidence_choices = list(wf_var_series[method].keys()) if isinstance(wf_var_series[method], dict) else []
        if confidence_choices:
            confidence = st.selectbox("Confidence", confidence_choices,
                                      format_func=lambda x: f"{float(x):.0%}", key="bt_conf")
            var_s = series_of(wf_var_series[method][confidence])
            if var_s is not None:
                aligned = pd.concat([pnl.rename("P&L"), var_s.rename("VaR")], axis=1).dropna()
                if not aligned.empty:
                    # Notebook VaR is typically a lower P&L quantile (negative). Keep that convention for exceptions.
                    threshold = aligned["VaR"]
                    exceptions = aligned["P&L"] < threshold
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=aligned.index, y=aligned["P&L"], name="Realized P&L", mode="lines"))
                    fig.add_trace(go.Scatter(x=aligned.index, y=threshold, name="VaR threshold", mode="lines"))
                    fig.add_trace(go.Scatter(x=aligned.index[exceptions], y=aligned.loc[exceptions, "P&L"],
                                             name="Exceptions", mode="markers", marker=dict(symbol="x", size=8)))
                    fig.update_layout(title=f"{method} — {float(confidence):.0%} VaR backtest",
                                      template="plotly_dark", height=430, xaxis_title="Date", yaxis_title="P&L (₹)")
                    st.plotly_chart(fig, use_container_width=True, key="plot_5")
                    st.metric("Exceptions in plotted overlap", f"{int(exceptions.sum())} / {len(aligned)}",
                              help="Count of aligned realized P&L observations below the fitted VaR quantile. This assumes the stored VaR series is a lower-tail P&L quantile.")
    st.subheader("Backtesting tests — how to read them")
    explain("Kupiec POF test", "Whether the observed exception frequency matches the target exception probability.", "A p-value below 0.05 typically rejects correct unconditional coverage.", "The test checks exception frequency, not whether exceptions are clustered.")
    explain("Christoffersen independence test", "Whether VaR exceptions are independent over time.", "A low p-value suggests exceptions cluster, which is dangerous because breaches may concentrate during turbulent periods.", "The result can be unstable with very few exceptions.")
    explain("Conditional coverage test", "Combines exception frequency and exception independence.", "A low p-value indicates the model fails at least one of these aspects at the chosen significance level.", "Interpret it alongside exception counts and the forecast horizon.")
    explain("Basel traffic-light test", "A regulatory-style classification based on exception counts for a specified backtest window.", "Green generally indicates fewer exceptions; yellow and red indicate increasingly concerning counts.", "Regulatory rules are specific to the confidence level and observation window; use the notebook's implemented classification rather than extrapolating.")

# ---------------- PHASE 9 ----------------
with tabs[9]:
    st.header("Phase 9 — Historical & Hypothetical Stress Testing")
    st.write("Stress testing explores severe but plausible losses and historical episodes that may not be adequately represented by a single VaR number.")
    if pnl is not None and not pnl.empty:
        rolling_5 = pnl.rolling(5).sum().dropna()
        c1, c2, c3 = st.columns(3)
        show_metric("Worst 1-day P&L", fmt_money(pnl.min()),
                    "Worst single-day realized P&L in the observed sample. This is not a bound on future losses.")
        show_metric("Worst 5-day P&L", fmt_money(rolling_5.min()) if len(rolling_5) else "Not available",
                    "Worst rolling five-observation cumulative P&L; this is a historical window, not necessarily a calendar week.")
        show_metric("Worst 1-day loss", fmt_money(abs(min(0, pnl.min()))),
                    "Absolute magnitude of the worst observed daily loss, shown as a positive number for readability.")
        chart_line(rolling_5, "Rolling 5-Day Cumulative P&L", "5-day P&L",
                   "The lowest point identifies the most severe observed rolling five-day loss.")
    if isinstance(scenario_df, pd.DataFrame) and not scenario_df.empty:
        show_table(scenario_df, "Notebook stress scenario outputs",
                   "Scenario names, shock assumptions, and losses are displayed as calculated by the notebook.")
    # Show any notebook-created stress variables/tables even if names differ.
    stress_names = [n for n in ns if any(term in n.lower() for term in ["stress", "worst_5day", "worst_single_day", "reverse_stress"])]
    if stress_names:
        with st.expander("Other stress-testing outputs captured from the notebook"):
            for name in stress_names:
                obj = ns[name]
                if isinstance(obj, (pd.DataFrame, pd.Series)):
                    st.markdown(f"**{name}**")
                    st.dataframe(obj, use_container_width=True)
                elif np.isscalar(obj) and not callable(obj):
                    st.write(f"**{name}:** {obj}")
    st.subheader("Hypothetical shock illustration")
    shock_pct = st.slider("Illustrative portfolio shock (%)", 1, 50, 10, key="stress_shock")
    illustrative_loss = float(investment) * shock_pct / 100
    show_metric("Illustrative loss amount", fmt_money(illustrative_loss),
                "Initial investment multiplied by the selected shock percentage. This is an illustrative arithmetic scenario, not a market forecast and not necessarily the same as the notebook's stress models.")
    explain("Historical stress", "Replays or summarizes severe losses observed in the available historical sample.", "It helps show how the portfolio behaved in a real crisis period.", "A future crisis may differ from the historical sample.")
    explain("Hypothetical stress", "Applies an assumed shock to the portfolio or its risk factors.", "It shows sensitivity to a scenario chosen by the analyst.", "The shock is an assumption, not a probability forecast.")
    explain("Reverse stress testing", "Starts with a loss threshold and asks what shock or combination of shocks could produce it.", "It can reveal vulnerabilities that ordinary scenario selection overlooks.", "Only display reverse-stress results if the notebook has actually calculated them; this dashboard does not invent a reverse-stress model.")

st.divider()
st.caption("Interpretation tooltips and notes are educational guides. Model outputs retain the notebook's assumptions, data, units and sign conventions; review those conventions before using figures in a report or decision.")
