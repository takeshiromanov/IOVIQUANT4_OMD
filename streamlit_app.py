import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import tempfile
import time
from pathlib import Path
import config
import ioviquant_engine as eng
import omd_integration as omd_bridge
import omd_hunter_comparison as omd_compare
from shared_data import business_day_lag, load_cor1m_series

st.set_page_config(
    page_title="IOVIQUANT Pro - Unicorn Hunter",
    page_icon="🦄",
    layout="wide",
)
st.title("⚡ IOVIQUANT Pro: Scalar Sizing & Regime Tuning Studio")
st.caption(
    "Universo Xetra 600 · backtest sperimentale a scopo di ricerca, "
    "non consulenza finanziaria."
)

BASE_DIR = Path(__file__).resolve().parent
YF_CACHE_DIR = Path(tempfile.gettempdir()) / "ioviquant_yfinance_cache"
YF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Directory scrivibile sia in locale sia su Streamlit Community Cloud.
yf.set_tz_cache_location(str(YF_CACHE_DIR))

# Il motore contabilizza in EUR ma esegue in USD. I titoli quotati in valuta
# locale vengono quindi trasformati in OHLC USD prima del calcolo segnali.
# False = la coppia esprime USD per valuta locale (es. EURUSD).
# True = la coppia esprime valuta locale per USD (es. USDJPY) e va invertita.
CURRENCY_FX_SPEC = {
    "EUR": ("EURUSD=X", False),
    "GBP": ("GBPUSD=X", False),
    "AUD": ("AUDUSD=X", False),
    "CAD": ("USDCAD=X", True),
    "CHF": ("USDCHF=X", True),
    "DKK": ("USDDKK=X", True),
    "SEK": ("USDSEK=X", True),
    "JPY": ("USDJPY=X", True),
    "KRW": ("USDKRW=X", True),
    "TWD": ("USDTWD=X", True),
    "HKD": ("USDHKD=X", True),
    "INR": ("USDINR=X", True),
    "MXN": ("USDMXN=X", True),
    "SAR": ("USDSAR=X", True),
    "IDR": ("USDIDR=X", True),
    "MYR": ("USDMYR=X", True),
    "THB": ("USDTHB=X", True),
    "SGD": ("USDSGD=X", True),
}

current_year = datetime.date.today().year
default_start = pd.to_datetime(f"{current_year}-01-01")
default_end = pd.to_datetime("today")

# ==============================================================================
# --- TOGGLE DI LAYER (fuori dal form: reattivi subito, abilitano/disabilitano
#     visivamente gli slider dei rispettivi layer senza dover premere "Avvia")
# ==============================================================================
st.sidebar.header("🎛️ Configurazione Setup")
use_expanded_universe = st.sidebar.toggle(
    f"Attiva universo espanso Xetra ({len(config.EXPANDED_UNIVERSE)} titoli)",
    value=False,
    help="Spento: usa solo i 60 titoli core congelati. Acceso: usa "
         "esattamente i ticker di xetra_raw_deduplicated.csv.",
)
active_universe = (
    config.EXPANDED_UNIVERSE
    if use_expanded_universe
    else config.CORE_UNIVERSE
)
active_universe_file = (
    config.EXPANDED_UNIVERSE_FILE
    if use_expanded_universe
    else config.UNIVERSE_FILE
)
st.sidebar.caption(
    f"Universo selezionato: {len(active_universe)} titoli · fonte: {active_universe_file.name}"
)
# Layer potati dalla configurazione Lite: restano disponibili nel motore per
# riprodurre esperimenti storici, ma non generano piu' controlli nel pannello.
use_omd_entry_filter = False
use_omd_forced_sell_exit = False
# La modalita' Legacy e i suoi parametri restano nel motore come riferimento
# storico, ma non fanno parte della configurazione calibrabile Lite.
legacy_mode = False
use_scalar_sizing = st.sidebar.checkbox("Attiva Size Scalare Continua", value=True, disabled=legacy_mode)
use_2d_breadth = st.sidebar.checkbox("Attiva Breadth 2D (Livello + Momentum)", value=True, disabled=legacy_mode)
use_dynamic_sl = st.sidebar.checkbox("Attiva Trailing ATR Modulato dal Regime", value=True, disabled=legacy_mode)
use_equity_cap = st.sidebar.checkbox(
    "Attiva Cap di Sizing su Equity Corrente",
    value=bool(eng.RECOMMENDED_PARAMS["use_equity_cap"]),
    disabled=legacy_mode,
    help="Fix §7.6 (16/07/2026): il tetto per posizione (15%) usa l'equity CORRENTE invece "
         "del capitale iniziale fisso. Causale (nessun look-ahead). Se disattivato, si torna "
         "al comportamento storico (bug noto: il rischio proporzionale non si riduce in drawdown)."
)
with st.sidebar.form("backtest_params"):
    st.header("📅 Orizzonte Temporale")
    start_date = st.date_input("Data Inizio", default_start)
    end_date = st.date_input("Data Fine", default_end)
    st.caption(
        "La data iniziale e' il confine operativo: nessun trade viene aperto "
        "prima. I 400 giorni precedenti sono scaricati solo come warm-up per "
        "indicatori e parametri."
    )

    st.header("💰 Capitale & Allocazione")
    initial_capital = st.number_input("Capitale Iniziale (€)", value=10000.0, step=1000.0)
    base_pos = st.number_input("Posizione Base (€)", value=200.0, step=50.0)
    st.caption("Contabilità e risultati in EUR; esecuzione dei titoli USA in USD.")
    commission_fixed = st.number_input("Commissione fissa per ordine (USD)", value=1.0, step=0.25)
    spread_bps = st.number_input("Spread totale stimato (bps)", value=8.0, step=1.0)
    slippage_bps = st.number_input("Slippage per lato (bps)", value=5.0, step=1.0)
    fx_conversion_bps = st.number_input("Costo conversione iniziale EUR→USD (bps)", value=10.0, step=1.0)

    st.header("📈 Layer 1: Ingressi & Volatilità (VIX)")
    entry_threshold = st.slider("Soglia Minima d'Ingresso (Scalar Trigger)", 0.1, 0.9, 0.35, 0.05, disabled=legacy_mode,
                                 help="Verificato 18/07/2026: con floor_base=1.6 (default raccomandato), "
                                      "macro_regime_score da solo supera sempre questa soglia — il gate "
                                      "non filtra piu' nulla qui (bit-identico su tutto il range testato). "
                                      "Resta vincolante se floor_base torna basso (es. Scalare Default).")
    vix_threshold = st.slider("Soglia VIX (centro sigmoide)", 15, 40, 25, 1)
    vix_k = st.slider("Ripidità Sigmoide VIX", 0.05, 1.0, 0.25, 0.05, disabled=legacy_mode,
                       help="Più alto = transizione più brusca attorno alla soglia. In Legacy resta on/off rigido.")
    vix_floor = st.slider("Fattore VIX Minimo (alta volatilità)", 0.2, 0.9, 0.5, 0.05, disabled=legacy_mode)
    convexity_exp = st.slider(
        "Esponente Convessità Sizing",
        1.0,
        6.0,
        float(eng.RECOMMENDED_PARAMS["convexity_exp"]),
        0.1,
        disabled=legacy_mode,
        help="raw_score^esponente: >1 penalizza setup mediocri e amplifica quelli "
             "migliori. Ritarato a 2.0 il 16/08/2026 per evitare che il cap al "
             "15% schiacci la dispersione prodotta dal raw score.",
    )

    st.header("🦄 Layer 2: Filtro Unicorno (Estensione)")
    ext_threshold = st.slider("Soglia Estensione (ATR sopra EMA63)", 1.0, 9.0, 6.0, 0.5, disabled=legacy_mode)
    ext_k = st.slider("Decadimento Penalità Estensione", 0.1, 2.0, 0.5, 0.1, disabled=legacy_mode)

    st.header("🌍 Layer 3: Market Breadth")
    breadth_alpha = st.slider("Peso Componente Livello (vs Momentum)", 0.0, 1.0, 0.6, 0.1)
    breadth_k = st.slider("Lookback Momentum BF (giorni)", 10, 60, 30, 5)

    st.header("🤖 Layer 4: HMM Walk-Forward & Transizione di Regime")
    hmm_min_train = st.slider("Training Minimo HMM (giorni)", 100, 500, 252, 10, disabled=legacy_mode)
    hmm_refit_every = st.slider("Refit HMM ogni N giorni", 5, 60, 21, 1, disabled=legacy_mode)
    hmm_delta_lookback = st.slider("Lookback Delta P_Bull (giorni)", 5, 60, 21, 1, disabled=legacy_mode)
    k_transition = st.slider("Ripidità Sigmoide Transizione (k_transition)", 1.0, 15.0, 6.0, 0.5, disabled=legacy_mode)
    floor_base = st.slider("Floor Inviluppo Macro Regime", 0.0, 2.5, 1.6, 0.1, disabled=legacy_mode,
                            help="Prima era fisso a 0.4 (non esposto in UI). Calibrato a 1.6 il 16/07/2026 — "
                                 "insieme a ceiling_base, uno dei due parametri risultati essenziali nello Step A "
                                 "(rimuoverlo/azzerarlo peggiora Alpha e MaxDD in modo netto).")
    ceiling_base = st.slider("Ceiling Inviluppo Macro Regime", 0.0, 3.0, 2.0, 0.1, disabled=legacy_mode,
                              help="Prima era fisso a 0.8. Calibrato a 2.0 il 16/07/2026, al bordo della griglia testata.")
    st.header("🛡️ Layer 5: Stop dinamico")
    w_bf = st.slider("Peso Breadth (BF) su R_exit", 0.0, 1.0, 1.0, 0.1, disabled=legacy_mode,
                      help="Aggiornato a 1.0 il 18/07/2026: R_exit ora e' guidato solo dalla breadth. "
                           "Il livello P_Bull (w_hmm) e' stato tolto perche' soffre della staleness "
                           "dell'HMM (refit ogni 21gg) — validato che la rimozione migliora Alpha e "
                           "MaxDD insieme, sia su train sia su test.")
    k_min = st.slider("Moltiplicatore ATR Minimo (Stop Stretto)", 1.0, 3.0, 1.5, 0.1)
    k_max = st.slider("Moltiplicatore ATR Massimo (Lascia Correre)", 3.0, 6.0, 3.5, 0.1, disabled=legacy_mode)

    run_btn = st.form_submit_button("Avvia Elaborazione 🚀")

# ==============================================================================
# --- FETCH DATI (cache separata dal calcolo segnali: cambiare un parametro
#     di calibrazione NON forza un nuovo download da yfinance) ---
# ==============================================================================
@st.cache_data(ttl="6h", max_entries=6, show_spinner=False)
def fetch_raw_data(
    tickers, benchmark, start, end, local_fx_symbols=(),
    omd_history_years=0,
):
    start_offset = pd.to_datetime(start) - pd.Timedelta(days=400)
    if omd_history_years:
        omd_start = pd.to_datetime(start) - pd.DateOffset(
            years=int(omd_history_years)
        )
        start_offset = min(start_offset, omd_start)
    # yfinance considera end esclusiva: +1 include l'ultima data scelta.
    end_exclusive = pd.to_datetime(end) + pd.Timedelta(days=1)
    all_syms = list(dict.fromkeys(
        list(tickers) + [benchmark, "^VIX", "EURUSD=X"] + list(local_fx_symbols)
    ))
    data_dict = {}

    def collect(raw, symbols):
        if raw is None or raw.empty:
            return
        for tk in symbols:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    if tk in raw.columns.get_level_values(0):
                        frame = raw[tk]
                    elif tk in raw.columns.get_level_values(1):
                        frame = raw.xs(tk, axis=1, level=1)
                    else:
                        continue
                elif len(symbols) == 1:
                    frame = raw
                else:
                    continue
                columns = ["Open", "High", "Low", "Close"]
                if "Volume" in frame.columns:
                    columns.append("Volume")
                frame = frame[columns].dropna(
                    how="all", subset=["Open", "High", "Low", "Close"]
                )
                if frame.empty:
                    continue
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                data_dict[tk] = frame
            except (KeyError, TypeError, ValueError):
                continue

    # Blocchi piccoli e download sequenziale: con universi ampi la concorrenza
    # interna di yfinance puo' bloccare i DB SQLite o esaurire i thread DNS.
    # La prima esecuzione e' meno rapida, ma deterministica e recuperabile.
    for offset in range(0, len(all_syms), 12):
        chunk = all_syms[offset:offset + 12]
        raw = yf.download(
            chunk,
            start=start_offset,
            end=end_exclusive,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
            timeout=30,
        )
        collect(raw, chunk)

    # Secondo tentativo strettamente sequenziale per eventuali simboli saltati.
    missing = [tk for tk in all_syms if tk not in data_dict]
    for tk in missing:
        time.sleep(0.15)
        retry = yf.download(
            [tk],
            start=start_offset,
            end=end_exclusive,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="ticker",
            timeout=30,
        )
        collect(retry, [tk])
    return data_dict


@st.cache_data(ttl="1h", max_entries=2, show_spinner=False)
def load_cor1m():
    """Carica COR1M dalla fonte comune con fallback locale atomico."""
    return load_cor1m_series(BASE_DIR)


def build_param_pack():
    # La configurazione UI parte dalla stessa source of truth dell'harness.
    # Gli override sotto corrispondono solo ai controlli realmente esposti.
    p = eng.RECOMMENDED_PARAMS.copy()
    p.update({
        "vix_threshold": vix_threshold, "vix_k": vix_k, "vix_floor": vix_floor,
        "entry_threshold": entry_threshold, "use_scalar_sizing": use_scalar_sizing,
        "ext_threshold": ext_threshold, "ext_k": ext_k,
        "breadth_alpha": breadth_alpha, "breadth_k": breadth_k,
        "use_2d_breadth": use_2d_breadth,
        "hmm_min_train": hmm_min_train, "hmm_refit_every": hmm_refit_every,
        "hmm_delta_lookback": hmm_delta_lookback, "k_transition": k_transition,
        "floor_base": floor_base, "floor_breadth_w": 0.6, "ceiling_base": ceiling_base, "ceiling_breadth_w": 1.7,
        "convexity_exp": convexity_exp,
        "use_dynamic_sl": use_dynamic_sl, "w_bf": w_bf,
        "k_min": k_min, "k_max": k_max,
        "use_equity_cap": use_equity_cap,
        "use_fx_conversion": True,
        "base_currency": "EUR", "execution_currency": "USD",
        "commission_fixed": commission_fixed,
        "spread_bps": spread_bps, "slippage_bps": slippage_bps,
        "fx_conversion_bps": fx_conversion_bps,
        "block_same_open_reentry": True,
    })
    return p


def render_operational_metrics(df_trades, prefix=""):
    st.markdown(f"#### 🔎 Statistiche Operative {prefix}")
    closed = df_trades[df_trades["Stato"] == "Chiusa"] if not df_trades.empty else df_trades
    tot_trades = len(closed)
    if tot_trades == 0:
        st.warning("Nessun trade chiuso nel periodo.")
        return
    tot_win = int((closed["Profit"] > 0).sum())
    win_rate = tot_win / tot_trades * 100
    best_trade = closed["Profit"].max()
    worst_trade = closed["Profit"].min()
    ret_pct = (closed["Sell_Price"] - closed["Buy_Price"]) / closed["Buy_Price"]
    unicorn_rate = (ret_pct >= 0.50).mean() * 100
    trades_100 = int((ret_pct >= 1.00).sum())
    holding_days = (pd.to_datetime(closed["Exit_Date"]) - pd.to_datetime(closed["Entry_Date"])).dt.days
    avg_holding = holding_days.mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Totale Trade Chiusi", f"{tot_trades}")
    c2.metric("Win Rate %", f"{win_rate:.1f}%", f"Vincenti: {tot_win}")
    c3.metric("🦄 Unicorn Rate (≥+50%)", f"{unicorn_rate:.1f}%")
    c4.metric("Trade ≥ +100%", f"{trades_100}")
    c5.metric("Holding Medio", f"{avg_holding:.0f} gg" if pd.notna(avg_holding) else "-")

    c6, c7 = st.columns(2)
    c6.metric("Miglior Trade", f"€ {best_trade:,.2f}")
    c7.metric("Peggior Trade", f"€ {worst_trade:,.2f}")


def calc_metrics(series):
    ret = series.pct_change().fillna(0)
    days = (series.index[-1] - series.index[0]).days
    cagr = (series.iloc[-1] / series.iloc[0]) ** (365.25 / days) - 1 if days > 0 else 0
    vol = ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    mdd = ((series - series.cummax()) / series.cummax()).min()
    return cagr, sharpe, mdd, series.iloc[-1] - series.iloc[0]


def render_audit_log(df_trades):
    if df_trades.empty:
        st.warning("Nessuna operazione registrata.")
        return
    df_disp = df_trades.copy()
    df_disp["Entry_Date"] = df_disp["Entry_Date"].dt.strftime("%Y-%m-%d")
    df_disp["Exit_Date"] = df_disp["Exit_Date"].apply(lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else "-")
    df_disp = df_disp.sort_values(by="Entry_Date", ascending=False)

    st.dataframe(
        df_disp.style.format({
            "Shares": "{:,.0f}", "Buy_Price": "$ {:.2f}",
            "Sell_Price": "$ {:.2f}", "Profit": "€ {:.2f}"
        }).map(
            lambda x: "color: #00CC96" if x == "Aperta" else (
                "color: #EF553B" if x == "SL Netto" else (
                    "color: #636EFA" if x == "Trend Break" else "color: gray")),
            subset=["Stato", "Motivo_Chiusura"]
        ),
        width='stretch'
    )

# ==============================================================================
# --- CONTROLLO FLUSSO ---
# ==============================================================================
if run_btn:
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        st.error("La data iniziale deve precedere o coincidere con la data finale.")
        st.stop()
    with st.spinner("Download dati e calcolo segnali in corso..."):
        active_currencies = {
            config.TICKER_CURRENCY.get(ticker, "USD")
            for ticker in active_universe
        }
        unsupported_currencies = sorted(
            currency for currency in active_currencies
            if currency != "USD" and currency not in CURRENCY_FX_SPEC
        )
        if unsupported_currencies:
            st.error(
                "Valute senza coppia FX configurata: " + ", ".join(unsupported_currencies)
            )
            st.stop()
        local_fx_symbols = tuple(sorted({
            CURRENCY_FX_SPEC[currency][0]
            for currency in active_currencies
            if currency != "USD"
        }))
        raw = fetch_raw_data(
            tuple(active_universe), config.BENCHMARK, start_date, end_date,
            local_fx_symbols,
            5 if use_omd_entry_filter else 0,
        )
        p_pack = build_param_pack()

        required_market = [config.BENCHMARK, "^VIX", "EURUSD=X", *local_fx_symbols]
        missing_market = [ticker for ticker in required_market if ticker not in raw]
        if missing_market:
            st.error("Download incompleto dei dati obbligatori: " + ", ".join(missing_market))
            st.stop()
        missing_stocks = [ticker for ticker in active_universe if ticker not in raw]
        if missing_stocks:
            st.warning(
                f"{len(missing_stocks)} titoli senza dati e quindi esclusi: "
                + ", ".join(missing_stocks)
            )

        dd = {}
        conversion_failures = []
        for ticker in active_universe:
            if ticker not in raw:
                continue
            currency = config.TICKER_CURRENCY.get(ticker, "USD")
            if currency == "USD":
                dd[ticker] = raw[ticker]
                continue
            fx_symbol, inverse_quote = CURRENCY_FX_SPEC[currency]
            try:
                converted = eng.convert_ohlc_to_usd(
                    raw[ticker], raw[fx_symbol], inverse_quote=inverse_quote
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                conversion_failures.append(ticker)
                continue
            if converted.empty:
                conversion_failures.append(ticker)
                continue
            if "Volume" in raw[ticker].columns:
                converted["Volume"] = raw[ticker]["Volume"].reindex(
                    converted.index
                )
            dd[ticker] = converted
        if conversion_failures:
            st.warning(
                f"{len(conversion_failures)} titoli esclusi per conversione FX incompleta: "
                + ", ".join(conversion_failures)
            )
        dd["__BENCHMARK__"] = raw[config.BENCHMARK]
        dd["__VIX__"] = raw["^VIX"]
        dd["__FX__"] = raw["EURUSD=X"]
        cor1m, cor1m_source = load_cor1m()
        cor1m_lag_business_days = business_day_lag(
            cor1m.index.max(), pd.Timestamp(end_date)
        )
        if cor1m_lag_business_days > 2:
            st.warning(
                f"COR1M ({cor1m_source}) fermo al {cor1m.index.max():%Y-%m-%d} "
                f"({cor1m_lag_business_days} giorni feriali prima della data finale). "
                "Il motore mantiene l'ultimo valore disponibile: verifica il "
                "workflow IOVIQUANT_DATA prima di interpretarlo come segnale corrente."
            )
        dd["__COR1M__"] = cor1m

        omd_schedule = None
        if use_omd_entry_filter:
            try:
                with st.spinner(
                    "Calcolo cluster mensili OMD (K=30, refit invariato)..."
                ):
                    omd_inputs = omd_bridge.build_omd_inputs(
                        dd, active_universe
                    )
                    omd_prepared = omd_bridge.build_omd_features(omd_inputs)
                    omd_schedule = omd_bridge.generate_monthly_clusters(
                        omd_prepared,
                        asof=pd.Timestamp(end_date),
                        settings=omd_bridge.OMDSettings(k_book=30),
                    )
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"Calcolo OMD non completato: {exc}")
                st.stop()
            if omd_schedule.empty:
                st.error(
                    "OMD non ha prodotto cluster utilizzabili. Servono almeno "
                    "36 mesi di training e 90 titoli con segnali completi."
                )
                st.stop()

        signals, macro = eng.assemble_all_signals(dd, active_universe, p_pack)

        omd_variant_results = None
        omd_comparison = None
        if use_omd_entry_filter:
            with st.spinner(
                "Confronto UH puro, filtro OMD e filtro OMD + Sell forzato..."
            ):
                omd_variant_results, omd_comparison = (
                    omd_compare.run_three_way_comparison(
                        signals=signals,
                        universe=active_universe,
                        initial_capital=initial_capital,
                        base_size=base_pos,
                        params=p_pack,
                        benchmark_ohlc=raw[config.BENCHMARK],
                        start_date=pd.Timestamp(start_date),
                        end_date=pd.Timestamp(end_date),
                        omd_schedule=omd_schedule,
                    )
                )

        st.session_state.signals = signals
        st.session_state.macro = macro
        st.session_state.bench_ohlc = raw[config.BENCHMARK]
        st.session_state.p_pack = p_pack
        st.session_state.initial_capital = initial_capital
        st.session_state.base_pos = base_pos
        st.session_state.run_start_date = pd.Timestamp(start_date)
        st.session_state.run_end_date = pd.Timestamp(end_date)
        st.session_state.cor1m_first_date = cor1m.index.min()
        st.session_state.cor1m_last_date = cor1m.index.max()
        st.session_state.cor1m_source = cor1m_source
        st.session_state.loaded_tickers = list(dd)
        st.session_state.run_currencies = sorted(active_currencies)
        st.session_state.run_universe = list(active_universe)
        st.session_state.run_universe_expanded = use_expanded_universe
        st.session_state.omd_schedule = omd_schedule
        st.session_state.omd_variant_results = omd_variant_results
        st.session_state.omd_comparison = omd_comparison
        st.session_state.run_done = True

if st.session_state.get("run_done", False):
    signals = st.session_state.signals
    p_pack = st.session_state.p_pack
    initial_capital = st.session_state.initial_capital
    base_pos = st.session_state.base_pos
    run_universe = st.session_state.run_universe
    run_start_date = st.session_state.run_start_date
    run_end_date = st.session_state.run_end_date

    tab1, tab2 = st.tabs(["📊 Portafoglio Globale", "🔍 Deep Dive & Modello Isolato"])

    with tab1:
        st.caption(
            f"COR1M ({st.session_state.cor1m_source}): "
            f"{st.session_state.cor1m_first_date:%Y-%m-%d} → "
            f"{st.session_state.cor1m_last_date:%Y-%m-%d} · "
            f"{len(signals)}/{len(run_universe)} titoli caricati · "
            f"universo {'espanso' if st.session_state.run_universe_expanded else 'core 60'} · "
            f"valute: {', '.join(st.session_state.run_currencies)}"
        )
        variant_results = st.session_state.get("omd_variant_results")
        if variant_results:
            active_variant = omd_compare.selected_variant_name(
                bool(p_pack.get("use_omd_entry_filter", False)),
                bool(p_pack.get("use_omd_forced_sell_exit", False)),
            )
            selected_result = variant_results[active_variant]
            hist_g = selected_result.history
            ev_g = selected_result.events
            all_t_g = selected_result.all_trades
            closed_g = selected_result.closed_trades
            st.caption(f"Configurazione visualizzata: **{active_variant}**")
        else:
            hist_g, ev_g, all_t_g, closed_g = eng.run_simulation(
                signals,
                run_universe,
                initial_capital,
                base_pos,
                p_pack,
                start_date=run_start_date,
                end_date=run_end_date,
                omd_schedule=st.session_state.get("omd_schedule"),
            )

        bench_eq, bench_entry_date, bench_entry_open = eng.build_open_entry_benchmark_equity(
            st.session_state.bench_ohlc,
            initial_capital,
            run_start_date,
            run_end_date,
        )
        res_g = hist_g.join(bench_eq).ffill()
        st.caption(
            f"Inizio operativo strategia: {hist_g.index.min():%Y-%m-%d} · "
            f"VWCE acquistato all'Open del {bench_entry_date:%Y-%m-%d} "
            f"a € {bench_entry_open:,.2f}. Lo storico precedente è solo warm-up."
        )
        kpi = eng.calc_metrics_extended(
            res_g["Equity"], res_g["VWCE_Eq"], closed_g,
            events=ev_g, initial_equity=initial_capital,
        )
        nt_s = res_g["Equity"].iloc[-1] - initial_capital
        nt_b = res_g["VWCE_Eq"].iloc[-1] - initial_capital

        omd_comparison = st.session_state.get("omd_comparison")
        if omd_comparison is not None:
            st.markdown("### Confronto OMD × Unicorn Hunter")
            comparison_display = omd_comparison.copy()
            percent_columns = [
                "CAGR", "Alpha_vs_VWCE", "MaxDD",
            ]
            st.dataframe(
                comparison_display.style.format({
                    **{column: "{:.2%}" for column in percent_columns},
                    "Sharpe": "{:.2f}",
                    "Unicorn_Rate_%": "{:.1f}%",
                    "Entry_MFE_>=50%_%": "{:.1f}%",
                    "Unicorn_Conversion_%": "{:.1f}%",
                    "Win_Rate_%": "{:.1f}%",
                    "Fast_Loss_3S_Rate_%": "{:.1f}%",
                    "Costo_Annuo_%": "{:.2f}%",
                    "Turnover_Annuo_x": "{:.2f}x",
                    "Capitale_Finale_EUR": "€ {:,.2f}",
                }),
                width="stretch",
            )
            schedule = st.session_state.get("omd_schedule")
            if schedule is not None and not schedule.empty:
                latest = schedule.sort_values("Effective_Date").iloc[-1]
                st.caption(
                    f"Ultimo cluster: segnale {latest['Signal_Date']:%Y-%m-%d}, "
                    f"efficace dal {latest['Effective_Date']:%Y-%m-%d} · "
                    f"Buy {latest['Buy_Count']} · Sell {latest['Sell_Count']} · "
                    f"regime OMD {latest['Regime']}"
                )
                with st.expander("Vedi composizione e calendario cluster OMD"):
                    buy_names = list(latest["Buy_Tickers"])
                    sell_names = list(latest["Sell_Tickers"])
                    max_names = max(len(buy_names), len(sell_names))
                    cluster_table = pd.DataFrame({
                        "Buy": buy_names + [None] * (max_names - len(buy_names)),
                        "Sell": sell_names + [None] * (max_names - len(sell_names)),
                    })
                    st.dataframe(cluster_table, width="stretch", hide_index=True)
                    st.dataframe(
                        schedule[[
                            "Signal_Date", "Effective_Date", "Buy_Count",
                            "Sell_Count", "Regime", "Refit",
                        ]].sort_values("Effective_Date", ascending=False),
                        width="stretch",
                        hide_index=True,
                    )

        cols = st.columns(4)
        cols[0].metric("Net Profit Strat", f"€ {nt_s:,.2f}", f"vs € {nt_b:,.2f} (Bench)")
        cols[1].metric("CAGR netto EUR", f"{kpi['CAGR_strategia']:.2%}",
                       f"{kpi['Excess_CAGR_net_EUR']:+.2%} vs VWCE")
        cols[2].metric("Sharpe", f"{kpi['Sharpe_strategia']:.2f}",
                       f"Bench: {kpi['Sharpe_benchmark']:.2f}")
        cols[3].metric("Max Drawdown", f"{kpi['MaxDD_strategia']:.2%}",
                       f"Bench: {kpi['MaxDD_benchmark']:.2%}")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Entry: MFE ≥ +50%", f"{kpi['Entry_MFE_>=50%_%']:.1f}%")
        k2.metric("Exit: conversione unicorn", f"{kpi['Unicorn_Conversion_%']:.1f}%")
        k3.metric("Giveback mediano", f"{kpi['Median_Peak_Giveback_pp']:.1f} pp")
        k4.metric("Costo annuo stimato", f"{kpi['Annualized_Cost_Drag_%']:.2f}%")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Runner attivati", f"{kpi['Runner_Activation_Events']}")
        r2.metric("Runner chiusi", f"{kpi['Runner_Activated_Trades']}")
        r3.metric("Runner chiusi ≥ +50%", f"{kpi['Runner_Close_Conversion_%']:.1f}%")
        r4.metric(
            "Rendimento mediano runner",
            f"{kpi['Runner_Median_Return_%']:.1f}%"
            if pd.notna(kpi["Runner_Median_Return_%"]) else "-",
        )

        cash_base = res_g["Cash_Base"].astype(float)
        cash_share = (cash_base / res_g["Equity"]).replace([np.inf, -np.inf], np.nan).dropna()
        max_deployed = (1.0 - cash_share.min()) * 100.0 if not cash_share.empty else 0.0
        capacity_saturated_sessions = (
            (cash_base < base_pos).mean() * 100.0 if not cash_base.empty else 0.0
        )
        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Liquidità finale", f"€ {cash_base.iloc[-1]:,.2f}")
        l2.metric("Liquidità minima", f"€ {cash_base.min():,.2f}")
        l3.metric("Impiego massimo capitale", f"{max_deployed:.1f}%")
        l4.metric(
            "Sedute senza una size base libera",
            f"{capacity_saturated_sessions:.1f}%",
            help="Quota di sedute in cui la liquidita' disponibile e' inferiore "
                 "alla Posizione Base selezionata.",
        )

        render_operational_metrics(closed_g, prefix="(Portafoglio Globale)")

        st.markdown("---")

        fig_g = make_subplots(specs=[[{"secondary_y": True}]])
        fig_g.add_trace(go.Scatter(x=res_g.index, y=res_g["Equity"], name="Strategia Selezionata",
                                    line=dict(color="#00CC96", width=2)), secondary_y=False)
        fig_g.add_trace(go.Scatter(x=res_g.index, y=res_g["VWCE_Eq"], name="VWCE (Buy&Hold)",
                                    line=dict(color="#636EFA", dash="dot")), secondary_y=False)
        fig_g.add_trace(go.Scatter(x=res_g.index, y=res_g["Cash_Base"], name="Liquidità disponibile",
                                    line=dict(color="#F2C94C", width=1.5, dash="dash")), secondary_y=False)
        macro_p_bull = st.session_state.macro["p_bull_bench"].loc[
            lambda x: (x.index >= run_start_date) & (x.index <= run_end_date)
        ]
        macro_breadth = st.session_state.macro["pct_above_200"].loc[
            lambda x: (x.index >= run_start_date) & (x.index <= run_end_date)
        ]
        fig_g.add_trace(go.Scatter(x=macro_p_bull.index,
                                    y=macro_p_bull * 100, name="P_Bull HMM ×100",
                                    line=dict(color="rgba(128, 128, 128, 0.35)", width=1, dash="dot")), secondary_y=True)
        fig_g.add_trace(go.Scatter(x=macro_breadth.index,
                                    y=macro_breadth, name="Breadth %",
                                    line=dict(color="rgba(255, 165, 0, 0.3)", width=1)), secondary_y=True)

        fig_g.update_layout(title="Confronto Performance Curve & Macro Overlay", template="plotly_dark", height=500)
        fig_g.update_yaxes(title_text="Capitale (€)", secondary_y=False)
        fig_g.update_yaxes(title_text="P_Bull×100 / Breadth %", secondary_y=True, range=[0, 100])
        st.plotly_chart(fig_g, width='stretch')

        st.markdown("### 📝 Trading Journal Completo")
        render_audit_log(all_t_g)

    with tab2:
        valid_tickers = [t for t in run_universe if t in signals]
        selected_ticker = st.selectbox("Seleziona Titolo per Deep Dive", valid_tickers)

        if selected_ticker:
            hist_s, ev_s, all_t_s, closed_s = eng.run_simulation(
                signals,
                [selected_ticker],
                initial_capital,
                base_pos,
                p_pack,
                start_date=run_start_date,
                end_date=run_end_date,
                omd_schedule=st.session_state.get("omd_schedule"),
            )

            df_single = signals[selected_ticker].loc[
                lambda x: (x.index >= run_start_date) & (x.index <= run_end_date)
            ]
            ticker_entry_open = float(df_single["Open"].dropna().iloc[0])
            tick_eq = (df_single["Close"] / ticker_entry_open) * initial_capital
            res_s = hist_s.join(tick_eq.rename("Ticker_BH")).ffill()

            st.markdown(f"## Analisi Isolata: {selected_ticker}")
            render_operational_metrics(closed_s, prefix=f"({selected_ticker})")

            fig_dd = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.6, 0.4],
                specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
            )

            fig_dd.add_trace(go.Scatter(x=df_single.index, y=df_single["Close"], name="Prezzo",
                                         line=dict(color="gray", width=1.5)), row=1, col=1, secondary_y=False)
            fig_dd.add_trace(go.Scatter(x=df_single.index, y=df_single["EMA5"], name="EMA 5",
                                         line=dict(color="#B6D552", width=1)), row=1, col=1, secondary_y=False)
            fig_dd.add_trace(go.Scatter(x=df_single.index, y=df_single["EMA21"], name="EMA 21",
                                         line=dict(color="#FFA500", width=1.2)), row=1, col=1, secondary_y=False)
            fig_dd.add_trace(go.Scatter(x=df_single.index, y=df_single["EMA63"], name="EMA 63",
                                         line=dict(color="#FF0000", width=1.5)), row=1, col=1, secondary_y=False)
            fig_dd.add_trace(go.Scatter(x=df_single.index, y=df_single["p_bull_bench"], name="P_Bull HMM (benchmark)",
                                         line=dict(color="rgba(128, 128, 128, 0.35)", dash="dot")), row=1, col=1, secondary_y=True)
            fig_dd.add_trace(go.Scatter(x=df_single.index, y=df_single["macro_regime_score"], name="Macro Regime Score",
                                         line=dict(color="rgba(180, 100, 255, 0.5)", dash="dash")), row=1, col=1, secondary_y=True)

            breakout_days = df_single[df_single["breakout_bonus"] > 0]
            if not breakout_days.empty:
                fig_dd.add_trace(go.Scatter(x=breakout_days.index, y=breakout_days["Close"], mode="markers",
                                             name="Breakout Filtro Unicorno",
                                             marker=dict(symbol="diamond", size=8, color="#FFD700")),
                                  row=1, col=1, secondary_y=False)

            if not ev_s.empty:
                buys = ev_s[ev_s["Action"] == "BUY"]
                sells = ev_s[ev_s["Action"] == "SELL"]
                fig_dd.add_trace(go.Scatter(x=buys["Date"], y=buys["Price"], mode="markers", name="Buy Trigger",
                                             marker=dict(symbol="triangle-up", size=11, color="#00CC96")),
                                  row=1, col=1, secondary_y=False)
                fig_dd.add_trace(go.Scatter(x=sells["Date"], y=sells["Price"], mode="markers", name="Sell Trigger",
                                             marker=dict(symbol="triangle-down", size=11, color="#EF553B")),
                                  row=1, col=1, secondary_y=False)

            fig_dd.add_trace(go.Scatter(x=res_s.index, y=res_s["Equity"], name=f"Strat su {selected_ticker}",
                                         line=dict(color="#00CC96", width=2)), row=2, col=1)
            fig_dd.add_trace(go.Scatter(x=res_s.index, y=res_s["Ticker_BH"], name=f"B&H {selected_ticker}",
                                         line=dict(color="#FF8C00", dash="dot")), row=2, col=1)

            fig_dd.update_layout(height=650, template="plotly_dark", hovermode="x unified", margin=dict(t=30, b=10))
            st.plotly_chart(fig_dd, width='stretch')

            st.markdown(f"### 📝 Trading Journal Filtrato: {selected_ticker}")
            render_audit_log(all_t_s)
