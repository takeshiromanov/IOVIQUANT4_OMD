"""
IOVIQUANT Pro - Motore di calcolo (core engine)
================================================
Modulo puro Python/pandas, senza dipendenza da Streamlit, cosi' da poter
essere importato sia dalla webapp (app.py) sia da script di calibrazione
headless (locali o nel sandbox di Claude).

Fix di causalita' rispetto alla versione precedente:
- Tutte le variabili di segnale (ema_signal, ATR, extension_penalty,
  breakout_bonus, vix_factor, macro_regime_score, breadth_factor, P_Bull)
  vengono calcolate "a fine giornata" e poi SHIFTATE di 1 prima di essere
  usate nel loop di simulazione: la decisione per il giorno D usa sempre
  dati noti alla chiusura di D-1.
- Fill di entrata/uscita eseguiti sull'Open del giorno D (non piu' sulla
  Close dello stesso giorno usato per decidere).
- Mark-to-market di fine giornata sulla Close reale di D.
- HMM: un solo modello Gaussiano a 2 stati fittato sui log-return del
  benchmark (non piu' uno per titolo), con expanding window causale e
  refit periodico. L'amplificatore usa il delta a N giorni di P_Bull
  (sigmoide sulla transizione), non il livello.
- VIX: sigmoide continua invece di soglia binaria on/off.
- Filtro Unicorno: extension penalty + breakout bonus, entrambi assenti
  nella versione precedente.
- Sizing: esponente di convessita' (convexity_exp) applicato al raw_score.
- Stop profit-aware: oltre una soglia di profitto aperto, il moltiplicatore
  ATR passa a un cap superiore dedicato.
- Layer 6 (nuovo, 11/07/2026): freno di portafoglio basato sulla correlazione
  realizzata a coppie dell'universo (rolling, causale). Quando la correlazione
  sale, riduce il sizing delle nuove entrate (leva a) E stringe simultaneamente
  lo stop di TUTTE le posizioni aperte (leva b), indipendentemente dai segnali
  tecnici del singolo titolo. Disattivo di default (use_portfolio_brake=False).

--- Layer sperimentali attivi (stato aggiornato 27/07/2026, audit di ridondanza) ---
- EMA a finestre calibrabili (ema_fast_n/ema_mid_n/ema_slow_n, default
  5/21/63): prima erano hardcoded. I nomi di colonna EMA5/EMA21/EMA63 restano
  invariati per retrocompatibilita' con Legacy Mode e con app.py (che li usa
  per il plotting), anche se ora rappresentano periodi parametrici — vedi
  commento in compute_ticker_signals. Implementata, mai sweeppata: prossimo
  passo aperto.
- Layer AR-weighted momentum (Cap. 11-12 del libro Zakamulin-Giner) —
  RIDISEGNATO 27/07/2026. Prima versione (23/07) applicava i pesi AR ai
  rendimenti giornalieri del SINGOLO TITOLO pre-evento raro: respinta,
  nessuna struttura di autocorrelazione stabile (vedi stima_pesi_ar.py e
  PROGETTO_IOVIQUANT_RIASSUNTO.md, esito 23/07). Diagnosi: il framework del
  libro presuppone un processo AGGREGATO (indice di mercato) con persistenza
  stimabile su base mensile/lungo periodo, non un singolo titolo condizionato
  a un evento raro. Corretto: il segnale AR si calcola ora UNA SOLA VOLTA sui
  log-return del BENCHMARK (stesso pattern di vix_factor/cor1m_factor — una
  serie unica, applicata a tutti i ticker), non piu' per-ticker sul proprio
  Close. Ancora "da testare": ar_weights=None di default (nessun peso
  calibrato sull'aggregato), use_ar_weighted_momentum=False. Va rieseguita la
  stima offline (stima_pesi_ar.py, funzione aggregata) prima di attivare.

--- Layer rimossi in questa sessione (27/07/2026, audit di ridondanza) ---
Rimossi per intero dal motore (non solo disattivati) — entrambi con verdetto
gia' chiuso nel riassunto progetto, nessuna delle due rimozioni tocca
RECOMMENDED_PARAMS/DEFAULT_PARAMS (erano gia' spenti ovunque, sanity check
bit-identico confermato dopo la rimozione):
- Layer 6ter/6ter-bis (freno COR1M-basso, 22/07/2026): 6ter (su k_profit_cap)
  confermato strutturalmente inerte; 6ter-bis (modulazione soglia uscita
  parziale) sweep non conclusivo, superficie di calibrazione rumorosa/non
  monotona. Nessuno dei due mai promosso.
- Layer cluster brake settoriale (23/07/2026): riconosciuto esplicitamente
  come remake del Layer 6 (correlazione realizzata) gia' superato da COR1M,
  non un'idea "dal libro" nonostante la presentazione iniziale. Esito tecnico
  positivo su FULL ma mai elevato a priorita' — rimosso per non portare
  avanti un modulo morto senza base teorica dichiarata.
"""

import logging
import numpy as np
import pandas as pd
try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:  # consente test di accounting/KPI senza il layer HMM
    GaussianHMM = None

logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # silenzia i warning di non-convergenza (attesi su finestre corte in warm-up)


def _align_asof_ffill(series, target_index):
    """Allinea una serie sparsa all'ultima osservazione disponibile.

    L'unione degli indici è indispensabile per fonti settimanali datate nel
    weekend (come il COR1M storico): un semplice ``reindex(target).ffill()``
    eliminerebbe prima tutte le righe domenicali e non avrebbe più nulla da
    propagare alle sedute successive.
    """
    source = pd.Series(series).sort_index()
    target = pd.DatetimeIndex(target_index)
    union = source.index.union(target).sort_values()
    return source.reindex(union).ffill().reindex(target)


def convert_ohlc_to_usd(local_ohlc, fx_ohlc, inverse_quote=False):
    """Converte OHLC da valuta locale a USD senza usare osservazioni future.

    ``fx_ohlc`` deve esprimere USD per unita' locale quando ``inverse_quote``
    e' False (es. EURUSD), oppure unita' locali per USD quando e' True
    (es. USDJPY). Per High/Low una quotazione inversa richiede di scambiare
    Low e High prima del reciproco. Il risultato viene infine reso coerente
    imponendo High >= Open/Close e Low <= Open/Close.
    """
    columns = ["Open", "High", "Low", "Close"]
    missing_local = set(columns).difference(local_ohlc.columns)
    missing_fx = set(columns).difference(fx_ohlc.columns)
    if missing_local or missing_fx:
        raise ValueError(
            "OHLC incompleto per conversione FX: "
            f"prezzo={sorted(missing_local)}, fx={sorted(missing_fx)}"
        )

    local = local_ohlc[columns].astype(float).sort_index()
    fx = fx_ohlc[columns].astype(float).sort_index()
    union = fx.index.union(local.index).sort_values()
    aligned = fx.reindex(union).ffill().reindex(local.index)

    if inverse_quote:
        rates = pd.DataFrame(index=aligned.index)
        rates["Open"] = 1.0 / aligned["Open"]
        rates["Close"] = 1.0 / aligned["Close"]
        rates["High"] = 1.0 / aligned["Low"]
        rates["Low"] = 1.0 / aligned["High"]
    else:
        rates = aligned

    rates = rates.replace([np.inf, -np.inf], np.nan)
    converted = local * rates
    converted["High"] = pd.concat(
        [converted["Open"], converted["Close"], converted["High"]], axis=1
    ).max(axis=1)
    converted["Low"] = pd.concat(
        [converted["Open"], converted["Close"], converted["Low"]], axis=1
    ).min(axis=1)
    return converted.dropna(subset=columns)


# ==============================================================================
# DEFAULT PARAMETERS (valori di partenza plausibili, da validare in calibrazione)
# ==============================================================================
DEFAULT_PARAMS = {
    # Layer 1: EMA + VIX
    "vix_threshold": 25.0,
    "vix_k": 0.25,          # steepness sigmoide VIX
    "vix_floor": 0.5,       # asintoto inferiore del fattore VIX ad alta volatilita'
    "entry_threshold": 0.35,
    "use_scalar_sizing": True,
    # Gate opzionale sul prezzo realmente eseguibile all'Open T rispetto a
    # medie note alla Close T-1. Valori: "none", "fast" (EMA5_lag1),
    # "mid" (EMA21_lag1), "mid_atr_hard" e "mid_atr_soft". Le due varianti
    # ATR usano z=(Open-EMA21_lag1)/ATR_lag1: hard richiede z sopra la soglia;
    # soft riduce linearmente la size tra z=0 e il floor. Disattivo di default
    # finche' l'ablazione non viene promossa.
    "entry_open_ema_gate": "none",
    "entry_open_ema_atr_threshold": -0.5,
    "entry_open_ema_atr_soft_floor": -1.0,
    "use_entry_close_ema21_atr_gate": False,
    "entry_close_ema21_atr_threshold": -0.15,
    "use_entry_ema_spread_atr_gate": False,
    "entry_ema_spread_atr_threshold": 0.05,
    "entry_fast_loss_gate_regime": "all",
    # Overlay OMD opzionale. Il calendario dei cluster viene passato
    # esplicitamente a run_simulation: i default False preservano bit-per-bit
    # la baseline Unicorn Hunter standalone.
    "use_omd_entry_filter": False,
    "use_omd_forced_sell_exit": False,
    # Freno macro di transizione (sperimentale). La regola usa livelli e
    # delta noti alla Close T-1: COR1M basso, oppure COR1M in risalita rapida
    # con VIX in accelerazione. Disattivo nei default; l'ablazione dedicata
    # decide se e come promuoverlo.
    "use_macro_state_entry_brake": False,
    "macro_state_entry_mode": "low_or_fracture",
    "macro_state_entry_size_factor": 0.0,
    "macro_cor1m_low_threshold": 10.0,
    "macro_cor1m_rise_5d_threshold": 5.0,
    "macro_vix_rise_5d_threshold": 0.10,
    "macro_breadth_fall_5d_threshold": -5.0,
    "macro_hmm_fall_5d_threshold": -0.05,
    # Acceleratore d'uscita: stringe lo stop dei non-runner e il giveback dei
    # runner, ma non introduce un take-profit fisso che troncherebbe la coda.
    "use_macro_state_exit_accelerator": False,
    "macro_state_exit_mode": "low_or_fracture",
    "macro_state_exit_k_mult": 0.65,
    "macro_state_runner_giveback": 0.10,
    "macro_state_exit_policy": "stop_mult",
    "macro_state_exit_min_profit": 0.15,
    "macro_state_exit_require_micro_weakness": True,
    "macro_profit_lock_fraction": 0.50,
    "macro_partial_exit_pct": 0.25,
    "macro_partial_min_profit": 0.25,
    # Cap dinamico all'esposizione per le sole NUOVE entrate. Non vende e non
    # riduce posizioni esistenti se il loro valore supera il cap per effetto di
    # un rally: limita soltanto il reinvestimento del cash nei regimi fragili.
    # Lo stato risk-off e' causale (feature note alla Close T-1), stringe subito
    # e recupera gradualmente dopo un breve hold. Disattivo nei default.
    "use_macro_exposure_cap": False,
    "macro_exposure_transition_mode": "broad_fracture",
    "macro_exposure_riskoff_mode": "confirmed_fracture",
    "macro_exposure_transition_cap": 0.60,
    "macro_exposure_riskoff_cap": 0.35,
    "macro_exposure_hold_sessions": 5,
    "macro_exposure_recovery_step": 0.10,
    "use_failed_launch_exit": False,
    "failed_launch_days": 3,
    "failed_launch_atr_threshold": -1.0,

    # EMA a finestre calibrabili (nuovo, 23/07/2026): default = valori
    # precedentemente hardcoded, per garantire output bit-identico. Il libro
    # (Cap. 5-6, Zakamulin-Giner) mostra che NSF/ALT dipendono direttamente
    # dalla dimensione della finestra — non c'e' motivo a priori che 5/21/63
    # siano ottime per QUESTO universo. Non ancora sweeppate.
    "ema_fast_n": 5,
    "ema_mid_n": 21,
    "ema_slow_n": 63,
    # Costanti storicamente hardcoded, ora nominate per consentire alla pagina
    # diagnostica di misurarne la sensibilita'. I default preservano il
    # comportamento precedente in modo bit-identico.
    "atr_window": 14,
    "ema_signal_steepness": 2.0,

    # Layer 2: Filtro Unicorno
    "ext_threshold": 3.0,       # ATR sopra EMA63 oltre cui scatta la penalita'
    "ext_k": 0.5,                # decadimento penalita' estensione
    "breakout_lookback": 20,     # giorni per "nuovo massimo"
    "compression_ratio": 0.75,   # ATR% < 0.75 * media_63 = "compresso"
    "breakout_bonus_pct": 0.30,  # bonus moltiplicativo +30%
    "atr_pct_avg_window": 63,

    # Layer 3: Breadth 2D
    "breadth_alpha": 0.6,
    "breadth_k": 30,
    "breadth_sma_window": 200,
    "breadth_level_steepness": 0.1,
    "breadth_center": 50.0,
    "breadth_momentum_steepness": 1.0,
    "old_breadth_threshold": 40.0,   # usato solo in Legacy / breadth non-2D

    # Layer 4: HMM walk-forward + transizione
    "hmm_min_train": 252,
    "hmm_refit_every": 21,
    "hmm_n_components": 2,
    "hmm_n_iter": 100,
    "hmm_tol": 1e-2,
    "hmm_random_state": 42,
    "hmm_delta_lookback": 21,
    "k_transition": 6.0,
    "floor_base": 0.4, "floor_breadth_w": 0.6,
    "ceiling_base": 0.8, "ceiling_breadth_w": 1.7,
    "hmm_w": 1.5,   # solo Legacy: 1 + hmm_w*(P_Bull-0.5)

    # Sizing
    "convexity_exp": 1.5,
    # Cap di sizing (§7.6, fix 16/07/2026): di default il cap sul target_size
    # resta ancorato al capitale INIZIALE (comportamento storico, bug noto: in
    # drawdown il rischio proporzionale sale invece di scendere). Con
    # use_equity_cap=True il cap usa l'equity CORRENTE (cash + MTM posizioni
    # aperte, causale: valutata all'ultima Close nota prima della decisione di
    # oggi) — si restringe automaticamente in drawdown. Default False per non
    # alterare la riproducibilita' di nessun risultato precedente a questa fix.
    "use_equity_cap": False,
    "position_cap_pct": 0.15,

    # Layer 5: Stop dinamico + profit-aware
    "use_dynamic_sl": True,
    "w_bf": 0.5, "w_hmm": 0.5,
    "k_min": 1.5, "k_max": 3.5,
    "use_profit_aware": True,
    "profit_threshold": 0.50,   # +50% non realizzato
    "k_profit_cap": 5.0,

    # Layer 5bis (nuovo, 20/07/2026): Extension-on-exit -- simmetrico al
    # profit-aware sopra. Stringe lo stop (moltiplica k_used per
    # extension_penalty_lag1, stessa formula/soglia del Filtro Unicorno
    # d'ingresso, ext_threshold/ext_k) quando la posizione e' sia estesa sia
    # ancora priva di un cuscinetto di profitto (unrealized return sotto
    # ext_exit_gate). Idea originaria: §7.1 del riassunto progetto.
    # ESITO ABLAZIONE (20/07/2026): testato su train SUB_A, risultato
    # bit-identico al baseline -- MAI si attiva. Diagnosticato: con
    # ext_threshold=6.0 (soglia calibrata per l'ingresso), extension_penalty_lag1
    # e' <1.0 solo nell'1.5% dei ticker-day di tutto l'universo, e "esteso"
    # (rally recente forte) e' quasi mutuamente esclusivo con "ancora in
    # rosso/pari". Non e' un problema di soglia del gate di profitto, e' la
    # soglia di estensione stessa (ereditata dall'ingresso) troppo estrema
    # per questo scopo. NON adottato. Disattivo di default, nessuna
    # regressione -- lasciato nel motore come riferimento/possibile base per
    # una versione con soglia di estensione dedicata (piu' bassa) in futuro.
    "use_extension_on_exit": False,
    "ext_exit_gate": 0.0,   # 0.0 = ancora in rosso o pari

    # Layer 5ter (nuovo, 20/07/2026): ema-decay-on-exit -- vedi commento inline
    # in run_simulation. threshold=0.6 (poco sopra il bordo min_gap=0, dove
    # ema_signal_lag1=0.5), k=8.0 (sigmoide abbastanza ripida nello spazio 0-1),
    # floor=0.5 (non stringe mai oltre meta' del k_dynamic base).
    # ESITO ABLAZIONE (20/07/2026): testato su train SUB_A con 4 soglie
    # (0.55-0.7) e con parametri molto piu' blandi (floor 0.85-0.9, k 3-4) --
    # RESPINTO in ogni configurazione. Stessa firma di whipsaw gia' vista con
    # k_min stretto (§ sessione 20/07): N trade +30-45%, MaxDD peggiora
    # ovunque (-14.00%->-21/-24% su train), Alpha crolla, Unicorn Rate a
    # zero nella versione aggressiva. Confermato su SUB_B/FULL con la
    # variante piu' blanda: Alpha SUB_B +26.10%->+14.93%, MaxDD FULL
    # -18.31%->-23.40%. Diagnosi: ema_signal_lag1 in zona 0.5-0.7 e' rumore
    # ordinario dentro trend sani, non un segnale di inversione -- stringere
    # lo stop li' taglia sistematicamente posizioni che si sarebbero
    # riprese. NON adottato. Disattivo di default, nessuna regressione.
    "use_ema_decay_on_exit": False,
    "ema_decay_threshold": 0.6,
    "ema_decay_k": 8.0,
    "ema_decay_floor": 0.5,

    # Layer 5quater (nuovo, 20/07/2026): uscita scalata -- vedi commento inline
    # in run_simulation. partial_exit_pct = frazione di azioni (intere) vendute
    # al traguardo profit_threshold. Riusa la soglia gia' calibrata del
    # profit-aware, nessun nuovo grado di liberta' sulla soglia. Disattivo di
    # default, nessuna regressione.
    "use_partial_exit": False,
    "partial_exit_pct": 0.33,

    # Layer 5quater-bis (31/07/2026): post-unicorn runner. Quando una Close
    # gia' osservata porta il trade almeno a runner_trigger, la posizione
    # passa in uno stato dedicato: il normale stop ATR viene sostituito da un
    # trailing sulle sole Close note e, opzionalmente, il Trend Break EMA viene
    # ignorato. L'uscita resta causale (segnale sulla Close T-1, fill Open T).
    # Disattivo di default finche' non supera la validazione nested-purged.
    "use_post_unicorn_runner": False,
    "runner_trigger": 0.50,
    "runner_giveback_pct": 0.30,
    "runner_min_locked_gain": 0.10,
    "runner_ignore_trend_break": True,
    # False = ablazione pulita: dopo il trigger ignora soltanto il Trend Break
    # e continua a usare lo stop ATR/profit-aware standard.
    "runner_use_pct_stop": True,
    # None mantiene la retrocompatibilita' con runner_use_pct_stop:
    # True -> "pct", False -> "atr". "hybrid" applica
    # max(profit_floor, min(percent_stop, baseline_ATR_stop)).
    # Ablazione 31/07/2026: hybrid non promosso (solo 50% dei contesti inner
    # conformi; worst inner MaxDD -35.59% vs -24.28% del runner pct).
    "runner_stop_mode": None,
    # Solo per runner_stop_mode="bounded": l'ATR puo' allargare il giveback
    # minimo, ma non oltre questa quota del massimo raggiunto.
    # Ablazione 31/07/2026 a 35%: non promosso; worst inner MaxDD -30.29%,
    # FULL excess +19.11% vs +20.55% del runner pct20.
    "runner_max_giveback_pct": 0.35,
    # Solo per runner_stop_mode="profit_ratchet": quota del massimo profitto
    # maturato che il trailing tenta di preservare. A 0.50 lo stop e' a meta'
    # strada tra prezzo di carico e massimo noto.
    # Ablazione 31/07/2026: non promosso; worst inner MaxDD -37.53%, FULL
    # MaxDD -41.18% nonostante excess CAGR +26.22%.
    "runner_profit_lock_fraction": 0.50,

    # Layer 5quinquies (nuovo, 30/07/2026, Cap. 14 del libro Zakamulin-Giner,
    # "Optimal Trading Frequency") -- vedi commento inline in run_simulation
    # per il meccanismo. exit_freq_days=5 = valuta il test di uscita ogni ~5
    # giorni di trading (settimanale). exit_freq_offset: solo per controllo di
    # robustezza (verifica che l'effetto non dipenda dalla fase di
    # allineamento), 0 in produzione.
    # ESITO ABLAZIONE (30/07/2026): testato a tutte le 5 fasi (offset 0-4) su
    # FULL/SUB_A/SUB_B -- RESPINTO in ogni singola combinazione fase x finestra
    # (15/15), nessuna eccezione. Objective (Alpha+0.3*MaxDD) sempre sotto il
    # baseline daily: FULL 0.082->[-0.026, 0.065] a seconda della fase, SUB_A
    # 0.024->[-0.070, 0.017], SUB_B 0.214->[0.062, 0.155]. N_Trade scende
    # (621-649 vs 896 su FULL) ma MaxDD peggiora quasi ovunque invece di
    # migliorare -- il meccanismo non filtra whipsaw "falso", ritarda uscite
    # vere lasciando correre le perdite fino al prossimo giorno di verifica.
    # Diagnosi: lo stop dinamico esistente (ATR trailing continuo + profit-
    # aware) gia' incorpora smoothing sufficiente; il costo di transazione in
    # questo motore (fee=1.0, trascurabile) non e' il vincolo che il Cap. 14
    # presuppone -- l'argomento del libro (frequenza quasi irrilevante in
    # Sharpe, ma breakeven sui costi molto piu' permissivo a bassa frequenza)
    # vale quando i costi di transazione sono il fattore limitante; qui non lo
    # sono. NON adottato. Disattivo di default, nessuna regressione.
    "use_weekly_exit_freq": False,
    "exit_freq_days": 5,
    "exit_freq_offset": 0,

    # Layer 6: Freno di portafoglio (correlazione realizzata) — valori di partenza
    # plausibili basati sulla distribuzione storica (mediana ~0.25, p75 ~0.34,
    # p90 ~0.49; bear market 2022 e shock aprile 2025 nel range 0.44-0.73),
    # da validare in calibrazione come tutti gli altri.
    "use_portfolio_brake": False,
    "corr_window": 20,
    "corr_threshold": 0.40,
    "corr_k": 15.0,
    "corr_floor": 0.4,

    # Layer 6bis: Freno di portafoglio via ^COR1M (Cboe 1-Month Implied Correlation,
    # forward-looking dalle opzioni) — alternativa/complemento alla correlazione
    # realizzata sopra. Valori di partenza da percentili storici (dati dal
    # 2022-03-31, mediana ~18.5, P75 ~31, P90 ~45; picchi giornalieri 49.9 in
    # aprile 2025 e 41.7 in marzo 2026), da validare in calibrazione.
    # Copertura dati: da 2022-03-31 in poi. Prima di quella data il fattore
    # resta neutro (1.0, nessun freno) per assenza di segnale, non per estrapolazione.
    "use_cor1m_brake": False,
    "cor1m_threshold": 35.0,
    "cor1m_k": 0.15,
    "cor1m_floor": 0.4,

    # Layer 6ter/6ter-bis (freno "complacency" via COR1M basso) — RIMOSSI
    # interamente il 27/07/2026 (audit di ridondanza). Entrambi mai promossi:
    # 6ter (su k_profit_cap) confermato strutturalmente inerte il 22/07 (lo
    # stop non e' mai il vincolo attivo, stessa firma gia' nota per
    # k_profit_cap dall'11/07); 6ter-bis (modulazione soglia uscita parziale)
    # sweep non conclusivo, superficie di calibrazione rumorosa/non monotona.
    # Vedi PROGETTO_IOVIQUANT_RIASSUNTO.md §6sexies per la diagnosi completa.

    # Modalita'
    "legacy_mode": False,
    "use_2d_breadth": True,

    # Contabilita' ed esecuzione. I default mantengono il comportamento
    # storico nominale; RECOMMENDED_PARAMS abilita il reporting EUR con
    # esecuzione dei titoli USA in USD.
    "use_fx_conversion": False,
    "base_currency": "NATIVE",
    "execution_currency": "NATIVE",
    "fx_conversion_bps": 0.0,
    "commission_fixed": 1.0,
    "spread_bps": 0.0,       # spread totale; al fill si applica meta' spread
    "slippage_bps": 0.0,
    "block_same_open_reentry": False,

    # Alias storico mantenuto per gli script esistenti.
    "fee": 1.0,

    # Layer 7 (nuovo, 19/07/2026): Gap Intensity -- percentile cross-sectionale
    # dell'intensita' media dei gap overnight (|Open_t - Close_t-1|/Close_t-1,
    # rolling gap_lookback gg). Validato in fase 2a: unico segnale con lift
    # stabile fino a 40gg di anticipo, indipendente da RS/momentum (corr~0.02).
    # Disattivo di default (use_gap_signal=False, nessuna regressione).
    "use_gap_signal": False,
    "gap_lookback": 10,
    "gap_threshold": 0.75,   # percentile (0-1) sopra cui il bonus inizia a salire
    "gap_k": 15.0,           # steepness sigmoide (scala 0-1, serve k alto)
    "gap_bonus_pct": 0.30,   # bonus moltiplicativo massimo (+30%), no penalita' sotto soglia

    # Layer 8 (nuovo, 19/07/2026): Distanza dai minimi 252gg -- percentile
    # cross-sectionale di (Close/rolling_min_252gg - 1). Validato come
    # complemento a Gap Intensity (corr 0.33, non ridondante): la combinazione
    # migliora la precisione rispetto a Gap da solo. Disattivo di default.
    "use_lowdist_signal": False,
    "lowdist_lookback": 252,
    "lowdist_threshold": 0.75,
    "lowdist_k": 15.0,
    "lowdist_bonus_pct": 0.30,

    # Layer 7bis/8bis (19/07/2026): versione GATE di Gap Intensity e Distanza-
    # dai-minimi -- filtro booleano AND accanto a ema_signal>0, NESSUNA
    # interazione con raw_score/convexity_exp (a differenza di use_gap_signal/
    # use_lowdist_signal sopra, che si sono rivelati solo un moltiplicatore di
    # size sulle stesse posizioni gia' vincenti, non un ampliamento reale
    # dell'insieme dei trade). Qui il candidato semplicemente non viene
    # considerato se non supera la soglia, indipendentemente da quanto sarebbe
    # stato dimensionato. Disattivo di default.
    "use_gap_gate": False,
    "gap_gate_threshold": 0.75,
    "use_lowdist_gate": False,
    "lowdist_gate_threshold": 0.75,

    # Layer 7ter/8ter (19/07/2026): condizionamento di regime per i gate sopra.
    # Diagnosticato: il gate esclude sistematicamente i difensivi (MRK, ABBV,
    # JNJ, PG, ecc.) che in baseline fanno da ammortizzatore in bear market --
    # confermato sul drawdown Dic-2022 (13 posizioni quasi tutte in utile in
    # baseline, vs 4 posizioni concentrate col gate). Prima versione con
    # p_bull_bench (HMM) fallita: staleness nota (refit ogni 21gg, stessa
    # lezione del 18/07 su w_hmm) -- l'HMM restava a ~0.99 per tutto nov-dic
    # 2022, riconosceva il crollo solo 1 giorno su ~40. Corretto con
    # breadth_factor_lag1 (aggiornato ogni giorno, nessuna staleness).
    # use_regime_conditional_gates attiva i gate SOLO quando breadth_factor_lag1
    # supera regime_bull_threshold; sotto soglia i gate sono trasparenti
    # (comportamento = baseline). Disattivo di default (nessuna regressione).
    "use_regime_conditional_gates": False,
    "regime_bull_threshold": 0.5,

    # --- Layer AR-weighted momentum (nuovo, 23/07/2026, Cap. 11-12 libro) ---
    # ar_weights: vettore di pesi stimato OFFLINE con stima_pesi_ar.py sui
    # 148 episodi "unicorno" reali. None finche' non viene calibrato -- con
    # ar_weights=None, compute_ar_weighted_signal ritorna 0.0 ovunque, quindi
    # use_ar_weighted_momentum=True senza pesi calibrati e' innocuo (non
    # solleva errori) ma inutile: va sempre accompagnato da un vettore reale.
    "use_ar_weighted_momentum": False,
    # ESITO TEST REALE (23/07/2026): stima_pesi_ar.py eseguito sui 195 episodi
    # "unicorno" estratti dai dati reali (target +50%/105gg, stessa def. di
    # §6quater). Risultato: NESSUNA struttura di autocorrelazione stabile nei
    # rendimenti pre-evento, a qualunque risoluzione di lag testata (10/15/
    # 20/30/60). Test di stabilita' tra le due onde macro (rally AI 2023-24 vs
    # supercycle memoria/AI 2025-26): correlazione tra i pesi stimati sulle
    # due onde SEMPRE <0.2 in valore assoluto, spesso NEGATIVA (es. -0.152 a
    # max_lag=10, il caso con rapporto osservazioni/parametri piu' favorevole,
    # 21.7). Non e' un problema di potenza statistica (rapporto gia' ampio a
    # max_lag basso) -- il segnale semplicemente non esiste in questa forma.
    # Diagnosi: il Cap. 11-12 del libro presuppone un processo AGGREGATO
    # (indice di mercato) con persistenza stimabile su base mensile/lungo
    # periodo (kappa~0.3 stimato su 150 anni di dati S&P mensili nel libro);
    # qui si tentava un AR su rendimenti GIORNALIERI di SINGOLI TITOLI
    # condizionati a un evento raro (195 episodi, spesso sovrapposti nel
    # tempo/settore, quindi con molta meno informazione indipendente del
    # numero nominale) -- contesto troppo rumoroso per questo framework.
    # NON adottato. Layer RIDISEGNATO il 27/07/2026 per applicarsi a un
    # processo aggregato invece che al singolo titolo pre-evento (vedi
    # compute_ar_weighted_signal, ora chiamata una sola volta sul benchmark
    # in assemble_all_signals) -- stima_pesi_ottimi_aggregato in
    # stima_pesi_ar.py. TEST ONESTO ESEGUITO lo stesso giorno sul benchmark
    # reale (VWCE.MI, 1647gg): autocorrelazioni ai vari lag minuscole (tutte
    # <0.05 in valore assoluto, rumore su un indice diversificato daily), e
    # la stabilita' tra le due meta' della serie e' accettabile solo a
    # max_lag=5 (corr=0.742) ma crolla a max_lag=10/15 (corr=0.173/-0.002).
    # Ripetuto a frequenza settimanale (338 osservazioni): ancora instabile
    # (corr=-0.880 a max_lag=4, 0.364 a max_lag=8). Nessuna delle due
    # frequenze mostra una struttura promuovibile in questo campione --
    # risultato negativo ma onesto, non un problema di dominio del framework
    # come nella versione per-titolo (qui il processo E' aggregato, come
    # richiesto dal libro), semplicemente VWCE.MI a queste frequenze non
    # mostra persistenza AR stabile. "Da testare" resta la sola strada
    # collegata al Cap. 14 (aggregazione a frequenze piu' basse, es.
    # mensile) mai ancora provata per mancanza di osservazioni sufficienti
    # a questa lunghezza di storico.
    "ar_weights": None,
    # ar_weight_scale: bonus moltiplicativo MASSIMO (stile gap_bonus_pct/
    # lowdist_bonus_pct, forma additiva 1.0 + bonus*sigmoide -- MAI <1.0).
    # Corretto fin dall'inizio in questa forma (non nella forma penalizzante
    # stile vix_factor) per evitare l'errore gia' fatto e corretto il
    # 19/07/2026 con gap_factor/lowdist_factor: con convexity_exp=5.0, un
    # fattore <1 applicato alla maggioranza dei titoli ogni giorno viene
    # amplificato in modo devastante (0.4^5~=0.01).
    "ar_weight_scale": 0.30,
    "ar_weight_k": 8.0,   # steepness sigmoide sul segnale AR (grezzo, non in [0,1])

    # Layer cluster brake settoriale — RIMOSSO interamente il 27/07/2026.
    # Riconosciuto esplicitamente (23/07, dopo rilievo diretto di Vins) come
    # remake del Layer 6 (correlazione realizzata) gia' superato da COR1M, non
    # un'idea "dal libro" nonostante la presentazione iniziale. Esito tecnico
    # positivo su FULL ma mai elevato a priorita'. Vedi
    # PROGETTO_IOVIQUANT_RIASSUNTO.md §6septies per la diagnosi completa.
}


# ==============================================================================
# CONFIGURAZIONE RACCOMANDATA (24/08/2026) — baseline COR1M-off
# ==============================================================================
# NON sovrascrive DEFAULT_PARAMS: DEFAULT_PARAMS resta il punto di partenza
# "vanilla" non calibrato, usato come baseline "Scalare Default" per i
# confronti (metodologia non negoziabile del progetto — se DEFAULT_PARAMS
# cambiasse, quella baseline perderebbe significato). RECOMMENDED_PARAMS e'
# invece il risultato calibrato di oggi, pensato come default dei controlli
# nella UI (app.py) e come config di riferimento per nuove sessioni.
#
# Percorso: config vincente 11/07 -> fix cap su equity (§7.6, 16/07) -> Step A
# (rimosso breakout_bonus_pct: stesso Alpha, MaxDD -2pp) -> Step B (rimossi
# profit_threshold/k_profit_cap: risultato bit-identico, meccanismo mai
# binding). extension_penalty e vix_factor ancora in valutazione (marginali,
# non ancora tagliati).
#
# Dal 24/08/2026 il freno COR1M e' spento nella baseline: sull'universo Xetra
# disponibile (590/600) il solo spegnimento porta Alpha FULL da -7.93% a
# +7.47% e Unicorn rate da 0.973% a 1.473%, a fronte di MaxDD da -30.67% a
# -31.64%. La priorita' deliberata e' il risultato finale. La serie ^COR1M
# resta supportata e richiesta soltanto quando il layer viene riattivato.
RECOMMENDED_PARAMS = {
    **DEFAULT_PARAMS,
    # Gate di qualita' dell'ingresso promosso il 01/08/2026 dopo ablazione
    # nested-purged: la Close conosciuta a T-1 deve trovarsi almeno 0.25 ATR
    # sopra EMA21(T-1). Nel test outer riduce i fast loss senza sacrificare
    # il right tail; il confronto FULL resta documentato come sensitivity.
    "use_entry_close_ema21_atr_gate": True,
    "entry_close_ema21_atr_threshold": 0.25,
    "ext_threshold": 6.0,
    "floor_base": 1.6, "ceiling_base": 2.0,
    "breakout_bonus_pct": 0.0,      # Step A: rimosso, stesso Alpha / MaxDD migliore (confermato su SUB_B)
    # Ritarato a 2.0 il 16/08/2026: a 5.0 il cap al 15% assorbiva gran parte
    # della dispersione del raw score. Il cap resta prudenzialmente attivo e
    # separato, cosi' la scelta puo' essere rivista senza ripristinare la
    # convessita' estrema.
    "convexity_exp": 2.0,
    "use_equity_cap": True,          # fix §7.6
    # Pruning 24/08/2026: con convexity_exp=2 e COR1M spento, il profit-aware e'
    # esattamente inerte su core-60 (FULL/SUB_A/SUB_B) e sull'universo Xetra
    # disponibile (590 titoli): stessi trade, equity e KPI dell'app. Il ramo
    # resta nel motore per riprodurre configurazioni storiche.
    "use_profit_aware": False,
    "use_cor1m_brake": False,
    "cor1m_threshold": 50.0,

    # Capitale/reporting EUR, strumenti eseguiti in USD. Le ipotesi di costo
    # sono intenzionalmente parametriche: non rappresentano un broker
    # specifico e vanno sottoposte a sensitivity analysis.
    "use_fx_conversion": True,
    "base_currency": "EUR",
    "execution_currency": "USD",
    "fx_conversion_bps": 10.0,
    "commission_fixed": 1.0,
    "spread_bps": 8.0,
    "slippage_bps": 5.0,
    "block_same_open_reentry": True,

    # Post-unicorn runner: promosso il 31/07/2026 dopo screening nested-inner
    # e confronto outer fissato. Soglia +50% su Close laggata; trailing 20%
    # sulle Close, floor di profitto +10%, Trend Break ignorato dopo il trigger.
    "use_post_unicorn_runner": True,
    "runner_trigger": 0.50,
    "runner_giveback_pct": 0.20,
    "runner_min_locked_gain": 0.10,
    "runner_ignore_trend_break": True,
    "runner_use_pct_stop": True,

    # --- Semplificazione 18/07/2026 (ablazioni di ridondanza, non di performance) ---
    # w_hmm=0 rimuove il livello P_Bull (soffre della staleness HMM, refit ogni
    # hmm_refit_every=21gg) dal termine di uscita R_exit, lasciando solo la breadth
    # (aggiornata ogni giorno, nessun lag). Validato: Alpha e MaxDD migliorano
    # INSIEME sia su train (SUB_A) sia su test (SUB_B) — non un compromesso.
    # Piccolo costo di MaxDD solo su FULL in continuo (-0.92pp), diagnosticato come
    # effetto di path-dependency dell'equity-cap (non un problema del cambio in se').
    # Applicato SOLO qui, non in DEFAULT_PARAMS (mai testato con questi pesi la').
    "w_bf": 1.0, "w_hmm": 0.0,

    # --- Layer sperimentali NON attivati qui: ereditano i default spenti da
    # DEFAULT_PARAMS (ema_fast_n=5/mid=21/slow=63 identici a prima,
    # use_ar_weighted_momentum=False). RECOMMENDED_PARAMS resta quindi
    # bit-identico al comportamento pre-refactor finche' questi non vengono
    # esplicitamente calibrati e attivati qui.
}

# Configurazione di controllo che riproduce l'esecuzione nominale usata per
# ottenere i risultati storici di RECOMMENDED_PARAMS prima del 31/07/2026.
# Serve solo per regressione/confronto: la logica di segnale e' identica.
RECOMMENDED_PARAMS_GROSS_NATIVE = {
    **RECOMMENDED_PARAMS,
    # Mantiene questo controllo storico realmente pre-promozione 01/08/2026.
    "use_entry_close_ema21_atr_gate": False,
    "use_fx_conversion": False,
    "base_currency": "NATIVE",
    "execution_currency": "NATIVE",
    "fx_conversion_bps": 0.0,
    "spread_bps": 0.0,
    "slippage_bps": 0.0,
    "block_same_open_reentry": False,
    "use_post_unicorn_runner": False,
}

# ==============================================================================
# NOTA 18/07/2026 — parametri esclusi dal set libero per future ottimizzazioni
# congiunte, perche' confermati ridondanti/non-vincolanti SOTTO RECOMMENDED_PARAMS
# specificamente (non sotto DEFAULT_PARAMS, che li mantiene tutti attivi come
# baseline "vanilla" intenzionale, invariata per non comprometterne il significato
# di confronto). Nessuna riga di codice rimossa dal motore: la stessa formula resta
# valida per Legacy/Scalare Default, che dipendono da questi parametri con valori
# diversi. Chi riprende la calibrazione NON deve includere questi 5 nei prossimi
# grid/sweep quando parte da RECOMMENDED_PARAMS:
#   - breakout_lookback, compression_ratio: alimentano solo breakout_bonus, che con
#     breakout_bonus_pct=0.0 (Step A) e' sempre zero per costruzione. Morti SOLO qui.
#   - entry_threshold: con floor_base=1.6, macro_regime_score da solo supera sempre
#     0.35 -> il gate non ha mai filtrato un trade (bit-identico 0.0/0.20/0.35 su
#     FULL/SUB_A/SUB_B). Morto SOLO con floor_base=1.6 (sotto DEFAULT_PARAMS,
#     floor_base=0.4, NON verificato — potrebbe tornare vincolante, non testato).
#   - k_profit_cap: satura esattamente al valore corrente (5.0) — nessun valore
#     superiore cambia una sola metrica su SUB_A o SUB_B. Non serve piu' tunarlo.
#   - w_bf, w_hmm: non piu' liberi, fissati sopra a 1.0/0.0 (vedi commento inline).
# Verificati invece VIVI e NON semplificabili in questo giro: vix_threshold, vix_k
# (sensibilita' reale, il default e' gia' un ottimo locale su train), floor_breadth_w
# /ceiling_breadth_w (ablarli a 0 fa crollare l'Alpha di ~11pp), k_transition (il
# meccanismo conta: azzerarlo cambia Alpha e MaxDD su SUB_B con un vero trade-off,
# stessa firma bull-tilt di cx/floor-ceiling — resta pero' bassa la sensibilita' al
# suo VALORE specifico, gia' noto da luglio, quindi va tenuto ma non va ri-tarato),
# profit_threshold (attivo sotto ~0.5, il default e' gia' al bordo del plateau
# inerte su SUB_A ma il meccanismo nel complesso resta necessario su SUB_B).
#
# Layer 6 (freno di portafoglio via correlazione realizzata: corr_window,
# corr_threshold, corr_k, corr_floor) resta nel motore come modulo storico/di
# riferimento, ma e' dormiente per costruzione (use_portfolio_brake=False sempre
# in RECOMMENDED_PARAMS) — superato dal Layer 6bis (^COR1M). Non rimosso dal
# codice: nessun costo di complessita' nel modello attivo, riattivabile se serve.
# ==============================================================================


# ==============================================================================
# 1. SEGNALI PER-TICKER (EMA, ATR, Filtro Unicorno)
# ==============================================================================
def compute_ticker_signals(df, p):
    """df deve avere colonne Open, High, Low, Close (index = date)."""
    df = df.copy()

    high_low = df["High"] - df["Low"]
    high_cp = (df["High"] - df["Close"].shift(1)).abs()
    low_cp = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=int(p.get("atr_window", 14))).mean().ffill()

    # EMA a finestre calibrabili (23/07/2026): prima erano hardcoded a 5/21/63.
    # I nomi di colonna restano EMA5/EMA21/EMA63 per retrocompatibilita' con
    # Legacy Mode (run_simulation confronta row["EMA5_lag1"] < row["EMA21_lag1"])
    # e con app.py (plotting), anche se ora il periodo effettivo e' governato
    # da ema_fast_n/ema_mid_n/ema_slow_n. Default 5/21/63 -> stesso identico
    # output di prima (.get con fallback per non rompere param_pack esistenti
    # che non passano ancora queste chiavi, es. calibration_harness cache key).
    n_fast = p.get("ema_fast_n", 5)
    n_mid = p.get("ema_mid_n", 21)
    n_slow = p.get("ema_slow_n", 63)
    df["EMA5"] = df["Close"].ewm(span=n_fast, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=n_mid, adjust=False).mean()
    df["EMA63"] = df["Close"].ewm(span=n_slow, adjust=False).mean()

    atr_safe = df["ATR"].replace(0, np.nan)
    gap_fast = (df["EMA5"] - df["EMA21"]) / atr_safe
    gap_mid = (df["EMA21"] - df["EMA63"]) / atr_safe
    min_gap = np.minimum(gap_fast, gap_mid)
    # Primitive conservate per la diagnostica: non cambiano la strategia, ma
    # permettono di ricalcolare le trasformazioni algebriche senza rigenerare
    # EMA/ATR a ogni sweep.
    df["gap_fast"] = gap_fast
    df["gap_mid"] = gap_mid
    df["min_gap"] = min_gap
    ema_steepness = float(p.get("ema_signal_steepness", 2.0))
    df["ema_signal"] = np.where(
        min_gap < 0,
        0.0,
        1 / (1 + np.exp(-ema_steepness * min_gap)),
    )
    df["ema_signal"] = df["ema_signal"].fillna(0.0)

    # --- Filtro Unicorno: extension penalty ---
    extension = (df["Close"] - df["EMA63"]) / atr_safe
    df["extension"] = extension
    excess = (extension - p["ext_threshold"]).clip(lower=0)
    df["extension_penalty"] = (1 / (1 + p["ext_k"] * excess)).fillna(1.0)

    # --- Filtro Unicorno: breakout bonus ---
    atr_pct = df["ATR"] / df["Close"]
    atr_pct_avg = atr_pct.rolling(int(p.get("atr_pct_avg_window", 63))).mean()
    is_compressed = atr_pct < (p["compression_ratio"] * atr_pct_avg)
    prior_high = df["Close"].rolling(p["breakout_lookback"]).max().shift(1)
    is_new_high = df["Close"] > prior_high
    df["atr_pct"] = atr_pct
    df["atr_pct_avg"] = atr_pct_avg
    df["is_new_high"] = is_new_high.astype(float)
    df["breakout_bonus"] = np.where(is_compressed & is_new_high, p["breakout_bonus_pct"], 0.0)

    return df


# ==============================================================================
# 1bis. LAYER AR-WEIGHTED MOMENTUM (Cap. 11-12, Zakamulin-Giner) — nuovo 23/07/2026
# ==============================================================================
def compute_ar_weighted_signal(close_series, ar_weights):
    """
    Applica una convoluzione causale dei rendimenti giornalieri passati con
    un vettore di pesi ar_weights stimato offline (vedi stima_pesi_ar.py),
    seguendo la logica del Cap. 11-12 del libro: se ar_weights riflette i
    coefficienti autoregressivi impliciti del processo di rendimento (invece
    di una forma euristica come extension_penalty), il segnale risultante e'
    la miglior combinazione LINEARE dei rendimenti passati per prevedere il
    segno del rendimento futuro (Eq. 11.25-11.28 del libro).

    close_series: Series di prezzi Close di un singolo titolo (index=date).
    ar_weights: array-like di pesi, lunghezza = ordine dell'indicatore
        (lag 0 = rendimento piu' recente). Tipicamente prodotto da
        stima_pesi_ar.stima_pesi_ottimi().

    Ritorna una Series allineata a close_series.index. Il valore a tempo t
    e' calcolato usando SOLO rendimenti fino a t (rolling.apply con la
    finestra terminante in t) — nessun look-ahead qui; lo shift di 1 verso
    il "_lag1" avviene comunque a valle in assemble_all_signals, come per
    tutti gli altri segnali del motore (doppia cautela causale intenzionale).
    """
    if ar_weights is None or len(ar_weights) == 0:
        return pd.Series(0.0, index=close_series.index)

    weights = np.asarray(ar_weights, dtype=float)
    n = len(weights)
    log_ret = np.log(close_series / close_series.shift(1)).fillna(0.0)

    # weights[0] deve moltiplicare il rendimento PIU' RECENTE nella finestra:
    # rolling.apply passa la finestra in ordine cronologico (piu' vecchio
    # prima), quindi i pesi vanno invertiti per allinearsi correttamente.
    weights_rev = weights[::-1]

    def _dot(x):
        return np.dot(x, weights_rev)

    signal = log_ret.rolling(window=n, min_periods=n).apply(_dot, raw=True)
    return signal.fillna(0.0)


def compute_ar_bonus_factor(ar_signal, p):
    """
    Converte il segnale AR grezzo (somma pesata di log-return, tipicamente
    piccolo e non delimitato) in un fattore BONUS additivo, mai <1.0 --
    stessa forma di compute_gap_factor/compute_lowdist_factor, non quella
    penalizzante di compute_vix_factor. Centrato a 0: segnale AR positivo
    (momentum atteso positivo secondo i pesi calibrati) spinge il fattore
    verso (1+ar_weight_scale); segnale negativo o nullo lascia il fattore a
    1.0 (nessuna penalita' -- il filtro ema_signal>0 gia' esclude i trend
    negativi a monte, qui si vuole solo premiare i migliori tra i positivi).
    """
    x = p["ar_weight_k"] * ar_signal
    sigmoid = 1 / (1 + np.exp(-x))
    factor = 1.0 + p["ar_weight_scale"] * sigmoid
    return factor


# ==============================================================================
# 2. BREADTH 2D (livello + momentum), calcolato sull'universo
# ==============================================================================
def compute_breadth_2d(close_wide, p):
    sma200 = close_wide.rolling(int(p.get("breadth_sma_window", 200))).mean()
    pct_above_200 = (close_wide > sma200).mean(axis=1) * 100

    level_comp = 1 / (
        1 + np.exp(
            -float(p.get("breadth_level_steepness", 0.1))
            * (pct_above_200 - float(p.get("breadth_center", 50.0)))
        )
    )
    breadth_sma = pct_above_200.rolling(window=p["breadth_k"]).mean()
    breadth_std = pct_above_200.rolling(window=p["breadth_k"]).std().replace(0, 1)
    mom_comp = (pct_above_200 - breadth_sma) / breadth_std
    mom_comp_scaled = 1 / (
        1 + np.exp(-float(p.get("breadth_momentum_steepness", 1.0)) * mom_comp)
    )

    breadth_factor = (p["breadth_alpha"] * level_comp) + ((1 - p["breadth_alpha"]) * mom_comp_scaled)
    breadth_factor = breadth_factor.clip(0, 1)
    return breadth_factor, pct_above_200


# ==============================================================================
# 3. VIX: sigmoide continua
# ==============================================================================
def compute_vix_factor(vix_series, p):
    x = p["vix_k"] * (vix_series - p["vix_threshold"])
    factor = p["vix_floor"] + (1 - p["vix_floor"]) / (1 + np.exp(x))
    return factor


def compute_vix_factor_binary(vix_series, p):
    """Versione rigida on/off, usata solo in Legacy Mode come baseline fedele."""
    return np.where(vix_series > p["vix_threshold"], 0.5, 1.0)


# ==============================================================================
# 3bis. LAYER 6: FRENO DI PORTAFOGLIO (correlazione realizzata a coppie)
# ==============================================================================
def compute_portfolio_correlation(close_wide, p):
    """
    Correlazione media a coppie tra i rendimenti giornalieri dell'universo,
    su finestra rolling causale. Implementazione efficiente O(n*T) per finestra
    (standardizzazione + somma dei quadrati) anziche' O(n^2*T) con matrice di
    correlazione completa ad ogni passo:
        avg_corr(t) = [mean(S_t^2) - n] / [n*(n-1)]
    dove S_t = somma dei rendimenti standardizzati (nella finestra) di tutti
    i titoli al giorno t. Verificato empiricamente contro episodi di stress
    noti (bear market 2022, shock aprile 2025): la correlazione media sale
    a 0.44-0.73 in quei periodi, contro una mediana storica di ~0.25.
    """
    log_ret = np.log(close_wide / close_wide.shift(1)).fillna(0.0)
    vals = log_ret.values
    T_total, n = vals.shape
    window = p["corr_window"]
    out = np.full(T_total, np.nan)

    for t in range(window - 1, T_total):
        w = vals[t - window + 1: t + 1, :]
        mean = w.mean(axis=0)
        std = w.std(axis=0)
        std_safe = np.where(std == 0, np.nan, std)
        z = np.nan_to_num((w - mean) / std_safe, nan=0.0)
        S = z.sum(axis=1)
        out[t] = (np.mean(S ** 2) - n) / (n * (n - 1))

    return pd.Series(out, index=close_wide.index)


def compute_corr_factor(corr_series, p):
    """Sigmoide: correlazione alta -> fattore verso corr_floor (freno attivo).
    Stessa forma funzionale di compute_vix_factor, stesso stile del progetto."""
    x = p["corr_k"] * (corr_series - p["corr_threshold"])
    factor = p["corr_floor"] + (1 - p["corr_floor"]) / (1 + np.exp(x))
    return factor


def compute_gap_factor(gap_pct_rank, p):
    """BONUS additivo (non fattore penalizzante): 1.0 sotto soglia (nessun
    effetto sui titoli con gap ordinario), sale verso (1+gap_bonus_pct) sopra
    soglia. Correzione 19/07/2026: la prima versione usava la forma di
    vix_factor (floor->1, solo penalizzante), che con convexity_exp=5.0
    scontava sistematicamente il ~75%% dei titoli sotto soglia ogni giorno --
    stessa forma di breakout_bonus, non di vix_factor."""
    x = p["gap_k"] * (gap_pct_rank - p["gap_threshold"])
    sigmoid = 1 / (1 + np.exp(x * -1))
    factor = 1.0 + p["gap_bonus_pct"] * sigmoid
    return factor


def compute_lowdist_factor(lowdist_pct_rank, p):
    """Stessa correzione: bonus additivo, non fattore penalizzante."""
    x = p["lowdist_k"] * (lowdist_pct_rank - p["lowdist_threshold"])
    sigmoid = 1 / (1 + np.exp(x * -1))
    factor = 1.0 + p["lowdist_bonus_pct"] * sigmoid
    return factor


def compute_cor1m_factor(cor1m_series, p):
    """Stessa forma della correlazione realizzata, ma sul segnale forward-looking
    ^COR1M (Cboe 1-Month Implied Correlation). La gestione del gap di copertura
    (dati assenti prima del 2022-03-31 -> fattore neutro 1.0, non estrapolato)
    avviene a valle in assemble_all_signals, stesso pattern di breadth/HMM/VIX."""
    x = p["cor1m_k"] * (cor1m_series - p["cor1m_threshold"])
    factor = p["cor1m_floor"] + (1 - p["cor1m_floor"]) / (1 + np.exp(x))
    return factor


def _macro_state_trigger(row, p, mode_key):
    """Stato di rischio macro noto a T-1, condiviso da entry ed exit."""
    mode = p.get(mode_key, "low_or_fracture")
    valid_modes = {
        "low", "fracture", "low_or_fracture",
        "cor_breadth_fracture", "low_or_cor_breadth_fracture",
        "broad_fracture", "confirmed_fracture",
        "low_confirmed_fracture",
    }
    if mode not in valid_modes:
        raise ValueError(
            f"{mode_key} non valido: {mode!r}. Valori ammessi: "
            f"{sorted(valid_modes)}."
        )

    cor_level = row.get("cor1m_level_lag1")
    cor_delta = row.get("cor1m_delta_5_lag1")
    vix_return = row.get("vix_return_5_lag1")
    breadth_delta = row.get("breadth_delta_5_lag1")
    hmm_delta = row.get("hmm_delta_5_lag1")
    cor_prior_level = row.get("cor1m_level_5_lag1")
    low = (
        pd.notna(cor_level)
        and float(cor_level) < float(p.get("macro_cor1m_low_threshold", 10.0))
    )
    cor_rising = (
        pd.notna(cor_delta)
        and float(cor_delta) > float(p.get("macro_cor1m_rise_5d_threshold", 5.0))
    )
    vix_rising = (
        pd.notna(vix_return)
        and float(vix_return) > float(p.get("macro_vix_rise_5d_threshold", 0.10))
    )
    breadth_falling = (
        pd.notna(breadth_delta)
        and float(breadth_delta) < float(
            p.get("macro_breadth_fall_5d_threshold", -5.0)
        )
    )
    hmm_falling = (
        pd.notna(hmm_delta)
        and float(hmm_delta) < float(
            p.get("macro_hmm_fall_5d_threshold", -0.05)
        )
    )
    fracture = cor_rising and vix_rising
    cor_breadth_fracture = cor_rising and breadth_falling
    confirmations = int(vix_rising) + int(breadth_falling) + int(hmm_falling)
    broad_fracture = cor_rising and confirmations >= 1
    confirmed_fracture = cor_rising and confirmations >= 2
    prior_low = (
        pd.notna(cor_prior_level)
        and float(cor_prior_level) < float(
            p.get("macro_cor1m_low_threshold", 10.0)
        )
    )
    return {
        "low": low,
        "fracture": fracture,
        "low_or_fracture": low or fracture,
        "cor_breadth_fracture": cor_breadth_fracture,
        "low_or_cor_breadth_fracture": low or cor_breadth_fracture,
        "broad_fracture": broad_fracture,
        "confirmed_fracture": confirmed_fracture,
        "low_confirmed_fracture": prior_low and confirmed_fracture,
    }[mode]


def _update_macro_exposure_cap(row, p, current_cap, hold_remaining):
    """Aggiorna causalmente il cap di esposizione per le nuove entrate.

    Il cap si stringe immediatamente quando lo stato peggiora. In miglioramento
    resta fermo per ``macro_exposure_hold_sessions`` e poi risale di
    ``macro_exposure_recovery_step`` per seduta. Il valore ritornato non causa
    vendite: viene consumato soltanto dalla sezione ingressi.
    """
    transition_cap = float(p.get("macro_exposure_transition_cap", 0.60))
    riskoff_cap = float(p.get("macro_exposure_riskoff_cap", 0.35))
    hold_sessions = int(p.get("macro_exposure_hold_sessions", 5))
    recovery_step = float(p.get("macro_exposure_recovery_step", 0.10))
    if not 0.0 <= riskoff_cap <= transition_cap <= 1.0:
        raise ValueError(
            "I cap macro devono rispettare 0 <= riskoff <= transition <= 1."
        )
    if hold_sessions < 0:
        raise ValueError("macro_exposure_hold_sessions deve essere >= 0.")
    if not 0.0 < recovery_step <= 1.0:
        raise ValueError("macro_exposure_recovery_step deve essere in (0, 1].")

    riskoff_active = _macro_state_trigger(
        row, p, "macro_exposure_riskoff_mode"
    )
    transition_active = _macro_state_trigger(
        row, p, "macro_exposure_transition_mode"
    )
    if riskoff_active:
        target_cap = riskoff_cap
        direct_state = "risk_off"
    elif transition_active:
        target_cap = transition_cap
        direct_state = "transition"
    else:
        target_cap = 1.0
        direct_state = "normal"

    current_cap = float(np.clip(current_cap, 0.0, 1.0))
    hold_remaining = max(0, int(hold_remaining))
    tolerance = 1e-12
    if target_cap < current_cap - tolerance:
        current_cap = target_cap
        hold_remaining = hold_sessions
    elif abs(target_cap - current_cap) <= tolerance and target_cap < 1.0:
        # Finche' lo stato resta attivo, l'hold decorre dalla sua ultima
        # osservazione, non dal primo giorno del regime.
        hold_remaining = hold_sessions
    elif target_cap > current_cap + tolerance:
        if hold_remaining > 0:
            hold_remaining -= 1
        else:
            current_cap = min(target_cap, current_cap + recovery_step)

    if direct_state == "normal" and current_cap < 1.0 - tolerance:
        effective_state = "recovery"
    elif direct_state == "transition" and current_cap < transition_cap - tolerance:
        effective_state = "risk_off_recovery"
    else:
        effective_state = direct_state
    return current_cap, hold_remaining, effective_state, target_cap


# ==============================================================================
# 4. HMM WALK-FORWARD CAUSALE SUL BENCHMARK (un solo modello, non per-ticker)
# ==============================================================================
def compute_benchmark_hmm_walkforward(bench_close, p, random_state=None):
    """
    Ritorna una Series P_Bull allineata a bench_close.index, calcolata in modo
    causale: ad ogni step i, il modello e' fittato SOLO su dati fino a i
    (expanding window), con refit ogni `hmm_refit_every` barre. La probabilita'
    filtrata a i e' l'ultimo elemento di predict_proba calcolato su [0..i]
    (proprieta' del forward-backward: a fine finestra, smoothed == filtered).
    """
    if GaussianHMM is None:
        raise ImportError(
            "Dipendenza mancante: installare 'hmmlearn' nell'ambiente del "
            "backtest. I test di accounting/KPI possono funzionare senza HMM, "
            "ma la generazione dei segnali no."
        )
    log_ret = np.log(bench_close / bench_close.shift(1)).fillna(0.0)
    X_full = log_ret.values.reshape(-1, 1)
    n = len(X_full)

    p_bull = pd.Series(index=bench_close.index, dtype=float)
    model = None
    bull_state = None
    last_refit = -10**9
    min_train = p["hmm_min_train"]
    refit_every = p["hmm_refit_every"]
    random_state = int(
        p.get("hmm_random_state", 42) if random_state is None else random_state
    )

    for i in range(n):
        if i < min_train:
            continue
        need_refit = (model is None) or (i - last_refit >= refit_every)
        if need_refit:
            try:
                m = GaussianHMM(
                    n_components=int(p.get("hmm_n_components", 2)),
                    covariance_type="full",
                    random_state=random_state,
                    n_iter=int(p.get("hmm_n_iter", 100)),
                    tol=float(p.get("hmm_tol", 1e-2)),
                )
                m.fit(X_full[: i + 1])
                model = m
                bull_state = int(np.argmax(model.means_))
                last_refit = i
            except Exception:
                pass  # tiene il modello precedente se il refit fallisce (instabilita' numerica)

        if model is not None:
            probs = model.predict_proba(X_full[: i + 1])
            p_bull.iloc[i] = probs[-1, bull_state]

    return p_bull.ffill()


# ==============================================================================
# 5. MACRO REGIME SCORE: transizione HMM (delta) x inviluppo Breadth
# ==============================================================================
def compute_macro_regime_score(p_bull_bench, breadth_factor_aligned, p):
    delta = p_bull_bench - p_bull_bench.shift(p["hmm_delta_lookback"])
    transition_signal = 1 / (1 + np.exp(-p["k_transition"] * delta))

    floor = p["floor_base"] + p["floor_breadth_w"] * breadth_factor_aligned
    ceiling = p["ceiling_base"] + p["ceiling_breadth_w"] * breadth_factor_aligned
    macro_regime_score = floor + (ceiling - floor) * transition_signal
    return macro_regime_score, transition_signal


# ==============================================================================
# 6. ASSEMBLAGGIO: unisce segnali per-ticker + macro, produce frame lag-1
# ==============================================================================
def assemble_all_signals(data_dict, universe, p):
    """
    data_dict: {ticker: df con colonne Open, High, Low, Close}
    Ritorna: dict {ticker: df con segnali grezzi + versione '_lag1' shiftata},
             piu' le serie macro (per grafici/debug).
    """
    # Usa esclusivamente l'universo esplicitamente richiesto. In particolare,
    # eventuali ticker ``validation_only`` presenti nel data_dict non devono
    # modificare breadth/correlazione del campione usato per la selezione.
    close_wide = pd.DataFrame({
        t: data_dict[t]["Close"] for t in universe
        if t in data_dict and not t.startswith("__")
    }).ffill()
    breadth_factor, pct_above_200 = compute_breadth_2d(close_wide, p)
    breadth_level_comp = 1 / (
        1 + np.exp(
            -float(p.get("breadth_level_steepness", 0.1))
            * (pct_above_200 - float(p.get("breadth_center", 50.0)))
        )
    )
    breadth_mean = pct_above_200.rolling(window=p["breadth_k"]).mean()
    breadth_std = pct_above_200.rolling(window=p["breadth_k"]).std().replace(0, 1)
    breadth_momentum_z = (pct_above_200 - breadth_mean) / breadth_std
    breadth_momentum_scaled = 1 / (
        1 + np.exp(
            -float(p.get("breadth_momentum_steepness", 1.0))
            * breadth_momentum_z
        )
    )

    bench = data_dict["__BENCHMARK__"]["Close"]
    p_bull_bench = compute_benchmark_hmm_walkforward(bench, p)

    vix = data_dict["__VIX__"]["Close"]
    vix_factor_cont = compute_vix_factor(vix, p)
    vix_factor_bin = pd.Series(compute_vix_factor_binary(vix, p), index=vix.index)

    portfolio_corr = compute_portfolio_correlation(close_wide, p)
    corr_factor = compute_corr_factor(portfolio_corr, p)

    if "__COR1M__" in data_dict:
        cor1m_raw = data_dict["__COR1M__"]["Close"]
        cor1m_factor = compute_cor1m_factor(cor1m_raw, p)
    else:
        cor1m_factor = pd.Series(1.0, index=close_wide.index)  # neutro se non fornito

    # Livelli e transizioni macro per le regole di stato. Le serie restano
    # grezze qui e vengono shiftate insieme agli altri segnali più sotto.
    # COR1M non viene retro-estrapolato prima della sua copertura reale.
    macro_calendar = close_wide.index
    vix_level = _align_asof_ffill(vix, macro_calendar)
    cor1m_level = _align_asof_ffill(cor1m_raw, macro_calendar) \
        if "__COR1M__" in data_dict else pd.Series(np.nan, index=macro_calendar)
    vix_return_5 = vix_level.pct_change(5, fill_method=None)
    cor1m_delta_5 = cor1m_level.diff(5)
    cor1m_level_5 = cor1m_level.shift(5)
    breadth_delta_5 = pct_above_200.reindex(macro_calendar).ffill().diff(5)
    hmm_delta_5 = p_bull_bench.reindex(macro_calendar).ffill().diff(5)

    if p.get("use_fx_conversion", False):
        if "__FX__" not in data_dict:
            raise ValueError(
                "use_fx_conversion=True richiede data_dict['__FX__'] con EURUSD "
                "(USD per EUR): Open per i fill e Close per il reporting."
            )
        fx_open = data_dict["__FX__"]["Open"].sort_index()
        fx_close = data_dict["__FX__"]["Close"].sort_index()
    else:
        fx_open = pd.Series(1.0, index=close_wide.index)
        fx_close = pd.Series(1.0, index=close_wide.index)

    # Layer AR-weighted momentum (Cap. 11-12 del libro) — RIDISEGNATO 27/07/2026:
    # calcolato UNA SOLA VOLTA sui log-return del BENCHMARK (processo aggregato,
    # non piu' per-ticker sul singolo Close), stesso pattern di vix_factor/
    # cor1m_factor: una serie unica, poi assegnata a tutti i ticker nel loop
    # sotto. Motivazione (diagnosi 23/07/2026): il framework del libro presuppone
    # persistenza stimabile su un processo aggregato (indice), non su rendimenti
    # giornalieri di singolo titolo condizionati a un evento raro -- vedi
    # PROGETTO_IOVIQUANT_RIASSUNTO.md §6septies. ar_weights=None di default ->
    # innocuo per costruzione (ar_bonus_factor resta 1.0 ovunque).
    if p.get("use_ar_weighted_momentum", False) and p.get("ar_weights") is not None:
        ar_weighted_signal_bench = compute_ar_weighted_signal(bench, p["ar_weights"])
        ar_bonus_factor_bench = compute_ar_bonus_factor(ar_weighted_signal_bench, p)
    else:
        ar_weighted_signal_bench = pd.Series(0.0, index=close_wide.index)
        ar_bonus_factor_bench = pd.Series(1.0, index=close_wide.index)

    # Layer 7: Gap Intensity (cross-sectionale, richiede Open oltre a Close)
    open_wide = pd.DataFrame({
        t: data_dict[t]["Open"] for t in universe
        if t in data_dict and not t.startswith("__")
    }).ffill()
    gap = (open_wide - close_wide.shift(1)) / close_wide.shift(1)
    gap_intensity = gap.abs().rolling(p["gap_lookback"]).mean()
    gap_pct_rank = gap_intensity.rank(axis=1, pct=True)
    gap_factor_wide = compute_gap_factor(gap_pct_rank, p)

    # Layer 8: Distanza dai minimi 252gg (cross-sectionale)
    pct_from_low = close_wide / close_wide.rolling(p["lowdist_lookback"]).min() - 1
    lowdist_pct_rank = pct_from_low.rank(axis=1, pct=True)
    lowdist_factor_wide = compute_lowdist_factor(lowdist_pct_rank, p)

    breadth_aligned = breadth_factor.reindex(vix_factor_cont.index.union(breadth_factor.index)).ffill()
    macro_regime_score, transition_signal = compute_macro_regime_score(
        p_bull_bench.reindex(breadth_aligned.index).ffill(), breadth_aligned, p
    )
    transition_delta = (
        p_bull_bench.reindex(breadth_aligned.index).ffill()
        - p_bull_bench.reindex(breadth_aligned.index).ffill().shift(p["hmm_delta_lookback"])
    )

    out = {}
    for t in universe:
        if t not in data_dict:
            continue
        df = compute_ticker_signals(data_dict[t], p)

        idx = df.index
        breadth_raw = breadth_factor.reindex(idx).ffill()
        macro_raw = macro_regime_score.reindex(idx).ffill()
        df["breadth_available"] = breadth_raw.notna().astype(float)
        df["macro_regime_available"] = macro_raw.notna().astype(float)
        df["breadth_factor"] = breadth_raw.fillna(0.5)
        df["pct_above_200"] = pct_above_200.reindex(idx).ffill().fillna(50.0)
        df["breadth_level_comp"] = breadth_level_comp.reindex(idx).ffill().fillna(0.5)
        df["breadth_momentum_z"] = breadth_momentum_z.reindex(idx).ffill().fillna(0.0)
        df["breadth_momentum_scaled"] = breadth_momentum_scaled.reindex(idx).ffill().fillna(0.5)
        df["p_bull_bench"] = p_bull_bench.reindex(idx).ffill().fillna(0.5)
        df["transition_delta"] = transition_delta.reindex(idx).ffill().fillna(0.0)
        df["transition_signal"] = transition_signal.reindex(idx).ffill().fillna(0.5)
        df["macro_regime_score"] = macro_raw.fillna(0.6)
        df["vix_factor"] = vix_factor_cont.reindex(idx).ffill().fillna(1.0)
        df["vix_factor_bin"] = vix_factor_bin.reindex(idx).ffill().fillna(1.0)
        df["portfolio_corr"] = portfolio_corr.reindex(idx).ffill()
        df["corr_factor"] = corr_factor.reindex(idx).ffill().fillna(1.0)
        df["cor1m_factor"] = _align_asof_ffill(
            cor1m_factor, idx
        ).fillna(1.0)
        df["vix_level"] = vix_level.reindex(idx).ffill()
        df["cor1m_level"] = cor1m_level.reindex(idx).ffill()
        df["vix_return_5"] = vix_return_5.reindex(idx).ffill()
        df["cor1m_delta_5"] = cor1m_delta_5.reindex(idx).ffill()
        df["cor1m_level_5"] = cor1m_level_5.reindex(idx).ffill()
        df["breadth_delta_5"] = breadth_delta_5.reindex(idx).ffill()
        df["hmm_delta_5"] = hmm_delta_5.reindex(idx).ffill()
        df["fx_open_usd_per_eur"] = fx_open.reindex(idx).ffill()
        df["fx_close_usd_per_eur"] = fx_close.reindex(idx).ffill()
        if t in gap_factor_wide.columns:
            gap_factor_raw = gap_factor_wide[t].reindex(idx).ffill()
            df["gap_available"] = gap_factor_raw.notna().astype(float)
            df["gap_factor"] = gap_factor_raw.fillna(1.0)
            df["gap_rank"] = gap_pct_rank[t].reindex(idx).ffill().fillna(0.5)
        else:
            df["gap_available"] = 0.0
            df["gap_factor"] = 1.0
            df["gap_rank"] = 0.5
        if t in lowdist_factor_wide.columns:
            lowdist_factor_raw = lowdist_factor_wide[t].reindex(idx).ffill()
            df["lowdist_available"] = lowdist_factor_raw.notna().astype(float)
            df["lowdist_factor"] = lowdist_factor_raw.fillna(1.0)
            df["lowdist_rank"] = lowdist_pct_rank[t].reindex(idx).ffill().fillna(0.5)
        else:
            df["lowdist_available"] = 0.0
            df["lowdist_factor"] = 1.0
            df["lowdist_rank"] = 0.5

        # Layer AR-weighted momentum: serie UNICA calcolata sul benchmark
        # (vedi sopra), semplicemente riallineata all'indice del ticker --
        # stesso pattern di broadcast di vix_factor/cor1m_factor. Se
        # use_ar_weighted_momentum=False o ar_weights=None, le serie sorgente
        # sono gia' costanti (0.0/1.0), quindi questo resta innocuo.
        df["ar_weighted_signal"] = ar_weighted_signal_bench.reindex(idx).ffill().fillna(0.0)
        df["ar_bonus_factor"] = ar_bonus_factor_bench.reindex(idx).ffill().fillna(1.0)

        sig_cols = ["ema_signal", "ATR", "extension_penalty", "breakout_bonus",
                    "gap_fast", "gap_mid", "min_gap", "extension",
                    "atr_pct", "atr_pct_avg", "is_new_high",
                    "vix_factor", "vix_factor_bin", "macro_regime_score",
                    "macro_regime_available", "breadth_factor", "breadth_available",
                    "pct_above_200", "breadth_level_comp",
                    "breadth_momentum_z", "breadth_momentum_scaled",
                    "p_bull_bench", "transition_delta", "transition_signal",
                    "portfolio_corr", "corr_factor", "cor1m_factor",
                    "vix_level", "cor1m_level", "vix_return_5",
                    "cor1m_delta_5", "breadth_delta_5", "hmm_delta_5",
                    "cor1m_level_5",
                    "ar_weighted_signal", "ar_bonus_factor",
                    "gap_factor", "gap_available", "lowdist_factor", "lowdist_available",
                    "gap_rank", "lowdist_rank",
                    "EMA5", "EMA21", "EMA63", "Close", "fx_close_usd_per_eur"]
        lag = df[sig_cols].shift(1)
        lag.columns = [c + "_lag1" for c in sig_cols]
        df = df.join(lag)

        out[t] = df

    return out, {
        "breadth_factor": breadth_factor, "pct_above_200": pct_above_200,
        "p_bull_bench": p_bull_bench, "macro_regime_score": macro_regime_score,
        "vix_factor": vix_factor_cont, "transition_signal": transition_signal,
        "portfolio_corr": portfolio_corr, "corr_factor": corr_factor,
        "cor1m_factor": cor1m_factor,
        "vix_level": vix_level, "cor1m_level": cor1m_level,
        "vix_return_5": vix_return_5, "cor1m_delta_5": cor1m_delta_5,
        "cor1m_level_5": cor1m_level_5,
        "breadth_delta_5": breadth_delta_5, "hmm_delta_5": hmm_delta_5,
        "fx_open_usd_per_eur": fx_open,
        "fx_close_usd_per_eur": fx_close,
    }


# ==============================================================================
# 7. SIMULAZIONE EVENT-DRIVEN CAUSALE (segnale T-1 -> fill Open T -> MTM Close T)
# ==============================================================================
def build_open_entry_benchmark_equity(
    benchmark_ohlc,
    capital,
    start_date,
    end_date=None,
):
    """Buy&hold investito alla prima Open disponibile nella finestra.

    I dati precedenti a ``start_date`` possono essere presenti nel dataframe,
    ma non contribuiscono ne' al prezzo d'ingresso ne' alla curva restituita.
    """
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date is not None else None
    frame = benchmark_ohlc.sort_index().loc[lambda x: x.index >= start_ts]
    if end_ts is not None:
        frame = frame.loc[frame.index <= end_ts]
    frame = frame.dropna(subset=["Open", "Close"])
    if frame.empty:
        raise ValueError("Benchmark senza dati OHLC nella finestra operativa selezionata.")
    entry_date = frame.index[0]
    entry_open = float(frame.iloc[0]["Open"])
    if not np.isfinite(entry_open) or entry_open <= 0:
        raise ValueError("Open iniziale del benchmark mancante o non valido.")
    equity = frame["Close"].astype(float) / entry_open * float(capital)
    equity.name = "VWCE_Eq"
    return equity, entry_date, entry_open


def run_simulation(
    signals,
    target_tickers,
    capital,
    base_size,
    p,
    start_date=None,
    end_date=None,
    omd_schedule=None,
):
    portfolio = {}
    history = []
    events = []
    closed_trades = []
    use_fx = bool(p.get("use_fx_conversion", False))
    commission = float(p.get("commission_fixed", p.get("fee", 1.0)))
    spread_bps = float(p.get("spread_bps", 0.0))
    slippage_bps = float(p.get("slippage_bps", 0.0))
    one_way_impact = ((spread_bps / 2.0) + slippage_bps) / 10000.0
    block_same_open = bool(p.get("block_same_open_reentry", False))
    use_omd_entry_filter = bool(p.get("use_omd_entry_filter", False))
    use_omd_forced_sell_exit = bool(
        p.get("use_omd_forced_sell_exit", False)
    )
    if use_omd_forced_sell_exit and not use_omd_entry_filter:
        raise ValueError(
            "use_omd_forced_sell_exit richiede use_omd_entry_filter=True."
        )

    omd_records = []
    if omd_schedule is not None:
        if not isinstance(omd_schedule, pd.DataFrame):
            raise TypeError("omd_schedule deve essere un DataFrame.")
        required_omd_columns = {
            "Effective_Date", "Buy_Tickers", "Sell_Tickers"
        }
        missing_omd_columns = required_omd_columns.difference(
            omd_schedule.columns
        )
        if missing_omd_columns:
            raise ValueError(
                "omd_schedule incompleto; colonne mancanti: "
                f"{sorted(missing_omd_columns)}"
            )
        normalized = omd_schedule.copy()
        normalized["Effective_Date"] = pd.to_datetime(
            normalized["Effective_Date"]
        ).dt.tz_localize(None)
        normalized = normalized.sort_values("Effective_Date")
        for _, record in normalized.iterrows():
            sell = set(record["Sell_Tickers"] or ())
            buy = set(record["Buy_Tickers"] or ()) - sell
            omd_records.append((
                pd.Timestamp(record["Effective_Date"]), buy, sell,
                record.get("Signal_Date", pd.NaT),
            ))
    if use_omd_entry_filter and not omd_records:
        raise ValueError(
            "Filtro OMD attivo ma nessun cluster mensile disponibile."
        )

    all_dates = pd.concat([df.index.to_series() for t, df in signals.items() if t in target_tickers]).unique()
    dates = sorted(all_dates)
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        dates = [date for date in dates if date >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        dates = [date for date in dates if date <= end_ts]

    def _fx_value(date, column):
        if not use_fx:
            return 1.0
        for ticker in target_tickers:
            if ticker in signals and date in signals[ticker].index:
                value = signals[ticker].loc[date].get(column)
                if pd.notna(value) and float(value) > 0:
                    return float(value)
        raise ValueError(f"Tasso EURUSD mancante/non valido il {date} ({column}).")

    def _sell_fill(mid_price):
        return float(mid_price) * (1.0 - one_way_impact)

    def _buy_fill(mid_price):
        return float(mid_price) * (1.0 + one_way_impact)

    def _close_position_at_open(ticker, pos, date, row, fx_open_today, reason):
        """Chiude integralmente una posizione all'Open con accounting standard."""
        nonlocal cash
        if pd.isna(row.get("Open")):
            return False
        mid_price = float(row["Open"])
        fill_price = _sell_fill(mid_price)
        revenue = (pos["shares"] * fill_price) - commission
        cash += revenue
        realized_partial = pos.get("realized_revenue", 0.0)
        if realized_partial > 0:
            total_revenue = realized_partial + revenue
            orig_shares = pos.get("orig_shares", pos["shares"])
            profit = total_revenue - pos["cost_basis"]
            report_shares = orig_shares
            report_sell_price = (
                total_revenue / orig_shares
                if orig_shares > 0 else fill_price
            )
        else:
            profit = revenue - pos["cost_basis"]
            report_shares = pos["shares"]
            report_sell_price = fill_price
        total_revenue_base = (
            pos.get("realized_revenue_base", 0.0)
            + revenue / fx_open_today
        )
        profit_base = total_revenue_base - pos["cost_basis_base"]
        realized_return = report_sell_price / pos["buy_price"] - 1.0
        mfe = pos["mfe_highest_price"] / pos["buy_price"] - 1.0
        mae = pos["lowest_price"] / pos["buy_price"] - 1.0
        closed_trades.append({
            "Entry_Date": pos["entry_date"], "Exit_Date": date,
            "Ticker": ticker, "Shares": report_shares,
            "Buy_Price": pos["buy_price"],
            "Sell_Price": report_sell_price,
            "Buy_Price_USD": pos["buy_price"],
            "Sell_Price_USD": report_sell_price,
            "Profit": profit_base if use_fx else profit,
            "Profit_Base": profit_base if use_fx else profit,
            "Profit_Exec": profit,
            "Return_%": realized_return * 100.0,
            "MFE_%": mfe * 100.0, "MAE_%": mae * 100.0,
            "Peak_Giveback_pp": (mfe - realized_return) * 100.0,
            "Runner_Activated": bool(pos.get("runner_active", False)),
            "Runner_Trigger_Date": pos.get("runner_trigger_date", pd.NaT),
            "Holding_Sessions": int(pos.get("sessions_held", 0)),
            "Entry_FX": pos["entry_fx"], "Exit_FX": fx_open_today,
            "Motivo_Chiusura": reason, "Stato": "Chiusa",
        })
        impact_cost = pos["shares"] * (mid_price - fill_price)
        events.append({
            "Date": date, "Ticker": ticker, "Action": "SELL",
            "Reason": reason,
            "Price": fill_price, "Mid_Price": mid_price,
            "Shares": pos["shares"],
            "Notional_Exec": pos["shares"] * fill_price,
            "Notional_Base": (
                pos["shares"] * fill_price / fx_open_today
            ),
            "Costs_Exec": commission + impact_cost,
            "Costs_Base": (
                commission + impact_cost
            ) / fx_open_today,
            "FX_USD_per_EUR": fx_open_today,
        })
        return True

    if dates:
        initial_fx = _fx_value(dates[0], "fx_open_usd_per_eur")
    else:
        initial_fx = 1.0
    fx_conversion_rate = float(p.get("fx_conversion_bps", 0.0)) / 10000.0 if use_fx else 0.0
    gross_initial_cash = capital * initial_fx
    initial_fx_cost = gross_initial_cash * fx_conversion_rate
    cash = gross_initial_cash - initial_fx_cost
    macro_exposure_cap = 1.0
    macro_exposure_hold_remaining = 0
    macro_risk_state = "normal"
    macro_target_cap = 1.0
    if use_fx and dates:
        events.append({
            "Date": dates[0], "Ticker": "__CASH__", "Action": "FX_CONVERSION",
            "Price": initial_fx, "Mid_Price": initial_fx, "Shares": np.nan,
            "Notional_Exec": gross_initial_cash, "Notional_Base": capital,
            "Costs_Exec": initial_fx_cost, "Costs_Base": initial_fx_cost / initial_fx,
            "FX_USD_per_EUR": initial_fx,
        })

    omd_record_index = -1
    current_omd_buy: set[str] = set()
    current_omd_sell: set[str] = set()
    for date_idx, date in enumerate(dates):
        fx_open_today = _fx_value(date, "fx_open_usd_per_eur")
        fx_close_today = _fx_value(date, "fx_close_usd_per_eur")
        exited_tickers_today = set()
        while (omd_record_index + 1 < len(omd_records)
               and omd_records[omd_record_index + 1][0] <= date):
            omd_record_index += 1
            effective_date, buy_cluster, sell_cluster, signal_date = (
                omd_records[omd_record_index]
            )
            current_omd_sell = set(sell_cluster)
            current_omd_buy = set(buy_cluster) - current_omd_sell
            events.append({
                "Date": date,
                "Ticker": "__OMD__",
                "Action": "OMD_REBALANCE",
                "Signal_Date": signal_date,
                "Effective_Date": effective_date,
                "Buy_Count": len(current_omd_buy),
                "Sell_Count": len(current_omd_sell),
                "Price": np.nan,
                "Mid_Price": np.nan,
                "Shares": np.nan,
                "Notional_Exec": 0.0,
                "Notional_Base": 0.0,
                "Costs_Exec": 0.0,
                "Costs_Base": 0.0,
                "FX_USD_per_EUR": fx_open_today,
            })
        # --- 1. USCITE ---
        # Layer 5quinquies (nuovo, 30/07/2026, Cap. 14 del libro Zakamulin-Giner,
        # "Optimal Trading Frequency"): quando attivo, il TEST di uscita (SL/Trend
        # Break/uscita parziale) viene valutato solo ogni exit_freq_days giorni,
        # usando l'indice POSIZIONALE nella sequenza 'dates' gia' nota (non un
        # calendario) -- robusto a festivita'/buchi dati (vedi bug noto date USA),
        # causale per costruzione: dipende solo dalla posizione nella sequenza
        # gia' osservata, mai da eventi futuri. exit_freq_offset permette di
        # testare piu' fasi (0..exit_freq_days-1) per verificare che un eventuale
        # effetto non sia un artefatto di allineamento casuale con un particolare
        # giorno della settimana -- controllo di robustezza, non un parametro da
        # calibrare in produzione (default 0). Il trailing high
        # (pos["highest_price"]) resta aggiornato OGNI giorno indipendentemente
        # da questo gate (e' tracciamento dati, non una decisione di trading):
        # se aggiornato solo nei giorni di verifica, lo stop calcolato quel
        # giorno userebbe un massimo trascinato stantio, sottostimando il vero
        # trailing high raggiunto negli inframezzi. Disattivo di default
        # (use_weekly_exit_freq=False) -> bit-identico al comportamento precedente,
        # exit_freq_days/offset irrilevanti quando lo switch e' off.
        check_exits_today = (not p.get("use_weekly_exit_freq", False)) or \
            ((date_idx + p.get("exit_freq_offset", 0)) % p.get("exit_freq_days", 5) == 0)

        to_remove = []
        for t, pos in portfolio.items():
            if date not in signals[t].index:
                continue
            row = signals[t].loc[date]

            # La posizione entra dopo la fase uscite del giorno zero. Ogni
            # successiva riga negoziabile del ticker incrementa quindi di una
            # seduta l'eta' osservabile, senza usare il calendario futuro.
            pos["sessions_held"] = pos.get("sessions_held", 0) + 1

            pos["highest_price"] = max(pos["highest_price"], pos.get("_last_close", pos["highest_price"]))

            if (use_omd_forced_sell_exit
                    and t in current_omd_sell
                    and _close_position_at_open(
                        t, pos, date, row, fx_open_today, "OMD Sell"
                    )):
                to_remove.append(t)
                exited_tickers_today.add(t)
                continue

            if pd.isna(row.get("ATR_lag1")):
                continue

            if not check_exits_today:
                continue

            # Deve esistere anche in Legacy: viene consultato nella causale di
            # uscita comune, mentre solo il ramo moderno puo' impostarlo True.
            runner_stop_applied = False
            macro_exit_accel_applied = False
            macro_profit_lock_applied = False
            if p["legacy_mode"]:
                sl_level = pos["highest_price"] - (p["k_min"] * row["ATR_lag1"])
                is_tech_exit = row["EMA5_lag1"] < row["EMA21_lag1"]
            else:
                R_exit = (p["w_bf"] * row["breadth_factor_lag1"]) + (p["w_hmm"] * row["p_bull_bench_lag1"])
                k_dynamic = p["k_min"] + R_exit * (p["k_max"] - p["k_min"]) if p["use_dynamic_sl"] else p["k_min"]

                unreal_ret = (row["Close_lag1"] - pos["buy_price"]) / pos["buy_price"]
                macro_exit_policy = p.get(
                    "macro_state_exit_policy", "stop_mult"
                )
                valid_macro_exit_policies = {
                    "stop_mult", "profit_ratchet", "partial_profit",
                    "partial_plus_ratchet",
                }
                if macro_exit_policy not in valid_macro_exit_policies:
                    raise ValueError(
                        "macro_state_exit_policy non valida: "
                        f"{macro_exit_policy!r}."
                    )
                macro_state_active = (
                    p.get("use_macro_state_exit_accelerator", False)
                    and _macro_state_trigger(
                        row, p, "macro_state_exit_mode"
                    )
                )
                macro_micro_weak = (
                    pd.notna(row.get("Close_lag1"))
                    and pd.notna(row.get("EMA5_lag1"))
                    and float(row["Close_lag1"]) < float(row["EMA5_lag1"])
                )
                macro_exit_accel_applied = bool(
                    macro_state_active
                    and unreal_ret >= float(
                        p.get("macro_state_exit_min_profit", 0.15)
                    )
                    and (
                        not p.get(
                            "macro_state_exit_require_micro_weakness", True
                        )
                        or macro_micro_weak
                    )
                )

                # Stato post-unicorn: l'attivazione usa esclusivamente la Close
                # laggata. Il massimo di trailing contiene al piu' la Close di
                # ieri, perche' il mark-to-market della giornata corrente avviene
                # soltanto dopo tutte le decisioni e i fill.
                if p.get("use_post_unicorn_runner", False):
                    if (not pos.get("runner_active", False)
                            and unreal_ret >= p.get("runner_trigger", 0.50)):
                        pos["runner_active"] = True
                        pos["runner_trigger_date"] = date
                        events.append({
                            "Date": date, "Ticker": t, "Action": "RUNNER_ON",
                            "Price": float(row["Close_lag1"]),
                            "Mid_Price": float(row["Close_lag1"]),
                            "Shares": pos["shares"],
                            "Notional_Exec": 0.0, "Notional_Base": 0.0,
                            "Costs_Exec": 0.0, "Costs_Base": 0.0,
                            "FX_USD_per_EUR": fx_open_today,
                        })

                # Layer 5quater (nuovo, 20/07/2026): uscita scalata -- vende una QUOTA
                # (azioni intere, arrotondamento per difetto) al traguardo di profitto
                # profit_threshold: stesso trigger gia' usato dal profit-aware sotto
                # (il "profitto maturato" del motore), nessuna nuova soglia libera.
                # Blocca parte del gain SENZA chiudere la posizione: le azioni restanti
                # continuano a essere gestite dal Layer 5 (stop dinamico/profit-aware)
                # come se nulla fosse successo. Un solo scale-out per posizione. Fill
                # causale sull'Open di oggi, stessa convenzione di ogni esecuzione nel
                # motore. Il ricavo si accumula in pos["realized_revenue"] e confluisce
                # in un'UNICA riga di closed_trades alla chiusura finale (Sell_Price =
                # media ponderata sulle tranche) -- non due righe separate, altrimenti
                # ogni vincitore genererebbe un "unicorno" extra per costruzione
                # contabile, non per un vero evento di coda. Disattivo di default.
                # partial_exit_done si marca solo al trigger effettivo (non alla
                # semplice verifica), cosi' che la soglia resti disponibile per i
                # giorni successivi finche' non scatta davvero.
                if p.get("use_partial_exit", False) and not pos.get("partial_exit_done", False) \
                        and pd.notna(row.get("Open")):
                    effective_profit_threshold = p["profit_threshold"]
                    if unreal_ret >= effective_profit_threshold:
                        shares_to_sell = int(pos["shares"] * p.get("partial_exit_pct", 0.33))
                        if shares_to_sell > 0:
                            if "orig_shares" not in pos:
                                pos["orig_shares"] = pos["shares"]
                            mid_price_partial = float(row["Open"])
                            fill_price_partial = _sell_fill(mid_price_partial)
                            revenue_partial = (shares_to_sell * fill_price_partial) - commission
                            cash += revenue_partial
                            pos["realized_revenue"] = pos.get("realized_revenue", 0.0) + revenue_partial
                            pos["realized_revenue_base"] = (
                                pos.get("realized_revenue_base", 0.0)
                                + revenue_partial / fx_open_today
                            )
                            pos["shares"] -= shares_to_sell
                            impact_cost = shares_to_sell * (mid_price_partial - fill_price_partial)
                            events.append({
                                "Date": date, "Ticker": t, "Action": "PARTIAL_SELL",
                                "Price": fill_price_partial, "Mid_Price": mid_price_partial,
                                "Shares": shares_to_sell,
                                "Notional_Exec": shares_to_sell * fill_price_partial,
                                "Notional_Base": shares_to_sell * fill_price_partial / fx_open_today,
                                "Costs_Exec": commission + impact_cost,
                                "Costs_Base": (commission + impact_cost) / fx_open_today,
                                "FX_USD_per_EUR": fx_open_today,
                            })
                        pos["partial_exit_done"] = True

                # De-risking macro mirato: una sola presa parziale, soltanto
                # quando lo stato macro è attivo, il trade è già in profitto e
                # (di default) il titolo ha perso EMA5 sulla Close nota.
                if (macro_exit_accel_applied
                        and macro_exit_policy in {
                            "partial_profit", "partial_plus_ratchet"
                        }
                        and not pos.get("macro_partial_exit_done", False)
                        and unreal_ret >= float(
                            p.get("macro_partial_min_profit", 0.25)
                        )
                        and pd.notna(row.get("Open"))):
                    macro_partial_pct = float(
                        p.get("macro_partial_exit_pct", 0.25)
                    )
                    if not 0 < macro_partial_pct < 1:
                        raise ValueError(
                            "macro_partial_exit_pct deve essere in (0, 1)."
                        )
                    shares_to_sell = int(pos["shares"] * macro_partial_pct)
                    if shares_to_sell > 0:
                        if "orig_shares" not in pos:
                            pos["orig_shares"] = pos["shares"]
                        mid_price_partial = float(row["Open"])
                        fill_price_partial = _sell_fill(mid_price_partial)
                        revenue_partial = (
                            shares_to_sell * fill_price_partial
                        ) - commission
                        cash += revenue_partial
                        pos["realized_revenue"] = (
                            pos.get("realized_revenue", 0.0) + revenue_partial
                        )
                        pos["realized_revenue_base"] = (
                            pos.get("realized_revenue_base", 0.0)
                            + revenue_partial / fx_open_today
                        )
                        pos["shares"] -= shares_to_sell
                        impact_cost = shares_to_sell * (
                            mid_price_partial - fill_price_partial
                        )
                        events.append({
                            "Date": date, "Ticker": t,
                            "Action": "MACRO_PARTIAL_SELL",
                            "Price": fill_price_partial,
                            "Mid_Price": mid_price_partial,
                            "Shares": shares_to_sell,
                            "Notional_Exec": shares_to_sell * fill_price_partial,
                            "Notional_Base": (
                                shares_to_sell * fill_price_partial
                                / fx_open_today
                            ),
                            "Costs_Exec": commission + impact_cost,
                            "Costs_Base": (
                                commission + impact_cost
                            ) / fx_open_today,
                            "FX_USD_per_EUR": fx_open_today,
                        })
                    pos["macro_partial_exit_done"] = True

                if p["use_profit_aware"] and unreal_ret >= p["profit_threshold"]:
                    k_used = p["k_profit_cap"]
                else:
                    k_used = k_dynamic
                    if p.get("use_extension_on_exit", False) and unreal_ret < p.get("ext_exit_gate", 0.0):
                        # Layer 5bis (nuovo, 20/07/2026): simmetrico al profit-aware sopra.
                        # Quello allarga lo stop su chi e' gia' in utile; questo lo stringe
                        # su chi e' ESTESO (extension_penalty_lag1 basso, stessa formula/
                        # soglia del Filtro Unicorno in ingresso, ext_threshold/ext_k) E
                        # ancora privo di un cuscinetto di profitto (sotto ext_exit_gate,
                        # default 0.0 = ancora in rosso o pari). Mutuamente esclusivo col
                        # ramo profit-aware per costruzione (si entra qui solo se
                        # unreal_ret < profit_threshold). Disattivo di default
                        # (use_extension_on_exit=False), nessuna regressione.
                        k_used = k_used * row["extension_penalty_lag1"]

                    if p.get("use_ema_decay_on_exit", False):
                        # Layer 5ter (nuovo, 20/07/2026): stringe lo stop in funzione del
                        # TREND DEL SINGOLO TITOLO (ema_signal_lag1: 0 esattamente al
                        # momento del Trend Break, sale verso 1.0 con trend solido), non
                        # solo della breadth di mercato (R_exit sopra, identica per tutti
                        # i titoli in un dato giorno). Cattura il gap EMA5/21/63 che si
                        # restringe ma non ha ancora incrociato -- anticipazione del Trend
                        # Break, non un duplicato (il Trend Break scatta comunque quando
                        # ema_signal_lag1 tocca 0). Stessa forma funzionale di vix_factor/
                        # cor1m_factor. Applicato SOLO nel ramo non-profit-aware, stessa
                        # logica del Layer 5bis: non tocca lo stop allargato dei vincitori
                        # gia' in utile, per non penalizzare l'Unicorn Rate. Disattivo di
                        # default, nessuna regressione.
                        x = p["ema_decay_k"] * (row["ema_signal_lag1"] - p["ema_decay_threshold"])
                        ema_decay_factor = p["ema_decay_floor"] + (1 - p["ema_decay_floor"]) / (1 + np.exp(x))
                        k_used = k_used * ema_decay_factor

                if p["use_portfolio_brake"]:
                    # Leva (b): quando la correlazione realizzata sale, lo stop si stringe
                    # SIMULTANEAMENTE su tutte le posizioni aperte (corr_factor_lag1 e'
                    # lo stesso valore, universo-wide, per ogni ticker in questa data)
                    k_used = k_used * row["corr_factor_lag1"]

                if p["use_cor1m_brake"]:
                    # Stessa leva, guidata dal segnale forward-looking ^COR1M
                    k_used = k_used * row["cor1m_factor_lag1"]

                if (macro_exit_accel_applied
                        and macro_exit_policy == "stop_mult"):
                    macro_k_mult = float(
                        p.get("macro_state_exit_k_mult", 0.65)
                    )
                    if not 0 < macro_k_mult <= 1:
                        raise ValueError(
                            "macro_state_exit_k_mult deve essere in (0, 1]."
                        )
                    k_used = k_used * macro_k_mult

                sl_level = pos["highest_price"] - (k_used * row["ATR_lag1"])
                is_tech_exit = row["ema_signal_lag1"] == 0

                if pos.get("runner_active", False):
                    runner_stop_mode = p.get("runner_stop_mode")
                    if runner_stop_mode is None:
                        runner_stop_mode = (
                            "pct" if p.get("runner_use_pct_stop", True) else "atr"
                        )
                    if runner_stop_mode not in {
                        "pct", "atr", "hybrid", "bounded",
                        "profit_ratchet",
                    }:
                        raise ValueError(
                            "runner_stop_mode deve essere 'pct', 'atr', "
                            "'hybrid', 'bounded' o 'profit_ratchet', ricevuto "
                            f"{runner_stop_mode!r}."
                        )
                    if runner_stop_mode in {
                        "pct", "hybrid", "bounded", "profit_ratchet"
                    }:
                        runner_giveback = p.get("runner_giveback_pct", 0.30)
                        if (macro_exit_accel_applied
                                and macro_exit_policy == "stop_mult"):
                            runner_giveback = min(
                                runner_giveback,
                                float(p.get("macro_state_runner_giveback", 0.10)),
                            )
                        runner_pct_stop = pos["highest_price"] * (
                            1.0 - runner_giveback
                        )
                        runner_floor_stop = pos["buy_price"] * (
                            1.0 + p.get("runner_min_locked_gain", 0.10)
                        )
                        if runner_stop_mode == "hybrid":
                            sl_level = max(
                                runner_floor_stop,
                                min(runner_pct_stop, sl_level),
                            )
                        elif runner_stop_mode == "bounded":
                            min_giveback = p.get(
                                "runner_giveback_pct", 0.20
                            )
                            max_giveback = p.get(
                                "runner_max_giveback_pct", 0.35
                            )
                            if not 0 <= min_giveback <= max_giveback < 1:
                                raise ValueError(
                                    "Il giveback runner bounded richiede "
                                    "0 <= minimo <= massimo < 1."
                                )
                            atr_distance = max(
                                pos["highest_price"] - sl_level, 0.0
                            )
                            effective_distance = np.clip(
                                atr_distance,
                                min_giveback * pos["highest_price"],
                                max_giveback * pos["highest_price"],
                            )
                            sl_level = max(
                                runner_floor_stop,
                                pos["highest_price"] - effective_distance,
                            )
                        elif runner_stop_mode == "profit_ratchet":
                            lock_fraction = p.get(
                                "runner_profit_lock_fraction", 0.50
                            )
                            if not 0 <= lock_fraction <= 1:
                                raise ValueError(
                                    "runner_profit_lock_fraction deve essere "
                                    "compreso tra 0 e 1."
                                )
                            ratchet_stop = pos["buy_price"] + (
                                lock_fraction
                                * (pos["highest_price"] - pos["buy_price"])
                            )
                            sl_level = max(
                                runner_floor_stop, ratchet_stop
                            )
                        else:
                            sl_level = max(
                                runner_pct_stop, runner_floor_stop
                            )
                        runner_stop_applied = True
                    if p.get("runner_ignore_trend_break", True):
                        is_tech_exit = False

                macro_profit_lock_applied = False
                if (macro_exit_accel_applied
                        and macro_exit_policy in {
                            "profit_ratchet", "partial_plus_ratchet"
                        }):
                    lock_fraction = float(
                        p.get("macro_profit_lock_fraction", 0.50)
                    )
                    if not 0 <= lock_fraction <= 1:
                        raise ValueError(
                            "macro_profit_lock_fraction deve essere in [0, 1]."
                        )
                    macro_ratchet_stop = pos["buy_price"] + (
                        lock_fraction
                        * (pos["highest_price"] - pos["buy_price"])
                    )
                    if macro_ratchet_stop > sl_level:
                        sl_level = macro_ratchet_stop
                        macro_profit_lock_applied = True

            is_sl = row["Close_lag1"] < sl_level
            is_failed_launch = False
            if (not p["legacy_mode"] and p.get("use_failed_launch_exit", False)
                    and not pos.get("runner_active", False)):
                failed_launch_days = int(p.get("failed_launch_days", 3))
                failed_launch_threshold = float(
                    p.get("failed_launch_atr_threshold", -1.0)
                )
                if failed_launch_days < 1:
                    raise ValueError("failed_launch_days deve essere >= 1.")
                if failed_launch_threshold > 0:
                    raise ValueError(
                        "failed_launch_atr_threshold deve essere <= 0."
                    )
                entry_atr = pos.get("entry_atr")
                if (1 <= pos.get("sessions_held", 0) <= failed_launch_days
                        and pd.notna(entry_atr) and float(entry_atr) > 0):
                    failed_launch_level = pos["buy_price"] + (
                        failed_launch_threshold * float(entry_atr)
                    )
                    is_failed_launch = row["Close_lag1"] < failed_launch_level

            if (is_sl or is_tech_exit or is_failed_launch) and pd.notna(row.get("Open")):
                mid_price = float(row["Open"])
                fill_price = _sell_fill(mid_price)
                revenue = (pos["shares"] * fill_price) - commission
                cash += revenue
                realized_partial = pos.get("realized_revenue", 0.0)
                if realized_partial > 0:
                    # Layer 5quater era scattato su questa posizione: accorpa le tranche
                    # (vedi commento sopra) in un'unica riga, prezzo medio ponderato.
                    total_revenue = realized_partial + revenue
                    orig_shares = pos.get("orig_shares", pos["shares"])
                    profit = total_revenue - pos["cost_basis"]
                    report_shares = orig_shares
                    report_sell_price = total_revenue / orig_shares if orig_shares > 0 else fill_price
                else:
                    profit = revenue - pos["cost_basis"]
                    report_shares = pos["shares"]
                    report_sell_price = fill_price
                total_revenue_base = pos.get("realized_revenue_base", 0.0) + revenue / fx_open_today
                profit_base = total_revenue_base - pos["cost_basis_base"]
                realized_return = report_sell_price / pos["buy_price"] - 1.0
                mfe = pos["mfe_highest_price"] / pos["buy_price"] - 1.0
                mae = pos["lowest_price"] / pos["buy_price"] - 1.0
                if is_sl and macro_profit_lock_applied:
                    motivo = "Macro Profit Lock"
                elif is_sl and runner_stop_applied:
                    motivo = "Runner Stop"
                elif (is_sl and macro_exit_accel_applied
                        and macro_exit_policy == "stop_mult"):
                    motivo = "Macro Accelerated Stop"
                elif is_failed_launch:
                    motivo = "Failed Launch"
                else:
                    motivo = "SL Netto" if is_sl else "Trend Break"
                closed_trades.append({
                    "Entry_Date": pos["entry_date"], "Exit_Date": date, "Ticker": t,
                    "Shares": report_shares, "Buy_Price": pos["buy_price"], "Sell_Price": report_sell_price,
                    "Buy_Price_USD": pos["buy_price"], "Sell_Price_USD": report_sell_price,
                    "Profit": profit_base if use_fx else profit,
                    "Profit_Base": profit_base if use_fx else profit,
                    "Profit_Exec": profit,
                    "Return_%": realized_return * 100.0,
                    "MFE_%": mfe * 100.0, "MAE_%": mae * 100.0,
                    "Peak_Giveback_pp": (mfe - realized_return) * 100.0,
                    "Runner_Activated": bool(pos.get("runner_active", False)),
                    "Runner_Trigger_Date": pos.get("runner_trigger_date", pd.NaT),
                    "Holding_Sessions": int(pos.get("sessions_held", 0)),
                    "Entry_FX": pos["entry_fx"], "Exit_FX": fx_open_today,
                    "Motivo_Chiusura": motivo, "Stato": "Chiusa",
                })
                impact_cost = pos["shares"] * (mid_price - fill_price)
                events.append({
                    "Date": date, "Ticker": t, "Action": "SELL",
                    "Price": fill_price, "Mid_Price": mid_price,
                    "Shares": pos["shares"],
                    "Notional_Exec": pos["shares"] * fill_price,
                    "Notional_Base": pos["shares"] * fill_price / fx_open_today,
                    "Costs_Exec": commission + impact_cost,
                    "Costs_Base": (commission + impact_cost) / fx_open_today,
                    "FX_USD_per_EUR": fx_open_today,
                })
                to_remove.append(t)
                exited_tickers_today.add(t)

        for t in to_remove:
            del portfolio[t]

        # Equity corrente causale per il cap di sizing (§7.6): cash disponibile
        # dopo le uscite di oggi + mark-to-market delle posizioni ancora aperte,
        # valutate all'ultima Close nota (quella di ieri: il MTM di oggi avviene
        # solo nella sezione 3, piu' sotto). Nessun look-ahead: a questo punto del
        # loop la Close di oggi non e' ancora stata osservata per queste posizioni.
        current_equity_exec = cash
        for t, pos in portfolio.items():
            current_equity_exec += pos["shares"] * pos.get("_last_close", pos["buy_price"])
        if use_fx:
            try:
                fx_for_cap = _fx_value(date, "fx_close_usd_per_eur_lag1")
            except ValueError:
                fx_for_cap = fx_open_today
        else:
            fx_for_cap = 1.0
        current_equity_base = current_equity_exec / fx_for_cap

        # --- 2. INGRESSI ---
        pct_b_lag1 = None
        market_row_today = None
        for t in target_tickers:
            if t in signals and date in signals[t].index:
                candidate_row = signals[t].loc[date]
                if market_row_today is None:
                    market_row_today = candidate_row
                if pd.notna(candidate_row.get("pct_above_200_lag1")):
                    pct_b_lag1 = candidate_row["pct_above_200_lag1"]
                    break

        if p.get("use_macro_exposure_cap", False):
            if market_row_today is None:
                market_row_today = pd.Series(dtype=float)
            (
                macro_exposure_cap,
                macro_exposure_hold_remaining,
                macro_risk_state,
                macro_target_cap,
            ) = _update_macro_exposure_cap(
                market_row_today,
                p,
                macro_exposure_cap,
                macro_exposure_hold_remaining,
            )
        else:
            macro_exposure_cap = 1.0
            macro_exposure_hold_remaining = 0
            macro_risk_state = "normal"
            macro_target_cap = 1.0

        gross_exposure_exec_before_entries = sum(
            pos["shares"] * pos.get("_last_close", pos["buy_price"])
            for pos in portfolio.values()
        )
        if (p.get("use_macro_exposure_cap", False)
                and macro_exposure_cap < 1.0 - 1e-12):
            macro_entry_budget_exec = max(
                0.0,
                macro_exposure_cap * current_equity_exec
                - gross_exposure_exec_before_entries,
            )
        else:
            macro_entry_budget_exec = float("inf")
        macro_entry_budget_start_exec = macro_entry_budget_exec
        macro_cap_constrained_orders = 0
        position_cap_eligible_orders = 0
        position_cap_constrained_orders = 0

        if p["legacy_mode"]:
            market_halt = (pct_b_lag1 is not None) and (pct_b_lag1 < p["old_breadth_threshold"])
        else:
            market_halt = (not p["use_2d_breadth"]) and (pct_b_lag1 is not None) and (pct_b_lag1 < p["old_breadth_threshold"])

        if not market_halt:
            daily_signals = []
            for t in target_tickers:
                if t not in signals or date not in signals[t].index or t in portfolio:
                    continue
                if use_omd_entry_filter and t not in current_omd_buy:
                    continue
                if block_same_open and t in exited_tickers_today:
                    continue
                row = signals[t].loc[date]
                if pd.isna(row.get("ema_signal_lag1")) or pd.isna(row.get("Open")):
                    continue

                ema_sig = row["ema_signal_lag1"]
                if ema_sig <= 0:
                    continue

                entry_size_factor = 1.0
                entry_open_ema_gate = p.get("entry_open_ema_gate", "none")
                gate_columns = {
                    "none": None,
                    "fast": "EMA5_lag1",
                    "mid": "EMA21_lag1",
                    "mid_atr_hard": "EMA21_lag1",
                    "mid_atr_soft": "EMA21_lag1",
                }
                if entry_open_ema_gate not in gate_columns:
                    raise ValueError(
                        "entry_open_ema_gate deve essere 'none', 'fast', "
                        "'mid', 'mid_atr_hard' o 'mid_atr_soft', "
                        f"ricevuto {entry_open_ema_gate!r}."
                    )
                gate_column = gate_columns[entry_open_ema_gate]
                if entry_open_ema_gate in {"fast", "mid"}:
                    gate_value = row.get(gate_column)
                    if pd.isna(gate_value) or float(row["Open"]) <= float(gate_value):
                        continue
                elif entry_open_ema_gate in {"mid_atr_hard", "mid_atr_soft"}:
                    gate_value = row.get(gate_column)
                    atr_value = row.get("ATR_lag1")
                    if (pd.isna(gate_value) or pd.isna(atr_value)
                            or float(atr_value) <= 0):
                        continue
                    open_ema_atr = (
                        (float(row["Open"]) - float(gate_value))
                        / float(atr_value)
                    )
                    if entry_open_ema_gate == "mid_atr_hard":
                        threshold = float(
                            p.get("entry_open_ema_atr_threshold", -0.5)
                        )
                        if open_ema_atr <= threshold:
                            continue
                    else:
                        soft_floor = float(
                            p.get("entry_open_ema_atr_soft_floor", -1.0)
                        )
                        if soft_floor >= 0:
                            raise ValueError(
                                "entry_open_ema_atr_soft_floor deve essere < 0."
                            )
                        entry_size_factor = float(np.clip(
                            (open_ema_atr - soft_floor) / (0.0 - soft_floor),
                            0.0,
                            1.0,
                        ))
                        if entry_size_factor <= 0:
                            continue

                atr_value = row.get("ATR_lag1")
                fast_loss_gate_regime = p.get(
                    "entry_fast_loss_gate_regime", "all"
                )
                if fast_loss_gate_regime not in {"all", "bull"}:
                    raise ValueError(
                        "entry_fast_loss_gate_regime deve essere 'all' o "
                        f"'bull', ricevuto {fast_loss_gate_regime!r}."
                    )
                fast_loss_regime_is_bull = (
                    row.get("breadth_factor_lag1", 1.0)
                    >= p.get("regime_bull_threshold", 0.5)
                )
                apply_fast_loss_gates = (
                    fast_loss_gate_regime == "all"
                    or fast_loss_regime_is_bull
                )
                if (apply_fast_loss_gates
                        and p.get("use_entry_close_ema21_atr_gate", False)):
                    ema21_value = row.get("EMA21_lag1")
                    close_value = row.get("Close_lag1")
                    if (pd.isna(ema21_value) or pd.isna(close_value)
                            or pd.isna(atr_value) or float(atr_value) <= 0):
                        continue
                    close_ema21_atr = (
                        (float(close_value) - float(ema21_value))
                        / float(atr_value)
                    )
                    if close_ema21_atr <= float(
                        p.get("entry_close_ema21_atr_threshold", -0.15)
                    ):
                        continue

                if (apply_fast_loss_gates
                        and p.get("use_entry_ema_spread_atr_gate", False)):
                    ema5_value = row.get("EMA5_lag1")
                    ema21_value = row.get("EMA21_lag1")
                    if (pd.isna(ema5_value) or pd.isna(ema21_value)
                            or pd.isna(atr_value) or float(atr_value) <= 0):
                        continue
                    ema_spread_atr = (
                        (float(ema5_value) - float(ema21_value))
                        / float(atr_value)
                    )
                    if ema_spread_atr <= float(
                        p.get("entry_ema_spread_atr_threshold", 0.05)
                    ):
                        continue

                if (p.get("use_macro_state_entry_brake", False)
                        and _macro_state_trigger(
                            row, p, "macro_state_entry_mode"
                        )):
                    macro_size_factor = float(
                        p.get("macro_state_entry_size_factor", 0.0)
                    )
                    if not 0 <= macro_size_factor <= 1:
                        raise ValueError(
                            "macro_state_entry_size_factor deve essere in [0, 1]."
                        )
                    entry_size_factor *= macro_size_factor
                    if entry_size_factor <= 0:
                        continue

                if not p["legacy_mode"]:
                    regime_is_bull = row.get("breadth_factor_lag1", 1.0) >= p.get("regime_bull_threshold", 0.5)
                    apply_gates = (not p.get("use_regime_conditional_gates", False)) or regime_is_bull
                    if apply_gates:
                        if p.get("use_gap_gate", False) and row.get("gap_rank_lag1", 1.0) < p["gap_gate_threshold"]:
                            continue
                        if p.get("use_lowdist_gate", False) and row.get("lowdist_rank_lag1", 1.0) < p["lowdist_gate_threshold"]:
                            continue

                if p["legacy_mode"]:
                    hmm_amp = 1.0 + p["hmm_w"] * (row["p_bull_bench_lag1"] - 0.5)
                    target_size = base_size * hmm_amp * row["vix_factor_bin_lag1"]
                    daily_signals.append({
                        "ticker": t, "price": float(row["Open"]),
                        "target_size": target_size, "score": 1.0,
                        "entry_size_factor": entry_size_factor,
                        "entry_atr": row.get("ATR_lag1"),
                    })
                else:
                    raw_score = (ema_sig * row["macro_regime_score_lag1"]
                                 * (1 + row["breakout_bonus_lag1"])
                                 * row["extension_penalty_lag1"]
                                 * row["vix_factor_lag1"])
                    if p["use_portfolio_brake"]:
                        raw_score = raw_score * row["corr_factor_lag1"]
                    if p["use_cor1m_brake"]:
                        raw_score = raw_score * row["cor1m_factor_lag1"]
                    if p.get("use_gap_signal", False):
                        raw_score = raw_score * row["gap_factor_lag1"]
                    if p.get("use_lowdist_signal", False):
                        raw_score = raw_score * row["lowdist_factor_lag1"]
                    if p.get("use_ar_weighted_momentum", False):
                        # Layer AR-weighted momentum (nuovo, 23/07/2026): bonus
                        # additivo (mai <1.0, vedi compute_ar_bonus_factor),
                        # stessa forma sicura di gap_factor/lowdist_factor.
                        raw_score = raw_score * row["ar_bonus_factor_lag1"]
                    if raw_score > p["entry_threshold"]:
                        position_scalar = raw_score ** p["convexity_exp"]
                        target_size = base_size * position_scalar if p["use_scalar_sizing"] else base_size
                        daily_signals.append({
                            "ticker": t, "price": float(row["Open"]),
                            "target_size": target_size, "score": raw_score,
                            "entry_size_factor": entry_size_factor,
                            "entry_atr": row.get("ATR_lag1"),
                        })

            daily_signals = sorted(daily_signals, key=lambda x: x["score"], reverse=True)
            cap_base = current_equity_base if p.get("use_equity_cap", False) else capital
            position_cap_eligible_orders = len(daily_signals)
            for sig in daily_signals:
                if sig["target_size"] > (
                    cap_base * float(p.get("position_cap_pct", 0.15)) + 1e-12
                ):
                    position_cap_constrained_orders += 1
                size_final_base = (
                    min(
                        sig["target_size"],
                        cap_base * float(p.get("position_cap_pct", 0.15)),
                    )
                    * sig.get("entry_size_factor", 1.0)
                )
                size_final_exec = size_final_base * fx_open_today
                if (p.get("use_macro_exposure_cap", False)
                        and np.isfinite(macro_entry_budget_exec)):
                    # Il residuo viene assegnato per score: il sort precedente
                    # garantisce che i segnali migliori abbiano priorita'.
                    if macro_entry_budget_exec < size_final_exec - 1e-12:
                        macro_cap_constrained_orders += 1
                    size_final_exec = min(
                        size_final_exec, macro_entry_budget_exec
                    )
                mid_price = sig["price"]
                fill_price = _buy_fill(mid_price)
                # Mantiene la regola storica di prenotazione della size: si
                # apre l'ordine solo se il cash copre l'intero target teorico
                # piu' la commissione, prima dell'arrotondamento delle azioni.
                # E' necessaria per la parita' del path di portafoglio.
                if cash < size_final_exec + commission:
                    continue
                if fill_price > 0:
                    shares = int(size_final_exec / fill_price)
                    if shares > 0:
                        actual_cost = (shares * fill_price) + commission
                        if cash < actual_cost:
                            continue
                        cash -= actual_cost
                        if (p.get("use_macro_exposure_cap", False)
                                and np.isfinite(macro_entry_budget_exec)):
                            macro_entry_budget_exec = max(
                                0.0,
                                macro_entry_budget_exec - shares * fill_price,
                            )
                        portfolio[sig["ticker"]] = {
                            "shares": shares, "entry_date": date, "buy_price": fill_price,
                            "highest_price": fill_price, "lowest_price": fill_price,
                            "mfe_highest_price": fill_price,
                            "runner_active": False,
                            "runner_trigger_date": pd.NaT,
                            "cost_basis": actual_cost,
                            "cost_basis_base": actual_cost / fx_open_today,
                            "entry_fx": fx_open_today,
                            "entry_atr": sig.get("entry_atr"),
                            "sessions_held": 0,
                            "_last_close": mid_price,
                        }
                        impact_cost = shares * (fill_price - mid_price)
                        events.append({
                            "Date": date, "Ticker": sig["ticker"], "Action": "BUY",
                            "Price": fill_price, "Mid_Price": mid_price,
                            "Shares": shares,
                            "Notional_Exec": shares * fill_price,
                            "Notional_Base": shares * fill_price / fx_open_today,
                            "Costs_Exec": commission + impact_cost,
                            "Costs_Base": (commission + impact_cost) / fx_open_today,
                            "FX_USD_per_EUR": fx_open_today,
                            "Macro_Risk_State": macro_risk_state,
                            "Exposure_Cap": macro_exposure_cap,
                        })

        # --- 3. MARK-TO-MARKET (Close reale di oggi) ---
        daily_equity_exec = cash
        for t, pos in portfolio.items():
            if date in signals[t].index and pd.notna(signals[t].loc[date, "Close"]):
                c = float(signals[t].loc[date, "Close"])
                pos["_last_close"] = c
                pos["highest_price"] = max(pos["highest_price"], c)
                high = float(signals[t].loc[date].get("High", c))
                low = float(signals[t].loc[date].get("Low", c))
                pos["mfe_highest_price"] = max(pos["mfe_highest_price"], high)
                pos["lowest_price"] = min(pos["lowest_price"], low)
            else:
                # Calendari misti: se il ticker non quota oggi, la posizione
                # non scompare dal NAV. Si mantiene causalmente l'ultima Close
                # osservata, gia' salvata in _last_close. Nessun backfill futuro.
                c = float(pos.get("_last_close", pos["buy_price"]))
            daily_equity_exec += pos["shares"] * c
        daily_equity_base = daily_equity_exec / fx_close_today
        gross_exposure_exec = daily_equity_exec - cash
        exposure_ratio = (
            gross_exposure_exec / daily_equity_exec
            if daily_equity_exec > 0 else np.nan
        )
        history.append({
            "Date": date,
            "Equity": daily_equity_base,
            "Equity_Base": daily_equity_base,
            "Equity_Exec": daily_equity_exec,
            "Cash_Exec": cash,
            "Cash_Base": cash / fx_close_today,
            "Gross_Exposure_Exec": gross_exposure_exec,
            "Gross_Exposure_Base": gross_exposure_exec / fx_close_today,
            "Exposure_Ratio": exposure_ratio,
            "Exposure_Cap": macro_exposure_cap,
            "Macro_Target_Cap": macro_target_cap,
            "Macro_Risk_State": macro_risk_state,
            "Macro_Hold_Remaining": macro_exposure_hold_remaining,
            "Macro_Entry_Budget_Start_Base": (
                macro_entry_budget_start_exec / fx_open_today
                if np.isfinite(macro_entry_budget_start_exec) else np.nan
            ),
            "Macro_Entry_Budget_End_Base": (
                macro_entry_budget_exec / fx_open_today
                if np.isfinite(macro_entry_budget_exec) else np.nan
            ),
            "Macro_Cap_Constrained_Orders": macro_cap_constrained_orders,
            "Position_Cap_Eligible_Orders": position_cap_eligible_orders,
            "Position_Cap_Constrained_Orders": position_cap_constrained_orders,
            "FX_USD_per_EUR": fx_close_today,
        })

    # --- Trade ancora aperti a fine periodo ---
    open_trades = []
    if dates:
        last_date = dates[-1]
        for t, pos in portfolio.items():
            if (last_date in signals[t].index
                    and pd.notna(signals[t].loc[last_date, "Close"])):
                curr_price = float(signals[t].loc[last_date, "Close"])
            else:
                curr_price = float(pos.get("_last_close", pos["buy_price"]))
            if pd.notna(curr_price):
                realized_partial = pos.get("realized_revenue", 0.0)
                exit_fx = _fx_value(last_date, "fx_close_usd_per_eur")
                estimated_exit_fill = _sell_fill(curr_price)
                mtm_value = (pos["shares"] * estimated_exit_fill) - commission
                if realized_partial > 0:
                    total_value = realized_partial + mtm_value
                    report_shares = pos.get("orig_shares", pos["shares"])
                    unrealized = total_value - pos["cost_basis"]
                    report_sell_price = total_value / report_shares if report_shares > 0 else curr_price
                else:
                    unrealized = mtm_value - pos["cost_basis"]
                    report_shares = pos["shares"]
                    report_sell_price = estimated_exit_fill
                unrealized_base = (
                    pos.get("realized_revenue_base", 0.0)
                    + mtm_value / exit_fx
                    - pos["cost_basis_base"]
                )
                realized_return = report_sell_price / pos["buy_price"] - 1.0
                mfe = pos["mfe_highest_price"] / pos["buy_price"] - 1.0
                mae = pos["lowest_price"] / pos["buy_price"] - 1.0
                open_trades.append({
                    "Entry_Date": pos["entry_date"], "Exit_Date": pd.NaT, "Ticker": t,
                    "Shares": report_shares, "Buy_Price": pos["buy_price"], "Sell_Price": report_sell_price,
                    "Buy_Price_USD": pos["buy_price"], "Sell_Price_USD": report_sell_price,
                    "Profit": unrealized_base if use_fx else unrealized,
                    "Profit_Base": unrealized_base if use_fx else unrealized,
                    "Profit_Exec": unrealized,
                    "Return_%": realized_return * 100.0,
                    "MFE_%": mfe * 100.0, "MAE_%": mae * 100.0,
                    "Peak_Giveback_pp": (mfe - realized_return) * 100.0,
                    "Runner_Activated": bool(pos.get("runner_active", False)),
                    "Runner_Trigger_Date": pos.get("runner_trigger_date", pd.NaT),
                    "Holding_Sessions": int(pos.get("sessions_held", 0)),
                    "Entry_FX": pos["entry_fx"], "Exit_FX": exit_fx,
                    "Motivo_Chiusura": "-", "Stato": "Aperta",
                })

    df_closed = pd.DataFrame(closed_trades)
    df_open = pd.DataFrame(open_trades)
    df_all = pd.concat([df_closed, df_open], ignore_index=True) if not df_closed.empty or not df_open.empty else pd.DataFrame()

    return pd.DataFrame(history).set_index("Date"), pd.DataFrame(events), df_all, df_closed


# ==============================================================================
# 8. METRICHE ESTESE (rendimento netto, rischio, entrata e uscita)
# ==============================================================================
def calc_metrics_extended(
    equity_series,
    bench_equity_series,
    df_closed,
    events=None,
    initial_equity=None,
):
    """Calcola KPI in quattro famiglie.

    Outcome: CAGR e excess CAGR (differenza aritmetica vs benchmark).
    Guardrail: MaxDD e durata del drawdown.
    Entrata: quota di trade che raggiunge MFE >= +50%.
    Uscita: conversione in close >= +50%, peak capture e giveback.

    ``Alpha`` resta come alias retrocompatibile di ``Excess_CAGR_net_EUR``:
    non e' Jensen alpha.
    """

    def _cagr_sharpe_mdd(series, start_value=None):
        series = pd.Series(series).dropna().sort_index()
        if series.empty:
            return 0.0, 0.0, 0.0, 0
        start_value = float(start_value) if start_value is not None else float(series.iloc[0])
        augmented = pd.concat([
            pd.Series([start_value], index=[series.index[0] - pd.Timedelta(microseconds=1)]),
            series,
        ])
        ret = augmented.pct_change().dropna()
        days = (series.index[-1] - series.index[0]).days
        cagr = (series.iloc[-1] / start_value) ** (365.25 / days) - 1 if days > 0 else 0.0
        ret_std = ret.std()
        sharpe = (ret.mean() / ret_std) * np.sqrt(252) if ret_std > 0 else 0.0
        drawdown = augmented / augmented.cummax() - 1.0
        mdd = float(drawdown.min())

        max_duration = 0
        current_duration = 0
        for value in drawdown.iloc[1:]:
            if value < 0:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        return cagr, sharpe, mdd, max_duration

    bench_start = initial_equity if initial_equity is not None else None
    cagr_s, sharpe_s, mdd_s, dd_duration_s = _cagr_sharpe_mdd(equity_series, initial_equity)
    cagr_b, sharpe_b, mdd_b, dd_duration_b = _cagr_sharpe_mdd(bench_equity_series, bench_start)

    n_trades = len(df_closed)
    if n_trades > 0:
        win_trades = df_closed[df_closed["Profit"] > 0]
        win_rate = len(win_trades) / n_trades * 100
        profit_for_fast_loss = pd.to_numeric(
            df_closed.get("Profit_Base", df_closed["Profit"]), errors="coerce"
        )
        if "Holding_Sessions" in df_closed:
            holding_sessions = pd.to_numeric(
                df_closed["Holding_Sessions"], errors="coerce"
            )
            fast_loss_mask = (
                profit_for_fast_loss < 0
            ) & holding_sessions.between(1, 3)
            fast_loss_count = int(fast_loss_mask.sum())
            fast_loss_rate = float(fast_loss_mask.mean() * 100.0)
        else:
            fast_loss_count = 0
            fast_loss_rate = np.nan
        if "Return_%" in df_closed:
            ret_pct = pd.to_numeric(df_closed["Return_%"], errors="coerce") / 100.0
        else:
            ret_pct = (df_closed["Sell_Price"] - df_closed["Buy_Price"]) / df_closed["Buy_Price"]
        unicorn_rate = (ret_pct >= 0.50).mean() * 100
        trades_100 = int((ret_pct >= 1.00).sum())
        holding_days = (pd.to_datetime(df_closed["Exit_Date"]) - pd.to_datetime(df_closed["Entry_Date"])).dt.days
        avg_holding = holding_days.mean()
        avg_holding_winners = holding_days[df_closed["Profit"] > 0].mean() if len(win_trades) > 0 else np.nan

        if "MFE_%" in df_closed:
            mfe_pct = pd.to_numeric(df_closed["MFE_%"], errors="coerce") / 100.0
            reached = mfe_pct >= 0.50
            entry_precision = reached.mean() * 100.0
            unicorn_conversion = (
                (ret_pct[reached] >= 0.50).mean() * 100.0 if reached.any() else 0.0
            )
            giveback_pp = (mfe_pct[reached] - ret_pct[reached]) * 100.0
            median_giveback = giveback_pp.median() if reached.any() else np.nan
            # Exit KPI calcolato solo sui trade che hanno davvero raggiunto la
            # soglia unicorn: sugli altri confonderebbe entry failure e exit.
            capture = ret_pct[reached] / mfe_pct[reached]
            median_peak_capture = capture.median() * 100.0 if reached.any() else np.nan
        else:
            entry_precision = unicorn_conversion = 0.0
            median_giveback = median_peak_capture = np.nan

        positive_profit = pd.to_numeric(df_closed["Profit"], errors="coerce").clip(lower=0)
        positive_profit_total = positive_profit.sum()
        tail_pnl_contribution = (
            positive_profit[ret_pct >= 0.50].sum() / positive_profit_total * 100.0
            if positive_profit_total > 0 else 0.0
        )
        top5_pnl_contribution = (
            positive_profit.nlargest(5).sum() / positive_profit_total * 100.0
            if positive_profit_total > 0 else 0.0
        )
        if "Runner_Activated" in df_closed:
            runner_mask = df_closed["Runner_Activated"].fillna(False).astype(bool)
        else:
            runner_mask = pd.Series(False, index=df_closed.index)
        runner_activated_trades = int(runner_mask.sum())
        runner_activation_rate = runner_mask.mean() * 100.0
        runner_close_conversion = (
            (ret_pct[runner_mask] >= 0.50).mean() * 100.0
            if runner_activated_trades else 0.0
        )
        runner_median_return = (
            ret_pct[runner_mask].median() * 100.0
            if runner_activated_trades else np.nan
        )
        runner_stop_exits = int(
            df_closed.get(
                "Motivo_Chiusura", pd.Series("", index=df_closed.index)
            ).eq("Runner Stop").sum()
        )
    else:
        win_rate = unicorn_rate = trades_100 = avg_holding = avg_holding_winners = 0
        entry_precision = unicorn_conversion = tail_pnl_contribution = top5_pnl_contribution = 0
        median_giveback = median_peak_capture = np.nan
        runner_activated_trades = runner_stop_exits = 0
        runner_activation_rate = runner_close_conversion = 0.0
        runner_median_return = np.nan
        fast_loss_count = 0
        fast_loss_rate = 0.0

    same_open_reentries = 0
    runner_activation_events = 0
    turnover_ann = cost_drag_ann = total_costs = 0.0
    if events is not None and not events.empty:
        ev = events.copy()
        runner_activation_events = int(ev["Action"].eq("RUNNER_ON").sum())
        order_ev = ev[ev["Action"].isin(["BUY", "SELL", "PARTIAL_SELL"])]
        if not order_ev.empty:
            grouped_actions = order_ev.groupby(["Date", "Ticker"])["Action"].agg(set)
            same_open_reentries = int(grouped_actions.apply(lambda x: "BUY" in x and "SELL" in x).sum())
        if "Costs_Base" in ev:
            total_costs = float(pd.to_numeric(ev["Costs_Base"], errors="coerce").fillna(0).sum())
        if "Notional_Base" in order_ev:
            total_notional = float(pd.to_numeric(order_ev["Notional_Base"], errors="coerce").fillna(0).sum())
            days = max((equity_series.index[-1] - equity_series.index[0]).days, 1)
            years = days / 365.25
            avg_equity = float(pd.Series(equity_series).mean())
            if avg_equity > 0 and years > 0:
                turnover_ann = total_notional / avg_equity / years
                cost_drag_ann = total_costs / avg_equity / years * 100.0

    return {
        "CAGR_strategia": cagr_s, "CAGR_benchmark": cagr_b,
        "Excess_CAGR_net_EUR": cagr_s - cagr_b,
        "Alpha": cagr_s - cagr_b,
        "Sharpe_strategia": sharpe_s, "Sharpe_benchmark": sharpe_b,
        "MaxDD_strategia": mdd_s, "MaxDD_benchmark": mdd_b,
        "MaxDD_Duration_Strategy_sessions": dd_duration_s,
        "MaxDD_Duration_Benchmark_sessions": dd_duration_b,
        "N_Trade_Chiusi": n_trades, "Win_Rate_%": win_rate,
        "Fast_Loss_3S_Count": fast_loss_count,
        "Fast_Loss_3S_Rate_%": fast_loss_rate,
        "Fast_Loss_3S_Avoidance_%": 100.0 - fast_loss_rate,
        "Unicorn_Rate_%": unicorn_rate, "Trade_>=100%": trades_100,
        "Entry_MFE_>=50%_%": entry_precision,
        "Unicorn_Conversion_%": unicorn_conversion,
        "Median_Peak_Capture_%": median_peak_capture,
        "Median_Peak_Giveback_pp": median_giveback,
        "Tail_Positive_PnL_Contribution_%": tail_pnl_contribution,
        "Top5_Positive_PnL_Contribution_%": top5_pnl_contribution,
        "Runner_Activated_Trades": runner_activated_trades,
        "Runner_Activation_Events": runner_activation_events,
        "Runner_Activation_Rate_%": runner_activation_rate,
        "Runner_Close_Conversion_%": runner_close_conversion,
        "Runner_Median_Return_%": runner_median_return,
        "Runner_Stop_Exits": runner_stop_exits,
        "Same_Open_Reentry_Count": same_open_reentries,
        "Annual_Turnover_x": turnover_ann,
        "Total_Explicit_and_Impact_Costs_Base": total_costs,
        "Annualized_Cost_Drag_%": cost_drag_ann,
        "Holding_Medio_gg": avg_holding, "Holding_Medio_Vincenti_gg": avg_holding_winners,
    }
