"""
OMD-Portfolio -- Fase 5 (fedele): catene condizionate, generatore continuo
---------------------------------------------------------------------------
Replica diretta di Eq. 9-16 del paper:

  - Eq. 9:  bucketing decile/quintile delle covariate, K_f (qui = K = 5,
            per la stessa ragione di ampiezza universo della catena target)
  - Eq. 11: intensita' di transizione log-lineari
            q_ab(x) = exp(eta_ab + gamma_ab . phi(x)),  phi = livelli-bucket centrati
  - Eq. 12-15: stima di massima verosimiglianza "alla Lando-Skodeberg":
            si scompone in una regressione di Poisson INDIPENDENTE per ogni
            coppia ordinata (a,b), a!=b, usando R_a(x) (mesi di esposizione,
            qui =1 per osservazione) e Delta N_ab(x) (transizioni osservate)
            come statistiche sufficienti. Nessun EM, nessun cammino latente:
            il pannello mensile stesso fornisce esposizioni e conteggi.
  - Eq. 10: la transizione a un mese e' P(x) = expm(Q(x)), non un'approssimazione
            lineare -- Q(x) e' la matrice generatrice assemblata dai q_ab(x) stimati.

Refit walk-forward ogni 12 mesi (Tabella 1: "refit: 12mo"), non mensile.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import expm
from sklearn.linear_model import PoissonRegressor

K = 5     # catena target: quintili (vedi nota ampiezza universo)
KF = 5    # bucketing covariate: stessa logica, invece del K_f=10 del paper
REFIT_MONTHS = 12  # Tabella 1 del paper


def centered_phi(bucket: pd.DataFrame, kf: int = KF) -> pd.DataFrame:
    """phi(x) = (x - (K_f+1)/2) / K_f -- livelli-bucket centrati, Eq. 11."""
    return (bucket - (kf + 1) / 2) / kf


def compute_bucketed_features(feature_dict_daily: dict, active: pd.DataFrame,
                               dates: pd.DatetimeIndex, kf: int = KF) -> dict:
    """Covariate bucketizzate cross-sezionalmente (Eq. 9) e centrate (Eq. 11),
    riusabili sia per costruire il pannello di training sia per il lookup
    diretto nei giorni di test (griglia mensile)."""
    from omd.covariates import bucket_cross_sectional
    bucketed = {}
    for name, df in feature_dict_daily.items():
        common = df.index.intersection(dates)
        b = bucket_cross_sectional(df.loc[common], active.loc[common], k=kf,
                                    ascending_is_best=True, min_names=15)
        bucketed[name] = centered_phi(b.reindex(dates), kf=kf)
    return bucketed


def build_daily_panel(class_daily: pd.DataFrame, bucketed: dict, active: pd.DataFrame,
                       k: int = K) -> pd.DataFrame:
    """Come build_panel, ma su griglia GIORNALIERA (transizione t -> giorno di
    trading successivo), per calibrare correttamente le statistiche sufficienti
    del generatore (R_a(x), Delta N_ab(x)) -- vedi discussione sull'artefatto
    di sovra-persistenza da campionamento solo mensile. Vettorizzato con
    stack() invece di un loop per riga, per tenere tempi ragionevoli."""
    dates = class_daily.index
    a_df = class_daily
    b_df = class_daily.shift(-1)
    act = active.reindex(dates).fillna(False)

    valid = a_df.notna() & b_df.notna() & act
    for name in bucketed:
        valid &= bucketed[name].reindex(dates).notna()

    a_long = a_df.where(valid).stack()
    b_long = b_df.where(valid).stack()
    panel = pd.DataFrame({"a": a_long, "b": b_long})
    for name in bucketed:
        panel[f"phi_{name}"] = bucketed[name].reindex(dates).where(valid).stack()
    panel = panel.dropna().reset_index()
    panel.columns = ["date", "ticker", "a", "b"] + [f"phi_{n}" for n in bucketed]
    panel["a"] = panel["a"].astype(int)
    panel["b"] = panel["b"].astype(int)
    return panel


def build_panel(class_df_monthly: pd.DataFrame, feature_dict: dict, active: pd.DataFrame,
                 k: int = K, kf: int = KF) -> pd.DataFrame:
    """Pannello lungo: una riga per (titolo, mese) con classe di origine a,
    classe di arrivo b (mese dopo), e le covariate bucket-centrate phi_*.
    Le covariate sono bucketizzate cross-sezionalmente PER MESE (Eq. 9)."""
    from omd.covariates import bucket_cross_sectional

    dates = class_df_monthly.index
    valid_dates = [d for d in dates if d in active.index]

    bucketed = {}
    for name, df in feature_dict.items():
        common = df.index.intersection(valid_dates)
        sub = df.loc[common]
        act_sub = active.loc[common]
        b = bucket_cross_sectional(sub, act_sub, k=kf, ascending_is_best=True, min_names=15)
        bucketed[name] = centered_phi(b, kf=kf)

    feat_names = list(feature_dict.keys())
    rows = []
    for j in range(len(dates) - 1):
        dt_a, dt_b = dates[j], dates[j + 1]
        if dt_a not in active.index:
            continue
        a_s, b_s = class_df_monthly.loc[dt_a], class_df_monthly.loc[dt_b]
        phi_cols = {}
        ok = True
        for name in feat_names:
            if dt_a not in bucketed[name].index:
                ok = False
                break
            phi_cols[name] = bucketed[name].loc[dt_a]
        if not ok:
            continue
        phi_df = pd.DataFrame(phi_cols)
        common = phi_df.dropna().index.intersection(a_s.dropna().index).intersection(b_s.dropna().index)
        for tkr in common:
            row = {"date": dt_b, "ticker": tkr, "a": int(a_s[tkr]), "b": int(b_s[tkr])}
            for name in feat_names:
                row[f"phi_{name}"] = phi_df.loc[tkr, name]
            rows.append(row)
    return pd.DataFrame(rows)


def fit_generator_pairs(train_df: pd.DataFrame, k: int = K, alpha: float = 1.0,
                         min_obs: int = 30, min_pos: int = 5):
    """Stima eta_ab, gamma_ab per ogni coppia (a,b), a!=b, via regressione di
    Poisson (Eq. 12-15). Ogni riga con origine a e' un'osservazione con
    esposizione R=1; la variabile risposta e' l'indicatore di transizione a->b.

    Se una coppia ha troppe poche transizioni osservate per stimare una
    regressione affidabile (tipico per salti "lontani", es. dalla coda peggiore
    alla migliore in un solo passo), il fallback e' un tasso quasi-nullo
    Laplace-smoothed -- NON un tasso "moderato" 1/K, che sovrastimerebbe
    grossolanamente una transizione rara e farebbe collassare la probabilita'
    di restare nello stato di origine (bug trovato e corretto: vedi discussione)."""
    phi_cols = [c for c in train_df.columns if c.startswith("phi_")]
    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a in range(1, k + 1):
            sub_a = train_df[train_df["a"] == a]
            exposure_a = len(sub_a)
            if exposure_a < min_obs:
                for b in range(1, k + 1):
                    if b != a:
                        models[(a, b)] = ("rate", np.log(0.5 / max(exposure_a, 1)))
                continue
            X = sub_a[phi_cols].values
            for b in range(1, k + 1):
                if b == a:
                    continue
                y = (sub_a["b"] == b).astype(int).values
                if y.sum() < min_pos:
                    # troppo rara per stimare gamma in modo affidabile: tasso
                    # quasi-nullo Laplace-smoothed, coerente con la scarsita' osservata
                    models[(a, b)] = ("rate", np.log(max(y.sum(), 0.5) / exposure_a))
                    continue
                reg = PoissonRegressor(alpha=alpha, max_iter=300)
                reg.fit(X, y)
                models[(a, b)] = ("glm", (reg.intercept_, reg.coef_))
    return models


def transition_matrix_for(models: dict, phi_x: np.ndarray, k: int = K,
                           fallback_rate: float = 0.01, cap: float = 5.0,
                           delta: float = 1.0) -> np.ndarray:
    """Assembla Q(x) dai q_ab(x) stimati (intensita' GIORNALIERE) e ritorna
    P(x) = expm(delta * Q(x)), Eq. 10-11. delta = orizzonte in giorni di borsa
    (1 per un giorno, ~21 per un mese -- usare il conteggio esatto quando noto).
    cap=5/giorno e' gia' molto generoso data la scala naturale osservata (~0.1-0.5/giorno)."""
    Q = np.zeros((k, k))
    for a in range(1, k + 1):
        for b in range(1, k + 1):
            if a == b:
                continue
            m = models.get((a, b))
            if m is None:
                q = fallback_rate
            elif m[0] == "rate":
                q = np.exp(np.clip(m[1], -20, np.log(cap)))
            else:
                intercept, coef = m[1]
                q = np.exp(np.clip(intercept + coef @ phi_x, -20, np.log(cap)))
            Q[a - 1, b - 1] = q
        Q[a - 1, a - 1] = -Q[a - 1].sum()
    P = expm(Q * delta)
    P = np.clip(P, 1e-8, None)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def walk_forward_daily_to_monthly(daily_panel: pd.DataFrame, class_monthly: pd.DataFrame,
                                   bucketed_daily: dict, k: int = K,
                                   refit_months: int = REFIT_MONTHS, min_train_days: int = 750):
    """Allena il generatore sulle transizioni GIORNALIERE (< data di refit),
    valuta sulla transizione MENSILE (mese m -> mese m+1) usando il numero
    esatto di giorni di borsa Delta tra le due date di griglia mensile:
    P(mese m -> m+1 | x) = expm(Delta * Q(x)). Refit ogni refit_months mesi."""
    phi_cols = [c for c in daily_panel.columns if c.startswith("phi_")]
    monthly_dates = class_monthly.index
    baseline_llh = np.log(1.0 / k)
    records = []

    models = None
    last_refit_month_idx = -10 ** 9
    for i in range(len(monthly_dates) - 1):
        dt_a, dt_b = monthly_dates[i], monthly_dates[i + 1]

        n_train_days = daily_panel.loc[daily_panel["date"] < dt_a, "date"].nunique()
        train_ready = n_train_days >= min_train_days
        need_refit = train_ready and (models is None or (i - last_refit_month_idx) >= refit_months)
        if need_refit:
            train_df = daily_panel[daily_panel["date"] < dt_a]
            models = fit_generator_pairs(train_df, k=k)
            last_refit_month_idx = i
        if models is None:
            continue

        a_s, b_s = class_monthly.loc[dt_a], class_monthly.loc[dt_b]
        # phi al giorno di griglia mensile dt_a (lookup diretto, non dal pannello)
        phi_today = {}
        ok = True
        for name in bucketed_daily:
            if dt_a not in bucketed_daily[name].index:
                ok = False
                break
            phi_today[name] = bucketed_daily[name].loc[dt_a]
        if not ok:
            continue
        phi_df = pd.DataFrame(phi_today)
        common = phi_df.dropna().index.intersection(a_s.dropna().index).intersection(b_s.dropna().index)
        if len(common) == 0:
            continue

        # Delta = n. di giorni di trading tra dt_a e dt_b nella griglia giornaliera originale
        daily_dates = bucketed_daily[list(bucketed_daily.keys())[0]].index
        delta_days = int(((daily_dates > dt_a) & (daily_dates <= dt_b)).sum())
        if delta_days <= 0:
            delta_days = 21  # fallback prudente

        for tkr in common:
            a_val, b_val = int(a_s[tkr]), int(b_s[tkr])
            phi_x = phi_df.loc[tkr].values.astype(float)
            P = transition_matrix_for(models, phi_x, k=k, delta=delta_days)
            p_true = max(P[a_val - 1, b_val - 1], 1e-6)
            records.append({"date": dt_b, "start": a_val, "end": b_val,
                             "llh_model": np.log(p_true), "llh_base": baseline_llh})

    return pd.DataFrame(records)


def fit_generator_nocov(train_df: pd.DataFrame, k: int = K, min_obs: int = 30):
    """Baseline SENZA covariate, stessa metodologia (statistiche sufficienti
    dal pannello giornaliero): q_ab = (transizioni a->b) / (esposizione in a).
    Serve per calcolare il guadagno INCREMENTALE delle covariate a parita' di
    metodo di calibrazione (niente piu' il confronto sbilanciato di prima)."""
    models = {}
    for a in range(1, k + 1):
        sub_a = train_df[train_df["a"] == a]
        exposure = len(sub_a)
        if exposure < min_obs:
            for b in range(1, k + 1):
                if b != a:
                    models[(a, b)] = None
            continue
        for b in range(1, k + 1):
            if b == a:
                continue
            n_ab = (sub_a["b"] == b).sum()
            models[(a, b)] = ("rate", np.log(max(n_ab, 0.5) / exposure))
    return models


def walk_forward_baseline_nocov(daily_panel: pd.DataFrame, class_monthly: pd.DataFrame,
                                 bucketed_daily: dict, k: int = K,
                                 refit_months: int = REFIT_MONTHS, min_train_days: int = 750):
    """Come walk_forward_daily_to_monthly ma senza covariate -- stessa identica
    metodologia di calibrazione (giornaliera -> expm a orizzonte mensile),
    cambia solo l'assenza di phi. Baseline corretta per il confronto."""
    monthly_dates = class_monthly.index
    baseline_llh = np.log(1.0 / k)
    records = []
    models = None
    last_refit_month_idx = -10 ** 9
    for i in range(len(monthly_dates) - 1):
        dt_a, dt_b = monthly_dates[i], monthly_dates[i + 1]
        n_train_days = daily_panel.loc[daily_panel["date"] < dt_a, "date"].nunique()
        train_ready = n_train_days >= min_train_days
        need_refit = train_ready and (models is None or (i - last_refit_month_idx) >= refit_months)
        if need_refit:
            train_df = daily_panel[daily_panel["date"] < dt_a]
            models = fit_generator_nocov(train_df, k=k)
            last_refit_month_idx = i
        if models is None:
            continue
        a_s, b_s = class_monthly.loc[dt_a], class_monthly.loc[dt_b]
        common = a_s.dropna().index.intersection(b_s.dropna().index)
        daily_dates = bucketed_daily[list(bucketed_daily.keys())[0]].index
        delta_days = int(((daily_dates > dt_a) & (daily_dates <= dt_b)).sum()) or 21
        P = transition_matrix_for(models, np.zeros(0), k=k, delta=delta_days)
        for tkr in common:
            a_val, b_val = int(a_s[tkr]), int(b_s[tkr])
            p_true = max(P[a_val - 1, b_val - 1], 1e-6)
            records.append({"date": dt_b, "start": a_val, "end": b_val,
                             "llh_model": np.log(p_true), "llh_base": baseline_llh})
    return pd.DataFrame(records)


def walk_forward_generator(panel: pd.DataFrame, k: int = K, refit_months: int = REFIT_MONTHS,
                            min_train_months: int = 36):
    """Walk-forward con refit ogni refit_months (Tabella 1: 12). Il generatore
    stimato all'ultimo refit viene usato per tutte le predizioni fino al refit
    successivo -- mai una transizione di training usata come test."""
    phi_cols = [c for c in panel.columns if c.startswith("phi_")]
    dates = sorted(panel["date"].unique())
    baseline_llh = np.log(1.0 / k)
    records = []

    models = None
    last_refit_idx = -10 ** 9
    for i, dt in enumerate(dates):
        train_ready = i >= min_train_months
        need_refit = train_ready and (models is None or (i - last_refit_idx) >= refit_months)
        if need_refit:
            train_dates = dates[:i]  # tutte le transizioni gia' osservate, MAI quella corrente
            train_df = panel[panel["date"].isin(train_dates)]
            models = fit_generator_pairs(train_df, k=k)
            last_refit_idx = i
        if models is None:
            continue

        test_rows = panel[panel["date"] == dt]
        for _, row in test_rows.iterrows():
            a_val, b_val = int(row["a"]), int(row["b"])
            phi_x = row[phi_cols].values.astype(float)
            P = transition_matrix_for(models, phi_x, k=k)
            p_true = max(P[a_val - 1, b_val - 1], 1e-6)
            records.append({"date": dt, "start": a_val, "end": b_val,
                             "llh_model": np.log(p_true), "llh_base": baseline_llh})

    return pd.DataFrame(records)
