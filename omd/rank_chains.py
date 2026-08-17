"""
OMD-Portfolio -- Fase 3: catene di rank (non condizionate)
--------------------------------------------------------------
Bucketing cross-sezionale in K=5 quintili (invece dei decili del paper,
vedi nota sull'ampiezza dell'universo), stima della matrice di transizione
pooled su griglia mensile, entropy production (Eq. 6), e un check di
calibrazione one-step nello stile della Fig. 3 del paper: rendimento
medio-prossimo-quintile realizzato per ciascun quintile di partenza.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

K = 5  # quintili, non decili: vedi nota sull'ampiezza dell'universo (N~55-60)
LAPLACE_EPS = 0.5  # pseudo-count per evitare celle a zero nella KL divergence


def rolling_stat(returns: pd.DataFrame, window: int, kind: str) -> pd.DataFrame:
    """kind='mean' -> rendimento medio trailing (statistica di performance)
       kind='vol'  -> volatilita' realizzata trailing (statistica di rischio)"""
    if kind == "mean":
        return returns.rolling(window, min_periods=window).mean()
    elif kind == "vol":
        return returns.rolling(window, min_periods=window).std()
    raise ValueError(kind)


def cross_sectional_quintile(stat: pd.DataFrame, active: pd.DataFrame,
                              ascending_is_best: bool, k: int = K,
                              min_names: int = 15) -> pd.DataFrame:
    """Bucketing per data: rank 1 = migliore. Per il rendimento "migliore"
    = piu' alto (ascending_is_best=False, si rank-a decrescente).
    Per la volatilita' "migliore" = piu' bassa (ascending_is_best=True).
    Giorni con meno di min_names titoli attivi validi vengono saltati (NaN)."""
    out = pd.DataFrame(index=stat.index, columns=stat.columns, dtype=float)
    for dt in stat.index:
        row = stat.loc[dt].where(active.loc[dt]).dropna()
        if len(row) < min_names:
            continue
        ranks = row.rank(ascending=ascending_is_best, method="first")
        buckets = np.ceil(k * ranks / len(row)).clip(1, k)
        out.loc[dt, buckets.index] = buckets
    return out


def monthly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Ultimo giorno di trading di ogni mese, come nel paper (griglia mensile)."""
    return df.resample("ME").last()


def completed_monthly_dates(index: pd.DatetimeIndex,
                            asof: pd.Timestamp | None = None) -> pd.DatetimeIndex:
    """Ultimo giorno disponibile di ogni mese di calendario gia' concluso.

    ``resample('ME')`` include anche il mese corrente e gli assegna come
    valore l'ultima seduta disponibile. Per una strategia che ribilancia solo
    a fine mese quella seduta non e' ancora una vera data di ribilanciamento.
    """
    dates = pd.DatetimeIndex(index).dropna().unique().sort_values()
    if dates.empty:
        return dates

    dates_naive = dates.tz_localize(None) if dates.tz is not None else dates
    cutoff = pd.Timestamp.now() if asof is None else pd.Timestamp(asof)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)
    cutoff_period = cutoff.to_period("M")
    periods = dates_naive.to_period("M")

    last_dates = [dates_naive[periods == period][-1]
                  for period in periods.unique() if period < cutoff_period]
    return pd.DatetimeIndex(last_dates)


def build_transition_and_entropy(class_df: pd.DataFrame, k: int = K):
    """Transizioni a un passo (mese t -> mese t+1), pooled su tutti i titoli
    e tutte le date. Ritorna: mu (joint pooled), P (transizione), sigma (entropy production, Eq.6)."""
    mu = np.full((k, k), LAPLACE_EPS)
    dates = class_df.index
    for i in range(len(dates) - 1):
        a = class_df.iloc[i]
        b = class_df.iloc[i + 1]
        both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        for _, row in both.iterrows():
            ai, bi = int(row["a"]) - 1, int(row["b"]) - 1
            mu[ai, bi] += 1
    mu = mu / mu.sum()
    P = mu / mu.sum(axis=1, keepdims=True)
    sigma = np.sum(mu * np.log(mu / mu.T))
    return mu, P, sigma


def calibration_curve(class_df: pd.DataFrame, k: int = K):
    """Per ogni quintile di partenza a: media del quintile REALIZZATO al mese
    successivo (Fig. 3 del paper) e media PREVISTA dalla catena pooled (P)."""
    _, P, _ = build_transition_and_entropy(class_df, k)
    dates = class_df.index
    realized = {a: [] for a in range(1, k + 1)}
    for i in range(len(dates) - 1):
        a_s, b_s = class_df.iloc[i], class_df.iloc[i + 1]
        both = pd.concat([a_s, b_s], axis=1, keys=["a", "b"]).dropna()
        for a_val, grp in both.groupby("a"):
            realized[int(a_val)].extend(grp["b"].tolist())
    realized_mean = {a: np.mean(v) if v else np.nan for a, v in realized.items()}
    predicted_mean = {a: np.sum(P[a - 1] * np.arange(1, k + 1)) for a in range(1, k + 1)}
    return realized_mean, predicted_mean

