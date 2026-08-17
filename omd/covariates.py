"""
OMD-Portfolio -- Fase 4: covariate cross-sezionali
--------------------------------------------------------
Le covariate "di mercato" del paper (size, beta, illiquidita' Amihud,
momentum, volatilita' multi-finestra) piu' le due lette dalla matrice di
distanza (market loading, distance centrality) gia' costruite in Fase 2.

NOTA: il dataset non contiene market cap / shares outstanding, quindi
"size" e' approssimata con il volume in dollari trailing (proxy ragionevole:
i titoli piu' grandi tendono a scambiare piu' dollari, ma non e' lo stesso
identico segnale del paper -- deviazione dichiarata, come per ^COR1M).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def amihud_illiquidity(log_ret: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame,
                        window: int = 63) -> pd.DataFrame:
    """Media trailing di |rendimento| / volume_in_dollari. Alta = illiquido."""
    dollar_vol = (close * volume).replace(0, np.nan)
    daily_amihud = log_ret.abs() / dollar_vol
    return daily_amihud.rolling(window, min_periods=window).mean()


def size_proxy(close: pd.DataFrame, volume: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Log del volume in dollari medio trailing, proxy di "size" in assenza di market cap."""
    dollar_vol = (close * volume).replace(0, np.nan)
    avg_dollar_vol = dollar_vol.rolling(window, min_periods=window).mean()
    return np.log(avg_dollar_vol)


def trailing_beta(log_ret: pd.DataFrame, market_ret: pd.Series, window: int = 252) -> pd.DataFrame:
    """Beta trailing rispetto all'indice equal-weight interno, per rolling OLS univariata
    (cov(titolo, mercato) / var(mercato)), calcolata titolo per titolo."""
    mkt = market_ret.reindex(log_ret.index)
    out = pd.DataFrame(index=log_ret.index, columns=log_ret.columns, dtype=float)
    for col in log_ret.columns:
        joint = pd.concat([log_ret[col], mkt], axis=1).dropna()
        if joint.empty:
            continue
        roll_cov = joint.iloc[:, 0].rolling(window, min_periods=window).cov(joint.iloc[:, 1])
        roll_var = joint.iloc[:, 1].rolling(window, min_periods=window).var()
        beta = (roll_cov / roll_var).reindex(log_ret.index)
        out[col] = beta
    return out


def momentum_12_1(log_ret: pd.DataFrame, skip: int = 21, lookback: int = 252) -> pd.DataFrame:
    """Momentum "12 meno 1": rendimento cumulato sui 252 giorni trailing, escludendo
    l'ultimo mese (skip=21gg) per evitare l'effetto reversal a breve termine."""
    cum_long = log_ret.rolling(lookback, min_periods=lookback).sum()
    cum_short = log_ret.rolling(skip, min_periods=skip).sum()
    return cum_long - cum_short


def bucket_cross_sectional(feature: pd.DataFrame, active: pd.DataFrame, k: int = 5,
                            ascending_is_best: bool = True, min_names: int = 15) -> pd.DataFrame:
    """Bucketing decile/quintile generico (Eq. 9 del paper), riusabile per
    qualunque covariata. ascending_is_best=True vuol dire "valore piu' basso
    = bucket 1" (es. illiquidita', dove basso e' meglio); False il contrario."""
    out = pd.DataFrame(index=feature.index, columns=feature.columns, dtype=float)
    for dt in feature.index:
        row = feature.loc[dt].where(active.loc[dt]).dropna()
        if len(row) < min_names:
            continue
        ranks = row.rank(ascending=ascending_is_best, method="first")
        buckets = np.ceil(k * ranks / len(row)).clip(1, k)
        out.loc[dt, buckets.index] = buckets
    return out


