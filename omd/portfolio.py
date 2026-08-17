"""
omd/portfolio.py -- score, sleeve, no-trade band, tilt di diversificazione
------------------------------------------------------------------------------
Include il tilt gamma (Sezione 4 del paper) MAI implementato nel resto della
conversazione: selezione greedy che bilancia punteggio e distanza residua
(componente di mercato rimossa) tra i titoli scelti per la sleeve long-only.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from omd.generator import fit_generator_pairs, transition_matrix_for
from omd.distance_matrix import distance_matrix_from_corr

K_BOOK = 5
TOL = 0.08
LOOKBACK_REGIME = 63
GAMMA = 0.5  # peso di diversificazione (paper: punto ottimale single-peaked)


def latest_complete_signal_date(bucketed_daily: dict, class_daily,
                                asof: pd.Timestamp, min_names: int = 1):
    """Ultima data <= ``asof`` con classi e covariate complete.

    I dati live possono contenere un'ultima riga ancora parziale (per esempio
    volumi non consolidati). In quel caso le covariate bucketizzate sono NaN e
    uno snapshot calcolato meccanicamente sull'ultima data risulta vuoto.
    ``class_daily`` puo' essere un singolo DataFrame o una sequenza di
    DataFrame quando piu' segnali devono essere disponibili nello stesso
    giorno.
    """
    class_frames = ([class_daily] if isinstance(class_daily, pd.DataFrame)
                    else list(class_daily))
    if not class_frames:
        return None

    candidate_dates = class_frames[0].index[class_frames[0].index <= asof]
    for dt in reversed(candidate_dates):
        names = set(class_frames[0].loc[dt].dropna().index)
        for frame in class_frames[1:]:
            if dt not in frame.index:
                names.clear()
                break
            names.intersection_update(frame.loc[dt].dropna().index)
        for feature in bucketed_daily.values():
            if not names or dt not in feature.index:
                names.clear()
                break
            names.intersection_update(feature.loc[dt].dropna().index)
        if len(names) >= min_names:
            return pd.Timestamp(dt)
    return None


def market_regime(market_ret: pd.Series, asof: pd.Timestamp, lookback: int = LOOKBACK_REGIME) -> str:
    hist = market_ret.loc[:asof].tail(lookback)
    cum = (1 + hist).prod() - 1
    return "rising" if cum >= 0 else "falling"


def lambda_theta_for_regime(regime: str):
    if regime == "rising":
        return 0.0, 1.0
    return 0.75, 0.4


def compute_pi_best(models: dict, bucketed_daily: dict, class_daily: pd.DataFrame,
                     asof: pd.Timestamp, k: int = 5, delta: int = 21) -> pd.Series:
    """pi_i = P_{a_i,1}(x_i): probabilita' di essere nel quintile migliore
    il mese prossimo, dato lo stato corrente e le covariate."""
    real_dt = latest_complete_signal_date(bucketed_daily, class_daily, asof)
    if real_dt is None:
        return pd.Series(dtype=float)
    a_today = class_daily.loc[real_dt].dropna()

    phi_today = {}
    for name in bucketed_daily:
        phi_today[name] = bucketed_daily[name].loc[real_dt]
    phi_df = pd.DataFrame(phi_today)

    out = {}
    for tkr in a_today.index:
        if tkr not in phi_df.index or phi_df.loc[tkr].isna().any():
            continue
        a_val = int(a_today[tkr])
        phi_x = phi_df.loc[tkr].values.astype(float)
        P = transition_matrix_for(models, phi_x, k=k, delta=delta)
        out[tkr] = P[a_val - 1, 0]
    return pd.Series(out)


def apply_no_trade_band(new_score: pd.Series, prev_holdings: set, n_select: int,
                         ascending: bool = False, tol: float = TOL) -> set:
    ranked = new_score.sort_values(ascending=ascending)
    all_names = list(ranked.index)
    held = [n for n in prev_holdings if n in new_score.index]
    held_sorted = sorted(held, key=lambda n: new_score[n], reverse=not ascending)

    result = list(held_sorted)
    for cand in all_names:
        if len(result) >= n_select:
            break
        if cand not in result:
            result.append(cand)

    changed = True
    while changed:
        changed = False
        held_in_result = [n for n in result if n in prev_holdings]
        if not held_in_result:
            break
        worst_held = (min if not ascending else max)(held_in_result, key=lambda n: new_score[n])
        outsiders = [n for n in all_names if n not in result]
        if not outsiders:
            break
        best_outsider = outsiders[0]
        beats = (new_score[best_outsider] > new_score[worst_held] * (1 + tol)) if not ascending else \
                (new_score[best_outsider] < new_score[worst_held] * (1 - tol))
        if beats:
            result.remove(worst_held)
            result.append(best_outsider)
            changed = True
    return set(result[:n_select])


def residual_distance_matrix(returns: pd.DataFrame, market_ret: pd.Series, active: pd.DataFrame,
                              asof: pd.Timestamp, window: int = 252, min_names: int = 15):
    """NUOVO: matrice di distanza sui rendimenti RESIDUI (componente di
    mercato single-factor rimossa), base per il tilt di diversificazione
    gamma -- non usata altrove nella conversazione."""
    hist = returns.loc[:asof].tail(window)
    if hist.shape[0] < window:
        return None
    active_today = active.loc[asof]
    cols = active_today[active_today].index
    hist = hist[cols].ffill(limit=2).dropna(axis=1, how="any")
    if hist.shape[1] < min_names:
        return None
    mkt = market_ret.reindex(hist.index)
    var_mkt = mkt.var()
    resid = pd.DataFrame(index=hist.index, columns=hist.columns, dtype=float)
    for col in hist.columns:
        beta = hist[col].cov(mkt) / var_mkt if var_mkt > 0 else 0.0
        resid[col] = hist[col] - beta * mkt
    C = resid.corr()
    return distance_matrix_from_corr(C)


def diversify_selection(score: pd.Series, dist_matrix: pd.DataFrame, n_select: int,
                         gamma: float = GAMMA) -> list:
    """NUOVO: selezione greedy che bilancia punteggio (score) e distanza
    residua media dai titoli gia' scelti. gamma=0 -> solo score (comportamento
    originale); gamma=1 -> solo diversificazione."""
    if dist_matrix is None or gamma <= 0:
        return score.sort_values(ascending=False).head(n_select).index.tolist()

    candidates = [c for c in score.sort_values(ascending=False).index if c in dist_matrix.index]
    if not candidates:
        return score.sort_values(ascending=False).head(n_select).index.tolist()

    score_norm = (score - score.min()) / (score.max() - score.min() + 1e-9)
    selected = [candidates[0]]
    remaining = candidates[1:]

    while len(selected) < n_select and remaining:
        best_val, best_cand = -np.inf, None
        for cand in remaining:
            avg_dist = dist_matrix.loc[cand, selected].mean() / np.pi  # normalizzato [0,1]
            val = (1 - gamma) * score_norm.get(cand, 0) + gamma * avg_dist
            if val > best_val:
                best_val, best_cand = val, cand
        selected.append(best_cand)
        remaining.remove(best_cand)

    return selected


def build_score_and_sleeves(pi_R: pd.Series, pi_V: pd.Series, lam: float, k_book: int = K_BOOK,
                             dist_matrix: pd.DataFrame = None, gamma: float = 0.0):
    common = pi_R.index.intersection(pi_V.index)
    pi_R, pi_V = pi_R.loc[common], pi_V.loc[common]

    score_longonly = (1 - lam) * pi_R + lam * pi_V
    score_longshort = pi_R

    if gamma > 0 and dist_matrix is not None:
        longonly_names = diversify_selection(score_longonly, dist_matrix, 2 * k_book, gamma=gamma)
    else:
        longonly_names = score_longonly.sort_values(ascending=False).head(2 * k_book).index.tolist()

    long_names = score_longshort.sort_values(ascending=False).head(k_book).index.tolist()
    short_names = score_longshort.sort_values(ascending=True).head(k_book).index.tolist()

    return {
        "score_longonly": score_longonly.sort_values(ascending=False),
        "score_longshort": score_longshort.sort_values(ascending=False),
        "longonly_holdings": set(longonly_names),
        "long_holdings": set(long_names),
        "short_holdings": set(short_names),
    }
