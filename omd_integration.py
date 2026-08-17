"""Integrazione causale OMD -> universo mensile di Unicorn Hunter.

OMD genera un cluster Buy (top 2*K della sleeve long-only) e un cluster Sell
(bottom K della probabilita' di rendimento). I segnali di un mese concluso
diventano operativi soltanto alla prima seduta successiva.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from omd.covariates import (
    amihud_illiquidity,
    momentum_12_1,
    size_proxy,
    trailing_beta,
)
from omd.distance_matrix import (
    distance_centrality,
    distance_matrix_from_corr,
    leading_eigenvector_centrality,
    trailing_correlation,
)
from omd.generator import (
    build_daily_panel,
    compute_bucketed_features,
    fit_generator_pairs,
)
from omd.portfolio import (
    apply_no_trade_band,
    build_score_and_sleeves,
    compute_pi_best,
    lambda_theta_for_regime,
    latest_complete_signal_date,
    market_regime,
)
from omd.rank_chains import (
    completed_monthly_dates,
    cross_sectional_quintile,
    rolling_stat,
)
from omd.transfer_entropy import market_bucket_expanding, leadlag_score_for_name


DEFAULT_K_BOOK = 30
DEFAULT_REFIT_MONTHS = 12
DEFAULT_MIN_TRAIN_MONTHS = 36
DEFAULT_NO_TRADE_TOL = 0.08
DEFAULT_MIN_HISTORY_DAYS = 252


@dataclass(frozen=True)
class OMDSettings:
    k_book: int = DEFAULT_K_BOOK
    refit_months: int = DEFAULT_REFIT_MONTHS
    min_train_months: int = DEFAULT_MIN_TRAIN_MONTHS
    no_trade_tol: float = DEFAULT_NO_TRADE_TOL
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS
    gamma: float = 0.0


def _wide_field(
    data_dict: dict[str, pd.DataFrame],
    universe: list[str] | tuple[str, ...],
    field: str,
) -> pd.DataFrame:
    columns = {
        ticker: data_dict[ticker][field]
        for ticker in universe
        if ticker in data_dict and field in data_dict[ticker].columns
    }
    if not columns:
        return pd.DataFrame()
    frame = pd.DataFrame(columns).sort_index()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.loc[~frame.index.duplicated(keep="last")]


def build_omd_inputs(
    data_dict: dict[str, pd.DataFrame],
    universe: list[str] | tuple[str, ...],
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> dict[str, object]:
    """Costruisce matrici OMD usando gli stessi prezzi USD di Unicorn Hunter."""
    close = _wide_field(data_dict, universe, "Close")
    volume = _wide_field(data_dict, universe, "Volume")
    common = close.columns.intersection(volume.columns)
    close = close[common]
    volume = volume[common].reindex(close.index)
    if close.empty or len(common) < 2:
        raise ValueError("OMD richiede prezzi e volumi per almeno due ticker.")

    log_ret = np.log(close / close.shift(1)).ffill(limit=2)
    has_price = close.notna()
    active = has_price & (has_price.cumsum() >= int(min_history_days))
    valid_days = active.sum(axis=1) > 0
    return {
        "close": close.loc[valid_days],
        "volume": volume.loc[valid_days],
        "ret": log_ret.loc[valid_days],
        "active": active.loc[valid_days],
    }


def build_omd_features(inputs: dict[str, object]) -> dict[str, object]:
    """Replica la pipeline di feature dell'app OMD senza dipendenze Streamlit."""
    close = inputs["close"]
    volume = inputs["volume"]
    ret = inputs["ret"]
    active = inputs["active"]
    market_ret = ret.where(active).mean(axis=1).rename("market_ret")

    beta = trailing_beta(ret, market_ret, window=252)
    illiquidity = amihud_illiquidity(ret, close, volume, window=63)
    size = size_proxy(close, volume, window=63)
    momentum = momentum_12_1(ret)

    market_loading_rows: dict[pd.Timestamp, pd.Series] = {}
    distance_centrality_rows: dict[pd.Timestamp, pd.Series] = {}
    for nominal_month_end in ret.resample("ME").last().index:
        real_dates = ret.index[ret.index <= nominal_month_end]
        if real_dates.empty:
            continue
        real_date = real_dates[-1]
        corr = trailing_correlation(ret, active, real_date, window=252)
        if corr is None:
            continue
        distance = distance_matrix_from_corr(corr)
        market_loading_rows[real_date] = leading_eigenvector_centrality(corr)
        distance_centrality_rows[real_date] = distance_centrality(distance)

    market_loading = (
        pd.DataFrame(market_loading_rows).T.reindex(ret.index).ffill()
        if market_loading_rows else pd.DataFrame(index=ret.index, columns=ret.columns)
    )
    dist_centrality = (
        pd.DataFrame(distance_centrality_rows).T.reindex(ret.index).ffill()
        if distance_centrality_rows else pd.DataFrame(index=ret.index, columns=ret.columns)
    )

    market_bucket = market_bucket_expanding(market_ret)
    class_r1 = cross_sectional_quintile(
        rolling_stat(ret, 21, "mean"), active, ascending_is_best=False
    )
    leadlag = pd.DataFrame({
        ticker: leadlag_score_for_name(
            class_r1[ticker], market_bucket, window=252, step=10
        )
        for ticker in ret.columns
    })

    feature_dict = {
        "beta": beta,
        "illiquidity": illiquidity,
        "size": size,
        "momentum": momentum,
        "mkt_loading": market_loading,
        "dist_centrality": dist_centrality,
        "leadlag_te": leadlag,
    }
    bucketed = compute_bucketed_features(feature_dict, active, ret.index)
    class_r6 = cross_sectional_quintile(
        rolling_stat(ret, 126, "mean"), active, ascending_is_best=False
    )
    class_v = cross_sectional_quintile(
        rolling_stat(ret, 63, "vol"), active, ascending_is_best=True
    )
    panel_r = build_daily_panel(class_r6, bucketed, active)
    panel_v = build_daily_panel(class_v, bucketed, active)
    return {
        **inputs,
        "market_ret": market_ret,
        "bucketed": bucketed,
        "class_r6": class_r6,
        "class_v": class_v,
        "panel_r": panel_r,
        "panel_v": panel_v,
    }


def generate_monthly_clusters(
    prepared: dict[str, object],
    asof: str | pd.Timestamp,
    settings: OMDSettings = OMDSettings(),
) -> pd.DataFrame:
    """Genera cluster mensili walk-forward con refit OMD invariato a 12 mesi."""
    if settings.k_book < 1:
        raise ValueError("OMD k_book deve essere >= 1.")
    if settings.refit_months < 1 or settings.min_train_months < 1:
        raise ValueError("Le finestre OMD devono essere positive.")
    if not 0 <= settings.no_trade_tol < 1:
        raise ValueError("La no-trade band OMD deve essere in [0, 1).")
    if settings.gamma != 0:
        raise ValueError(
            "L'integrazione Unicorn usa gamma=0, che e' il default attuale OMD."
        )

    ret = prepared["ret"]
    active = prepared["active"]
    market_ret = prepared["market_ret"]
    bucketed = prepared["bucketed"]
    class_r6 = prepared["class_r6"]
    class_v = prepared["class_v"]
    panel_r = prepared["panel_r"]
    panel_v = prepared["panel_v"]
    monthly_dates = completed_monthly_dates(ret.index, asof=pd.Timestamp(asof))

    models_r = models_v = None
    last_refit = -10**9
    previous_buy: set[str] = set()
    previous_sell: set[str] = set()
    rows: list[dict[str, object]] = []
    minimum_names = 3 * settings.k_book

    for month_index in range(settings.min_train_months, len(monthly_dates)):
        signal_date = pd.Timestamp(monthly_dates[month_index])
        next_sessions = ret.index[ret.index > signal_date]
        if next_sessions.empty:
            continue
        effective_date = pd.Timestamp(next_sessions[0])

        need_refit = (
            models_r is None
            or (month_index - last_refit) >= settings.refit_months
        )
        if need_refit:
            models_r = fit_generator_pairs(panel_r[panel_r["date"] < signal_date])
            models_v = fit_generator_pairs(panel_v[panel_v["date"] < signal_date])
            last_refit = month_index

        model_asof = latest_complete_signal_date(
            bucketed,
            [class_r6, class_v],
            signal_date,
            min_names=minimum_names,
        )
        if model_asof is None:
            continue
        pi_r = compute_pi_best(models_r, bucketed, class_r6, model_asof, delta=21)
        pi_v = compute_pi_best(models_v, bucketed, class_v, model_asof, delta=21)
        common = pi_r.index.intersection(pi_v.index)
        if len(common) < minimum_names:
            continue

        regime = market_regime(market_ret, model_asof)
        lam, theta = lambda_theta_for_regime(regime)
        sleeves = build_score_and_sleeves(
            pi_r.loc[common],
            pi_v.loc[common],
            lam=lam,
            k_book=settings.k_book,
            gamma=0.0,
        )

        sell = apply_no_trade_band(
            sleeves["score_longshort"],
            previous_sell,
            settings.k_book,
            ascending=True,
            tol=settings.no_trade_tol,
        )
        # Sell prevale sempre. La sleeve Buy viene riempita con i successivi
        # migliori candidati per conservare 2*K nomi quando la cross-section
        # dispone di almeno 3*K segnali completi.
        eligible_buy_score = sleeves["score_longonly"].drop(
            labels=list(sell), errors="ignore"
        )
        buy = apply_no_trade_band(
            eligible_buy_score,
            previous_buy - sell,
            2 * settings.k_book,
            ascending=False,
            tol=settings.no_trade_tol,
        )
        buy -= sell
        if len(buy) != 2 * settings.k_book or len(sell) != settings.k_book:
            continue

        rows.append({
            "Signal_Date": signal_date,
            "Model_AsOf": pd.Timestamp(model_asof),
            "Effective_Date": effective_date,
            "Buy_Tickers": tuple(sorted(buy)),
            "Sell_Tickers": tuple(sorted(sell)),
            "Buy_Count": len(buy),
            "Sell_Count": len(sell),
            "Regime": regime,
            "Lambda": float(lam),
            "Theta": float(theta),
            "Refit": bool(need_refit),
            "Common_Signals": int(len(common)),
        })
        previous_buy, previous_sell = set(buy), set(sell)

    return pd.DataFrame(rows)


def cluster_state_for_date(
    schedule: pd.DataFrame,
    date: str | pd.Timestamp,
) -> tuple[set[str], set[str]]:
    """Restituisce l'ultimo cluster gia' efficace alla data richiesta."""
    if schedule is None or schedule.empty:
        return set(), set()
    cutoff = pd.Timestamp(date)
    eligible = schedule[pd.to_datetime(schedule["Effective_Date"]) <= cutoff]
    if eligible.empty:
        return set(), set()
    row = eligible.sort_values("Effective_Date").iloc[-1]
    sell = set(row["Sell_Tickers"])
    buy = set(row["Buy_Tickers"]) - sell
    return buy, sell

