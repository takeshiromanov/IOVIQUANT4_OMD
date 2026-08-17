"""Confronto riproducibile fra Unicorn Hunter puro e varianti OMD."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import ioviquant_engine as eng


@dataclass
class VariantResult:
    history: pd.DataFrame
    events: pd.DataFrame
    all_trades: pd.DataFrame
    closed_trades: pd.DataFrame
    equity_with_benchmark: pd.DataFrame
    kpi: dict[str, float]


VARIANTS = (
    ("UH puro", False, False),
    ("OMD filtro Buy", True, False),
    ("OMD filtro Buy + Sell forzato", True, True),
)


def run_three_way_comparison(
    signals: dict[str, pd.DataFrame],
    universe: list[str] | tuple[str, ...],
    initial_capital: float,
    base_size: float,
    params: dict,
    benchmark_ohlc: pd.DataFrame,
    start_date,
    end_date,
    omd_schedule: pd.DataFrame,
) -> tuple[dict[str, VariantResult], pd.DataFrame]:
    """Esegue le tre configurazioni a parita' di dati, costi e parametri UH."""
    if omd_schedule is None or omd_schedule.empty:
        raise ValueError("Il confronto OMD richiede un calendario cluster non vuoto.")

    benchmark_equity, _, _ = eng.build_open_entry_benchmark_equity(
        benchmark_ohlc, initial_capital, start_date, end_date
    )
    results: dict[str, VariantResult] = {}
    rows: list[dict[str, object]] = []

    for name, use_filter, force_sell in VARIANTS:
        variant_params = params.copy()
        variant_params.update({
            "use_omd_entry_filter": use_filter,
            "use_omd_forced_sell_exit": force_sell,
        })
        history, events, all_trades, closed_trades = eng.run_simulation(
            signals,
            universe,
            initial_capital,
            base_size,
            variant_params,
            start_date=start_date,
            end_date=end_date,
            omd_schedule=omd_schedule if use_filter else None,
        )
        aligned = history.join(benchmark_equity).ffill()
        kpi = eng.calc_metrics_extended(
            aligned["Equity"],
            aligned["VWCE_Eq"],
            closed_trades,
            events=events,
            initial_equity=initial_capital,
        )
        results[name] = VariantResult(
            history=history,
            events=events,
            all_trades=all_trades,
            closed_trades=closed_trades,
            equity_with_benchmark=aligned,
            kpi=kpi,
        )
        rows.append({
            "Variante": name,
            "CAGR": kpi["CAGR_strategia"],
            "Alpha_vs_VWCE": kpi["Excess_CAGR_net_EUR"],
            "MaxDD": kpi["MaxDD_strategia"],
            "Sharpe": kpi["Sharpe_strategia"],
            "Unicorn_Rate_%": kpi["Unicorn_Rate_%"],
            "Entry_MFE_>=50%_%": kpi["Entry_MFE_>=50%_%"],
            "Unicorn_Conversion_%": kpi["Unicorn_Conversion_%"],
            "Trade_Chiusi": int(kpi["N_Trade_Chiusi"]),
            "Win_Rate_%": kpi["Win_Rate_%"],
            "Fast_Loss_3S_Rate_%": kpi["Fast_Loss_3S_Rate_%"],
            "Costo_Annuo_%": kpi["Annualized_Cost_Drag_%"],
            "Turnover_Annuo_x": kpi["Annual_Turnover_x"],
            "Capitale_Finale_EUR": float(aligned["Equity"].iloc[-1]),
        })

    return results, pd.DataFrame(rows).set_index("Variante")


def selected_variant_name(use_filter: bool, force_sell: bool) -> str:
    if not use_filter:
        return "UH puro"
    if force_sell:
        return "OMD filtro Buy + Sell forzato"
    return "OMD filtro Buy"
