#!/usr/bin/env python3
"""Aggiorna il CSV giornaliero COR1M con l'ultima osservazione Yahoo.

Yahoo espone soltanto l'ultima seduta disponibile per ``^COR1M``. Lo script
usa la data della quotazione restituita, non la data del runner, e fa un upsert:
inserisce la riga se assente oppure aggiorna quella gia' presente. Unicorn
Hunter usa soltanto il Close; per le nuove righe Open/High/Low sono quindi
impostati allo stesso valore.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_SYMBOL = "^COR1M"
DEFAULT_CSV = "CBOE_1Month_Implied_Correlation_Historical_Data.csv"
VALUE_COLUMNS = ("Price", "Open", "High", "Low")


@dataclass(frozen=True)
class LatestQuote:
    session_date: date
    close: float


@dataclass(frozen=True)
class UpdateResult:
    changed: bool
    action: str
    session_date: date
    close: float


def business_day_lag(observation_date, asof_date=None) -> int:
    """Conta i soli lunedi-venerdi successivi all'ultima osservazione.

    I weekend non generano staleness. Il calendario non tenta di ricostruire le
    festivita' USA: la soglia del workflow tollera fino a due giorni feriali.
    """
    observation = pd.Timestamp(observation_date).normalize()
    asof = pd.Timestamp(asof_date or date.today()).normalize()
    if asof <= observation:
        return 0
    first_possible = observation + pd.offsets.BDay(1)
    if first_possible > asof:
        return 0
    return int(len(pd.bdate_range(first_possible, asof)))


def fetch_latest_quote(symbol: str = DEFAULT_SYMBOL) -> LatestQuote:
    """Legge l'ultima seduta COR1M disponibile tramite yfinance."""
    history = yf.Ticker(symbol).history(
        period="5d",
        interval="1d",
        auto_adjust=False,
        actions=False,
    )
    if history is None or history.empty or "Close" not in history:
        raise RuntimeError(f"Yahoo non ha restituito dati utilizzabili per {symbol}.")

    closes = pd.to_numeric(history["Close"], errors="coerce").dropna()
    if closes.empty:
        raise RuntimeError(f"Yahoo non ha restituito un Close valido per {symbol}.")

    timestamp = pd.Timestamp(closes.index[-1])
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("America/New_York").tz_localize(None)
    session_date = timestamp.date()
    close = round(float(closes.iloc[-1]), 2)

    if session_date.weekday() >= 5:
        raise RuntimeError(
            f"Yahoo ha restituito una data non feriale per {symbol}: {session_date}."
        )
    if not np.isfinite(close) or not -100.0 <= close <= 100.0:
        raise RuntimeError(f"Valore COR1M non plausibile: {close!r}.")
    return LatestQuote(session_date=session_date, close=close)


def _change_percent(close: float, previous_close: float | None) -> str:
    if previous_close is None or not np.isfinite(previous_close) or previous_close == 0:
        return ""
    return f"{((close / previous_close) - 1.0) * 100.0:.2f}%"


def upsert_quote(
    csv_path: str | Path,
    quote: LatestQuote,
    *,
    write: bool = True,
) -> UpdateResult:
    """Inserisce o aggiorna una seduta, mantenendo schema e ordine cronologico."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV COR1M non trovato: {path}")

    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    required = {"Date", *VALUE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV COR1M incompleto; colonne mancanti: {sorted(missing)}")
    for optional in ("Vol.", "Change %"):
        if optional not in frame:
            frame[optional] = ""

    frame["_parsed_date"] = pd.to_datetime(
        frame["Date"], format="%m/%d/%Y", errors="raise"
    )
    date_order = frame["_parsed_date"].dropna()
    descending_order = bool(
        len(date_order) >= 2 and date_order.iloc[0] > date_order.iloc[-1]
    )
    duplicate_mask = frame["_parsed_date"].duplicated(keep="last")
    changed = bool(duplicate_mask.any())
    if changed:
        frame = frame.loc[~duplicate_mask].copy()

    target = pd.Timestamp(quote.session_date)
    formatted_close = f"{quote.close:.2f}"
    target_mask = frame["_parsed_date"].eq(target)

    if target_mask.any():
        row_index = frame.index[target_mask][-1]
        action = "updated"
        for column in VALUE_COLUMNS:
            old_value = pd.to_numeric(
                pd.Series([frame.at[row_index, column]]), errors="coerce"
            ).iloc[0]
            if pd.isna(old_value) or round(float(old_value), 2) != quote.close:
                changed = True
            frame.at[row_index, column] = formatted_close
    else:
        action = "inserted"
        changed = True
        new_row = {column: "" for column in frame.columns}
        new_row.update({
            "Date": quote.session_date.strftime("%m/%d/%Y"),
            "Price": formatted_close,
            "Open": formatted_close,
            "High": formatted_close,
            "Low": formatted_close,
            "_parsed_date": target,
        })
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)

    # Conserva l'ordinamento gia' usato dal file sorgente. I CSV scaricati
    # manualmente sono spesso newest-first; invertirli a ogni upsert produrrebbe
    # un diff enorme e un redeploy inutile pur senza cambiare i dati.
    frame = frame.sort_values(
        "_parsed_date", ascending=not descending_order
    ).reset_index(drop=True)
    target_index = frame.index[frame["_parsed_date"].eq(target)][-1]
    previous_rows = frame.loc[
        frame["_parsed_date"] < target, ["_parsed_date", "Price"]
    ].sort_values("_parsed_date")
    previous_prices = pd.to_numeric(previous_rows["Price"], errors="coerce").dropna()
    previous_close = float(previous_prices.iloc[-1]) if not previous_prices.empty else None
    change_text = _change_percent(quote.close, previous_close)
    if frame.at[target_index, "Change %"] != change_text:
        changed = True
        frame.at[target_index, "Change %"] = change_text

    if not changed:
        action = "unchanged"
    elif write:
        output = frame.drop(columns="_parsed_date")
        output.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )

    return UpdateResult(
        changed=changed,
        action=action,
        session_date=quote.session_date,
        close=quote.close,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV, help="CSV giornaliero da aggiornare")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Ticker Yahoo")
    parser.add_argument(
        "--max-business-day-lag",
        type=int,
        default=2,
        help="Massimo ritardo ammesso, contando solo lunedi-venerdi",
    )
    parser.add_argument("--dry-run", action="store_true", help="Non scrive il CSV")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    quote = fetch_latest_quote(args.symbol)
    lag = business_day_lag(quote.session_date)
    if lag > args.max_business_day_lag:
        raise RuntimeError(
            f"Quotazione {args.symbol} ferma al {quote.session_date}: "
            f"ritardo di {lag} giorni feriali."
        )
    result = upsert_quote(args.csv, quote, write=not args.dry_run)
    mode = "dry-run" if args.dry_run else "write"
    print(
        f"COR1M {result.session_date} close={result.close:.2f} "
        f"action={result.action} mode={mode} business_lag={lag}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
