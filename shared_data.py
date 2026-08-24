"""Loader resilienti per i dataset condivisi IOVIQUANT."""

from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Callable, Sequence
from urllib.request import Request, urlopen

import pandas as pd


COR1M_REMOTE_FILES = (
    "https://raw.githubusercontent.com/takeshiromanov/IOVIQUANT_DATA/main/data/cor1m/daily.csv",
    "https://raw.githubusercontent.com/takeshiromanov/IOVIQUANT_DATA/main/data/cor1m/weekly.csv",
)
COR1M_LOCAL_FILES = (
    "CBOE_1Month_Implied_Correlation_Historical_Data.csv",
    "CBOE_1Month_Implied_Correlation_Historical_Data2.csv",
)
REQUIRED_COR1M_COLUMNS = {"Date", "Price", "Open", "High", "Low"}


def business_day_lag(observation_date, asof_date=None) -> int:
    """Conta i soli lunedi-venerdi successivi all'ultima osservazione."""
    observation = pd.Timestamp(observation_date).normalize()
    asof = pd.Timestamp(asof_date or date.today()).normalize()
    if asof <= observation:
        return 0
    first_possible = observation + pd.offsets.BDay(1)
    if first_possible > asof:
        return 0
    return int(len(pd.bdate_range(first_possible, asof)))


def read_csv_source(source: str | Path) -> pd.DataFrame:
    """Legge un CSV locale o HTTP con timeout esplicito."""
    if isinstance(source, str) and source.startswith(("https://", "http://")):
        request = Request(source, headers={"User-Agent": "IOVIQUANT/1.0"})
        with urlopen(request, timeout=15) as response:
            return pd.read_csv(BytesIO(response.read()), encoding="utf-8-sig")
    return pd.read_csv(source, encoding="utf-8-sig")


def _parse_cor1m_pair(
    sources: Sequence[str | Path],
    reader: Callable[[str | Path], pd.DataFrame],
) -> pd.DataFrame:
    if len(sources) != 2:
        raise ValueError("COR1M richiede esattamente i file daily e weekly.")
    daily, weekly = (reader(source).copy() for source in sources)
    for frame in (daily, weekly):
        absent = REQUIRED_COR1M_COLUMNS.difference(frame.columns)
        if absent:
            raise ValueError(
                f"CSV COR1M non valido; colonne mancanti: {sorted(absent)}"
            )
        frame["Date"] = pd.to_datetime(
            frame["Date"], format="%m/%d/%Y", errors="raise"
        )
        for column in ("Price", "Open", "High", "Low"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # Gli export storici contenevano talvolta una copia del venerdi datata nel
    # weekend. La serie weekly, invece, usa legittimamente date di domenica.
    daily = daily.loc[daily["Date"].dt.weekday.lt(5)]
    daily = daily.set_index("Date").sort_index()
    weekly = weekly.set_index("Date").sort_index()
    if daily.empty:
        raise ValueError("Il CSV COR1M daily non contiene righe feriali valide.")

    ohlc = ["Open", "High", "Low", "Price"]
    combined = pd.concat([
        weekly.loc[weekly.index < daily.index.min(), ohlc],
        daily[ohlc],
    ]).sort_index()
    combined = combined.loc[~combined.index.duplicated(keep="last")]
    combined = combined.dropna(subset=ohlc)
    if combined.empty:
        raise ValueError("I CSV COR1M non contengono righe OHLC utilizzabili.")
    return combined.rename(columns={"Price": "Close"})


def load_cor1m_series(
    base_dir: str | Path,
    *,
    remote_files: Sequence[str] = COR1M_REMOTE_FILES,
    local_files: Sequence[str] = COR1M_LOCAL_FILES,
    reader: Callable[[str | Path], pd.DataFrame] = read_csv_source,
) -> tuple[pd.DataFrame, str]:
    """Carica la coppia remota e ripiega atomicamente su quella locale."""
    try:
        return _parse_cor1m_pair(remote_files, reader), "IOVIQUANT_DATA"
    except Exception as remote_error:
        local_paths = tuple(Path(base_dir) / filename for filename in local_files)
        missing = [path.name for path in local_paths if not path.is_file()]
        if missing:
            raise RuntimeError(
                "COR1M centrale non disponibile e fallback locale incompleto: "
                + ", ".join(missing)
            ) from remote_error
        try:
            return _parse_cor1m_pair(local_paths, reader), "fallback locale"
        except Exception as local_error:
            raise RuntimeError(
                "Impossibile caricare COR1M sia dalla fonte centrale sia dal fallback locale."
            ) from local_error
