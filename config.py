"""Configurazione operativa dell'app IOVIQUANT.

I 60 titoli core restano congelati in ``universe.csv``. L'universo espanso
viene invece letto da ``xetra_raw_deduplicated.csv`` e non contamina il core.
"""

from __future__ import annotations

import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
UNIVERSE_FILE = BASE_DIR / "universe.csv"
EXPANDED_UNIVERSE_FILE = BASE_DIR / "xetra_raw_deduplicated.csv"


def load_universe(
    path: str | Path = UNIVERSE_FILE,
    groups: set[str] | None = None,
) -> list[str]:
    """Carica e valida i ticker abilitati, preservando l'ordine del CSV."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"File universo non trovato: {source}. "
            "Mantieni universe.csv nella stessa cartella di app.py."
        )

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "ticker" not in reader.fieldnames:
            raise ValueError(f"{source.name} deve contenere una colonna 'ticker'.")

        tickers: list[str] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            ticker = (row.get("ticker") or "").strip().upper()
            enabled = (row.get("enabled") or "true").strip().lower()
            group = (row.get("group") or "").strip().lower()
            if not ticker or ticker.startswith("#") or enabled in {"0", "false", "no", "n"}:
                continue
            if groups is not None and group not in groups:
                continue
            if ticker in seen:
                continue
            if any(char.isspace() for char in ticker):
                raise ValueError(
                    f"Ticker non valido alla riga {row_number} di {source.name}: {ticker!r}."
                )
            seen.add(ticker)
            tickers.append(ticker)

    if not tickers:
        raise ValueError(f"Nessun ticker abilitato in {source.name}.")
    return tickers


def load_universe_metadata(path: str | Path = UNIVERSE_FILE) -> dict[str, dict[str, str]]:
    """Restituisce i metadati delle sole righe abilitate, indicizzati per ticker."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        metadata: dict[str, dict[str, str]] = {}
        for row in reader:
            ticker = (row.get("ticker") or "").strip().upper()
            enabled = (row.get("enabled") or "true").strip().lower()
            if not ticker or ticker.startswith("#") or enabled in {"0", "false", "no", "n"}:
                continue
            metadata[ticker] = {
                key: (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
    return metadata


def _xetra_ticker_column(fieldnames: list[str] | None, source: Path) -> str:
    """Accetta sia ``ticker`` sia il nome ``Tickler`` presente nel CSV fonte."""
    normalized = {
        str(field).strip().lower(): str(field)
        for field in (fieldnames or [])
        if field is not None
    }
    for candidate in ("ticker", "tickler"):
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        f"{source.name} deve contenere una colonna 'ticker' o 'Tickler'."
    )


def infer_yahoo_currency(ticker: str) -> str:
    """Deduce la valuta del listing Yahoo dai suffissi presenti nel CSV Xetra."""
    suffix = ticker.rsplit(".", 1)[1] if "." in ticker else ""
    suffix_currency = {
        "AS": "EUR",
        "BR": "EUR",
        "DE": "EUR",
        "F": "EUR",
        "HE": "EUR",
        "HM": "EUR",
        "IR": "EUR",
        "MC": "EUR",
        "MI": "EUR",
        "PA": "EUR",
        "SG": "EUR",
        "VI": "EUR",
        "L": "GBP",
        "SW": "CHF",
        "CO": "DKK",
        "ST": "SEK",
        "TO": "CAD",
        "HK": "HKD",
        # I due simboli .IL del file sono International Order Book di Londra,
        # quotati in USD; non sono listing israeliani in ILS.
        "IL": "USD",
    }
    return suffix_currency.get(suffix, "USD")


def load_expanded_universe(
    path: str | Path = EXPANDED_UNIVERSE_FILE,
) -> list[str]:
    """Carica esattamente i ticker del CSV Xetra, preservandone l'ordine."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"File universo espanso non trovato: {source}. "
            "Mantienilo nella stessa cartella dell'app."
        )

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        ticker_column = _xetra_ticker_column(reader.fieldnames, source)
        tickers: list[str] = []
        seen: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            ticker = (row.get(ticker_column) or "").strip().upper()
            if not ticker or ticker.startswith("#"):
                continue
            if any(char.isspace() for char in ticker):
                raise ValueError(
                    f"Ticker non valido alla riga {row_number} di {source.name}: {ticker!r}."
                )
            if ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)

    if not tickers:
        raise ValueError(f"Nessun ticker utilizzabile in {source.name}.")
    return tickers


def load_expanded_metadata(
    path: str | Path = EXPANDED_UNIVERSE_FILE,
) -> dict[str, dict[str, str]]:
    """Restituisce i metadati Xetra indicizzati per ticker Yahoo Finance."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        ticker_column = _xetra_ticker_column(reader.fieldnames, source)
        metadata: dict[str, dict[str, str]] = {}
        for row in reader:
            ticker = (row.get(ticker_column) or "").strip().upper()
            if not ticker or ticker.startswith("#") or ticker in metadata:
                continue
            metadata[ticker] = {
                key: (value or "").strip()
                for key, value in row.items()
                if key is not None
            }
            metadata[ticker]["ticker"] = ticker
            metadata[ticker]["currency"] = infer_yahoo_currency(ticker)
            metadata[ticker]["group"] = "xetra_expanded"
    return metadata


CORE_UNIVERSE = load_universe(groups={"core"})
EXPANDED_UNIVERSE = load_expanded_universe()
# Alias storico usato dall'app: ora rappresenta l'opzione espansa completa,
# non l'unione con i 60 core.
UNIVERSE = list(EXPANDED_UNIVERSE)

_core_metadata_all = load_universe_metadata()
TICKER_METADATA = load_expanded_metadata()
TICKER_METADATA.update({
    ticker: _core_metadata_all[ticker]
    for ticker in CORE_UNIVERSE
    if ticker in _core_metadata_all
})
TICKER_CURRENCY = {
    ticker: (
        metadata.get("currency")
        or infer_yahoo_currency(ticker)
    ).upper()
    for ticker, metadata in TICKER_METADATA.items()
}
BENCHMARK = "VWCE.MI"
