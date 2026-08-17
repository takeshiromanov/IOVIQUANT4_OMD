"""
OMD-Portfolio -- Fase 2: matrice di distanza
-----------------------------------------------
Correlazione trailing -> arccos -> matrice di distanza geodetica (Eq. 1 del paper).
Da qui si leggono: eigenvector centrality (market loading) e distance centrality
(media di riga), i due covariate "strutturali" usati piu' avanti per condizionare
le catene di rank.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def trailing_correlation(returns: pd.DataFrame, active: pd.DataFrame,
                          asof_date: pd.Timestamp, window: int = 252,
                          min_active: int = 20) -> pd.DataFrame | None:
    """Correlazione di Pearson sui rendimenti trailing, solo tra titoli
    attivi in tutta la finestra e senza NaN residui. Ritorna None se
    l'universo attivo quel giorno e' troppo piccolo per una stima sensata."""
    hist = returns.loc[:asof_date].tail(window)
    if hist.shape[0] < window:
        return None
    active_today = active.loc[asof_date]
    cols = active_today[active_today].index
    hist = hist[cols]
    # buchi isolati (1-2 giorni, tipicamente artefatti della raccolta dati
    # attorno alle festivita') vengono riempiti in avanti, usando solo
    # informazione passata -> nessun look-ahead. Titoli con buchi piu' ampi
    # vengono comunque scartati dalla stima.
    hist = hist.ffill(limit=2).dropna(axis=1, how="any")
    if hist.shape[1] < min_active:
        return None
    return hist.corr()


def distance_matrix_from_corr(C: pd.DataFrame) -> pd.DataFrame:
    """M = arccos(C), Eq. (1). Clip per sicurezza numerica su [-1,1]."""
    G = C.values.copy()
    np.fill_diagonal(G, 1.0)
    G = np.clip(G, -1.0, 1.0)
    M = np.arccos(G)
    return pd.DataFrame(M, index=C.index, columns=C.columns)


def leading_eigenvector_centrality(C: pd.DataFrame) -> pd.Series:
    """Market loading: componente del titolo nell'autovettore dominante
    della matrice di correlazione (normalizzato, segno positivo)."""
    vals, vecs = np.linalg.eigh(C.values)
    top = vecs[:, np.argmax(vals)]
    if top.mean() < 0:
        top = -top
    top = np.abs(top)
    top = top / top.sum()
    return pd.Series(top, index=C.index, name="market_loading")


def distance_centrality(M: pd.DataFrame) -> pd.Series:
    """Media di riga della matrice di distanza: prossimita' angolare media
    di un titolo al resto della cross-section (bassa = "centrale/market-like")."""
    return M.mean(axis=1).rename("distance_centrality")


