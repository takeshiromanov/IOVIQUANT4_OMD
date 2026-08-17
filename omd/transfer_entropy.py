"""
OMD-Portfolio -- Fase 6: transfer entropy lead-lag score
------------------------------------------------------------
Eq. 19: TE(X->A) = sum p(a_t+1, a_t, x_t) * log[ p(a_t+1|a_t,x_t) / p(a_t+1|a_t) ]

Il "lead-lag score" (Sezione 2.3/3.3) e' TE(nome->mercato) - TE(mercato->nome):
positivo per un "bellwether" che il mercato segue, negativo per un "follower".
Calcolato su finestra trailing di un anno (252gg), come gli altri due covariate
letti dalla matrice di distanza (market loading, distance centrality).

NOTA IMPORTANTE (dal paper): questo covariate risulta NEUTRO per la previsione
delle catene -- il paper stesso lo segnala esplicitamente. Il suo valore reale
e' altrove (overlay assicurativo del portafoglio, rete stock-to-stock separata,
Fase 7). Qui lo implementiamo comunque per fedelta' e per verificare se anche
sul nostro universo risulta neutro -- se si', e' una conferma di fedelta', non
un fallimento.

Processo "mercato": discretizziamo il rendimento giornaliero dell'indice
equal-weight interno in K=5 bucket via quantili ESPANSIVI (solo dati passati,
niente look-ahead), stesso K usato per le catene di rank.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

K = 5


def market_bucket_expanding(market_ret: pd.Series, k: int = K, min_periods: int = 252) -> pd.Series:
    """Bucket del rendimento di mercato in K livelli, via quantili calcolati
    SOLO sui dati passati (espansivo) -- nessun look-ahead."""
    ranks = market_ret.expanding(min_periods=min_periods).rank(pct=True)
    buckets = np.ceil(ranks * k).clip(1, k)
    return buckets


def _joint_counts(a_next: np.ndarray, a_cur: np.ndarray, x_cur: np.ndarray, k: int, eps: float = 0.5):
    """Conteggi congiunti (a_t+1, a_t, x_t) con pseudo-count di Laplace."""
    counts = np.full((k, k, k), eps)
    for an, ac, xc in zip(a_next, a_cur, x_cur):
        if np.isnan(an) or np.isnan(ac) or np.isnan(xc):
            continue
        counts[int(an) - 1, int(ac) - 1, int(xc) - 1] += 1
    return counts


def transfer_entropy_pair(a_next: np.ndarray, a_cur: np.ndarray, x_cur: np.ndarray, k: int = K) -> float:
    """TE(X->A), Eq. 19, plug-in esatto su conteggi discreti."""
    joint = _joint_counts(a_next, a_cur, x_cur, k)  # p(a', a, x)
    p_axa = joint / joint.sum()
    p_ax = p_axa.sum(axis=0)          # p(a, x)  [a_cur, x_cur]
    p_a_next_given_ax = p_axa / p_ax[None, :, :]  # p(a'|a,x)
    p_a = p_axa.sum(axis=(0, 2))       # marginal su a_cur -> serve p(a'|a)
    joint_2d = p_axa.sum(axis=2)       # p(a', a)
    p_a_marg = joint_2d.sum(axis=0)    # p(a)
    p_a_next_given_a = joint_2d / p_a_marg[None, :]  # p(a'|a)

    te = 0.0
    for an in range(k):
        for ac in range(k):
            for xc in range(k):
                p = p_axa[an, ac, xc]
                if p <= 0:
                    continue
                num = p_a_next_given_ax[an, ac, xc]
                den = p_a_next_given_a[an, ac]
                if num <= 0 or den <= 0:
                    continue
                te += p * np.log(num / den)
    return te


def leadlag_score_for_name(name_bucket: pd.Series, market_bucket: pd.Series,
                            window: int = 252, k: int = K, step: int = 5) -> pd.Series:
    """TE(nome->mercato) - TE(mercato->nome) su finestra trailing, ricalcolata
    ogni `step` giorni (il segnale e' lento, come gli altri due covariate
    della matrice di distanza, e il calcolo esatto e' costoso: si ricalcola
    ogni pochi giorni e si riempie in avanti, pratica gia' usata per gli
    altri due covariate della matrice di distanza)."""
    both = pd.concat([name_bucket, market_bucket], axis=1, keys=["name", "mkt"]).dropna()
    idx = both.index
    out = pd.Series(index=idx, dtype=float)
    if len(idx) < window + 1:
        return out.reindex(name_bucket.index)
    for pos in range(window, len(idx), step):
        window_df = both.iloc[pos - window:pos + 1]
        a_name = window_df["name"].values
        a_mkt = window_df["mkt"].values
        a_name_next, a_name_cur = a_name[1:], a_name[:-1]
        a_mkt_next, a_mkt_cur = a_mkt[1:], a_mkt[:-1]
        te_name_to_mkt = transfer_entropy_pair(a_mkt_next, a_mkt_cur, a_name_cur, k=k)
        te_mkt_to_name = transfer_entropy_pair(a_name_next, a_name_cur, a_mkt_cur, k=k)
        out.iloc[pos] = te_name_to_mkt - te_mkt_to_name
    out = out.ffill()
    return out.reindex(name_bucket.index)




def build_stock_network_te(class_daily: pd.DataFrame, window: int = 252, k: int = K,
                            step: int = 10, min_names: int = 15) -> pd.DataFrame:
    """NUOVO (non testato nel resto della conversazione): rete diretta di
    transfer entropy titolo-titolo, TE(i->j) per ogni coppia, sulla finestra
    trailing piu' recente disponibile. Usata per l'overlay assicurativo
    (Sezione 4 del paper: 'directed stock-to-stock transfer-entropy network',
    punteggio hub HITS, 25% di peso sulla sleeve protettiva).

    Costo computazionale: O(N^2) chiamate a transfer_entropy_pair, ciascuna
    O(k^3 * window). Con N~60 e k=5 e' dell'ordine di qualche decina di
    secondi -- accettabile per un ricalcolo periodico (cache Streamlit),
    non per ogni interazione utente."""
    names = [c for c in class_daily.columns if class_daily[c].notna().sum() >= window]
    recent = class_daily[names].tail(window + 1)

    te = pd.DataFrame(index=names, columns=names, dtype=float)
    arrays = {n: recent[n].values for n in names}
    for ni in names:
        a_i = arrays[ni]
        if np.isnan(a_i).any():
            continue
        a_i_next, a_i_cur = a_i[1:], a_i[:-1]
        for nj in names:
            if ni == nj:
                continue
            a_j = arrays[nj]
            if np.isnan(a_j).any():
                continue
            a_j_cur = a_j[:-1]
            te.loc[ni, nj] = transfer_entropy_pair(a_i_next, a_i_cur, a_j_cur, k=k)
    return te


def compute_hub_scores(te_network: pd.DataFrame, max_iter: int = 500) -> pd.Series:
    """HITS sul grafo diretto pesato dalla transfer entropy: hub score alto
    = il titolo 'invia' molta informazione ad altri (un bellwether che il
    resto del mercato segue). Richiede networkx."""
    import networkx as nx
    G = nx.DiGraph()
    names = te_network.index.tolist()
    G.add_nodes_from(names)
    for i in names:
        for j in names:
            if i == j:
                continue
            w = te_network.loc[i, j]
            if pd.notna(w) and w > 0:
                G.add_edge(i, j, weight=w)
    if G.number_of_edges() == 0:
        return pd.Series(0.0, index=names)
    hubs, _authorities = nx.hits(G, max_iter=max_iter, normalized=True)
    return pd.Series(hubs).reindex(names).fillna(0.0)


def build_hub_sleeve(hub_scores: pd.Series, k_book: int) -> tuple:
    """Long i migliori hub (bellwether), short i peggiori -- sleeve
    'assicurativa' da miscelare col 25% di peso nella parte protettiva
    (Tabella 1 del paper)."""
    ranked = hub_scores.sort_values(ascending=False)
    long_hub = set(ranked.head(k_book).index)
    short_hub = set(ranked.tail(k_book).index)
    return long_hub, short_hub
