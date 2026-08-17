# Universo Xetra 600

Questa configurazione sostituisce il precedente Global Momentum 240.

- Flag spento: vengono usati esclusivamente i 60 ticker core congelati di
  `universe.csv`.
- Flag acceso: vengono usati esattamente i 600 ticker unici della colonna
  `Tickler` di `xetra_raw_deduplicated.csv`.
- I due universi sono alternativi: i 9 ticker core non presenti nel CSV Xetra
  non vengono aggiunti implicitamente all'universo espanso.
- 51 ticker sono presenti in entrambe le liste.

Il CSV espanso contiene società, capitalizzazione, prezzo, ricavi, ISIN, nome
ufficiale Xetra e MIC del mercato primario. La valuta operativa viene dedotta
dal suffisso Yahoo Finance; i listing locali vengono poi convertiti in USD dal
motore prima della simulazione.
