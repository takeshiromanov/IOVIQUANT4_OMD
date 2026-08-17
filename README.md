# IOVIQUANT Pro — Universo Xetra 600

Versione Streamlit del modello Unicorn Hunter con:

- 60 titoli core congelati, selezionabili con il flag spento;
- 600 ticker unici letti da `xetra_raw_deduplicated.csv`, selezionabili con il flag acceso;
- universi alternativi: il CSV espanso non viene unito implicitamente ai 60 core;
- conversione causale degli OHLC locali in USD;
- capitale e reporting in EUR;
- COR1M letto esclusivamente dai CSV locali.
- integrazione OMD opzionale sull'universo Xetra: cluster Buy mensile da 60
  titoli (`K=30`) e cluster Sell da 30;
- refit dei generatori OMD ogni 12 mesi, come nel progetto originale;
- filtro causale: il cluster del mese concluso è applicato dalla prima seduta
  successiva;
- uscita forzata dei titoli Sell attivabile separatamente;
- confronto KPI automatico fra UH puro, filtro OMD e filtro OMD + Sell forzato.

## Pubblicazione su GitHub e Streamlit

Caricare tutti i file di questa cartella nella radice del repository GitHub.
`streamlit_app.py`, `ioviquant_engine.py`, `omd_integration.py`,
`omd_hunter_comparison.py`, `requirements.txt`, `universe.csv`,
`xetra_raw_deduplicated.csv`, la cartella `omd/` e i due CSV COR1M devono essere
caricati conservando la struttura di questa cartella.

Su Streamlit Community Cloud usare:

- branch: `main`;
- entrypoint: `streamlit_app.py`;
- Python: **3.13**.

Se l'app esistente usa Python 3.14, eliminarla da Streamlit Community Cloud e
ridistribuirla selezionando Python 3.13 nelle impostazioni avanzate. Il
repository GitHub non deve essere eliminato.

## Avvertenze operative

Il primo download dell'universo Xetra 600 è sensibilmente più lento del core 60
e può richiedere diversi minuti. I download vengono memorizzati nella cache
Streamlit per sei ore. Singoli ticker non disponibili vengono segnalati ed esclusi.

OMD richiede l'universo espanso e almeno 36 mesi di training utilizzabile. Quando
è attivo, l'app scarica cinque anni aggiuntivi rispetto alla data iniziale: sono
warm-up e training soltanto, quindi Unicorn Hunter non può aprire posizioni prima
della data operativa selezionata.

I prezzi dei titoli locali vengono convertiti in USD con coppie FX giornaliere
Yahoo Finance; il benchmark VWCE.MI resta espresso in EUR. Commissioni e
microstruttura dei singoli mercati non sono ancora differenziate per borsa.

Il progetto è sperimentale e non costituisce consulenza finanziaria.
