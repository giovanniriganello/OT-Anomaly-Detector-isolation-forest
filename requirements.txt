"""
monitor.py
Client Modbus TCP + Isolation Forest per anomaly detection in tempo reale.
Si connette al PLC (reale o simulato) su 127.0.0.1:5020
Scrive un file .log con le sole anomalie rilevate.

Per usarlo con un PLC reale: cambia HOST e PORT nelle costanti sotto.
"""

import time
import random
import logging
import pandas as pd
from collections import deque
from datetime import datetime
from sklearn.ensemble import IsolationForest
from pymodbus.client import ModbusTcpClient

# --- CONFIGURAZIONE ---
HOST = "127.0.0.1"
PORT = 5020
SLAVE_ID = 1
REFIT_OGNI = 50        # riaddestra il modello ogni N letture
FINESTRA_ALERT = 5     # dimensione finestra temporale
SOGLIA_ALERT = 3       # minimo anomalie nella finestra per ALERT CONFERMATO

# --- LOGGING SU FILE ---
log_filename = f"anomalie_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def leggi_plc(plc: ModbusTcpClient) -> dict:
    """Legge temperatura e pressione dai holding registers 0 e 1."""
    risultato = plc.read_holding_registers(0, 2, slave=SLAVE_ID)
    temp = round(risultato.registers[0] / 100, 2)
    press = round(risultato.registers[1] / 1000, 2)
    return {"Temperatura": temp, "Pressione": press, "Tipo_Reale": "Live"}


def dati_calibrazione_sintetici(n=30) -> pd.DataFrame:
    """
    Genera n campioni normali sintetici per la calibrazione iniziale.
    Necessario perché all'avvio il PLC potrebbe avere registri a 0
    o dati non rappresentativi del regime stazionario.
    """
    dati = [
        {
            "Temperatura": round(random.normalvariate(50.0, 3.0), 2),
            "Pressione": round(random.normalvariate(4.0, 0.4), 2),
        }
        for _ in range(n)
    ]
    return pd.DataFrame(dati)


def calibra_modello(df: pd.DataFrame, contamination=0.05) -> IsolationForest:
    X = df[['Temperatura', 'Pressione']]
    modello = IsolationForest(contamination=contamination, random_state=42)
    modello.fit(X)
    return modello


if __name__ == "__main__":
    # Connessione al PLC
    plc = ModbusTcpClient(HOST, port=PORT)
    if not plc.connect():
        print(f"[ERRORE] Impossibile connettersi a {HOST}:{PORT}")
        exit(1)
    print(f"[OK] Connesso al PLC su {HOST}:{PORT}\n")

    # Fase 1 — Calibrazione
    print("=== FASE 1: Calibrazione (dati normali sintetici) ===")
    df_storico = dati_calibrazione_sintetici(30)
    modello = calibra_modello(df_storico)
    print(f"Modello calibrato su {len(df_storico)} campioni.\n")

    buffer_dati = deque(df_storico.to_dict('records'), maxlen=200)
    buffer_predizioni = deque(maxlen=FINESTRA_ALERT)
    contatore_nuovi = 0

    # Fase 2 — Monitoraggio
    print("=== FASE 2: Monitoraggio in tempo reale (CTRL+C per fermare) ===")
    print(f"Log anomalie → {log_filename}\n")
    print(f"{'Temp (°C)':<12}{'Press (bar)':<14}{'Score':<10}{'DIAGNOSI AI'}")
    print("-" * 55)

    try:
        while True:
            try:
                nuovo_dato = leggi_plc(plc)
            except Exception as e:
                print(f"[WARN] Errore lettura PLC: {e} — retry tra 2s")
                time.sleep(2)
                continue

            buffer_dati.append(nuovo_dato)
            contatore_nuovi += 1

            X_nuovo = pd.DataFrame(
                [[nuovo_dato['Temperatura'], nuovo_dato['Pressione']]],
                columns=['Temperatura', 'Pressione']
            )

            predizione = modello.predict(X_nuovo)[0]
            score = round(modello.decision_function(X_nuovo)[0], 4)
            buffer_predizioni.append(predizione)
            anomalie_in_finestra = list(buffer_predizioni).count(-1)

            if predizione == -1 and anomalie_in_finestra >= SOGLIA_ALERT:
                diagnosi = "🚨 ALERT CONFERMATO"
                logger.critical(
                    "ALERT_CONFERMATO | Temp=%.2f°C | Press=%.2f bar | "
                    "Score=%.4f | Anomalie_su_%d=%d",
                    nuovo_dato['Temperatura'], nuovo_dato['Pressione'],
                    score, FINESTRA_ALERT, anomalie_in_finestra
                )
            elif predizione == -1:
                diagnosi = "⚠️  outlier (attesa conferma)"
                logger.warning(
                    "OUTLIER | Temp=%.2f°C | Press=%.2f bar | "
                    "Score=%.4f | Anomalie_su_%d=%d",
                    nuovo_dato['Temperatura'], nuovo_dato['Pressione'],
                    score, FINESTRA_ALERT, anomalie_in_finestra
                )
            else:
                diagnosi = "✅ OK"

            print(f"{nuovo_dato['Temperatura']:<12}{nuovo_dato['Pressione']:<14}"
                  f"{score:<10}{diagnosi}")

            # Refit periodico sul buffer scorrevole
            if contatore_nuovi % REFIT_OGNI == 0:
                df_refit = pd.DataFrame(list(buffer_dati))
                modello = calibra_modello(df_refit)
                print(f"  [Modello riaddestrato su {len(buffer_dati)} campioni]\n")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\nMonitoraggio interrotto.")
        plc.close()
        logging.shutdown()
