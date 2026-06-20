"""
plc_simulator.py
Simula un PLC industriale via Modbus TCP su 127.0.0.1:5020
Holding registers:
  0 → Temperatura (valore * 100, es. 5034 = 50.34°C)
  1 → Pressione   (valore * 1000, es. 4012 = 4.012 bar)
"""

import random
import threading
import time
from pymodbus.server import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext, ModbusSequentialDataBlock

HOST = "127.0.0.1"
PORT = 5020


def aggiorna_registri(context):
    """Aggiorna i registri ogni secondo simulando dati normali + anomalie occasionali."""
    contatore = 0
    soglia_anomalia = random.randint(12, 18)

    while True:
        contatore += 1
        if contatore % soglia_anomalia == 0:
            # Picco anomalo
            temp_raw = int(random.uniform(85.0, 110.0) * 100)
            press_raw = int(random.uniform(0.5, 1.5) * 1000)
            soglia_anomalia = random.randint(12, 18)  # reset soglia
        else:
            # Funzionamento normale
            temp_raw = int(random.normalvariate(50.0, 3.0) * 100)
            press_raw = int(random.normalvariate(4.0, 0.4) * 1000)

        # Clamp a 0 per evitare valori negativi nei registri (unsigned 16-bit)
        temp_raw = max(0, temp_raw)
        press_raw = max(0, press_raw)

        context[0x01].setValues(3, 0, [temp_raw, press_raw])
        time.sleep(1)


def avvia_server():
    store = ModbusSlaveContext(hr=ModbusSequentialDataBlock(0, [0] * 10))
    context = ModbusServerContext(slaves={0x01: store}, single=False)

    # Thread che aggiorna i valori in background
    t = threading.Thread(target=aggiorna_registri, args=(context,), daemon=True)
    t.start()

    print(f"[PLC] Server Modbus TCP in ascolto su {HOST}:{PORT}")
    print("[PLC] Premi CTRL+C per fermare\n")
    StartTcpServer(context, address=(HOST, PORT))


if __name__ == "__main__":
    avvia_server()
