import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# BTC AI BOT V5.5
# PAPER TRADING - GITHUB ACTIONS
# ============================================================

CAPITAL_INICIAL = 10_000.0

UMBRAL_ENTRADA = 0.70
UMBRAL_SALIDA = 0.45

RIESGO = 0.01
STOP_PCT = 0.02
TAKE_PCT = 0.04

COMISION = 0.0015
MAX_POSICION = 0.50

ESTADO = "estado.csv"
TRADES = "trades.csv"
SENALES = "senales.csv"
EQUITY = "equity.csv"
LOG = "log.csv"


# ============================================================
# CARGAR ESTADO
# ============================================================

if os.path.exists(ESTADO):

    estado = pd.read_csv(ESTADO).iloc[0]

    cash = float(estado["cash"])
    btc_qty = float(estado["btc_qty"])
    entry_price = float(estado["entry_price"])
    stop_price = float(estado["stop_price"])
    take_price = float(estado["take_price"])
    peak_equity = float(estado["peak_equity"])
    ultima_fecha = str(estado["ultima_fecha"])

else:

    cash = CAPITAL_INICIAL
    btc_qty = 0.0
    entry_price = 0.0
    stop_price = 0.0
    take_price = 0.0
    peak_equity = CAPITAL_INICIAL
    ultima_fecha = ""


# ============================================================
# DESCARGAR BTC
# ============================================================

print("📥 Descargando Bitcoin...")

btc = yf.download(
    "BTC-USD",
    period="3y",
    interval="1d",
    auto_adjust=False,
    progress=False
)

if btc.empty:
    raise RuntimeError("No se pudieron descargar datos de BTC.")


if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)


btc = btc[
    ["Open", "High", "Low", "Close", "Volume"]
].copy()


# ============================================================
# FECHAS
# ============================================================

btc.index = pd.to_datetime(btc.index)

if getattr(btc.index, "tz", None) is not None:
    btc.index = btc.index.tz_localize(None)

btc.index = btc.index.normalize()

hoy = pd.Timestamp.now().normalize()

# Solo velas cerradas
btc = btc[btc.index < hoy]


# ============================================================
# FEATURES
# ============================================================

btc["Ret_1"] = btc["Close"].pct_change(1)
btc["Ret_3"] = btc["Close"].pct_change(3)
btc["Ret_5"] = btc["Close"].pct_change(5)
btc["Ret_7"] = btc["Close"].pct_change(7)
btc["Ret_14"] = btc["Close"].pct_change(14)

btc["MA10"] = btc["Close"].rolling(10).mean()
btc["MA20"] = btc["Close"].rolling(20).mean()
btc["MA50"] = btc["Close"].rolling(50).mean()
btc["MA100"] = btc["Close"].rolling(100).mean()
btc["MA200"] = btc["Close"].rolling(200).mean()

btc["Dist_MA10"] = btc["Close"] / btc["MA10"] - 1
btc["Dist_MA20"] = btc["Close"] / btc["MA20"] - 1
btc["Dist_MA50"] = btc["Close"] / btc["MA50"] - 1
btc["Dist_MA100"] = btc["Close"] / btc["MA100"] - 1
btc["Dist_MA200"] = btc["Close"] / btc["MA200"] - 1

btc["MA20_MA50"] = btc["MA20"] / btc["MA50"] - 1
btc["MA50_MA200"] = btc["MA50"] / btc["MA200"] - 1


# RSI
cambio = btc["Close"].diff()

subidas = cambio.clip(lower=0)
bajadas = -cambio.clip(upper=0)

media_subidas = subidas.rolling(14).mean()
media_bajadas = bajadas.rolling(14).mean()

RS = media_subidas / media_bajadas

btc["RSI"] = 100 - (100 / (1 + RS))


# Volatilidad
btc["Vol_5"] = btc["Ret_1"].rolling(5).std()
btc["Vol_10"] = btc["Ret_1"].rolling(10).std()
btc["Vol_20"] = btc["Ret_1"].rolling(20).std()


# Rango
btc["Rango"] = (
    btc["High"] - btc["Low"]
) / btc["Close"]

btc["Rango_Medio"] = btc["Rango"].rolling(20).mean()


# Volumen
btc["Volumen_Medio"] = btc["Volume"].rolling(20).mean()

btc["Ratio_Volumen"] = (
    btc["Volume"] / btc["Volumen_Medio"]
)


# ============================================================
# OBJETIVO
# ============================================================

btc["Retorno_3d_Futuro"] = (
    btc["Close"].shift(-3) /
    btc["Close"] - 1
)

btc["Objetivo"] = np.nan

mask = btc["Retorno_3d_Futuro"].notna()

btc.loc[mask, "Objetivo"] = (
    btc.loc[
        mask,
        "Retorno_3d_Futuro"
    ] >= 0.01
).astype(int)


features = [
    "Ret_1",
    "Ret_3",
    "Ret_5",
    "Ret_7",
    "Ret_14",
    "Dist_MA10",
    "Dist_MA20",
    "Dist_MA50",
    "Dist_MA100",
    "Dist_MA200",
    "MA20_MA50",
    "MA50_MA200",
    "RSI",
    "Vol_5",
    "Vol_10",
    "Vol_20",
    "Rango",
    "Rango_Medio",
    "Ratio_Volumen"
]


# ============================================================
# ENTRENAMIENTO
# ============================================================

train = btc.dropna(
    subset=features + ["Objetivo"]
).copy()

X = train[features]
y = train["Objetivo"].astype(int)

print(f"📚 Datos de entrenamiento: {len(train)}")


modelo = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

modelo.fit(X, y)


# ============================================================
# ÚLTIMA VELA CERRADA
# ============================================================

ultima = btc.dropna(
    subset=features
).iloc[-1]

fecha = ultima.name
fecha_texto = str(fecha.date())

precio = float(ultima["Close"])

X_actual = pd.DataFrame(
    [ultima[features].values],
    columns=features
)

probabilidad = float(
    modelo.predict_proba(X_actual)[0][1]
)

RSI_actual = float(ultima["RSI"])
MA50_actual = float(ultima["MA50"])
MA200_actual = float(ultima["MA200"])


print()
print("=" * 60)
print("🤖 BTC AI BOT V5.5")
print("=" * 60)
print(f"📅 Fecha: {fecha_texto}")
print(f"₿ BTC: ${precio:,.2f}")
print(f"🧠 Probabilidad: {probabilidad:.2%}")


# ============================================================
# EVITAR DUPLICADOS
# ============================================================

if ultima_fecha == fecha_texto:

    print()
    print("⚠️ Esta vela ya fue procesada.")
    print("No se hará ninguna operación.")
    print("El sistema termina correctamente.")

    raise SystemExit


accion = "ESPERAR"
motivo = ""


# ============================================================
# GESTIONAR POSICIÓN
# ============================================================

if btc_qty > 0:

    valor_btc = btc_qty * precio

    # STOP LOSS
    if precio <= stop_price:

        ingreso = valor_btc * (1 - COMISION)

        pnl = (
            ingreso -
            btc_qty * entry_price
        )

        cash += ingreso

        accion = "VENDER"
        motivo = "STOP LOSS"

        nuevo_trade = pd.DataFrame([{
            "Fecha": fecha_texto,
            "Tipo": "VENTA",
            "Motivo": motivo,
            "Precio": precio,
            "BTC": btc_qty,
            "Entrada": entry_price,
            "PnL": pnl
        }])

        if os.path.exists(TRADES):
            trades = pd.read_csv(TRADES)
        else:
            trades = pd.DataFrame()

        trades = pd.concat(
            [trades, nuevo_trade],
            ignore_index=True
        )

        trades.to_csv(TRADES, index=False)

        btc_qty = 0
        entry_price = 0
        stop_price = 0
        take_price = 0


    # TAKE PROFIT
    elif precio >= take_price:

        ingreso = valor_btc * (1 - COMISION)

        pnl = (
            ingreso -
            btc_qty * entry_price
        )

        cash += ingreso

        accion = "VENDER"
        motivo = "TAKE PROFIT"

        nuevo_trade = pd.DataFrame([{
            "Fecha": fecha_texto,
            "Tipo": "VENTA",
            "Motivo": motivo,
            "Precio": precio,
            "BTC": btc_qty,
            "Entrada": entry_price,
            "PnL": pnl
        }])

        if os.path.exists(TRADES):
            trades = pd.read_csv(TRADES)
        else:
            trades = pd.DataFrame()

        trades = pd.concat(
            [trades, nuevo_trade],
            ignore_index=True
        )

        trades.to_csv(TRADES, index=False)

        btc_qty = 0
        entry_price = 0
        stop_price = 0
        take_price = 0


    # SALIDA IA
    elif probabilidad < UMBRAL_SALIDA:

        ingreso = valor_btc * (1 - COMISION)

        pnl = (
            ingreso -
            btc_qty * entry_price
        )

        cash += ingreso

        accion = "VENDER"
        motivo = "SALIDA IA"

        nuevo_trade = pd.DataFrame([{
            "Fecha": fecha_texto,
            "Tipo": "VENTA",
            "Motivo": motivo,
            "Precio": precio,
            "BTC": btc_qty,
            "Entrada": entry_price,
            "PnL": pnl
        }])

        if os.path.exists(TRADES):
            trades = pd.read_csv(TRADES)
        else:
            trades = pd.DataFrame()

        trades = pd.concat(
            [trades, nuevo_trade],
            ignore_index=True
        )

        trades.to_csv(TRADES, index=False)

        btc_qty = 0
        entry_price = 0
        stop_price = 0
        take_price = 0

    else:

        accion = "MANTENER"
        motivo = "POSICIÓN ABIERTA"


# ============================================================
# ENTRADA
# ============================================================

if btc_qty == 0 and accion != "VENDER":

    if probabilidad >= UMBRAL_ENTRADA:

        equity_actual = cash

        riesgo_dinero = equity_actual * RIESGO

        distancia_stop = precio * STOP_PCT

        posicion_por_riesgo = (
            riesgo_dinero /
            distancia_stop
        ) * precio

        valor_posicion = min(
            posicion_por_riesgo,
            equity_actual * MAX_POSICION
        )

        btc_comprar = valor_posicion / precio

        costo = (
            btc_comprar *
            precio *
            (1 + COMISION)
        )

        if costo <= cash:

            cash -= costo

            btc_qty = btc_comprar

            entry_price = precio

            stop_price = (
                entry_price *
                (1 - STOP_PCT)
            )

            take_price = (
                entry_price *
                (1 + TAKE_PCT)
            )

            accion = "COMPRAR"
            motivo = "SEÑAL IA"

            nuevo_trade = pd.DataFrame([{
                "Fecha": fecha_texto,
                "Tipo": "COMPRA",
                "Motivo": motivo,
                "Precio": precio,
                "BTC": btc_qty,
                "Entrada": entry_price,
                "PnL": 0
            }])

            if os.path.exists(TRADES):
                trades = pd.read_csv(TRADES)
            else:
                trades = pd.DataFrame()

            trades = pd.concat(
                [trades, nuevo_trade],
                ignore_index=True
            )

            trades.to_csv(TRADES, index=False)

        else:

            accion = "ESPERAR"
            motivo = "CAPITAL INSUFICIENTE"

    else:

        accion = "ESPERAR"
        motivo = "PROBABILIDAD BAJO UMBRAL"


# ============================================================
# EQUITY
# ============================================================

equity = cash + btc_qty * precio

peak_equity = max(
    peak_equity,
    equity
)

rendimiento = (
    equity /
    CAPITAL_INICIAL - 1
)

drawdown = (
    equity /
    peak_equity - 1
)


# ============================================================
# GUARDAR SEÑAL
# ============================================================

nueva_senal = pd.DataFrame([{
    "Fecha": fecha_texto,
    "Precio": precio,
    "Probabilidad": probabilidad,
    "RSI": RSI_actual,
    "MA50": MA50_actual,
    "MA200": MA200_actual,
    "Señal": accion,
    "Motivo": motivo
}])

if os.path.exists(SENALES):
    señales = pd.read_csv(SENALES)
else:
    señales = pd.DataFrame()

señales = pd.concat(
    [señales, nueva_senal],
    ignore_index=True
)

señales.to_csv(
    SENALES,
    index=False
)


# ============================================================
# GUARDAR EQUITY
# ============================================================

nueva_equity = pd.DataFrame([{
    "Fecha": fecha_texto,
    "BTC": precio,
    "Equity": equity,
    "Cash": cash,
    "BTC_QTY": btc_qty
}])

if os.path.exists(EQUITY):
    equity_hist = pd.read_csv(EQUITY)

    equity_hist = equity_hist[
        equity_hist["Fecha"].astype(str)
        != fecha_texto
    ]

else:
    equity_hist = pd.DataFrame()

equity_hist = pd.concat(
    [equity_hist, nueva_equity],
    ignore_index=True
)

equity_hist.to_csv(
    EQUITY,
    index=False
)


# ============================================================
# GUARDAR ESTADO
# ============================================================

nuevo_estado = pd.DataFrame([{
    "cash": cash,
    "btc_qty": btc_qty,
    "entry_price": entry_price,
    "stop_price": stop_price,
    "take_price": take_price,
    "peak_equity": peak_equity,
    "ultima_fecha": fecha_texto
}])

nuevo_estado.to_csv(
    ESTADO,
    index=False
)


# ============================================================
# LOG
# ============================================================

nuevo_log = pd.DataFrame([{
    "Fecha": fecha_texto,
    "Precio": precio,
    "Probabilidad": probabilidad,
    "Accion": accion,
    "Motivo": motivo,
    "Equity": equity
}])

if os.path.exists(LOG):
    log = pd.read_csv(LOG)
else:
    log = pd.DataFrame()

log = pd.concat(
    [log, nuevo_log],
    ignore_index=True
)

log.to_csv(
    LOG,
    index=False
)


# ============================================================
# RESULTADO
# ============================================================

print()
print("=" * 60)
print("📊 RESULTADO")
print("=" * 60)

print(f"📢 Acción: {accion}")
print(f"📝 Motivo: {motivo}")

print()
print(f"RSI: {RSI_actual:.2f}")
print(f"MA50: ${MA50_actual:,.2f}")
print(f"MA200: ${MA200_actual:,.2f}")

print()
print(f"💵 Cash: ${cash:,.2f}")
print(f"₿ BTC: {btc_qty:.6f}")
print(f"💰 Equity: ${equity:,.2f}")
print(f"📈 Rendimiento: {rendimiento:.2%}")
print(f"📉 Drawdown: {drawdown:.2%}")

if btc_qty > 0:

    print()
    print("🟢 POSICIÓN ABIERTA")
    print(f"Entrada: ${entry_price:,.2f}")
    print(f"Stop: ${stop_price:,.2f}")
    print(f"Take Profit: ${take_price:,.2f}")

else:

    print()
    print("🟡 SIN POSICIÓN")

print()
print("💾 Datos guardados.")
print("⚠️ PAPER TRADING — SIN DINERO REAL.")
