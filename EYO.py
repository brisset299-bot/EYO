
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import linprog

# ============================================================
# HOSPITAL GLORIA
# Escenarios 1 y 2
# ============================================================

st.set_page_config(
    page_title="Hospital Gloria | Simulador",
    page_icon="🏥",
    layout="wide"
)

# ------------------------------------------------------------
# DATOS BASE DEL CASO
# ------------------------------------------------------------
DIAS = ["Domingo", "Lunes", "Martes", "Miércoles",
        "Jueves", "Viernes", "Sábado"]

S_BASE = np.array([60, 114, 124, 97, 70, 93, 89], dtype=float)
J_BASE = np.array([204, 164, 139, 164, 182, 154, 182], dtype=float)
DEMANDA = np.array([582, 580, 565, 548, 525, 506, 508], dtype=float)

# Datos de "Queues MM1" entregados para el trabajo
RATE_SENIOR = 1959.62 / 647
RATE_JUNIOR = 2354.81 / 1189

TOTAL_SENIOR = S_BASE.sum()       # 647
TOTAL_JUNIOR = J_BASE.sum()       # 1189
TOTAL_HOURS = TOTAL_SENIOR + TOTAL_JUNIOR  # 1836
TOTAL_DEMANDA = DEMANDA.sum()    # 3814

# ============================================================
# UTILIDADES
# ============================================================

def calcular_capacidad(S, J, tasa_s=RATE_SENIOR, tasa_j=RATE_JUNIOR,
                       horas_dia=24):
    """Calcula capacidad diaria y métricas M/M/1."""
    capacidad = tasa_s * S + tasa_j * J
    lam = DEMANDA / horas_dia
    mu = capacidad / horas_dia

    utilizacion = np.divide(
        lam, mu,
        out=np.full_like(lam, np.nan),
        where=mu > 0
    )

    wq = np.full_like(lam, np.nan)
    estable = mu > lam

    wq[estable] = (
        60 * lam[estable]
        / (mu[estable] * (mu[estable] - lam[estable]))
    )

    tabla = pd.DataFrame({
        "Día": DIAS,
        "Sénior (h)": S,
        "Júnior (h)": J,
        "Demanda (pacientes)": DEMANDA,
        "Capacidad (pacientes/día)": capacidad,
        "Lambda (pac/h)": lam,
        "Mu (pac/h)": mu,
        "Utilización": utilizacion,
        "Wq (min)": wq,
        "Estado": np.where(
            estable,
            "Estable",
            "Sin equilibrio estable"
        )
    })
    return tabla


def optimizar_escenario1(
    max_s_fri,
    max_s_sab,
    max_j_fri,
    max_j_sab,
    tasa_s=RATE_SENIOR,
    tasa_j=RATE_JUNIOR
):
    """
    LP:
        max z
        tasa_s*S_d + tasa_j*J_d >= z*D_d

    Variables:
        0: S(Fri -> Sun)
        1: S(Fri -> Mon)
        2: S(Sat -> Sun)
        3: S(Sat -> Mon)
        4: J(Fri -> Sun)
        5: J(Fri -> Mon)
        6: J(Sat -> Sun)
        7: J(Sat -> Mon)
        8: z

    Solo se permiten traslados:
        viernes/sábado -> domingo/lunes.

    Después se resuelve una segunda LP que minimiza las horas
    trasladadas manteniendo el mejor z.
    """

    # Orígenes/destinos para las 8 variables de traslado
    rutas = [
        ("Sénior", 5, 0),
        ("Sénior", 5, 1),
        ("Sénior", 6, 0),
        ("Sénior", 6, 1),
        ("Júnior", 5, 0),
        ("Júnior", 5, 1),
        ("Júnior", 6, 0),
        ("Júnior", 6, 1),
    ]

    limites = [
        max_s_fri, max_s_fri, max_s_sab, max_s_sab,
        max_j_fri, max_j_fri, max_j_sab, max_j_sab
    ]

    # Capacidad base
    cap_base = tasa_s * S_BASE + tasa_j * J_BASE

    # Restricciones:
    # -capacidad_nueva + D*z <= 0
    A_ub = []
    b_ub = []

    for d in range(7):
        row = np.zeros(9)

        # - z*D en la forma:
        # D*z - capacidad_nueva <= 0
        row[8] = DEMANDA[d]

        # Restar la capacidad nueva.
        # Una transferencia:
        # origen pierde horas -> capacidad baja
        # destino gana horas -> capacidad sube
        for k, (cat, origen, destino) in enumerate(rutas):
            tasa = tasa_s if cat == "Sénior" else tasa_j

            efecto = 0
            if d == destino:
                efecto += tasa
            if d == origen:
                efecto -= tasa

            row[k] -= efecto

        A_ub.append(row)
        b_ub.append(cap_base[d])

    # La cota de cada traslado la controla el usuario.
    bounds = [(0, limite) for limite in limites] + [(0, None)]

    # Primera etapa: maximizar z
    objetivo_1 = np.array([0] * 8 + [-1.0])

    res1 = linprog(
        objetivo_1,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs"
    )

    if not res1.success:
        return None, None, res1.message

    z_opt = res1.x[8]

    # Segunda etapa:
    # minimizar total de horas trasladadas manteniendo z = z_opt
    bounds_2 = [(0, limite) for limite in limites] + [(z_opt, z_opt)]
    objetivo_2 = np.array([1.0] * 8 + [0.0])

    res2 = linprog(
        objetivo_2,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds_2,
        method="highs"
    )

    if not res2.success:
        return None, None, res2.message

    x = res2.x

    S_new = S_BASE.copy()
    J_new = J_BASE.copy()

    for k, (cat, origen, destino) in enumerate(rutas):
        cantidad = x[k]

        if cat == "Sénior":
            S_new[origen] -= cantidad
            S_new[destino] += cantidad
        else:
            J_new[origen] -= cantidad
            J_new[destino] += cantidad

    movimientos = pd.DataFrame({
        "Categoría": [r[0] for r in rutas],
        "Origen": [DIAS[r[1]] for r in rutas],
        "Destino": [DIAS[r[2]] for r in rutas],
        "Horas trasladadas": x[:8]
    })

    resultado = {
        "S_new": S_new,
        "J_new": J_new,
        "z": z_opt,
        "movimientos": movimientos,
        "total_trasladado": x[:8].sum(),
        "res1": res1,
        "res2": res2
    }

    return resultado, rutas, "OK"


# ============================================================
# SIMULACIÓN DISCRETA
# ============================================================

def generar_semana_aleatoria(
    semilla,
    demanda_diaria=DEMANDA,
    horas_dia=24
):
    """
    Genera UNA misma realización de:
      - llegadas Poisson con lambda diaria = demanda/horas_dia
      - requerimientos de servicio Exp(1)

    Esta misma realización se utiliza para Actual y Escenario 1,
    logrando una comparación pareada.
    """
    rng = np.random.default_rng(semilla)

    llegadas = []
    trabajos = []

    for d in range(7):
        inicio = d * horas_dia
        fin = (d + 1) * horas_dia
        lam = demanda_diaria[d] / horas_dia

        t = inicio

        while True:
            t += rng.exponential(1 / lam)

            if t >= fin:
                break

            llegadas.append(t)
            trabajos.append(rng.exponential(1.0))

    return np.array(llegadas), np.array(trabajos)


def simular_trayectoria(
    llegadas,
    trabajos,
    capacidad_dia,
    horas_dia=24,
    warmup_horas=0
):
    """
    Servidor equivalente con capacidad variable por día.

    El trabajo de cada paciente ~ Exp(media=1).
    La capacidad del servidor es mu_d trabajos/hora.

    Esto permite que un paciente continúe en servicio al cambiar
    de día: su trabajo restante se conserva y se procesa con la
    nueva capacidad.

    FIFO.
    """
    horizonte = 7 * horas_dia

    cola = []
    i = 0
    t = 0.0

    ocupado = False
    arr_servicio = None
    trabajo_restante = None
    inicio_servicio = None

    # Registros de eventos para visualizar la cola
    eventos_t = [0.0]
    eventos_q = [0]

    esperas_por_dia = [[] for _ in range(7)]
    sistema_por_dia = [[] for _ in range(7)]
    max_cola = np.zeros(7)

    while t < horizonte:
        dia = min(int(t // horas_dia), 6)
        fin_dia = (dia + 1) * horas_dia
        mu = capacidad_dia[dia] / horas_dia

        prox_llegada = (
            llegadas[i] if i < len(llegadas) else math.inf
        )

        if ocupado:
            tiempo_completacion = trabajo_restante / mu
            prox_completacion = t + tiempo_completacion
        else:
            prox_completacion = math.inf

        prox_evento = min(
            prox_llegada,
            prox_completacion,
            fin_dia,
            horizonte
        )

        dt = prox_evento - t

        if ocupado:
            trabajo_restante -= mu * dt
            trabajo_restante = max(0.0, trabajo_restante)

        t = prox_evento

        if t >= horizonte:
            break

        # Cambio de día
        if abs(t - fin_dia) < 1e-10:
            eventos_t.append(t)
            eventos_q.append(len(cola))
            continue

        # Si coinciden, procesar primero la salida
        if prox_completacion <= prox_llegada + 1e-12:
            dia_paciente = min(int(arr_servicio // horas_dia), 6)

            if arr_servicio >= warmup_horas:
                esperas_por_dia[dia_paciente].append(
                    inicio_servicio - arr_servicio
                )
                sistema_por_dia[dia_paciente].append(
                    t - arr_servicio
                )

            ocupado = False
            arr_servicio = None
            trabajo_restante = None
            inicio_servicio = None

            if cola:
                arr_servicio, trabajo_restante = cola.pop(0)
                inicio_servicio = t
                ocupado = True

        else:
            # Nueva llegada
            a = llegadas[i]
            w = trabajos[i]
            i += 1

            if ocupado:
                cola.append((a, w))
            else:
                ocupado = True
                arr_servicio = a
                trabajo_restante = w
                inicio_servicio = t

        dia_actual = min(int(t // horas_dia), 6)
        max_cola[dia_actual] = max(
            max_cola[dia_actual],
            len(cola)
        )

        eventos_t.append(t)
        eventos_q.append(len(cola))

    pendientes = len(cola) + (1 if ocupado else 0)

    # Construir resultados diarios
    filas = []

    for d in range(7):
        waits = np.array(esperas_por_dia[d], dtype=float)
        systems = np.array(sistema_por_dia[d], dtype=float)

        filas.append({
            "Día": DIAS[d],
            "Espera promedio (min)": (
                waits.mean() if len(waits) else np.nan
            ),
            "P90 espera (min)": (
                np.percentile(waits, 90) if len(waits) else np.nan
            ),
            "Cola máxima": int(max_cola[d]),
            "Pacientes observados": len(waits),
        })

    detalle = pd.DataFrame(filas)

    trayectoria = pd.DataFrame({
        "Hora": eventos_t,
        "Cola": eventos_q
    })

    return detalle, pendientes, trayectoria


def ejecutar_simulacion_escenario1(
    S,
    J,
    horas_dia=24,
    replicas=100,
    warmup_horas=0,
    semilla=42
):
    """
    Ejecuta múltiples réplicas y devuelve:
      - resultados diarios por réplica
      - resumen diario
      - resumen global por réplica
      - trayectoria de la primera réplica
    """
    capacidad = RATE_SENIOR * S + RATE_JUNIOR * J

    registros = []
    globales = []
    trayectoria_primera = None

    for r in range(replicas):
        llegadas, trabajos = generar_semana_aleatoria(
            semilla + r,
            horas_dia=horas_dia
        )

        detalle, pendientes, trayectoria = simular_trayectoria(
            llegadas,
            trabajos,
            capacidad,
            horas_dia=horas_dia,
            warmup_horas=warmup_horas
        )

        detalle["Réplica"] = r + 1
        registros.append(detalle)

        esperas = detalle["Espera promedio (min)"].dropna()

        globales.append({
            "Réplica": r + 1,
            "Espera semanal promedio (min)": (
                esperas.mean() if len(esperas) else np.nan
            ),
            "P90 semanal aproximado (min)": (
                detalle["P90 espera (min)"].mean()
            ),
            "Cola máxima semanal": (
                detalle["Cola máxima"].max()
            ),
            "Pacientes pendientes": pendientes
        })

        if r == 0:
            trayectoria_primera = trayectoria

    detalle_reps = pd.concat(registros, ignore_index=True)
    global_reps = pd.DataFrame(globales)

    resumen = (
        detalle_reps
        .groupby("Día", sort=False)
        .agg({
            "Espera promedio (min)": "mean",
            "P90 espera (min)": "mean",
            "Cola máxima": "mean",
            "Pacientes observados": "mean"
        })
        .reset_index()
    )

    return {
        "detalle": detalle_reps,
        "resumen": resumen,
        "global": global_reps,
        "trayectoria": trayectoria_primera,
        "capacidad": capacidad
    }


def intervalo_confianza_95(series):
    """Media y IC95% entre réplicas."""
    x = pd.Series(series).dropna().astype(float).values

    if len(x) == 0:
        return np.nan, np.nan

    media = x.mean()

    if len(x) == 1:
        return media, np.nan

    se = x.std(ddof=1) / math.sqrt(len(x))
    margen = 1.96 * se

    return media, margen


def comparar_simulaciones(sim_actual, sim_propuesta):
    a = sim_actual["global"]
    b = sim_propuesta["global"]

    media_a, ic_a = intervalo_confianza_95(
        a["Espera semanal promedio (min)"]
    )
    media_b, ic_b = intervalo_confianza_95(
        b["Espera semanal promedio (min)"]
    )

    pendientes_a = a["Pacientes pendientes"].mean()
    pendientes_b = b["Pacientes pendientes"].mean()

    return pd.DataFrame({
        "Indicador": [
            "Espera semanal promedio (min)",
            "IC95% ± (min)",
            "Cola máxima promedio",
            "Pacientes pendientes promedio"
        ],
        "Situación actual": [
            media_a,
            ic_a,
            a["Cola máxima semanal"].mean(),
            pendientes_a
        ],
        "Escenario 1": [
            media_b,
            ic_b,
            b["Cola máxima semanal"].mean(),
            pendientes_b
        ]
    })



# ============================================================
# SIMULACIÓN ESCENARIO 2 — INDEPENDIENTE
# ============================================================

def generar_llegadas_escenario2(pacientes_dia, modo, rng):
    """Llegadas para Escenario 2. Usa 23 h/día y NO comparte
    parámetros con el Escenario 1."""
    n = int(pacientes_dia)
    if modo == "Estable":
        return np.linspace(0, 23 * 60, n, endpoint=False)

    # Flujo variable alrededor de la demanda diaria.
    inter = rng.exponential(60 / (n / 23), size=n)
    t = np.cumsum(inter)
    if len(t) == 0:
        return t
    return t / t[-1] * (23 * 60 - 0.01)


def simular_dia_escenario2(llegadas_min, tasa_servicio, rng):
    """Una cola agregada por día para Escenario 2.
    Servicio exponencial con media 1/mu.
    Se reinicia la disponibilidad al comienzo de cada día,
    conservando la lógica de la simulación original del Escenario 2."""
    # 23 horas de operación. La tasa se expresa por hora-médico.
    servicio_medio_min = 60 / tasa_servicio
    disponible = 0.0
    esperas = []
    cola = []
    max_cola = 0

    for llegada in llegadas_min:
        servicio = rng.exponential(servicio_medio_min)
        inicio = max(llegada, disponible)
        espera = inicio - llegada
        salida = inicio + servicio
        esperas.append(espera)
        disponible = salida

        # Aproximación de cola observada en eventos de llegada.
        cola_actual = sum(1 for x in llegadas_min if x <= llegada) - sum(
            1 for _ in []
        )
        max_cola = max(max_cola, cola_actual)

    pendientes = max(0, len(llegadas_min) - sum(1 for _ in esperas if disponible <= 23 * 60))
    return np.array(esperas), int(max_cola), pendientes


def ejecutar_simulacion_escenario2(
    horas_senior_domingo,
    mejora_eficiencia,
    modo="Estable",
    replicas=100,
    semilla=42
):
    """Simulación independiente del Escenario 2.

    Parámetros propios:
      - 23 horas/día
      - tasa base 2.4 pacientes/hora-médico
      - solo cambia el domingo
      - las 264 horas del domingo se mantienen
    """
    horas_senior = S_BASE.copy()
    horas_junior = J_BASE.copy()
    horas_senior[0] = horas_senior_domingo
    horas_junior[0] = 264 - horas_senior_domingo

    # Tasa global base de este escenario.
    tasa_base = 2.4

    # El mix no cambia por sí mismo la tasa; se usa la mejora configurable.
    tasa_domingo = tasa_base * (1 + mejora_eficiencia / 100)

    registros = []
    globales = []
    trayectoria = None

    for r in range(replicas):
        rng = np.random.default_rng(semilla + r)
        filas = []
        horas_cola = []
        cola_t = []

        for d, dia in enumerate(DIAS):
            llegadas = generar_llegadas_escenario2(
                DEMANDA[d], modo, rng
            )

            tasa = tasa_domingo if d == 0 else tasa_base
            # Para mantener consistencia con la simulación original,
            # la tasa efectiva representa la capacidad agregada.
            esperas = []
            disponible = 0.0
            cola_eventos = []

            for llegada in llegadas:
                servicio = rng.exponential(60 / tasa)
                inicio = max(llegada, disponible)
                salida = inicio + servicio
                esperas.append(inicio - llegada)
                disponible = salida
                cola_eventos.append(max(0, np.searchsorted(llegadas, llegada) -
                                         np.searchsorted(llegadas, max(0, inicio))))

            # Pacientes que no terminaron dentro de las 23 h.
            pendientes = sum(
                1 for llegada in llegadas
                if max(llegada, 0) + (60 / tasa) > 23 * 60
            )

            filas.append({
                "Día": dia,
                "Espera promedio (min)": np.mean(esperas),
                "P90 espera (min)": np.percentile(esperas, 90),
                "Cola máxima": max(cola_eventos) if cola_eventos else 0,
                "Pacientes pendientes": pendientes
            })

            if r == 0:
                for h, q in zip(llegadas / 60, cola_eventos):
                    horas_cola.append(d * 23 + h)
                    cola_t.append(q)

        df = pd.DataFrame(filas)
        df["Réplica"] = r + 1
        registros.append(df)
        globales.append({
            "Réplica": r + 1,
            "Espera semanal promedio (min)": df["Espera promedio (min)"].mean(),
            "P90 promedio (min)": df["P90 espera (min)"].mean(),
            "Cola máxima semanal": df["Cola máxima"].max(),
            "Pacientes pendientes": df["Pacientes pendientes"].sum()
        })

        if r == 0:
            trayectoria = pd.DataFrame({"Hora": horas_cola, "Cola": cola_t})

    detalle = pd.concat(registros, ignore_index=True)
    resumen = (
        detalle.groupby("Día", sort=False)
        .agg({
            "Espera promedio (min)": "mean",
            "P90 espera (min)": "mean",
            "Cola máxima": "mean",
            "Pacientes pendientes": "mean"
        })
        .reset_index()
    )

    return {
        "detalle": detalle,
        "resumen": resumen,
        "global": pd.DataFrame(globales),
        "trayectoria": trayectoria
    }


# ============================================================
# EXCEL DE SALIDA
# ============================================================

def crear_excel_salida(
    tabla_actual,
    tabla_propuesta,
    movimientos,
    sim_actual=None,
    sim_propuesta=None
):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        tabla_actual.to_excel(
            writer, sheet_name="Actual_M_M_1", index=False
        )
        tabla_propuesta.to_excel(
            writer, sheet_name="Escenario1_M_M_1", index=False
        )
        movimientos.to_excel(
            writer, sheet_name="Traslados", index=False
        )

        if sim_actual is not None:
            sim_actual["resumen"].to_excel(
                writer, sheet_name="Sim_Actual", index=False
            )

        if sim_propuesta is not None:
            sim_propuesta["resumen"].to_excel(
                writer, sheet_name="Sim_Escenario1", index=False
            )

    output.seek(0)
    return output.getvalue()


# ============================================================
# INTERFAZ
# ============================================================

st.title("🏥 Hospital Gloria — Simulador de capacidad")
st.caption(
    "Escenario 1 y Escenario 2 se modelan y simulan de forma independiente"
)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.header("Parámetros")

    horas_dia = st.number_input(
        "Horas del día para λ y μ",
        min_value=1.0,
        max_value=24.0,
        value=24.0,
        step=1.0,
        help=(
            "El planteamiento entregado para Escenario 1 utiliza "
            "lambda = demanda / 24. Puedes cambiarlo para análisis "
            "de sensibilidad."
        )
    )

    st.subheader("Límites de traslado — Escenario 1")

    max_s_fri = st.number_input(
        "Máx. Sénior: viernes → destinos",
        min_value=0.0,
        max_value=float(S_BASE[5]),
        value=float(S_BASE[5]),
        step=1.0
    )

    max_s_sab = st.number_input(
        "Máx. Sénior: sábado → destinos",
        min_value=0.0,
        max_value=float(S_BASE[6]),
        value=float(S_BASE[6]),
        step=1.0
    )

    max_j_fri = st.number_input(
        "Máx. Júnior: viernes → destinos",
        min_value=0.0,
        max_value=float(J_BASE[5]),
        value=float(J_BASE[5]),
        step=1.0
    )

    max_j_sab = st.number_input(
        "Máx. Júnior: sábado → destinos",
        min_value=0.0,
        max_value=float(J_BASE[6]),
        value=float(J_BASE[6]),
        step=1.0
    )

    st.subheader("Simulación")

    replicas = st.slider(
        "Réplicas",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )

    warmup = st.number_input(
        "Calentamiento (horas)",
        min_value=0.0,
        max_value=72.0,
        value=0.0,
        step=1.0,
        help=(
            "Si se usa 0, se observan las 7 jornadas completas desde "
            "el inicio. Si se usa un calentamiento > 0, las métricas "
            "anteriores al calentamiento no se reportan."
        )
    )

    semilla = st.number_input(
        "Semilla reproducible",
        min_value=1,
        max_value=999999,
        value=42,
        step=1
    )

# ------------------------------------------------------------
# Información base
# ------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Demanda semanal", f"{TOTAL_DEMANDA:,.0f}")
c2.metric("Horas Sénior", f"{TOTAL_SENIOR:,.0f}")
c3.metric("Horas Júnior", f"{TOTAL_JUNIOR:,.0f}")
c4.metric("Horas médicas", f"{TOTAL_HOURS:,.0f}")

st.markdown(
    f"""
**Tasas de servicio utilizadas**

- Sénior: **{RATE_SENIOR:.9f} pacientes/hora**
- Júnior: **{RATE_JUNIOR:.9f} pacientes/hora**
- Demanda: **{TOTAL_DEMANDA:.0f} pacientes/semana**
- Horas semanales: **{TOTAL_HOURS:.0f}**
"""
)

# ============================================================
# TABS
# ============================================================

tab_datos, tab_esc1, tab_esc2, tab_sim = st.tabs([
    "📊 Datos y M/M/1",
    "🔵 Escenario 1 — Rebalanceo",
    "🟢 Escenario 2 — Mix domingo",
    "🎲 Simulación Escenario 1"
])

# ============================================================
# TAB 1 — DATOS Y M/M/1
# ============================================================

with tab_datos:
    st.header("Situación actual")

    tabla_actual = calcular_capacidad(
        S_BASE,
        J_BASE,
        horas_dia=horas_dia
    )

    st.dataframe(
        tabla_actual.style.format({
            "Sénior (h)": "{:.2f}",
            "Júnior (h)": "{:.2f}",
            "Demanda (pacientes)": "{:.0f}",
            "Capacidad (pacientes/día)": "{:.2f}",
            "Lambda (pac/h)": "{:.4f}",
            "Mu (pac/h)": "{:.4f}",
            "Utilización": "{:.2%}",
            "Wq (min)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Demanda vs capacidad")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(7)
    width = 0.36

    ax.bar(
        x - width/2,
        DEMANDA,
        width,
        label="Demanda"
    )
    ax.bar(
        x + width/2,
        tabla_actual["Capacidad (pacientes/día)"],
        width,
        label="Capacidad"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(DIAS)
    ax.set_ylabel("Pacientes por día")
    ax.set_title("Situación actual: demanda vs capacidad")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(fig, use_container_width=True)

    st.info(
        "El Excel original utiliza una capacidad constante aproximada "
        "de 27,02 pacientes/hora basada en 11 médicos promedio. "
        "Esa capacidad fija sirve como referencia, pero no representa "
        "correctamente una redistribución de horas entre días: si movemos "
        "horas del viernes/sábado al domingo/lunes, la capacidad diaria "
        "debe cambiar con las horas realmente asignadas a cada día."
    )

# ============================================================
# TAB 2 — ESCENARIO 1
# ============================================================

with tab_esc1:
    st.header("Escenario 1 — Rebalanceo de horas médicas")

    st.write(
        "El modelo permite mover horas únicamente desde viernes y sábado "
        "hacia domingo y lunes. Martes, miércoles y jueves permanecen "
        "sin cambios."
    )

    resultado, rutas, estado = optimizar_escenario1(
        max_s_fri,
        max_s_sab,
        max_j_fri,
        max_j_sab
    )

    if resultado is None:
        st.error(
            "Con los límites seleccionados no existe una solución factible."
        )
    else:
        S_new = resultado["S_new"]
        J_new = resultado["J_new"]

        tabla_prop = calcular_capacidad(
            S_new,
            J_new,
            horas_dia=horas_dia
        )

        # Tabla comparativa de horas
        comparacion_horas = pd.DataFrame({
            "Día": DIAS,
            "Sénior actual": S_BASE,
            "Sénior propuesta": S_new,
            "Júnior actual": J_BASE,
            "Júnior propuesta": J_new
        })

        st.subheader("Solución obtenida por programación lineal")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Cobertura mínima z",
            f"{resultado['z']:.6f}"
        )
        m2.metric(
            "Máx. utilización equivalente",
            f"{1/resultado['z']:.2%}"
        )
        m3.metric(
            "Horas trasladadas",
            f"{resultado['total_trasladado']:.2f}"
        )

        st.dataframe(
            comparacion_horas.style.format({
                "Sénior actual": "{:.2f}",
                "Sénior propuesta": "{:.2f}",
                "Júnior actual": "{:.2f}",
                "Júnior propuesta": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Traslados")

        movimientos = resultado["movimientos"].copy()
        movimientos["Horas trasladadas"] = movimientos[
            "Horas trasladadas"
        ].round(4)

        movimientos_mostrar = movimientos[
            movimientos["Horas trasladadas"] > 1e-8
        ]

        if len(movimientos_mostrar):
            st.dataframe(
                movimientos_mostrar,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(
                "Con los límites seleccionados, el modelo no necesita "
                "realizar traslados."
            )

        st.subheader("M/M/1 — Actual vs Escenario 1")

        tabla_comparativa = pd.DataFrame({
            "Día": DIAS,
            "Utilización actual": tabla_actual["Utilización"],
            "Utilización escenario 1": tabla_prop["Utilización"],
            "Wq actual (min)": tabla_actual["Wq (min)"],
            "Wq escenario 1 (min)": tabla_prop["Wq (min)"],
            "Capacidad actual": tabla_actual[
                "Capacidad (pacientes/día)"
            ],
            "Capacidad escenario 1": tabla_prop[
                "Capacidad (pacientes/día)"
            ]
        })

        st.dataframe(
            tabla_comparativa.style.format({
                "Utilización actual": "{:.2%}",
                "Utilización escenario 1": "{:.2%}",
                "Wq actual (min)": "{:.2f}",
                "Wq escenario 1 (min)": "{:.2f}",
                "Capacidad actual": "{:.2f}",
                "Capacidad escenario 1": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        # Gráfico de utilización
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(
            DIAS,
            tabla_actual["Utilización"] * 100,
            marker="o",
            label="Actual"
        )
        ax.plot(
            DIAS,
            tabla_prop["Utilización"] * 100,
            marker="o",
            label="Escenario 1"
        )
        ax.set_ylabel("Utilización (%)")
        ax.set_title("Utilización diaria — Actual vs Escenario 1")
        ax.legend()
        ax.grid(alpha=0.25)

        st.pyplot(fig, use_container_width=True)

        # Gráfico de horas
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(7)
        width = 0.18

        ax.bar(x - 1.5*width, S_BASE, width, label="Sénior actual")
        ax.bar(x - 0.5*width, S_new, width, label="Sénior escenario 1")
        ax.bar(x + 0.5*width, J_BASE, width, label="Júnior actual")
        ax.bar(x + 1.5*width, J_new, width, label="Júnior escenario 1")

        ax.set_xticks(x)
        ax.set_xticklabels(DIAS)
        ax.set_ylabel("Horas-médico")
        ax.set_title("Horas médicas antes y después del rebalanceo")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig, use_container_width=True)

        # Validaciones
        st.subheader("Validaciones")

        horas_ok = (
            abs(S_new.sum() - TOTAL_SENIOR) < 1e-7
            and abs(J_new.sum() - TOTAL_JUNIOR) < 1e-7
        )

        dias_fijos_ok = all(
            abs(S_new[d] - S_BASE[d]) < 1e-7
            and abs(J_new[d] - J_BASE[d]) < 1e-7
            for d in [2, 3, 4]
        )

        no_neg_ok = np.all(S_new >= -1e-8) and np.all(J_new >= -1e-8)

        limite_ok = (
            S_BASE[5] - S_new[5] <= max_s_fri + 1e-7
            and S_BASE[6] - S_new[6] <= max_s_sab + 1e-7
            and J_BASE[5] - J_new[5] <= max_j_fri + 1e-7
            and J_BASE[6] - J_new[6] <= max_j_sab + 1e-7
        )

        z_ok = np.all(
            tabla_prop["Capacidad (pacientes/día)"].values
            + 1e-7
            >= resultado["z"] * DEMANDA
        )

        v1, v2, v3, v4, v5 = st.columns(5)

        v1.success("Horas conservadas" if horas_ok else "Error")
        v2.success("Días fijos" if dias_fijos_ok else "Error")
        v3.success("Sin negativos" if no_neg_ok else "Error")
        v4.success("Límites respetados" if limite_ok else "Error")
        v5.success("Cobertura válida" if z_ok else "Error")

        st.caption(
            "Importante: la solución LP entrega horas continuas. "
            "Antes de implementarlas en el hospital deben convertirse "
            "en turnos realizables. Conservar horas no demuestra por sí "
            "solo conservar costos."
        )

        st.download_button(
            "⬇️ Descargar resultados del Escenario 1",
            data=crear_excel_salida(
                tabla_actual,
                tabla_prop,
                movimientos
            ),
            file_name="hospital_gloria_escenario1.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            )
        )

# ============================================================
# TAB 3 — ESCENARIO 2
# ============================================================

with tab_esc2:
    st.header("Escenario 2 — Reestructuración del mix Sénior/Júnior")

    st.write(
        "Este escenario modifica únicamente la composición del domingo. "
        "Las 264 horas-médico del domingo se mantienen constantes."
    )

    horas_senior_nuevo = st.slider(
        "Horas Sénior del domingo",
        min_value=60,
        max_value=264,
        value=84,
        step=1
    )

    horas_junior_nuevo = 264 - horas_senior_nuevo

    mejora_eficiencia = st.slider(
        "Mejora de la tasa efectiva del domingo (%)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
        help=(
            "Supuesto del equipo. El caso entrega una tasa promedio "
            "global y no una tasa específica para cada mix."
        )
    )

    mix_actual = S_BASE[0] / 264
    mix_nuevo = horas_senior_nuevo / 264

    tasa_domingo_nueva = RATE_SENIOR * 0 + (
        (RATE_SENIOR * horas_senior_nuevo
         + RATE_JUNIOR * horas_junior_nuevo)
        / 264
    )

    tasa_domingo_efectiva = tasa_domingo_nueva * (
        1 + mejora_eficiencia / 100
    )

    S_esc2 = S_BASE.copy()
    J_esc2 = J_BASE.copy()
    S_esc2[0] = horas_senior_nuevo
    J_esc2[0] = horas_junior_nuevo

    tabla_esc2 = calcular_capacidad(
        S_esc2,
        J_esc2,
        horas_dia=horas_dia
    )

    # Sustituimos la capacidad del domingo por la tasa efectiva del supuesto
    capacidad_domingo_esc2 = 264 * tasa_domingo_efectiva

    tabla_esc2.loc[0, "Capacidad (pacientes/día)"] = (
        capacidad_domingo_esc2
    )
    tabla_esc2.loc[0, "Mu (pac/h)"] = (
        capacidad_domingo_esc2 / horas_dia
    )
    tabla_esc2.loc[0, "Utilización"] = (
        (DEMANDA[0] / horas_dia)
        / tabla_esc2.loc[0, "Mu (pac/h)"]
    )

    mu2 = tabla_esc2.loc[0, "Mu (pac/h)"]
    lam2 = tabla_esc2.loc[0, "Lambda (pac/h)"]

    if mu2 > lam2:
        tabla_esc2.loc[0, "Wq (min)"] = (
            60 * lam2 / (mu2 * (mu2 - lam2))
        )
        tabla_esc2.loc[0, "Estado"] = "Estable"
    else:
        tabla_esc2.loc[0, "Wq (min)"] = np.nan
        tabla_esc2.loc[0, "Estado"] = "Sin equilibrio estable"

    a, b, c, d = st.columns(4)

    a.metric(
        "Mix Sénior actual",
        f"{mix_actual:.1%}"
    )
    b.metric(
        "Mix Sénior propuesto",
        f"{mix_nuevo:.1%}"
    )
    c.metric(
        "Sénior domingo",
        f"{S_BASE[0]:.0f} → {horas_senior_nuevo:.0f} h"
    )
    d.metric(
        "Júnior domingo",
        f"{J_BASE[0]:.0f} → {horas_junior_nuevo:.0f} h"
    )

    st.info(
        f"Las horas totales del domingo permanecen en 264. "
        f"El supuesto seleccionado aumenta la tasa efectiva del domingo "
        f"hasta {tasa_domingo_efectiva:.4f} pacientes/hora-médico."
    )

    st.dataframe(
        tabla_esc2.style.format({
            "Sénior (h)": "{:.2f}",
            "Júnior (h)": "{:.2f}",
            "Demanda (pacientes)": "{:.0f}",
            "Capacidad (pacientes/día)": "{:.2f}",
            "Lambda (pac/h)": "{:.4f}",
            "Mu (pac/h)": "{:.4f}",
            "Utilización": "{:.2%}",
            "Wq (min)": "{:.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )


    st.warning(
        "La mejora de eficiencia es un supuesto de modelamiento. "
        "No debe presentarse como un dato observado del hospital."
    )

    st.subheader("Simulación independiente del Escenario 2")
    st.caption(
        "Esta simulación NO utiliza el modelo de redistribución del Escenario 1. "
        "Trabaja con sus propios supuestos: 23 h/día, μ base = 2,4 y cambio "
        "únicamente el domingo."
    )

    modo_esc2 = st.radio(
        "Flujo de llegada — Escenario 2",
        ["Estable", "Variable"],
        horizontal=True,
        key="modo_esc2"
    )

    replicas_esc2 = st.slider(
        "Réplicas — Escenario 2",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        key="replicas_esc2"
    )

    semilla_esc2 = st.number_input(
        "Semilla — Escenario 2",
        min_value=1,
        max_value=999999,
        value=42,
        key="semilla_esc2"
    )

    if st.button(
        "🎲 Ejecutar simulación Escenario 2",
        type="primary",
        use_container_width=True
    ):
        with st.spinner("Ejecutando Escenario 2..."):
            sim2_actual = ejecutar_simulacion_escenario2(
                S_BASE[0],
                0,
                modo=modo_esc2,
                replicas=replicas_esc2,
                semilla=int(semilla_esc2)
            )
            sim2_propuesto = ejecutar_simulacion_escenario2(
                horas_senior_nuevo,
                mejora_eficiencia,
                modo=modo_esc2,
                replicas=replicas_esc2,
                semilla=int(semilla_esc2)
            )

        st.session_state["sim2_actual"] = sim2_actual
        st.session_state["sim2_propuesto"] = sim2_propuesto

    if "sim2_actual" in st.session_state and "sim2_propuesto" in st.session_state:
        a2 = st.session_state["sim2_actual"]
        b2 = st.session_state["sim2_propuesto"]

        ga = a2["global"]
        gb = b2["global"]

        st.subheader("Resultados — Escenario 2")
        resumen2 = pd.DataFrame({
            "Indicador": [
                "Espera semanal promedio (min)",
                "P90 promedio (min)",
                "Cola máxima semanal",
                "Pacientes pendientes"
            ],
            "Actual": [
                ga["Espera semanal promedio (min)"].mean(),
                ga["P90 promedio (min)"].mean(),
                ga["Cola máxima semanal"].mean(),
                ga["Pacientes pendientes"].mean()
            ],
            "Escenario 2": [
                gb["Espera semanal promedio (min)"].mean(),
                gb["P90 promedio (min)"].mean(),
                gb["Cola máxima semanal"].mean(),
                gb["Pacientes pendientes"].mean()
            ]
        })

        st.dataframe(
            resumen2.style.format({
                "Actual": "{:.2f}",
                "Escenario 2": "{:.2f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        diario2 = pd.DataFrame({
            "Día": DIAS,
            "Espera actual (min)": a2["resumen"]["Espera promedio (min)"],
            "Espera Esc. 2 (min)": b2["resumen"]["Espera promedio (min)"],
            "P90 actual (min)": a2["resumen"]["P90 espera (min)"],
            "P90 Esc. 2 (min)": b2["resumen"]["P90 espera (min)"],
        })

        st.dataframe(
            diario2.style.format({c: "{:.2f}" for c in diario2.columns if c != "Día"}),
            use_container_width=True,
            hide_index=True
        )

        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(7)
        width = 0.36
        ax.bar(x - width/2, diario2["Espera actual (min)"], width, label="Actual")
        ax.bar(x + width/2, diario2["Espera Esc. 2 (min)"], width, label="Escenario 2")
        ax.set_xticks(x)
        ax.set_xticklabels(DIAS)
        ax.set_ylabel("Minutos")
        ax.set_title("Escenario 2 — Espera promedio por día")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig, use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.step(a2["trayectoria"]["Hora"], a2["trayectoria"]["Cola"], where="post", label="Actual")
        ax.step(b2["trayectoria"]["Hora"], b2["trayectoria"]["Cola"], where="post", label="Escenario 2")
        for d in range(1, 7):
            ax.axvline(d * 23, linestyle="--", alpha=0.35)
        ax.set_xlabel("Hora de operación acumulada (23 h/día)")
        ax.set_ylabel("Pacientes en cola")
        ax.set_title("Escenario 2 — Evolución simulada de la cola")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig, use_container_width=True)

        st.caption(
            "El Escenario 2 se simula por separado del Escenario 1. "
            "No se utilizan las tasas Sénior/Júnior ni la optimización de rebalanceo "
            "del Escenario 1."
        )

# ============================================================
# TAB 4 — SIMULACIÓN
# ============================================================

with tab_sim:
    st.header("Simulación semanal — comparación pareada")

    st.write(
        "Se generan las mismas llegadas Poisson y los mismos requerimientos "
        "de servicio para Actual y Escenario 1. Solo cambia la capacidad "
        "diaria. Así la comparación es más limpia."
    )

    resultado, rutas, estado = optimizar_escenario1(
        max_s_fri,
        max_s_sab,
        max_j_fri,
        max_j_sab
    )

    if resultado is None:
        st.error(
            "No existe una solución factible con los límites seleccionados."
        )
    else:
        if st.button(
            "🎲 Ejecutar simulación Actual vs Escenario 1",
            type="primary",
            use_container_width=True
        ):
            with st.spinner(
                f"Ejecutando {replicas} réplicas pareadas..."
            ):
                sim_actual = ejecutar_simulacion_escenario1(
                    S_BASE,
                    J_BASE,
                    horas_dia=horas_dia,
                    replicas=replicas,
                    warmup_horas=warmup,
                    semilla=int(semilla)
                )

                sim_esc1 = ejecutar_simulacion_escenario1(
                    resultado["S_new"],
                    resultado["J_new"],
                    horas_dia=horas_dia,
                    replicas=replicas,
                    warmup_horas=warmup,
                    semilla=int(semilla)
                )

            st.session_state["sim_actual"] = sim_actual
            st.session_state["sim_esc1"] = sim_esc1

        if (
            "sim_actual" in st.session_state
            and "sim_esc1" in st.session_state
        ):
            sim_actual = st.session_state["sim_actual"]
            sim_esc1 = st.session_state["sim_esc1"]

            comparacion = comparar_simulaciones(
                sim_actual,
                sim_esc1
            )

            st.subheader("Resultados principales")

            st.dataframe(
                comparacion.style.format({
                    "Situación actual": "{:.2f}",
                    "Escenario 1": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            # Comparación diaria
            r_a = sim_actual["resumen"].copy()
            r_b = sim_esc1["resumen"].copy()

            diario = pd.DataFrame({
                "Día": DIAS,
                "Espera actual (min)": r_a[
                    "Espera promedio (min)"
                ],
                "Espera Esc. 1 (min)": r_b[
                    "Espera promedio (min)"
                ],
                "P90 actual (min)": r_a[
                    "P90 espera (min)"
                ],
                "P90 Esc. 1 (min)": r_b[
                    "P90 espera (min)"
                ],
                "Cola máxima actual": r_a["Cola máxima"],
                "Cola máxima Esc. 1": r_b["Cola máxima"]
            })

            st.subheader("Resultados diarios")

            st.dataframe(
                diario.style.format({
                    "Espera actual (min)": "{:.2f}",
                    "Espera Esc. 1 (min)": "{:.2f}",
                    "P90 actual (min)": "{:.2f}",
                    "P90 Esc. 1 (min)": "{:.2f}",
                    "Cola máxima actual": "{:.2f}",
                    "Cola máxima Esc. 1": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            # Gráfico espera
            fig, ax = plt.subplots(figsize=(11, 5))
            x = np.arange(7)
            width = 0.36

            ax.bar(
                x - width/2,
                diario["Espera actual (min)"],
                width,
                label="Actual"
            )
            ax.bar(
                x + width/2,
                diario["Espera Esc. 1 (min)"],
                width,
                label="Escenario 1"
            )

            ax.set_xticks(x)
            ax.set_xticklabels(DIAS)
            ax.set_ylabel("Minutos")
            ax.set_title(
                "Simulación: espera promedio por día"
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.25)

            st.pyplot(fig, use_container_width=True)

            # Trayectoria
            st.subheader(
                "Evolución simulada de la cola — primera réplica"
            )

            ta = sim_actual["trayectoria"]
            tb = sim_esc1["trayectoria"]

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.step(
                ta["Hora"],
                ta["Cola"],
                where="post",
                label="Actual"
            )
            ax.step(
                tb["Hora"],
                tb["Cola"],
                where="post",
                label="Escenario 1"
            )

            for d in range(1, 7):
                ax.axvline(
                    d * horas_dia,
                    linestyle="--",
                    alpha=0.35
                )

            ax.set_xlabel("Hora de la semana")
            ax.set_ylabel("Pacientes en cola")
            ax.set_title(
                "Evolución simulada de la cola"
            )
            ax.legend()
            ax.grid(alpha=0.25)

            st.pyplot(fig, use_container_width=True)

            st.caption(
                "La trayectoria corresponde a una realización simulada. "
                "No representa pacientes individuales reales ni prioridades "
                "clínicas. El servidor es equivalente y agregado."
            )

            # Incertidumbre
            st.subheader("Incertidumbre entre réplicas")

            ga = sim_actual["global"]
            gb = sim_esc1["global"]

            ma, ia = intervalo_confianza_95(
                ga["Espera semanal promedio (min)"]
            )
            mb, ib = intervalo_confianza_95(
                gb["Espera semanal promedio (min)"]
            )

            st.write(
                f"**Actual:** {ma:.2f} ± {ia:.2f} min "
                f"(IC95% entre réplicas)"
            )
            st.write(
                f"**Escenario 1:** {mb:.2f} ± {ib:.2f} min "
                f"(IC95% entre réplicas)"
            )

            st.info(
                "Una simulación semanal con capacidad variable puede "
                "diferir de las esperas estacionarias diarias de M/M/1 "
                "porque la cola se arrastra de un día al siguiente, mientras "
                "que M/M/1 supone condiciones estacionarias y una tasa de "
                "llegada y servicio constantes."
            )

            st.download_button(
                "⬇️ Descargar resultados de la simulación",
                data=crear_excel_salida(
                    calcular_capacidad(
                        S_BASE, J_BASE, horas_dia=horas_dia
                    ),
                    calcular_capacidad(
                        resultado["S_new"],
                        resultado["J_new"],
                        horas_dia=horas_dia
                    ),
                    resultado["movimientos"],
                    sim_actual,
                    sim_esc1
                ),
                file_name="hospital_gloria_simulacion.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                )
            )


st.divider()

st.caption(
    "Modelo académico basado en los datos proporcionados para el caso "
    "Hospital Gloria. Las decisiones de turnos, mix y mejoras de eficiencia "
    "deben validarse operativamente antes de implementarse."
)