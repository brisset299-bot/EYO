import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# HOSPITAL GLORIA
# Simulador de 3 escenarios independientes
#
# ESCENARIO 1:
# Rebalanceo de horas médicas entre días.
# Modelo LP + M/M/1 + simulación.
#
# ESCENARIO 2:
# Reestructuración del mix Sénior/Júnior el domingo.
# Tasa base 2.40 pac/h-médico + simulación.
#
# ESCENARIO 3:
# Reestructuración semanal con horas totales objetivo,
# mix Sénior/Júnior configurable y M/M/c + simulación.
# ============================================================

st.set_page_config(
    page_title="Simulador dinámico — Hospital Gloria",
    page_icon="🏥",
    layout="wide",
)

# ============================================================
# DATOS COMUNES
# ============================================================

DIAS = [
    "Domingo", "Lunes", "Martes", "Miércoles",
    "Jueves", "Viernes", "Sábado"
]

DEMANDA = np.array([582, 580, 565, 548, 525, 506, 508], dtype=float)

# Datos base de horas
S_BASE = np.array([60, 114, 124, 97, 70, 93, 89], dtype=float)
J_BASE = np.array([204, 164, 139, 164, 182, 154, 182], dtype=float)
H_BASE = S_BASE + J_BASE

TOTAL_SENIOR = S_BASE.sum()   # 647
TOTAL_JUNIOR = J_BASE.sum()   # 1189
TOTAL_HORAS = H_BASE.sum()     # 1836
TOTAL_DEMANDA = DEMANDA.sum() # 3814

# Tasas indicadas en Queues MM1 para Escenario 1
RATE_SENIOR = 1959.62 / 647
RATE_JUNIOR = 2354.81 / 1189

# Tasa utilizada por Escenario 2 y como tasa global del Escenario 3
RATE_BASE = 2.40

# Horas del Escenario 3 proporcionadas por el equipo
ESC3_HORAS_DEFAULT = np.array([
    248.00, 247.15, 240.75, 233.51,
    223.71, 215.61, 427.27
], dtype=float)

# ============================================================
# FUNCIONES GENERALES
# ============================================================

def formato_estado(util):
    if np.isnan(util):
        return "Sin capacidad"
    if util < 1:
        return "Estable"
    return "Sin equilibrio estable"


def capacidad_por_dia(S, J, tasa_s, tasa_j):
    return tasa_s * S + tasa_j * J


# ============================================================
# M/M/1 — ESCENARIOS 1 Y 2
# ============================================================

def mm1_table(
    S,
    J,
    demanda,
    horas_dia,
    tasa_s,
    tasa_j,
    horas_lambda=None,
):
    """
    M/M/1 con servidor equivalente.

    Lambda = demanda / horas_lambda
    Mu = capacidad / horas_dia

    Si mu <= lambda, Wq no está definido como sistema estable.
    """
    if horas_lambda is None:
        horas_lambda = horas_dia

    capacidad = capacidad_por_dia(S, J, tasa_s, tasa_j)

    lam = demanda / horas_lambda
    mu = capacidad / horas_dia

    util = np.divide(
        lam,
        mu,
        out=np.full_like(lam, np.nan),
        where=mu > 0,
    )

    wq = np.full_like(lam, np.nan)
    estable = mu > lam

    wq[estable] = (
        60.0
        * lam[estable]
        / (
            mu[estable]
            * (mu[estable] - lam[estable])
        )
    )

    w = np.full_like(lam, np.nan)
    w[estable] = 60.0 / (mu[estable] - lam[estable])

    return pd.DataFrame({
        "Día": DIAS,
        "Sénior (h)": S,
        "Júnior (h)": J,
        "Horas médico": S + J,
        "Pacientes": demanda,
        "Tasa servicio": capacidad / (S + J),
        "Capacidad": capacidad,
        "Lambda (pac/h)": lam,
        "Mu (pac/h)": mu,
        "Utilización": util,
        "Wq (min)": wq,
        "W (min)": w,
        "Estado": [formato_estado(x) for x in util],
    })


# ============================================================
# ERLANG-C / M/M/C — ESCENARIO 3
# ============================================================

def erlang_c_mm_c(lam, mu, c):
    """
    Erlang-C para c servidores enteros.

    El Escenario 3 entrega c = horas/23 como "canales equivalentes",
    que puede ser fraccionario. Erlang-C clásico requiere c entero.
    Por transparencia, usamos c_operativo = ceil(c_equivalente).

    Devuelve:
        P(wait), Wq horas, W horas, rho
    """
    c = int(max(1, math.ceil(c)))

    if mu <= 0:
        return np.nan, np.nan, np.nan, np.nan

    rho = lam / (c * mu)

    if rho >= 1:
        return np.nan, np.nan, np.nan, rho

    # Suma de Erlang
    suma = 0.0
    a = lam / mu

    for n in range(c):
        suma += (a ** n) / math.factorial(n)

    ultimo = (a ** c) / math.factorial(c)
    p0 = 1.0 / (suma + ultimo / (1 - rho))
    pw = ultimo * p0 / (1 - rho)

    wq_h = pw / (c * mu - lam)
    w_h = wq_h + 1 / mu

    return pw, wq_h, w_h, rho


def mmc_table(
    horas,
    demanda,
    horas_dia,
    tasa_servicio,
    mix_senior=0.35,
):
    """
    M/M/c agregado para Escenario 3.

    c_equivalente = horas médicas / horas del día.
    Como Erlang-C requiere c entero, c_operativo = ceil(c_equivalente).

    La tasa de servicio por canal es la tasa global configurable.
    """
    lam = demanda / horas_dia
    c_equiv = horas / horas_dia
    c_op = np.ceil(c_equiv).astype(int)
    mu = np.full(7, tasa_servicio, dtype=float)

    capacidad = c_equiv * tasa_servicio
    util_equiv = np.divide(
        lam,
        capacidad / horas_dia,
        out=np.full(7, np.nan),
        where=capacidad > 0,
    )

    pwait = np.full(7, np.nan)
    wq_min = np.full(7, np.nan)
    w_min = np.full(7, np.nan)
    rho_mmc = np.full(7, np.nan)

    for d in range(7):
        pw, wqh, wh, rho = erlang_c_mm_c(
            lam[d],
            tasa_servicio,
            c_op[d],
        )
        pwait[d] = pw
        rho_mmc[d] = rho

        if not np.isnan(wqh):
            wq_min[d] = wqh * 60
            w_min[d] = wh * 60

    senior = horas * mix_senior
    junior = horas * (1 - mix_senior)

    return pd.DataFrame({
        "Día": DIAS,
        "Sénior (h)": senior,
        "Júnior (h)": junior,
        "Horas médico": horas,
        "Pacientes": demanda,
        "Tasa servicio": tasa_servicio,
        "Capacidad": capacidad,
        "Lambda (pac/h)": lam,
        "c equivalente": c_equiv,
        "c operativo Erlang-C": c_op,
        "Utilización": util_equiv,
        "P(wait)": pwait,
        "Wq (min)": wq_min,
        "W (min)": w_min,
        "Estado": [
            "Estable" if not np.isnan(x) and x < 1
            else "Sin equilibrio estable"
            for x in util_equiv
        ],
    })


# ============================================================
# OPTIMIZACIÓN — ESCENARIO 1
# ============================================================

def optimizar_escenario1(
    max_s_fri,
    max_s_sab,
    max_j_fri,
    max_j_sab,
):
    """
    LP de dos etapas.

    Variables de transferencia:
      0 S viernes -> domingo
      1 S viernes -> lunes
      2 S sábado  -> domingo
      3 S sábado  -> lunes
      4 J viernes -> domingo
      5 J viernes -> lunes
      6 J sábado  -> domingo
      7 J sábado  -> lunes
      8 z

    Etapa 1:
      maximizar z = mínima cobertura diaria.

    Etapa 2:
      minimizar horas trasladadas manteniendo z óptimo.

    Se implementa con scipy.optimize.linprog.
    """
    from scipy.optimize import linprog

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
        max_s_fri, max_s_fri,
        max_s_sab, max_s_sab,
        max_j_fri, max_j_fri,
        max_j_sab, max_j_sab,
    ]

    cap_base = capacidad_por_dia(
        S_BASE, J_BASE, RATE_SENIOR, RATE_JUNIOR
    )

    A_ub = []
    b_ub = []

    for d in range(7):
        row = np.zeros(9)

        # D_d*z - capacidad_nueva_d <= 0
        row[8] = DEMANDA[d]

        for k, (categoria, origen, destino) in enumerate(rutas):
            tasa = RATE_SENIOR if categoria == "Sénior" else RATE_JUNIOR

            efecto = 0.0

            if d == destino:
                efecto += tasa

            if d == origen:
                efecto -= tasa

            row[k] -= efecto

        A_ub.append(row)
        b_ub.append(cap_base[d])

    bounds = [(0, x) for x in limites] + [(0, None)]

    # Etapa 1: max z => min -z
    c1 = np.array([0.0] * 8 + [-1.0])

    res1 = linprog(
        c1,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds,
        method="highs",
    )

    if not res1.success:
        return None

    z_opt = res1.x[8]

    # Etapa 2
    bounds2 = [(0, x) for x in limites] + [(z_opt, z_opt)]
    c2 = np.array([1.0] * 8 + [0.0])

    res2 = linprog(
        c2,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        bounds=bounds2,
        method="highs",
    )

    if not res2.success:
        return None

    x = res2.x

    S_new = S_BASE.copy()
    J_new = J_BASE.copy()

    for k, (categoria, origen, destino) in enumerate(rutas):
        h = x[k]

        if categoria == "Sénior":
            S_new[origen] -= h
            S_new[destino] += h
        else:
            J_new[origen] -= h
            J_new[destino] += h

    movimientos = pd.DataFrame({
        "Categoría": [r[0] for r in rutas],
        "Origen": [DIAS[r[1]] for r in rutas],
        "Destino": [DIAS[r[2]] for r in rutas],
        "Horas trasladadas": x[:8],
    })

    return {
        "S": S_new,
        "J": J_new,
        "z": z_opt,
        "movimientos": movimientos,
        "total_trasladado": x[:8].sum(),
    }


# ============================================================
# SIMULACIÓN: M/M/1 AGREGADO
# ============================================================

def generar_llegadas_poisson(
    demanda_diaria,
    horas_dia,
    rng,
):
    """
    Llegadas Poisson con lambda constante dentro de cada día.
    Devuelve tiempos absolutos en horas desde el inicio de la semana.
    """
    llegadas = []

    for d in range(7):
        lam = demanda_diaria[d] / horas_dia

        t_local = 0.0
        fin = horas_dia

        while True:
            t_local += rng.exponential(1 / lam)

            if t_local >= fin:
                break

            llegadas.append(d * horas_dia + t_local)

    return np.array(llegadas)


def generar_trabajos_exponenciales(
    n,
    rng,
):
    """Trabajo normalizado: Exp(media=1)."""
    return rng.exponential(1.0, size=n)


def simular_servidor_equivalente(
    S,
    J,
    tasa_s,
    tasa_j,
    horas_dia,
    demanda,
    semilla,
    warmup_horas=0.0,
):
    """
    Servidor equivalente de capacidad variable.

    Cada paciente requiere trabajo Exp(1).
    La capacidad del día es:
        tasa_s*S + tasa_j*J

    La capacidad efectiva en trabajo/hora es:
        capacidad_dia / horas_dia

    FIFO y conservación del trabajo restante al cambiar de día.
    """
    rng = np.random.default_rng(semilla)

    llegadas = generar_llegadas_poisson(
        demanda,
        horas_dia,
        rng,
    )

    trabajos = generar_trabajos_exponenciales(
        len(llegadas),
        rng,
    )

    capacidad_dia = capacidad_por_dia(
        S, J, tasa_s, tasa_j
    )

    cola = []

    i = 0
    t = 0.0

    ocupado = False
    arr_servicio = None
    trabajo_restante = None
    inicio_servicio = None

    esperas = []
    sistemas = []

    espera_dia = [[] for _ in range(7)]
    sistema_dia = [[] for _ in range(7)]
    cola_max = np.zeros(7)

    trayectoria_t = [0.0]
    trayectoria_q = [0]

    horizonte = 7 * horas_dia

    while t < horizonte - 1e-12:
        dia = min(int(t // horas_dia), 6)
        fin_dia = (dia + 1) * horas_dia

        capacidad_h = capacidad_dia[dia] / horas_dia

        if capacidad_h <= 0:
            break

        prox_llegada = (
            llegadas[i]
            if i < len(llegadas)
            else math.inf
        )

        if ocupado:
            prox_salida = (
                t + trabajo_restante / capacidad_h
            )
        else:
            prox_salida = math.inf

        prox_evento = min(
            prox_llegada,
            prox_salida,
            fin_dia,
            horizonte,
        )

        dt = prox_evento - t

        if ocupado:
            trabajo_restante -= capacidad_h * dt
            trabajo_restante = max(
                trabajo_restante,
                0.0
            )

        t = prox_evento

        if t >= horizonte - 1e-12:
            break

        # Cambio de día: conservar cola y trabajo restante.
        if abs(t - fin_dia) < 1e-10:
            trayectoria_t.append(t)
            trayectoria_q.append(len(cola))
            continue

        # Salida
        if prox_salida <= prox_llegada + 1e-12:
            dia_paciente = min(
                int(arr_servicio // horas_dia),
                6,
            )

            espera = inicio_servicio - arr_servicio
            sistema = t - arr_servicio

            if arr_servicio >= warmup_horas:
                esperas.append(espera)
                sistemas.append(sistema)
                espera_dia[dia_paciente].append(espera)
                sistema_dia[dia_paciente].append(sistema)

            ocupado = False
            arr_servicio = None
            trabajo_restante = None
            inicio_servicio = None

            if cola:
                arr_servicio, trabajo_restante = cola.pop(0)
                inicio_servicio = t
                ocupado = True

        # Llegada
        else:
            arr = llegadas[i]
            trabajo = trabajos[i]
            i += 1

            if ocupado:
                cola.append((arr, trabajo))
            else:
                ocupado = True
                arr_servicio = arr
                trabajo_restante = trabajo
                inicio_servicio = t

        cola_max[dia] = max(
            cola_max[dia],
            len(cola),
        )

        trayectoria_t.append(t)
        trayectoria_q.append(len(cola))

    pendientes = len(cola) + (1 if ocupado else 0)

    filas = []

    for d in range(7):
        e = np.array(
            espera_dia[d],
            dtype=float,
        )

        s = np.array(
            sistema_dia[d],
            dtype=float,
        )

        filas.append({
            "Día": DIAS[d],
            "Espera promedio (min)": (
                e.mean() * 60 if len(e) else np.nan
            ),
            "P90 espera (min)": (
                np.percentile(e, 90) * 60
                if len(e) else np.nan
            ),
            "Sistema promedio (min)": (
                s.mean() * 60 if len(s) else np.nan
            ),
            "Cola máxima": int(cola_max[d]),
        })

    detalle = pd.DataFrame(filas)

    trayectoria = pd.DataFrame({
        "Hora": trayectoria_t,
        "Cola": trayectoria_q,
    })

    return detalle, pendientes, trayectoria


def ejecutar_simulacion_mm1(
    S,
    J,
    tasa_s,
    tasa_j,
    horas_dia,
    replicas,
    warmup,
    semilla,
):
    registros = []
    globales = []
    trayectoria_primera = None

    for r in range(replicas):
        detalle, pendientes, trayectoria = (
            simular_servidor_equivalente(
                S,
                J,
                tasa_s,
                tasa_j,
                horas_dia,
                DEMANDA,
                semilla + r,
                warmup,
            )
        )

        detalle["Réplica"] = r + 1
        registros.append(detalle)

        globales.append({
            "Réplica": r + 1,
            "Espera promedio semanal (min)": (
                detalle["Espera promedio (min)"].mean()
            ),
            "P90 promedio diario (min)": (
                detalle["P90 espera (min)"].mean()
            ),
            "Cola máxima semanal": (
                detalle["Cola máxima"].max()
            ),
            "Pacientes pendientes": pendientes,
        })

        if r == 0:
            trayectoria_primera = trayectoria

    detalle_rep = pd.concat(
        registros,
        ignore_index=True,
    )

    global_rep = pd.DataFrame(globales)

    resumen = (
        detalle_rep
        .groupby("Día", sort=False)
        .agg({
            "Espera promedio (min)": "mean",
            "P90 espera (min)": "mean",
            "Sistema promedio (min)": "mean",
            "Cola máxima": "mean",
        })
        .reset_index()
    )

    return {
        "resumen": resumen,
        "global": global_rep,
        "trayectoria": trayectoria_primera,
    }


# ============================================================
# SIMULACIÓN M/M/C — ESCENARIO 3
# ============================================================

def simular_mmc_variable(
    horas,
    tasa_servicio,
    horas_dia,
    demanda,
    semilla,
    warmup_horas=0.0,
):
    """
    Simulación FIFO con servidores equivalentes enteros.

    Para hacer compatible la simulación con Erlang-C:
      c_operativo = ceil(horas / horas_dia)

    Todos los servidores tienen tasa tasa_servicio.

    La capacidad no se "arrastra" como trabajo fraccionario entre días;
    la cola sí se conserva al cambiar de día.
    """
    rng = np.random.default_rng(semilla)

    llegadas = []
    for d in range(7):
        lam = demanda[d] / horas_dia
        t_local = 0.0

        while True:
            t_local += rng.exponential(1 / lam)

            if t_local >= horas_dia:
                break

            llegadas.append(
                d * horas_dia + t_local
            )

    llegadas = np.array(llegadas)

    trabajos = rng.exponential(
        1 / tasa_servicio,
        size=len(llegadas),
    )

    c = np.ceil(
        horas / horas_dia
    ).astype(int)

    # Para cada servidor se guarda el instante en que queda libre.
    libres = np.zeros(int(c.max()))

    espera_dia = [[] for _ in range(7)]
    sistema_dia = [[] for _ in range(7)]
    cola_por_dia = np.zeros(7)

    # Cola explícita solo para calcular evolución aproximada.
    # Asignamos cada paciente al servidor que primero queda libre.
    eventos = []

    for idx, llegada in enumerate(llegadas):
        dia = min(int(llegada // horas_dia), 6)
        nser = c[dia]

        # Solo se consideran los nser servidores activos.
        libres_dia = libres[:nser]

        servidor = int(np.argmin(libres_dia))
        inicio = max(
            llegada,
            libres_dia[servidor],
        )

        espera = inicio - llegada
        salida = inicio + trabajos[idx]

        libres[servidor] = salida

        if llegada >= warmup_horas:
            espera_dia[dia].append(
                espera * 60
            )
            sistema_dia[dia].append(
                (salida - llegada) * 60
            )

        # Aproximación de cola en el instante de llegada.
        en_cola = np.sum(
            libres_dia > llegada + 1e-12
        )

        if espera > 0:
            cola_por_dia[dia] = max(
                cola_por_dia[dia],
                en_cola,
            )

        eventos.append({
            "Hora": llegada,
            "Cola": max(0, int(en_cola - nser + 1)),
        })

    pendientes = 0

    # Pacientes cuyo servicio termina después del horizonte.
    horizonte = 7 * horas_dia
    pendientes = int(
        np.sum(
            libres[:int(c.max())] > horizonte
        )
    )

    filas = []

    for d in range(7):
        e = np.array(
            espera_dia[d],
            dtype=float,
        )
        s = np.array(
            sistema_dia[d],
            dtype=float,
        )

        filas.append({
            "Día": DIAS[d],
            "Espera promedio (min)": (
                e.mean() if len(e) else np.nan
            ),
            "P90 espera (min)": (
                np.percentile(e, 90)
                if len(e) else np.nan
            ),
            "Sistema promedio (min)": (
                s.mean() if len(s) else np.nan
            ),
            "Cola máxima": int(
                cola_por_dia[d]
            ),
        })

    detalle = pd.DataFrame(filas)

    trayectoria = pd.DataFrame(eventos)

    if len(trayectoria):
        trayectoria = (
            trayectoria
            .sort_values("Hora")
            .reset_index(drop=True)
        )

    return detalle, pendientes, trayectoria


def ejecutar_simulacion_mmc(
    horas,
    tasa_servicio,
    horas_dia,
    replicas,
    warmup,
    semilla,
):
    registros = []
    globales = []
    trayectoria_primera = None

    for r in range(replicas):
        detalle, pendientes, trayectoria = (
            simular_mmc_variable(
                horas,
                tasa_servicio,
                horas_dia,
                DEMANDA,
                semilla + r,
                warmup,
            )
        )

        detalle["Réplica"] = r + 1
        registros.append(detalle)

        globales.append({
            "Réplica": r + 1,
            "Espera promedio semanal (min)": (
                detalle["Espera promedio (min)"].mean()
            ),
            "P90 promedio diario (min)": (
                detalle["P90 espera (min)"].mean()
            ),
            "Cola máxima semanal": (
                detalle["Cola máxima"].max()
            ),
            "Pacientes pendientes": pendientes,
        })

        if r == 0:
            trayectoria_primera = trayectoria

    detalle_rep = pd.concat(
        registros,
        ignore_index=True,
    )

    global_rep = pd.DataFrame(globales)

    resumen = (
        detalle_rep
        .groupby("Día", sort=False)
        .agg({
            "Espera promedio (min)": "mean",
            "P90 espera (min)": "mean",
            "Sistema promedio (min)": "mean",
            "Cola máxima": "mean",
        })
        .reset_index()
    )

    return {
        "resumen": resumen,
        "global": global_rep,
        "trayectoria": trayectoria_primera,
    }


# ============================================================
# UTILIDADES DE SIMULACIÓN
# ============================================================

def media_ic95(x):
    x = pd.Series(x).dropna().astype(float)

    if len(x) == 0:
        return np.nan, np.nan

    media = x.mean()

    if len(x) == 1:
        return media, np.nan

    se = x.std(ddof=1) / np.sqrt(len(x))
    return media, 1.96 * se


def comparacion_simulaciones(
    sim_a,
    sim_b,
    nombre_a="Actual",
    nombre_b="Propuesta",
):
    ga = sim_a["global"]
    gb = sim_b["global"]

    ma, ia = media_ic95(
        ga["Espera promedio semanal (min)"]
    )
    mb, ib = media_ic95(
        gb["Espera promedio semanal (min)"]
    )

    return pd.DataFrame({
        "Indicador": [
            "Espera semanal promedio (min)",
            "IC95% ± (min)",
            "Cola máxima semanal promedio",
            "Pacientes pendientes promedio",
        ],
        nombre_a: [
            ma,
            ia,
            ga["Cola máxima semanal"].mean(),
            ga["Pacientes pendientes"].mean(),
        ],
        nombre_b: [
            mb,
            ib,
            gb["Cola máxima semanal"].mean(),
            gb["Pacientes pendientes"].mean(),
        ],
    })


# ============================================================
# EXCEL
# ============================================================

def excel_bytes(hojas):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:
        for nombre, df in hojas.items():
            df.to_excel(
                writer,
                sheet_name=nombre[:31],
                index=False,
            )

    output.seek(0)
    return output.getvalue()


# ============================================================
# INTERFAZ
# ============================================================

st.title("🏥 Simulador dinámico — Hospital Gloria")
st.caption(
    "Tres escenarios independientes para analizar capacidad, "
    "utilización, tiempos de espera y comportamiento de la cola."
)

# ============================================================
# SIDEBAR — SELECCIÓN DE ESCENARIO
# ============================================================

with st.sidebar:
    st.header("Escenarios")

    escenario = st.radio(
        "Selecciona el módulo:",
        [
            "Escenario 1 — Rebalanceo semanal",
            "Escenario 2 — Mix del domingo",
            "Escenario 3 — Reestructuración semanal",
        ],
    )

    st.divider()

    st.header("Parámetros de simulación")

    if escenario.startswith("Escenario 1"):
        horas_dia = st.number_input(
            "Horas para λ y operación",
            min_value=1.0,
            max_value=24.0,
            value=24.0,
            step=1.0,
        )

        st.subheader("Límites de traslado")

        max_s_fri = st.number_input(
            "Sénior: máximo desde viernes",
            0.0,
            float(S_BASE[5]),
            float(S_BASE[5]),
            1.0,
        )

        max_s_sab = st.number_input(
            "Sénior: máximo desde sábado",
            0.0,
            float(S_BASE[6]),
            float(S_BASE[6]),
            1.0,
        )

        max_j_fri = st.number_input(
            "Júnior: máximo desde viernes",
            0.0,
            float(J_BASE[5]),
            float(J_BASE[5]),
            1.0,
        )

        max_j_sab = st.number_input(
            "Júnior: máximo desde sábado",
            0.0,
            float(J_BASE[6]),
            float(J_BASE[6]),
            1.0,
        )

        st.subheader("Réplicas")

        replicas = st.slider(
            "Número de réplicas",
            10, 300, 100, 10
        )

        warmup = st.number_input(
            "Calentamiento (h)",
            0.0, 72.0, 0.0, 1.0
        )

        semilla = st.number_input(
            "Semilla",
            1, 999999, 42, 1
        )

    elif escenario.startswith("Escenario 2"):
        tasa_2 = st.number_input(
            "Tasa promedio μ (pac/h-médico)",
            min_value=0.1,
            max_value=10.0,
            value=2.40,
            step=0.05,
        )

        horas_dia_2 = st.number_input(
            "Horas de operación por día",
            min_value=1.0,
            max_value=24.0,
            value=23.0,
            step=1.0,
        )

        st.subheader("Domingo")

        senior_dom = st.slider(
            "Horas Sénior domingo",
            0,
            264,
            84,
            1,
        )

        mejora_dom = st.slider(
            "Mejora adicional de eficiencia (%)",
            0.0,
            30.0,
            5.0,
            0.5,
        )

        st.subheader("Réplicas")

        replicas_2 = st.slider(
            "Número de réplicas",
            10, 300, 100, 10
        )

        warmup_2 = st.number_input(
            "Calentamiento (h)",
            0.0, 72.0, 0.0, 1.0
        )

        semilla_2 = st.number_input(
            "Semilla",
            1, 999999, 42, 1
        )

    else:
        tasa_3 = st.number_input(
            "Tasa promedio μ (pac/h-médico)",
            min_value=0.1,
            max_value=10.0,
            value=2.40,
            step=0.05,
        )

        horas_dia_3 = st.number_input(
            "Horas de operación por día",
            min_value=1.0,
            max_value=24.0,
            value=23.0,
            step=1.0,
        )

        mix_3 = st.slider(
            "Mix Sénior objetivo (%)",
            0.0,
            100.0,
            35.0,
            1.0,
        ) / 100

        st.subheader("Horas objetivo Escenario 3")

        h3 = []
        for d, default in zip(
            DIAS,
            ESC3_HORAS_DEFAULT
        ):
            h3.append(
                st.number_input(
                    f"{d} (h)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=float(default),
                    step=0.01,
                )
            )

        h3 = np.array(h3)

        st.subheader("Réplicas")

        replicas_3 = st.slider(
            "Número de réplicas",
            10, 300, 100, 10
        )

        warmup_3 = st.number_input(
            "Calentamiento (h)",
            0.0, 72.0, 0.0, 1.0
        )

        semilla_3 = st.number_input(
            "Semilla",
            1, 999999, 42, 1
        )


# ============================================================
# ENCABEZADO DINÁMICO
# ============================================================

if escenario.startswith("Escenario 1"):
    st.subheader("🔵 Escenario 1 — Rebalanceo semanal")

    st.write(
        "Se redistribuyen horas médicas existentes desde viernes y "
        "sábado hacia domingo y lunes. Martes, miércoles y jueves "
        "permanecen sin cambios. El objetivo es maximizar la menor "
        "cobertura de demanda diaria."
    )

elif escenario.startswith("Escenario 2"):
    st.subheader("🟢 Escenario 2 — Reestructuración del mix del domingo")

    st.write(
        "Se mantiene el total de 264 horas-médico del domingo, pero "
        "se modifica la proporción Sénior/Júnior. El efecto sobre "
        "la tasa de atención se parametriza como un supuesto de "
        "eficiencia y se evalúa mediante una simulación independiente."
    )

else:
    st.subheader("🟣 Escenario 3 — Reestructuración semanal")

    st.write(
        "Se evalúa una distribución semanal de horas médicas propuesta, "
        "con un mix Sénior/Júnior objetivo configurable. La capacidad "
        "se analiza con un modelo M/M/c agregado y posteriormente "
        "con simulación."
    )


# ============================================================
# KPI COMUNES
# ============================================================

if escenario.startswith("Escenario 1"):
    demanda_kpi = TOTAL_DEMANDA
    horas_kpi = TOTAL_HORAS
    tasa_kpi = (
        "S/J"
    )
    operacion_kpi = f"{horas_dia:.0f} h/día"

elif escenario.startswith("Escenario 2"):
    demanda_kpi = TOTAL_DEMANDA
    horas_kpi = TOTAL_HORAS
    tasa_kpi = f"{tasa_2:.2f}"
    operacion_kpi = f"{horas_dia_2:.0f} h/día"

else:
    demanda_kpi = TOTAL_DEMANDA
    horas_kpi = h3.sum()
    tasa_kpi = f"{tasa_3:.2f}"
    operacion_kpi = f"{horas_dia_3:.0f} h/día"

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Demanda semanal (pacientes)",
    f"{demanda_kpi:,.0f}",
)

k2.metric(
    "Horas médico",
    f"{horas_kpi} h",
)

k3.metric(
    "Tasa base",
    tasa_kpi,
)

k4.metric(
    "Operación",
    operacion_kpi,
)


# ============================================================
# ESCENARIO 1
# ============================================================

if escenario.startswith("Escenario 1"):

    st.header("1. Situación actual")

    actual = mm1_table(
        S_BASE,
        J_BASE,
        DEMANDA,
        horas_dia=horas_dia,
        tasa_s=RATE_SENIOR,
        tasa_j=RATE_JUNIOR,
    )

    st.dataframe(
        actual.style.format({
            "Sénior (h)": "{:.2f}",
            "Júnior (h)": "{:.2f}",
            "Horas médico": "{:.2f}",
            "Pacientes": "{:.0f}",
            "Tasa servicio": "{:.4f}",
            "Capacidad": "{:.2f}",
            "Lambda (pac/h)": "{:.4f}",
            "Mu (pac/h)": "{:.4f}",
            "Utilización": "{:.2%}",
            "Wq (min)": "{:.2f}",
            "W (min)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.header("2. Optimización")

    if st.button(
        "⚙️ Calcular propuesta del Escenario 1",
        type="primary",
        use_container_width=True,
    ):
        resultado = optimizar_escenario1(
            max_s_fri,
            max_s_sab,
            max_j_fri,
            max_j_sab,
        )

        if resultado is None:
            st.error(
                "No existe una solución factible con los límites "
                "de traslado seleccionados."
            )
        else:
            st.session_state["esc1_resultado"] = resultado

    resultado = st.session_state.get(
        "esc1_resultado"
    )

    if resultado is not None:
        S_new = resultado["S"]
        J_new = resultado["J"]

        propuesta = mm1_table(
            S_new,
            J_new,
            DEMANDA,
            horas_dia=horas_dia,
            tasa_s=RATE_SENIOR,
            tasa_j=RATE_JUNIOR,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Cobertura mínima z",
            f"{resultado['z']:.6f}",
        )

        c2.metric(
            "Máxima utilización equivalente",
            f"{1 / resultado['z']:.2%}",
        )

        c3.metric(
            "Horas trasladadas",
            f"{resultado['total_trasladado']:.2f}",
        )

        st.subheader("Traslados realizados")

        movimientos = resultado["movimientos"]

        mostrar = movimientos[
            movimientos["Horas trasladadas"] > 1e-8
        ].copy()

        if len(mostrar):
            st.dataframe(
                mostrar.style.format({
                    "Horas trasladadas": "{:.2f}"
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Con los límites seleccionados, no se requieren traslados."
            )

        st.subheader("Actual vs Escenario 1")

        tabla_c = pd.DataFrame({
            "Día": DIAS,
            "Sénior actual": S_BASE,
            "Sénior Esc. 1": S_new,
            "Júnior actual": J_BASE,
            "Júnior Esc. 1": J_new,
            "Capacidad actual": actual["Capacidad"],
            "Capacidad Esc. 1": propuesta["Capacidad"],
            "Utilización actual": actual["Utilización"],
            "Utilización Esc. 1": propuesta["Utilización"],
            "Wq actual": actual["Wq (min)"],
            "Wq Esc. 1": propuesta["Wq (min)"],
        })

        st.dataframe(
            tabla_c.style.format({
                "Sénior actual": "{:.2f}",
                "Sénior Esc. 1": "{:.2f}",
                "Júnior actual": "{:.2f}",
                "Júnior Esc. 1": "{:.2f}",
                "Capacidad actual": "{:.2f}",
                "Capacidad Esc. 1": "{:.2f}",
                "Utilización actual": "{:.2%}",
                "Utilización Esc. 1": "{:.2%}",
                "Wq actual": "{:.2f}",
                "Wq Esc. 1": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # Gráfico de horas
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(7)
        width = 0.18

        ax.bar(
            x - 1.5 * width,
            S_BASE,
            width,
            label="Sénior actual",
        )
        ax.bar(
            x - 0.5 * width,
            S_new,
            width,
            label="Sénior Escenario 1",
        )
        ax.bar(
            x + 0.5 * width,
            J_BASE,
            width,
            label="Júnior actual",
        )
        ax.bar(
            x + 1.5 * width,
            J_new,
            width,
            label="Júnior Escenario 1",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(DIAS)
        ax.set_ylabel("Horas médico")
        ax.set_title(
            "Horas médicas antes y después — Escenario 1"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig, use_container_width=True)

        # Utilización
        fig, ax = plt.subplots(figsize=(11, 4.5))

        ax.plot(
            DIAS,
            actual["Utilización"] * 100,
            marker="o",
            label="Actual",
        )
        ax.plot(
            DIAS,
            propuesta["Utilización"] * 100,
            marker="o",
            label="Escenario 1",
        )

        ax.set_ylabel("Utilización (%)")
        ax.set_title(
            "Utilización diaria — Escenario 1"
        )
        ax.legend()
        ax.grid(alpha=0.25)

        st.pyplot(fig, use_container_width=True)

        st.subheader("3. Simulación")

        if st.button(
            "🎲 Ejecutar simulación Escenario 1",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                f"Ejecutando {replicas} réplicas..."
            ):
                sim_a = ejecutar_simulacion_mm1(
                    S_BASE,
                    J_BASE,
                    RATE_SENIOR,
                    RATE_JUNIOR,
                    horas_dia,
                    replicas,
                    warmup,
                    int(semilla),
                )

                sim_b = ejecutar_simulacion_mm1(
                    S_new,
                    J_new,
                    RATE_SENIOR,
                    RATE_JUNIOR,
                    horas_dia,
                    replicas,
                    warmup,
                    int(semilla),
                )

            st.session_state["esc1_sim_a"] = sim_a
            st.session_state["esc1_sim_b"] = sim_b

        if (
            "esc1_sim_a" in st.session_state
            and "esc1_sim_b" in st.session_state
        ):
            sim_a = st.session_state["esc1_sim_a"]
            sim_b = st.session_state["esc1_sim_b"]

            st.dataframe(
                comparacion_simulaciones(
                    sim_a,
                    sim_b,
                    "Actual",
                    "Escenario 1",
                ).style.format({
                    "Actual": "{:.2f}",
                    "Escenario 1": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            diario_a = sim_a["resumen"]
            diario_b = sim_b["resumen"]

            diario = pd.DataFrame({
                "Día": DIAS,
                "Espera actual (min)": diario_a[
                    "Espera promedio (min)"
                ],
                "Espera Esc. 1 (min)": diario_b[
                    "Espera promedio (min)"
                ],
                "P90 actual (min)": diario_a[
                    "P90 espera (min)"
                ],
                "P90 Esc. 1 (min)": diario_b[
                    "P90 espera (min)"
                ],
                "Sistema actual (min)": diario_a[
                    "Sistema promedio (min)"
                ],
                "Sistema Esc. 1 (min)": diario_b[
                    "Sistema promedio (min)"
                ],
            })

            st.subheader(
                "Resultados de simulación por día"
            )

            st.dataframe(
                diario.style.format({
                    "Espera actual (min)": "{:.2f}",
                    "Espera Esc. 1 (min)": "{:.2f}",
                    "P90 actual (min)": "{:.2f}",
                    "P90 Esc. 1 (min)": "{:.2f}",
                    "Sistema actual (min)": "{:.2f}",
                    "Sistema Esc. 1 (min)": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

            fig, ax = plt.subplots(figsize=(11, 5))

            x = np.arange(7)
            width = 0.35

            ax.bar(
                x - width / 2,
                diario["Espera actual (min)"],
                width,
                label="Actual",
            )
            ax.bar(
                x + width / 2,
                diario["Espera Esc. 1 (min)"],
                width,
                label="Escenario 1",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(DIAS)
            ax.set_ylabel("Minutos")
            ax.set_title(
                "Espera promedio simulada — Escenario 1"
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.25)

            st.pyplot(fig, use_container_width=True)

            st.subheader(
                "Evolución simulada de la cola"
            )

            ta = sim_a["trayectoria"]
            tb = sim_b["trayectoria"]

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.step(
                ta["Hora"],
                ta["Cola"],
                where="post",
                label="Actual",
            )
            ax.step(
                tb["Hora"],
                tb["Cola"],
                where="post",
                label="Escenario 1",
            )

            for d in range(1, 7):
                ax.axvline(
                    d * horas_dia,
                    linestyle="--",
                    alpha=0.3,
                )

            ax.set_xlabel(
                "Hora de la semana"
            )
            ax.set_ylabel(
                "Pacientes en cola"
            )
            ax.set_title(
                "Evolución de la cola — primera réplica"
            )
            ax.legend()
            ax.grid(alpha=0.25)

            st.pyplot(fig, use_container_width=True)

        st.download_button(
            "⬇️ Descargar resultados Escenario 1",
            data=excel_bytes({
                "Actual_M_M_1": actual,
                "Escenario1_M_M_1": propuesta,
                "Traslados": movimientos,
            }),
            file_name="hospital_gloria_escenario1.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

# ============================================================
# ESCENARIO 2
# ============================================================

elif escenario.startswith("Escenario 2"):

    senior_actual = 60.0
    junior_actual = 204.0

    senior_nuevo = float(senior_dom)
    junior_nuevo = 264.0 - senior_nuevo

    mix_actual = senior_actual / 264
    mix_nuevo = senior_nuevo / 264

    # Tasa efectiva derivada del mix.
    tasa_mix = (
        senior_nuevo * RATE_SENIOR
        + junior_nuevo * RATE_JUNIOR
    ) / 264.0

    tasa_efectiva = tasa_mix * (
        1 + mejora_dom / 100
    )

    S2 = S_BASE.copy()
    J2 = J_BASE.copy()

    S2[0] = senior_nuevo
    J2[0] = junior_nuevo

    # El modelo 2 conserva el enfoque de tasa global configurable.
    # Se aplica la mejora solo al domingo.
    tasas_dom = np.full(7, tasa_2)
    tasas_dom[0] = tasa_efectiva

    capacidad_2 = (
        (S2 + J2) * tasas_dom
    )

    lam_2 = DEMANDA / horas_dia_2
    mu_2 = capacidad_2 / horas_dia_2

    util_2 = lam_2 / mu_2

    wq_2 = np.full(7, np.nan)
    estable_2 = mu_2 > lam_2

    wq_2[estable_2] = (
        60 * lam_2[estable_2]
        / (
            mu_2[estable_2]
            * (mu_2[estable_2] - lam_2[estable_2])
        )
    )

    tabla_2 = pd.DataFrame({
        "Día": DIAS,
        "Sénior (h)": S2,
        "Júnior (h)": J2,
        "Horas médico": S2 + J2,
        "Pacientes": DEMANDA,
        "Tasa servicio": tasas_dom,
        "Capacidad": capacidad_2,
        "Utilización": util_2,
        "Wq (min)": wq_2,
        "Estado": [
            "Estable" if x < 1
            else "Sin equilibrio estable"
            for x in util_2
        ],
    })

    a, b, c, d = st.columns(4)

    a.metric(
        "Sénior domingo",
        f"{senior_actual:.0f} → {senior_nuevo:.0f} h",
    )

    b.metric(
        "Júnior domingo",
        f"{junior_actual:.0f} → {junior_nuevo:.0f} h",
    )

    c.metric(
        "Mix Sénior",
        f"{mix_actual:.0%} → {mix_nuevo:.0%}",
    )

    d.metric(
        "Tasa efectiva domingo",
        f"{tasa_2:.0f} → {tasa_efectiva:.0f}",
    )

    st.info(
        "Las 264 horas del domingo se mantienen. La mejora adicional "
        "de eficiencia es un supuesto configurable; no debe presentarse "
        "como una medición real del hospital."
    )

    st.subheader("Capacidad y utilización")

    st.dataframe(
        tabla_2.style.format({
            "Sénior (h)": "{:.2f}",
            "Júnior (h)": "{:.2f}",
            "Horas médico": "{:.2f}",
            "Pacientes": "{:.0f}",
            "Tasa servicio": "{:.4f}",
            "Capacidad": "{:.2f}",
            "Utilización": "{:.2%}",
            "Wq (min)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Comparación del domingo")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        ["Actual", "Escenario 2"],
        [
            DEMANDA[0] /
            ((S_BASE[0] + J_BASE[0]) * tasa_2),
            DEMANDA[0] /
            ((S2[0] + J2[0]) * tasa_efectiva),
        ],
    )

    ax.set_ylabel("Utilización")
    ax.set_title(
        "Utilización del domingo"
    )
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(fig, use_container_width=True)

    st.subheader("Simulación independiente")

    if st.button(
        "🎲 Ejecutar simulación Escenario 2",
        type="primary",
        use_container_width=True,
    ):
        # Situación actual
        tasas_actual = np.full(
            7,
            tasa_2,
            dtype=float,
        )

        S2_actual = S_BASE.copy()
        J2_actual = J_BASE.copy()

        # Para mantener el mismo modelo de servidor equivalente,
        # representamos el domingo actual con la tasa base.
        tasa_s_2_actual = np.full(
            7,
            tasa_2,
        )
        tasa_s_2_propuesta = np.full(
            7,
            tasa_2,
        )
        tasa_s_2_propuesta[0] = tasa_efectiva

        # Se convierte a horas equivalentes de un único recurso.
        sim_actual = ejecutar_simulacion_mm1(
            S2_actual,
            J2_actual,
            tasa_s_2_actual,
            np.zeros(7),
            horas_dia_2,
            replicas_2,
            warmup_2,
            int(semilla_2),
        )

        # En este escenario, tasa_s se usa como capacidad por hora-médico
        # y tasa_j=0; la suma S+J representa horas-médico.
        sim_propuesta = ejecutar_simulacion_mm1(
            S2,
            J2,
            tasa_s_2_propuesta,
            np.zeros(7),
            horas_dia_2,
            replicas_2,
            warmup_2,
            int(semilla_2),
        )

        st.session_state["esc2_sim_a"] = sim_actual
        st.session_state["esc2_sim_b"] = sim_propuesta

    if (
        "esc2_sim_a" in st.session_state
        and "esc2_sim_b" in st.session_state
    ):
        sim_a = st.session_state["esc2_sim_a"]
        sim_b = st.session_state["esc2_sim_b"]

        st.dataframe(
            comparacion_simulaciones(
                sim_a,
                sim_b,
                "Actual",
                "Escenario 2",
            ).style.format({
                "Actual": "{:.2f}",
                "Escenario 2": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        da = sim_a["resumen"]
        db = sim_b["resumen"]

        diario = pd.DataFrame({
            "Día": DIAS,
            "Espera Actual (min)": da[
                "Espera promedio (min)"
            ],
            "Espera Esc. 2 (min)": db[
                "Espera promedio (min)"
            ],
            "P90 Actual (min)": da[
                "P90 espera (min)"
            ],
            "P90 Esc. 2 (min)": db[
                "P90 espera (min)"
            ],
            "Sistema Actual (min)": da[
                "Sistema promedio (min)"
            ],
            "Sistema Esc. 2 (min)": db[
                "Sistema promedio (min)"
            ],
        })

        st.dataframe(
            diario.style.format({
                "Espera Actual (min)": "{:.2f}",
                "Espera Esc. 2 (min)": "{:.2f}",
                "P90 Actual (min)": "{:.2f}",
                "P90 Esc. 2 (min)": "{:.2f}",
                "Sistema Actual (min)": "{:.2f}",
                "Sistema Esc. 2 (min)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(11, 5))

        x = np.arange(7)
        width = 0.35

        ax.bar(
            x - width / 2,
            diario["Espera Actual (min)"],
            width,
            label="Actual",
        )
        ax.bar(
            x + width / 2,
            diario["Espera Esc. 2 (min)"],
            width,
            label="Escenario 2",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(DIAS)
        ax.set_ylabel("Minutos")
        ax.set_title(
            "Espera promedio simulada — Escenario 2"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig, use_container_width=True)

        ta = sim_a["trayectoria"]
        tb = sim_b["trayectoria"]

        fig, ax = plt.subplots(figsize=(12, 5))

        ax.step(
            ta["Hora"],
            ta["Cola"],
            where="post",
            label="Actual",
        )
        ax.step(
            tb["Hora"],
            tb["Cola"],
            where="post",
            label="Escenario 2",
        )

        for d in range(1, 7):
            ax.axvline(
                d * horas_dia_2,
                linestyle="--",
                alpha=0.3,
            )

        ax.set_xlabel("Hora de la semana")
        ax.set_ylabel("Pacientes en cola")
        ax.set_title(
            "Evolución de la cola — Escenario 2"
        )
        ax.legend()
        ax.grid(alpha=0.25)

        st.pyplot(fig, use_container_width=True)

    st.download_button(
        "⬇️ Descargar resultados Escenario 2",
        data=excel_bytes({
            "Escenario2": tabla_2,
        }),
        file_name="hospital_gloria_escenario2.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# ============================================================
# ESCENARIO 3
# ============================================================

else:

    # Distribución del escenario 3
    S3 = h3 * mix_3
    J3 = h3 * (1 - mix_3)

    # Tabla base para comparar
    tasa_por_hora = np.full(
        7,
        tasa_3,
        dtype=float,
    )

    mmc = mmc_table(
        h3,
        DEMANDA,
        horas_dia_3,
        tasa_3,
        mix_3,
    )

    base_s3 = pd.DataFrame({
        "Día": DIAS,
        "Demanda": DEMANDA,
        "Base Senior (h)": S_BASE,
        "Base Junior (h)": J_BASE,
        "Base Total (h)": H_BASE,
        "Esc. 3 Senior (h)": S3,
        "Esc. 3 Junior (h)": J3,
        "Esc. 3 Total (h)": h3,
    })

    st.subheader("Parámetros de la propuesta")

    st.dataframe(
        base_s3.style.format({
            "Demanda": "{:.0f}",
            "Base Senior (h)": "{:.2f}",
            "Base Junior (h)": "{:.2f}",
            "Base Total (h)": "{:.2f}",
            "Esc. 3 Senior (h)": "{:.2f}",
            "Esc. 3 Junior (h)": "{:.2f}",
            "Esc. 3 Total (h)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Resultados M/M/c")

    st.info(
        "El Escenario 3 utiliza M/M/c. Los canales equivalentes se "
        "calculan como horas médico / 23 h. Como Erlang-C clásico "
        "requiere un número entero de servidores, se utiliza "
        "c_operativo = ceil(c_equivalente). Esta conversión es un "
        "supuesto metodológico y se muestra explícitamente."
    )

    st.dataframe(
        mmc.style.format({
            "Sénior (h)": "{:.2f}",
            "Júnior (h)": "{:.2f}",
            "Horas médico": "{:.2f}",
            "Pacientes": "{:.0f}",
            "Tasa servicio": "{:.2f}",
            "Capacidad": "{:.2f}",
            "Lambda (pac/h)": "{:.4f}",
            "c equivalente": "{:.2f}",
            "c operativo Erlang-C": "{:.0f}",
            "Utilización": "{:.2%}",
            "P(wait)": "{:.2%}",
            "Wq (min)": "{:.2f}",
            "W (min)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Horas médicas: Base vs Escenario 3")

    fig, ax = plt.subplots(figsize=(11, 5))

    x = np.arange(7)
    width = 0.35

    ax.bar(
        x - width / 2,
        H_BASE,
        width,
        label="Base",
    )

    ax.bar(
        x + width / 2,
        h3,
        width,
        label="Escenario 3",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(DIAS)
    ax.set_ylabel("Horas médico")
    ax.set_title(
        "Horas médicas totales por día"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    st.pyplot(fig, use_container_width=True)

    st.subheader(
        "Demanda, capacidad y utilización"
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        DIAS,
        DEMANDA,
        marker="o",
        label="Demanda",
    )

    ax.plot(
        DIAS,
        mmc["Capacidad"],
        marker="o",
        label="Capacidad",
    )

    ax.set_ylabel("Pacientes por día")
    ax.set_title(
        "Demanda vs capacidad — Escenario 3"
    )
    ax.legend()
    ax.grid(alpha=0.25)

    st.pyplot(fig, use_container_width=True)

    fig, ax = plt.subplots(figsize=(11, 4.5))

    ax.plot(
        DIAS,
        mmc["Utilización"] * 100,
        marker="o",
        label="Utilización",
    )

    ax.axhline(
        100,
        linestyle="--",
        alpha=0.4,
        label="Límite 100%",
    )

    ax.set_ylabel("Utilización (%)")
    ax.set_title(
        "Utilización diaria — Escenario 3"
    )
    ax.legend()
    ax.grid(alpha=0.25)

    st.pyplot(fig, use_container_width=True)

    st.subheader("Simulación M/M/c")

    if st.button(
        "🎲 Ejecutar simulación Escenario 3",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            f"Ejecutando {replicas_3} réplicas..."
        ):
            # Base: horas médicas originales
            sim_base = ejecutar_simulacion_mmc(
                H_BASE,
                tasa_3,
                horas_dia_3,
                replicas_3,
                warmup_3,
                int(semilla_3),
            )

            # Propuesta
            sim_esc3 = ejecutar_simulacion_mmc(
                h3,
                tasa_3,
                horas_dia_3,
                replicas_3,
                warmup_3,
                int(semilla_3),
            )

        st.session_state["esc3_sim_base"] = sim_base
        st.session_state["esc3_sim"] = sim_esc3

    if (
        "esc3_sim_base" in st.session_state
        and "esc3_sim" in st.session_state
    ):
        sim_base = st.session_state["esc3_sim_base"]
        sim_esc3 = st.session_state["esc3_sim"]

        st.subheader(
            "Resultados globales de simulación"
        )

        st.dataframe(
            comparacion_simulaciones(
                sim_base,
                sim_esc3,
                "Base",
                "Escenario 3",
            ).style.format({
                "Base": "{:.2f}",
                "Escenario 3": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        db = sim_base["resumen"]
        dp = sim_esc3["resumen"]

        diario3 = pd.DataFrame({
            "Día": DIAS,
            "Espera Base (min)": db[
                "Espera promedio (min)"
            ],
            "Espera Esc. 3 (min)": dp[
                "Espera promedio (min)"
            ],
            "P90 Base (min)": db[
                "P90 espera (min)"
            ],
            "P90 Esc. 3 (min)": dp[
                "P90 espera (min)"
            ],
            "Sistema Base (min)": db[
                "Sistema promedio (min)"
            ],
            "Sistema Esc. 3 (min)": dp[
                "Sistema promedio (min)"
            ],
        })

        st.subheader(
            "Resultados de simulación por día"
        )

        st.dataframe(
            diario3.style.format({
                "Espera Base (min)": "{:.2f}",
                "Espera Esc. 3 (min)": "{:.2f}",
                "P90 Base (min)": "{:.2f}",
                "P90 Esc. 3 (min)": "{:.2f}",
                "Sistema Base (min)": "{:.2f}",
                "Sistema Esc. 3 (min)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(11, 5))

        x = np.arange(7)
        width = 0.35

        ax.bar(
            x - width / 2,
            diario3["Espera Base (min)"],
            width,
            label="Base",
        )

        ax.bar(
            x + width / 2,
            diario3["Espera Esc. 3 (min)"],
            width,
            label="Escenario 3",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(DIAS)
        ax.set_ylabel("Minutos")
        ax.set_title(
            "Espera promedio — Base vs Escenario 3"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        st.pyplot(fig, use_container_width=True)

        ta = sim_base["trayectoria"]
        tp = sim_esc3["trayectoria"]

        fig, ax = plt.subplots(figsize=(12, 5))

        if len(ta):
            ax.step(
                ta["Hora"],
                ta["Cola"],
                where="post",
                label="Base",
            )

        if len(tp):
            ax.step(
                tp["Hora"],
                tp["Cola"],
                where="post",
                label="Escenario 3",
            )

        for d in range(1, 7):
            ax.axvline(
                d * horas_dia_3,
                linestyle="--",
                alpha=0.3,
            )

        ax.set_xlabel(
            "Hora de la semana"
        )
        ax.set_ylabel(
            "Pacientes en cola"
        )
        ax.set_title(
            "Evolución simulada de la cola — Escenario 3"
        )
        ax.legend()
        ax.grid(alpha=0.25)

        st.pyplot(fig, use_container_width=True)

        st.info(
            "La simulación puede diferir de M/M/c porque la simulación "
            "representa una semana finita, conserva la cola entre días "
            "y utiliza realizaciones aleatorias. M/M/c es un resultado "
            "teórico estacionario."
        )

    st.download_button(
        "⬇️ Descargar resultados Escenario 3",
        data=excel_bytes({
            "Datos_Escenario3": base_s3,
            "MMc_Escenario3": mmc,
        }),
        file_name="hospital_gloria_escenario3.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )


# ============================================================
# NOTA METODOLÓGICA FINAL
# ============================================================

st.divider()

st.caption(
    "Los tres escenarios son módulos independientes. Los resultados "
    "de un escenario no modifican los parámetros ni la simulación de "
    "los otros. Los modelos utilizan datos agregados y supuestos; "
    "no representan médicos individuales, prioridades clínicas ni "
    "tiempos observados de pacientes reales."
)
