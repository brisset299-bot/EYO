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

# Datos base: cantidad de personal Sénior/Júnior (usados como horas en Escenario 1 según su formulación)
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
        "Sénior (médicos)": S,
        "Júnior (médicos)": J,
        "Total médicos": S + J,
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
        "Médicos trasladados": x[:8],
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
# SIMULACIÓN TIPO ARENA — ESCENARIO 2
# ============================================================

def generar_llegadas_arena(pacientes_dia, horas_dia, modo, rng):
    """
    Genera exactamente la demanda diaria, pero con llegadas aleatorias.

    Estable:
        Proceso homogéneo: dado que conocemos exactamente N pacientes,
        sus tiempos de llegada se generan como N puntos aleatorios
        ordenados dentro de la jornada. Esto equivale a un proceso
        Poisson condicionado a N llegadas.

    Variable:
        Se generan interarribos exponenciales y se normalizan para que
        exactamente N pacientes lleguen durante la jornada, permitiendo
        mayor agrupación aleatoria de llegadas.
    """
    n = int(pacientes_dia)
    if n <= 0:
        return np.array([], dtype=float)

    horizonte_min = float(horas_dia) * 60.0

    if modo == "Estable":
        llegadas = np.sort(
            rng.uniform(0.0, horizonte_min, size=n)
        )
        return llegadas

    inter = rng.exponential(scale=1.0, size=n)
    tiempos = np.cumsum(inter)

    if tiempos[-1] <= 0:
        return np.sort(
            rng.uniform(0.0, horizonte_min, size=n)
        )

    return tiempos / tiempos[-1] * horizonte_min


def simular_semana_arena(
    pacientes,
    personal_senior,
    personal_junior,
    tasa_senior,
    tasa_junior,
    horas_dia,
    variabilidad_pct,
    modo_llegadas,
    replicas,
    semilla,
):
    """
    Simulación tipo Arena para el Escenario 2.

    Lógica:
        Llegadas -> Cola FCFS -> capacidad médica agregada -> Atención -> Salida

    Corrección metodológica importante:
    - Sénior/Júnior = número de médicos, NO horas.
    - La capacidad diaria se mantiene consistente con la tabla:
          capacidad_día = médicos_totales * tasa_mix
    - Esa capacidad diaria se convierte a capacidad por hora:
          capacidad_hora = capacidad_día / horas_dia
    - Los pacientes llegan aleatoriamente, no cada cierto número fijo
      de minutos.
    - El tiempo de atención es aleatorio y heterogéneo, manteniendo
      como media la capacidad del sistema.
    - Cada día se simula de manera independiente, como en la lógica
      planteada originalmente para el Escenario 2.
    - La demanda diaria se conserva exactamente.
    """
    resultados = []

    pacientes = np.asarray(pacientes, dtype=float)
    personal_senior = np.asarray(personal_senior, dtype=float)
    personal_junior = np.asarray(personal_junior, dtype=float)

    cv = max(float(variabilidad_pct), 0.0) / 100.0

    # Parámetro de dispersión lognormal.
    sigma_ln = (
        math.sqrt(math.log(1.0 + cv ** 2))
        if cv > 0
        else 0.0
    )

    for rep in range(int(replicas)):
        rng = np.random.default_rng(
            int(semilla) + rep
        )

        for d, dia in enumerate(DIAS):
            n = int(pacientes[d])
            ns = int(round(personal_senior[d]))
            nj = int(round(personal_junior[d]))
            total_personal = ns + nj

            if total_personal <= 0 or n <= 0:
                continue

            proporcion_senior = ns / total_personal

            # Tasa media del mix de personal.
            tasa_mix = (
                proporcion_senior * tasa_senior
                + (1.0 - proporcion_senior) * tasa_junior
            )

            # Capacidad diaria coherente con la tabla del caso.
            capacidad_dia = total_personal * tasa_mix

            # Capacidad equivalente por hora.
            capacidad_hora = capacidad_dia / float(horas_dia)

            if capacidad_hora <= 0:
                continue

            # Llegadas aleatorias manteniendo exactamente N pacientes.
            llegadas = generar_llegadas_arena(
                n,
                horas_dia,
                modo_llegadas,
                rng,
            )

            # ====================================================
            # TIEMPOS DE SERVICIO
            # ====================================================
            #
            # El servicio medio del sistema es:
            #
            #   60 / capacidad_hora   minutos/paciente
            #
            # La variabilidad representa médicos más rápidos
            # y más lentos sin cambiar la capacidad media.
            #
            # Se usa una variable de velocidad alrededor de 1.
            # Se normaliza de modo que E[1/factor] = 1, preservando
            # el tiempo medio de servicio.
            # ====================================================

            if sigma_ln > 0:
                raw = rng.lognormal(
                    mean=-0.5 * sigma_ln ** 2,
                    sigma=sigma_ln,
                    size=n,
                )

                normalizador = np.mean(1.0 / raw)
                factores_velocidad = raw * normalizador
            else:
                factores_velocidad = np.ones(n)

            # Componente exponencial del tiempo de atención.
            trabajos = rng.exponential(
                scale=1.0,
                size=n,
            )

            servicio_base_h = 1.0 / capacidad_hora

            tiempos_servicio_min = (
                trabajos
                * servicio_base_h
                * 60.0
                / factores_velocidad
            )

            # ====================================================
            # COLA FCFS
            # ====================================================

            disponible = 0.0

            esperas = []
            sistemas = []
            salidas = []

            # Para calcular correctamente la cantidad de pacientes
            # que estaban esperando cuando llega cada paciente.
            salidas_anteriores = []

            puntero_salida = 0
            cola_max = 0

            for idx, (llegada, servicio) in enumerate(
                zip(llegadas, tiempos_servicio_min)
            ):
                # Liberamos conceptualmente todos los pacientes
                # terminados antes de esta llegada.
                while (
                    puntero_salida < len(salidas_anteriores)
                    and salidas_anteriores[puntero_salida]
                    <= llegada
                ):
                    puntero_salida += 1

                pacientes_en_sistema = (
                    idx - puntero_salida
                )

                # En un único servidor equivalente, uno puede estar
                # siendo atendido; el resto forma la cola.
                cola_actual = max(
                    0,
                    pacientes_en_sistema - 1,
                )

                cola_max = max(
                    cola_max,
                    cola_actual,
                )

                inicio = max(
                    llegada,
                    disponible,
                )

                espera = inicio - llegada
                salida = inicio + servicio

                esperas.append(espera)
                sistemas.append(
                    salida - llegada
                )
                salidas.append(salida)
                salidas_anteriores.append(salida)

                disponible = salida

            esperas = np.asarray(
                esperas,
                dtype=float,
            )

            sistemas = np.asarray(
                sistemas,
                dtype=float,
            )

            salidas = np.asarray(
                salidas,
                dtype=float,
            )

            horizonte_min = float(horas_dia) * 60.0

            # Pacientes que todavía no terminaron al cerrar la jornada.
            pendientes = int(
                np.sum(salidas > horizonte_min)
            )

            utilizacion = (
                n / capacidad_dia
                if capacidad_dia > 0
                else np.nan
            )

            resultados.append({
                "Réplica": rep + 1,
                "Día": dia,
                "Pacientes": n,
                "Sénior": ns,
                "Júnior": nj,
                "Total médicos": total_personal,
                "Tasa mix": tasa_mix,
                "Capacidad": capacidad_dia,
                "Utilización": utilizacion,
                "Espera promedio (min)": np.mean(esperas),
                "Espera P90 (min)": np.percentile(
                    esperas,
                    90,
                ),
                "Tiempo sistema promedio (min)": np.mean(
                    sistemas
                ),
                "Máxima cola": int(cola_max),
                "Pacientes pendientes": pendientes,
            })

    detalle = pd.DataFrame(resultados)

    resumen = (
        detalle
        .groupby("Día", sort=False)
        .agg({
            "Pacientes": "mean",
            "Sénior": "mean",
            "Júnior": "mean",
            "Total médicos": "mean",
            "Tasa mix": "mean",
            "Capacidad": "mean",
            "Utilización": "mean",
            "Espera promedio (min)": "mean",
            "Espera P90 (min)": "mean",
            "Tiempo sistema promedio (min)": "mean",
            "Máxima cola": "mean",
            "Pacientes pendientes": "mean",
        })
        .reset_index()
    )

    # ============================================================
    # RESULTADOS GLOBALES
    # ============================================================
    #
    # La espera semanal se pondera por pacientes, no como promedio
    # simple de los siete promedios diarios.
    # ============================================================

    globales = []

    for rep, grupo in detalle.groupby(
        "Réplica",
        sort=False,
    ):
        total_pacientes = grupo["Pacientes"].sum()

        espera_ponderada = (
            np.sum(
                grupo["Espera promedio (min)"]
                * grupo["Pacientes"]
            )
            / total_pacientes
            if total_pacientes > 0
            else np.nan
        )

        globales.append({
            "Réplica": rep,
            "Espera promedio semanal (min)": espera_ponderada,
            "P90 promedio diario (min)": grupo[
                "Espera P90 (min)"
            ].mean(),
            "Cola máxima semanal": grupo[
                "Máxima cola"
            ].max(),
            "Pacientes pendientes": grupo[
                "Pacientes pendientes"
            ].sum(),
        })

    globales = pd.DataFrame(globales)

    return {
        "resumen": resumen,
        "detalle": detalle,
        "global": globales,
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
        max_s_fri = st.number_input("Sénior: máximo desde viernes", 0.0, float(S_BASE[5]), float(S_BASE[5]), 1.0)
        max_s_sab = st.number_input("Sénior: máximo desde sábado", 0.0, float(S_BASE[6]), float(S_BASE[6]), 1.0)
        max_j_fri = st.number_input("Júnior: máximo desde viernes", 0.0, float(J_BASE[5]), float(J_BASE[5]), 1.0)
        max_j_sab = st.number_input("Júnior: máximo desde sábado", 0.0, float(J_BASE[6]), float(J_BASE[6]), 1.0)
        st.subheader("Réplicas")
        replicas = st.slider("Número de réplicas", 10, 300, 100, 10)
        warmup = st.number_input("Calentamiento (h)", 0.0, 72.0, 0.0, 1.0)
        semilla = st.number_input("Semilla", 1, 999999, 42, 1)

    elif escenario.startswith("Escenario 2"):
        tasa_2 = st.number_input(
            "Tasa promedio (pac/h-médico)",
            min_value=0.1,
            max_value=10.0,
            value=2.41,
            step=0.01,
        )
        horas_dia_2 = st.number_input(
            "Horas de operación por día",
            min_value=1.0,
            max_value=24.0,
            value=24.0,
            step=1.0,
        )
        st.subheader("Domingo")
        senior_dom = st.slider(
            "Médicos Sénior domingo",
            min_value=0,
            max_value=264,
            value=60,
            step=1,
        )
        variabilidad_2 = st.slider(
            "Variabilidad de velocidad entre médicos (%)",
            min_value=0.0,
            max_value=40.0,
            value=15.0,
            step=1.0,
        )
        diferencia_mix_2 = st.slider(
            "Diferencia de velocidad Sénior vs Júnior (%)",
            min_value=0.0,
            max_value=40.0,
            value=10.0,
            step=1.0,
        )
        modo_llegadas_2 = st.radio(
            "Flujo de llegada",
            ["Estable", "Variable"],
            index=0,
        )
        st.subheader("Réplicas")
        replicas_2 = st.slider("Número de réplicas", 10, 500, 100, 10)
        semilla_2 = st.number_input("Semilla", 1, 999999, 42, 1)

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
        for d, default in zip(DIAS, ESC3_HORAS_DEFAULT):
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
        replicas_3 = st.slider("Número de réplicas", 10, 300, 100, 10)
        warmup_3 = st.number_input("Calentamiento (h)", 0.0, 72.0, 0.0, 1.0)
        semilla_3 = st.number_input("Semilla", 1, 999999, 42, 1)

# ============================================================
# ENCABEZADO DINÁMICO
# ============================================================

if escenario.startswith("Escenario 1"):
    st.subheader("🔵 Escenario 1 — Rebalanceo semanal")
    st.write(
        "Se redistribuyen médicos disponibles desde viernes y sábado "
        "hacia domingo y lunes. Martes, miércoles y jueves permanecen "
        "sin cambios. El objetivo es maximizar la menor cobertura de "
        "demanda diaria sin aumentar el personal semanal."
    )
elif escenario.startswith("Escenario 2"):
    st.subheader("🟢 Escenario 2 — Reestructuración del mix del domingo")
    st.write(
        "Se mantiene el total de 264 médicos del domingo, pero se modifica "
        "la proporción Sénior/Júnior. El efecto se evalúa mediante una "
        "simulación independiente."
    )
else:
    st.subheader("🟣 Escenario 3 — Reestructuración semanal")
    st.write(
        "Se evalúa una distribución semanal de horas médicas propuesta, "
        "con un mix Sénior/Júnior objetivo configurable."
    )

# ============================================================
# KPI COMUNES
# ============================================================

if escenario.startswith("Escenario 1"):
    demanda_kpi = TOTAL_DEMANDA
    personal_kpi = TOTAL_HORAS
    tasa_kpi = "S/J diferenciada"
    operacion_kpi = f"{horas_dia:.0f} h/día"
elif escenario.startswith("Escenario 2"):
    demanda_kpi = TOTAL_DEMANDA
    personal_kpi = TOTAL_HORAS
    tasa_kpi = f"{tasa_2:.2f}"
    operacion_kpi = f"{horas_dia_2:.0f} h/día"
else:
    demanda_kpi = TOTAL_DEMANDA
    personal_kpi = h3.sum()
    tasa_kpi = f"{tasa_3:.2f}"
    operacion_kpi = f"{horas_dia_3:.0f} h/día"

k1, k2, k3, k4 = st.columns(4)
k1.metric("Demanda semanal", f"{demanda_kpi:,.0f} pacientes")
k2.metric("Personal / horas objetivo", f"{personal_kpi:,.0f}")
k3.metric("Tasa base", tasa_kpi)
k4.metric("Operación", operacion_kpi)

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
            "Sénior (médicos)": "{:.0f}",
            "Júnior (médicos)": "{:.0f}",
            "Total médicos": "{:.0f}",
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
            "Médicos trasladados",
            f"{resultado['total_trasladado']:.2f}",
        )

        st.subheader("Traslados realizados")

        movimientos = resultado["movimientos"]

        mostrar = movimientos[
            movimientos["Médicos trasladados"] > 1e-8
        ].copy()

        if len(mostrar):
            st.dataframe(
                mostrar.style.format({
                    "Médicos trasladados": "{:.2f}"
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
        ax.set_ylabel("Médicos")
        ax.set_title(
            "Personal médico antes y después — Escenario 1"
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
    total_domingo = 264.0

    senior_nuevo = float(senior_dom)
    junior_nuevo = total_domingo - senior_nuevo

    # Tasas por tipo de personal: se conserva la relación histórica
    # Senior/Junior usada en el modelo y se normaliza para que el mix
    # actual 60/204 tenga promedio 2.41 pac/h-médico.
    tasa_s_base = RATE_SENIOR
    tasa_j_base = RATE_JUNIOR
    mix_actual_p = senior_actual / total_domingo
    promedio_base = mix_actual_p * tasa_s_base + (1.0 - mix_actual_p) * tasa_j_base
    factor_calibracion = tasa_2 / promedio_base

    tasa_s_2 = tasa_s_base * factor_calibracion
    tasa_j_2 = tasa_j_base * factor_calibracion

    # La propuesta puede usar un supuesto adicional de diferencia
    # de velocidad entre Sénior y Júnior. Se calibra nuevamente para
    # mantener 2.41 en el mix actual.
    ventaja = float(diferencia_mix_2) / 100.0
    tasa_s_2 = tasa_s_2 * (1.0 + ventaja)
    promedio_actual_ajustado = mix_actual_p * tasa_s_2 + (1.0 - mix_actual_p) * tasa_j_2
    tasa_j_2 = (tasa_2 - mix_actual_p * tasa_s_2) / max(1.0 - mix_actual_p, 1e-9)

    # Si la ventaja solicitada es demasiado alta para mantener tasa_j positiva,
    # se limita automáticamente a un valor técnicamente válido.
    if tasa_j_2 <= 0:
        tasa_s_2 = tasa_2 * 1.5
        tasa_j_2 = (tasa_2 - mix_actual_p * tasa_s_2) / (1.0 - mix_actual_p)

    mix_nuevo_p = senior_nuevo / total_domingo
    tasa_mix_actual = mix_actual_p * tasa_s_2 + (1.0 - mix_actual_p) * tasa_j_2
    tasa_mix_nuevo = mix_nuevo_p * tasa_s_2 + (1.0 - mix_nuevo_p) * tasa_j_2

    capacidad_actual = total_domingo * tasa_mix_actual
    capacidad_esc2_dom = total_domingo * tasa_mix_nuevo

    # Para los demás días no se modifica el personal ni el mix.
    total_personal = S_BASE + J_BASE
    tasa_mix_dias = np.full(7, tasa_2, dtype=float)
    tasa_mix_dias[0] = tasa_mix_nuevo

    capacidad_actual_sem = total_personal * tasa_2
    capacidad_esc2 = total_personal * tasa_mix_dias

    util_actual = DEMANDA / capacidad_actual_sem
    util_esc2 = DEMANDA / capacidad_esc2

    lam_2 = DEMANDA / horas_dia_2
    mu_actual_2 = capacidad_actual_sem / horas_dia_2
    mu_esc2_2 = capacidad_esc2 / horas_dia_2

    wq_actual = np.full(7, np.nan)
    wq_esc2 = np.full(7, np.nan)

    estable_actual = mu_actual_2 > lam_2
    estable_esc2 = mu_esc2_2 > lam_2

    wq_actual[estable_actual] = (
        60.0 * lam_2[estable_actual]
        / (mu_actual_2[estable_actual] * (mu_actual_2[estable_actual] - lam_2[estable_actual]))
    )
    wq_esc2[estable_esc2] = (
        60.0 * lam_2[estable_esc2]
        / (mu_esc2_2[estable_esc2] * (mu_esc2_2[estable_esc2] - lam_2[estable_esc2]))
    )

    S2 = S_BASE.copy()
    J2 = J_BASE.copy()
    S2[0] = senior_nuevo
    J2[0] = junior_nuevo

    tabla_2 = pd.DataFrame({
        "Día": DIAS,
        "Sénior (médicos)": S2,
        "Júnior (médicos)": J2,
        "Total médicos": total_personal,
        "Pacientes": DEMANDA,
        "Demanda/hora": lam_2,
        "Tasa mix (pac/h-médico)": tasa_mix_dias,
        "Capacidad diaria": capacidad_esc2,
        "Utilización": util_esc2,
        "Wq (min)": wq_esc2,
        "Estado": ["Estable" if x < 1 else "Sin equilibrio estable" for x in util_esc2],
    })

    a, b, c, d = st.columns(4)
    a.metric("Sénior domingo", f"{senior_actual:.0f} → {senior_nuevo:.0f}")
    b.metric("Júnior domingo", f"{junior_actual:.0f} → {junior_nuevo:.0f}")
    c.metric("Mix Sénior", f"{mix_actual_p:.1%} → {mix_nuevo_p:.1%}")
    d.metric("Tasa domingo", f"{tasa_mix_actual:.2f} → {tasa_mix_nuevo:.2f}")

    st.info(
        f"El domingo parte de 60 médicos Sénior y 204 Júnior (264 en total). "
        f"El escenario cambia únicamente el mix, manteniendo los 264 médicos. "
        f"La tasa promedio de referencia es {tasa_2:.2f} pac/h-médico. "
        f"Para representar que existen médicos más rápidos y más lentos, "
        f"la simulación introduce una variabilidad de {variabilidad_2:.0f}%. "
        f"La diferencia de velocidad Sénior/Júnior ({diferencia_mix_2:.0f}%) es un supuesto experimental, no un dato observado."
    )

    st.subheader("Capacidad y utilización")
    st.dataframe(
        tabla_2.style.format({
            "Sénior (médicos)": "{:.0f}",
            "Júnior (médicos)": "{:.0f}",
            "Total médicos": "{:.0f}",
            "Pacientes": "{:.0f}",
            "Demanda/hora": "{:.2f}",
            "Tasa mix (pac/h-médico)": "{:.2f}",
            "Capacidad diaria": "{:.2f}",
            "Utilización": "{:.2%}",
            "Wq (min)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Comparación del domingo")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(["Actual", "Escenario 2"], [util_actual[0], util_esc2[0]])
    ax.set_ylabel("Utilización")
    ax.set_title("Utilización del domingo")
    ax.set_ylim(0, max(1.0, float(max(util_actual[0], util_esc2[0])) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Simulación tipo Arena")
    st.caption(
        "Lógica: llegadas → cola FCFS → pool de atención médica → atención → salida. "
        "Los médicos no se interpretan como horas: son personal. La tasa promedio de 2.41 "
        "pac/h-médico se combina con una variabilidad individual para representar médicos "
        "más rápidos y más lentos."
    )

    if st.button("🎲 Ejecutar simulación Escenario 2", type="primary", use_container_width=True):
        with st.spinner(f"Ejecutando {replicas_2} réplicas..."):
            sim_actual = simular_semana_arena(
                pacientes=DEMANDA,
                personal_senior=S_BASE,
                personal_junior=J_BASE,
                tasa_senior=tasa_s_2,
                tasa_junior=tasa_j_2,
                horas_dia=horas_dia_2,
                variabilidad_pct=variabilidad_2,
                modo_llegadas=modo_llegadas_2,
                replicas=replicas_2,
                semilla=int(semilla_2),
            )

            S_actual_sim = S_BASE.copy()
            J_actual_sim = J_BASE.copy()

            S_esc_sim = S_BASE.copy()
            J_esc_sim = J_BASE.copy()
            S_esc_sim[0] = senior_nuevo
            J_esc_sim[0] = junior_nuevo

            sim_esc2 = simular_semana_arena(
                pacientes=DEMANDA,
                personal_senior=S_esc_sim,
                personal_junior=J_esc_sim,
                tasa_senior=tasa_s_2,
                tasa_junior=tasa_j_2,
                horas_dia=horas_dia_2,
                variabilidad_pct=variabilidad_2,
                modo_llegadas=modo_llegadas_2,
                replicas=replicas_2,
                semilla=int(semilla_2),
            )

        st.session_state["esc2_sim_a"] = sim_actual
        st.session_state["esc2_sim_b"] = sim_esc2

    if "esc2_sim_a" in st.session_state and "esc2_sim_b" in st.session_state:
        sim_a = st.session_state["esc2_sim_a"]
        sim_b = st.session_state["esc2_sim_b"]
        ga = sim_a["global"]
        gb = sim_b["global"]

        espera_a = ga["Espera promedio semanal (min)"].mean()
        espera_b = gb["Espera promedio semanal (min)"].mean()
        reduccion = (1.0 - espera_b / espera_a) * 100.0 if espera_a > 0 else 0.0

        comp = pd.DataFrame({
            "Indicador": [
                "Espera promedio semanal (min)",
                "P90 promedio diario (min)",
                "Cola máxima semanal promedio",
                "Pacientes pendientes",
                "Reducción de espera (%)",
            ],
            "Actual": [
                espera_a, ga["P90 promedio diario (min)"].mean(),
                ga["Cola máxima semanal"].mean(), ga["Pacientes pendientes"].mean(), 0.0,
            ],
            "Escenario 2": [
                espera_b, gb["P90 promedio diario (min)"].mean(),
                gb["Cola máxima semanal"].mean(), gb["Pacientes pendientes"].mean(), reduccion,
            ],
        })

        st.subheader("Resultados globales de la simulación")
        st.dataframe(comp.style.format({"Actual": "{:.2f}", "Escenario 2": "{:.2f}"}), use_container_width=True, hide_index=True)

        da = sim_a["resumen"]
        db = sim_b["resumen"]
        diario = pd.DataFrame({
            "Día": DIAS,
            "Espera Actual (min)": da["Espera promedio (min)"],
            "Espera Esc. 2 (min)": db["Espera promedio (min)"],
            "P90 Actual (min)": da["Espera P90 (min)"],
            "P90 Esc. 2 (min)": db["Espera P90 (min)"],
            "Sistema Actual (min)": da["Tiempo sistema promedio (min)"],
            "Sistema Esc. 2 (min)": db["Tiempo sistema promedio (min)"],
        })

        st.subheader("Resultados de simulación por día")
        st.dataframe(
            diario.style.format({
                "Espera Actual (min)": "{:.2f}", "Espera Esc. 2 (min)": "{:.2f}",
                "P90 Actual (min)": "{:.2f}", "P90 Esc. 2 (min)": "{:.2f}",
                "Sistema Actual (min)": "{:.2f}", "Sistema Esc. 2 (min)": "{:.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

        x = np.arange(7); width = 0.35
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(x - width/2, diario["Espera Actual (min)"], width, label="Actual")
        ax.bar(x + width/2, diario["Espera Esc. 2 (min)"], width, label="Escenario 2")
        ax.set_xticks(x); ax.set_xticklabels(DIAS); ax.set_ylabel("Minutos")
        ax.set_title("Tiempo promedio de espera — Actual vs Escenario 2")
        ax.legend(); ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig, use_container_width=True)

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.bar(x - width/2, da["Máxima cola"], width, label="Actual")
        ax.bar(x + width/2, db["Máxima cola"], width, label="Escenario 2")
        ax.set_xticks(x); ax.set_xticklabels(DIAS); ax.set_ylabel("Pacientes")
        ax.set_title("Máxima cola aproximada por día")
        ax.legend(); ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig, use_container_width=True)

        st.success(
            f"Con el mix {senior_nuevo:.0f} Sénior / {junior_nuevo:.0f} Júnior, "
            f"la espera promedio semanal simulada pasa de {espera_a:.2f} a {espera_b:.2f} min "
            f"({reduccion:.2f}% de reducción). El total dominical se mantiene en 264 médicos."
        )

    st.download_button(
        "⬇️ Descargar resultados Escenario 2",
        data=excel_bytes({"Escenario2_Capacidad": tabla_2}),
        file_name="hospital_gloria_escenario2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
