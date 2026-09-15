import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Simulador Hospital Gloria", layout="wide")

# ============================================================
# DATOS DEL CASO
# ============================================================
DIAS = ["Domingo", "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]
PACIENTES = np.array([582, 580, 565, 548, 525, 506, 508], dtype=float)
HORAS_MEDICO = np.array([264, 278, 263, 261, 252, 247, 271], dtype=float)
SENIOR = np.array([60, 114, 124, 97, 70, 93, 89], dtype=float)
JUNIOR = np.array([204, 164, 139, 164, 182, 154, 182], dtype=float)

HORAS_DIA = 23.0
TASA_BASE = 2.4  # pacientes por hora-médico

# ============================================================
# FUNCIONES
# ============================================================
def generar_llegadas(pacientes_dia, modo, rng):
    """
    Genera exactamente el número de pacientes diarios.
    - Estable: interarribos uniformes para representar flujo estable.
    - Variable: interarribos exponenciales, manteniendo exactamente
      el número de pacientes del día.
    """
    if pacientes_dia <= 0:
        return np.array([])

    if modo == "Estable":
        return np.linspace(0, HORAS_DIA * 60, int(pacientes_dia), endpoint=False)

    # Llegadas variables: exponenciales normalizadas para que el último
    # paciente permanezca dentro de las 23 horas.
    inter = rng.exponential(scale=1.0, size=int(pacientes_dia))
    tiempos = np.cumsum(inter)
    if tiempos[-1] == 0:
        return np.linspace(0, HORAS_DIA * 60, int(pacientes_dia), endpoint=False)
    tiempos = tiempos / tiempos[-1] * (HORAS_DIA * 60 - 0.01)
    return tiempos


def simular_semana(
    pacientes=PACIENTES,
    horas_medico=HORAS_MEDICO,
    tasa_base=TASA_BASE,
    ganancia_domingo=0.0,
    mix_senior_domingo=None,
    modo_llegadas="Estable",
    replicas=100,
    semilla=42,
):
    """
    Modelo de cola agregado por capacidad.

    Cada día:
      capacidad/h = horas_médico / 23 * tasa de servicio
      tiempo medio de servicio por paciente = 1 / capacidad/h

    Se modela un servidor agregado (pool de capacidad médica), porque
    el caso entrega HORAS-MÉDICO por día y una tasa promedio de 2.4
    pacientes/hora-médico, pero no entrega los horarios individuales
    de cada médico.

    El escenario 2 puede aumentar la tasa efectiva del domingo.
    """
    resultados = []

    for rep in range(replicas):
        rng = np.random.default_rng(semilla + rep)

        for d, dia in enumerate(DIAS):
            n = int(pacientes[d])
            horas = float(horas_medico[d])

            # Tasa efectiva.
            tasa = tasa_base

            if d == 0 and ganancia_domingo > 0:
                tasa = tasa_base * (1 + ganancia_domingo / 100.0)

            # Si se especifica un nuevo mix, se calcula solo como indicador.
            # No se altera la tasa automáticamente porque el caso NO entrega
            # una tasa separada para senior y junior.
            llegadas = generar_llegadas(n, modo_llegadas, rng)

            # Capacidad equivalente de médicos durante el día.
            med_equiv = horas / HORAS_DIA
            capacidad_hora = med_equiv * tasa
            servicio_medio_min = 60.0 / capacidad_hora

            # Simulación de una cola FCFS agregada.
            # Se genera un tiempo de servicio exponencial alrededor de la
            # capacidad promedio.
            disponible = 0.0
            esperas = []
            tiempos_sistema = []

            for t_llegada in llegadas:
                servicio = rng.exponential(servicio_medio_min)
                inicio = max(t_llegada, disponible)
                espera = inicio - t_llegada
                salida = inicio + servicio

                esperas.append(espera)
                tiempos_sistema.append(salida - t_llegada)
                disponible = salida

            capacidad_dia = horas * tasa
            utilizacion_teorica = n / capacidad_dia if capacidad_dia > 0 else np.nan

            resultados.append({
                "Rep": rep + 1,
                "Día": dia,
                "Pacientes": n,
                "Horas médico": horas,
                "Médicos equivalentes": med_equiv,
                "Tasa servicio": tasa,
                "Capacidad": capacidad_dia,
                "Utilización teórica": utilizacion_teorica,
                "Espera promedio (min)": np.mean(esperas) if esperas else 0,
                "Espera P90 (min)": np.percentile(esperas, 90) if esperas else 0,
                "Tiempo sistema promedio (min)": np.mean(tiempos_sistema) if tiempos_sistema else 0,
                "Máxima cola aproximada": np.nan,  # se calcula en otra función
            })

    df = pd.DataFrame(resultados)

    # Promedio de réplicas
    resumen = (
        df.groupby("Día", sort=False)
        .agg({
            "Pacientes": "mean",
            "Horas médico": "mean",
            "Médicos equivalentes": "mean",
            "Tasa servicio": "mean",
            "Capacidad": "mean",
            "Utilización teórica": "mean",
            "Espera promedio (min)": "mean",
            "Espera P90 (min)": "mean",
            "Tiempo sistema promedio (min)": "mean",
        })
        .reset_index()
    )

    return resumen, df


def calcular_capacidad_tabla(tasa_domingo=TASA_BASE):
    tasas = np.full(7, TASA_BASE, dtype=float)
    tasas[0] = tasa_domingo

    capacidad = HORAS_MEDICO * tasas
    utilizacion = PACIENTES / capacidad

    return pd.DataFrame({
        "Día": DIAS,
        "Senior (h)": SENIOR,
        "Junior (h)": JUNIOR,
        "Horas médico": HORAS_MEDICO,
        "Pacientes": PACIENTES,
        "Tasa servicio": tasas,
        "Capacidad": capacidad,
        "Utilización": utilizacion
    })


# ============================================================
# INTERFAZ
# ============================================================
st.title("Simulador dinámico — Hospital Gloria")
st.caption(
    "Simulación semanal (23 h/día) para comparar la situación actual "
    "con una reestructuración del mix Sénior/Júnior el domingo."
)

with st.sidebar:
    st.header("Parámetros del caso")

    tasa = st.number_input(
        "Tasa promedio de servicio (pacientes/hora-médico)",
        min_value=0.1,
        max_value=10.0,
        value=2.4,
        step=0.1,
    )

    horas_dia = st.number_input(
        "Horas de operación por día",
        min_value=1.0,
        max_value=24.0,
        value=23.0,
        step=1.0,
    )

    # Se mantiene la variable global sincronizada con el supuesto.
    HORAS_DIA = horas_dia

    st.header("Escenario 2 — Domingo")

    nuevo_mix = st.slider(
        "Horas Sénior el domingo",
        min_value=60,
        max_value=120,
        value=84,
        step=1,
    )

    horas_senior_actual = 60
    horas_total_domingo = 264
    horas_junior_nuevo = horas_total_domingo - nuevo_mix

    mix_actual = horas_senior_actual / horas_total_domingo * 100
    mix_nuevo = nuevo_mix / horas_total_domingo * 100

    ganancia = st.slider(
        "Ganancia de eficiencia por mayor supervisión (%)",
        min_value=0.0,
        max_value=30.0,
        value=5.0,
        step=0.5,
        help=(
            "Supuesto del equipo: el caso entrega una tasa promedio de 2.4 "
            "pacientes/hora-médico, pero no entrega una tasa separada para "
            "Sénior y Júnior. Por eso este parámetro permite probar cuánto "
            "aumentaría la tasa efectiva del domingo."
        ),
    )

    modo = st.radio(
        "Flujo de llegada",
        ["Estable", "Variable"],
        index=0,
        help=(
            "Estable: pacientes distribuidos uniformemente en las 23 horas. "
            "Variable: llegadas aleatorias alrededor del promedio diario."
        ),
    )

    replicas = st.slider(
        "Réplicas de simulación",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )

# ============================================================
# RESUMEN DEL ESCENARIO
# ============================================================
tasa_esc2 = tasa * (1 + ganancia / 100)
cap_actual_domingo = HORAS_MEDICO[0] * tasa
cap_esc2_domingo = HORAS_MEDICO[0] * tasa_esc2

util_actual = PACIENTES[0] / cap_actual_domingo
util_esc2 = PACIENTES[0] / cap_esc2_domingo

c1, c2, c3, c4 = st.columns(4)
c1.metric("Demanda semanal", f"{PACIENTES.sum():,.0f} pacientes")
c2.metric("Horas médico/semana", f"{HORAS_MEDICO.sum():,.0f} h")
c3.metric("Tasa base", f"{tasa:.2f} pac/h-médico")
c4.metric("Operación", f"{horas_dia:.0f} h/día")

st.subheader("Escenario 2 — Cambio del domingo")

a, b, c, d = st.columns(4)
a.metric("Mix Sénior actual", f"{mix_actual:.1f}%")
b.metric("Mix Sénior propuesto", f"{mix_nuevo:.1f}%")
c.metric("Capacidad domingo", f"{cap_actual_domingo:.0f} → {cap_esc2_domingo:.0f}")
d.metric("Utilización domingo", f"{util_actual:.1%} → {util_esc2:.1%}")

st.info(
    f"Supuesto de simulación: el aumento de Sénior de {horas_senior_actual} h "
    f"a {nuevo_mix} h mantiene las {horas_total_domingo} h médico totales. "
    f"Se modela una mejora de {ganancia:.1f}% en la tasa efectiva del domingo "
    f"(μ = {tasa_esc2:.2f} pac/h-médico). El porcentaje de mejora es una "
    f"suposición del equipo, no un dato entregado directamente por el caso."
)

# ============================================================
# TABLA DE CAPACIDAD
# ============================================================
tabla = calcular_capacidad_tabla(tasa_domingo=tasa_esc2)

tabla["Utilización"] = tabla["Utilización"].map(lambda x: f"{x:.1%}")
tabla["Tasa servicio"] = tabla["Tasa servicio"].map(lambda x: f"{x:.2f}")
tabla["Capacidad"] = tabla["Capacidad"].map(lambda x: f"{x:.0f}")

st.subheader("Capacidad semanal")

st.dataframe(
    tabla,
    use_container_width=True,
    hide_index=True,
)

# ============================================================
# SIMULACIONES
# ============================================================
if st.button("▶ Ejecutar simulación", type="primary", use_container_width=True):

    with st.spinner("Ejecutando simulaciones..."):

        # Actual: domingo sin ganancia
        resumen_actual, detalle_actual = simular_semana(
            pacientes=PACIENTES,
            horas_medico=HORAS_MEDICO,
            tasa_base=tasa,
            ganancia_domingo=0,
            modo_llegadas=modo,
            replicas=replicas,
            semilla=42,
        )

        # Escenario 2: solo domingo mejora
        resumen_esc2, detalle_esc2 = simular_semana(
            pacientes=PACIENTES,
            horas_medico=HORAS_MEDICO,
            tasa_base=tasa,
            ganancia_domingo=ganancia,
            modo_llegadas=modo,
            replicas=replicas,
            semilla=42,
        )

    # ========================================================
    # COMPARACIÓN
    # ========================================================
    comp = resumen_actual[[
        "Día",
        "Espera promedio (min)",
        "Espera P90 (min)",
        "Tiempo sistema promedio (min)"
    ]].copy()

    comp = comp.rename(columns={
        "Espera promedio (min)": "Espera actual (min)",
        "Espera P90 (min)": "P90 actual (min)",
        "Tiempo sistema promedio (min)": "Sistema actual (min)"
    })

    comp2 = resumen_esc2[[
        "Día",
        "Espera promedio (min)",
        "Espera P90 (min)",
        "Tiempo sistema promedio (min)"
    ]].copy()

    comp2 = comp2.rename(columns={
        "Espera promedio (min)": "Espera escenario 2 (min)",
        "Espera P90 (min)": "P90 escenario 2 (min)",
        "Tiempo sistema promedio (min)": "Sistema escenario 2 (min)"
    })

    comparacion = comp.merge(comp2, on="Día")

    comparacion["Reducción espera (%)"] = np.where(
        comparacion["Espera actual (min)"] > 0,
        (1 - comparacion["Espera escenario 2 (min)"] /
         comparacion["Espera actual (min)"]) * 100,
        0
    )

    st.subheader("Resultados de la simulación")

    st.dataframe(
        comparacion.round(2),
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # GRÁFICOS
    # ========================================================
    st.subheader("Impacto del Escenario 2")

    g1, g2 = st.columns(2)

    with g1:
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        x = np.arange(len(DIAS))
        w = 0.35

        espera1 = comparacion["Espera actual (min)"].values
        espera2 = comparacion["Espera escenario 2 (min)"].values

        ax1.bar(x - w/2, espera1, width=w, label="Actual")
        ax1.bar(x + w/2, espera2, width=w, label="Escenario 2")
        ax1.set_xticks(x)
        ax1.set_xticklabels(DIAS, rotation=35, ha="right")
        ax1.set_ylabel("Minutos")
        ax1.set_title("Tiempo promedio de espera")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.25)
        st.pyplot(fig1, use_container_width=True)

    with g2:
        fig2, ax2 = plt.subplots(figsize=(7, 4))

        tasa_actual = np.full(7, tasa)
        tasa_nueva = np.full(7, tasa)
        tasa_nueva[0] = tasa_esc2

        ax2.plot(DIAS, tasa_actual, marker="o", label="Actual")
        ax2.plot(DIAS, tasa_nueva, marker="o", label="Escenario 2")
        ax2.set_ylabel("Pacientes/hora-médico")
        ax2.set_title("Tasa efectiva de servicio")
        ax2.tick_params(axis="x", rotation=35)
        ax2.legend()
        ax2.grid(alpha=0.25)
        st.pyplot(fig2, use_container_width=True)

    # ========================================================
    # MENSAJE PARA EXPOSICIÓN
    # ========================================================
    espera_dom_actual = comparacion.loc[
        comparacion["Día"] == "Domingo", "Espera actual (min)"
    ].iloc[0]

    espera_dom_esc2 = comparacion.loc[
        comparacion["Día"] == "Domingo", "Espera escenario 2 (min)"
    ].iloc[0]

    reduccion_dom = (
        (1 - espera_dom_esc2 / espera_dom_actual) * 100
        if espera_dom_actual > 0 else 0
    )

    st.success(
        f"Conclusión de la corrida: al mantener la demanda y las horas médico "
        f"totales constantes, el Escenario 2 modifica solo el domingo. Con "
        f"{nuevo_mix} h Sénior ({mix_nuevo:.1f}% del total) y una mejora "
        f"supuesta de {ganancia:.1f}% en la tasa efectiva, la capacidad "
        f"dominical pasa de {cap_actual_domingo:.1f} a {cap_esc2_domingo:.1f} "
        f"pacientes/día. La espera promedio dominical pasa de "
        f"{espera_dom_actual:.1f} a {espera_dom_esc2:.1f} minutos "
        f"(reducción aproximada de {reduccion_dom:.1f}%)."
    )

    st.caption(
        "Importante: los resultados de espera son resultados del modelo, "
        "no cifras observadas del hospital. Para el trabajo deben reportarse "
        "como resultados de simulación bajo los supuestos elegidos."
    )

st.divider()

st.markdown(
    """
### Lógica del modelo

**Llegadas → cola → capacidad médica → atención → salida**

- La semana se modela como **7 días × 23 horas = 161 horas**.
- La demanda diaria utiliza los pacientes del cuadro proporcionado.
- La capacidad se calcula como **horas-médico × 2,4 pacientes/hora-médico**.
- El domingo del Escenario 2 mantiene las **264 horas médico totales**, pero aumenta las horas Sénior.
- Como el caso no entrega una tasa independiente para Sénior y Júnior, el efecto del nuevo mix se representa mediante una **ganancia de eficiencia configurable**. Esto debe presentarse como supuesto de simulación.
"""
)