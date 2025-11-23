import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from avanzadas_programa import (
    f, grad_f, hess_f,
    gradiente_descendente,
    newton,
    cuasi_newton_bfgs,
    powell_conjugate_directions_robusto
)

def graficar_funcion(x_opt, y_opt, radio=15):

    x = np.linspace(x_opt - radio, x_opt + radio, 100)
    y = np.linspace(y_opt - radio, y_opt + radio, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "contour"}]],
        subplot_titles=("Superficie 3D", "Curvas de Nivel")
    )

    # Superficie
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale="Viridis",
        showscale=False,
        opacity=0.9
    ), row=1, col=1)

    # Punto mínimo en 3D
    fig.add_trace(go.Scatter3d(
        x=[x_opt], y=[y_opt], z=[f(x_opt, y_opt)],
        mode="markers",
        marker=dict(size=6, color="red", symbol="circle")
    ), row=1, col=1)

    # Contour
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale="Viridis",
        showscale=False,
        line=dict(width=1.2),
        contours=dict(showlabels=True)
    ), row=1, col=2)

    # Marcar el mínimo
    fig.add_trace(go.Scatter(
        x=[x_opt], y=[y_opt],
        mode="markers",
        marker=dict(size=10, color="red", symbol="x")
    ), row=1, col=2)

    fig.update_layout(height=600, width=1100)

    return fig


st.title("🔍 Optimizador Multimétodo con Gráficas Interactivas")
st.write("Selecciona un método, define el punto inicial")

metodo = st.selectbox(
    "Selecciona un método",
    [
        "Gradiente Descendente",
        "Newton",
        "BFGS (Cuasi-Newton)",
        "Powell (Direcciones Conjugadas)"
    ]
)

col1, col2 = st.columns(2)
x0 = col1.number_input("x₀", value=2.0)
y0 = col2.number_input("y₀", value=2.0)

x0_vec = [x0, y0]

tol = st.number_input("Tolerancia", value=1e-6, format="%.1e")
max_iter = st.number_input("Máximo de iteraciones", value=100, step=10)

alpha = st.number_input("Alpha inicial (solo GD / BFGS)", value=1.0)

st.divider()


if st.button("Ejecutar método"):

    if metodo == "Gradiente Descendente":
        sol, tabla = gradiente_descendente(f, grad_f, x0_vec, alpha, tol, max_iter)

    elif metodo == "Newton":
        sol, tabla = newton(f, grad_f, hess_f, x0_vec, tol, max_iter)

    elif metodo == "BFGS (Cuasi-Newton)":
        sol, tabla = cuasi_newton_bfgs(f, grad_f, x0_vec, tol, max_iter)

    elif metodo == "Powell (Direcciones Conjugadas)":
        sol, tabla = powell_conjugate_directions_robusto(f, x0_vec, tol, max_iter)

    x_opt, y_opt = sol
    st.success(f"📌 Mínimo encontrado: (x = {x_opt:.6f}, y = {y_opt:.6f})")
    st.write(f"Valor de la función f(x,y) = {f(x_opt, y_opt):.6f}")

    df = pd.DataFrame(tabla)
    st.subheader("📋 Tabla de iteraciones")
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Gráfica de la función alrededor del mínimo")
    fig = graficar_funcion(x_opt, y_opt, radio=15)
    st.plotly_chart(fig, use_container_width=True)