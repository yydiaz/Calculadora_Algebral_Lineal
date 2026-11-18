"""Tablas de iteraciones.

Pandas se usa para ordenar la tabla y exportarla a CSV. La interfaz solo
muestra ese resultado; no calcula el método.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from calculadora.metodos_iterativos import ResultadoMetodo


def formatear_numero(valor) -> str:
    if valor is None:
        return "—"
    try:
        if pd.isna(valor):
            return "—"
    except (TypeError, ValueError):
        pass
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return str(valor)
    if not math.isfinite(numero):
        return "no finito"
    if numero == 0:
        return "0"
    if abs(numero) >= 1e6 or abs(numero) < 1e-5:
        return f"{numero:.6e}"
    texto = f"{numero:.8f}".rstrip("0").rstrip(".")
    return texto if texto else "0"


def vector_a_texto(vector: np.ndarray | None) -> str:
    if vector is None:
        return "no disponible"
    return "[ " + ", ".join(formatear_numero(componente) for componente in vector) + " ]"


def tabla_desde_resultado(resultado: ResultadoMetodo) -> pd.DataFrame:
    if not resultado.historial:
        return pd.DataFrame(columns=["Iteración", "Error", "Residuo"])

    n = len(resultado.historial[0]["x"])
    filas = []
    for registro in resultado.historial:
        fila = {"Iteración": registro["iteracion"]}
        for i in range(n):
            fila[f"x{i + 1}"] = registro["x"][i]
        fila["Error"] = np.nan if registro["error"] is None else registro["error"]
        fila["Residuo"] = registro["residuo"]
        filas.append(fila)
    return pd.DataFrame(filas)


def guardar_csv(resultado: ResultadoMetodo, ruta: str) -> None:
    tabla = tabla_desde_resultado(resultado)
    tabla.to_csv(ruta, index=False, encoding="utf-8-sig")
