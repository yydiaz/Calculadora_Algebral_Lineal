"""Tres casos de prueba obligatorios del Programa 8.

Los números de iteraciones, errores y residuos no se escriben aquí. La
interpretación los toma de la ejecución real de los algoritmos.
"""

from __future__ import annotations

import numpy as np

from calculadora.metodos_iterativos import ResultadoMetodo
from calculadora.referencia import SolucionReferencia, diferencia_con_referencia
from calculadora.tablas import formatear_numero, vector_a_texto
from calculadora.validaciones import Preparacion

CASOS = [
    {
        "clave": "convergencia",
        "titulo": "Caso 1. Convergencia",
        "A": [
            [3.0, -0.1, -0.2],
            [0.1, 7.0, -0.3],
            [0.3, -0.2, 10.0],
        ],
        "b": [7.85, -19.3, 71.4],
        "x0": [0.0, 0.0, 0.0],
        "tol": 0.0001,
        "max_iter": 30,
        "objetivo": (
            "Usar un sistema estrictamente diagonal dominante por filas y comprobar "
            "que Jacobi y Gauss-Seidel se acercan a la misma solución. Después se "
            "comparan iteraciones, errores, residuos y la solución de referencia."
        ),
    },
    {
        "clave": "reordenamiento",
        "titulo": "Caso 2. Reordenamiento",
        "A": [
            [0.0, 7.0, 1.0],
            [8.0, 1.0, 1.0],
            [1.0, 1.0, 9.0],
        ],
        "b": [17.0, 13.0, 30.0],
        "x0": [0.0, 0.0, 0.0],
        "tol": 0.0001,
        "max_iter": 30,
        "objetivo": (
            "Mostrar un sistema cuya diagonal inicial tiene un cero. La calculadora "
            "debe detectarlo, buscar un orden válido de las ecuaciones, reordenar "
            "también b y solo entonces ejecutar los métodos."
        ),
    },
    {
        "clave": "divergencia",
        "titulo": "Caso 3. No convergencia",
        "A": [
            [1.0, 3.0],
            [2.0, 3.0],
        ],
        "b": [4.0, 5.0],
        "x0": [0.0, 0.0],
        "tol": 0.0001,
        "max_iter": 8,
        "objetivo": (
            "Observar un sistema que no es diagonal dominante en ningún orden de "
            "filas y cuyos errores crecen. La ausencia de dominancia no se usa como "
            "prueba de divergencia: la advertencia sale del comportamiento del error. "
            "El proceso debe terminar de forma controlada."
        ),
    },
]


def caso_por_clave(clave: str) -> dict:
    for caso in CASOS:
        if caso["clave"] == clave:
            return caso
    raise KeyError(clave)


def _linea_metodo(resultado: ResultadoMetodo, referencia: SolucionReferencia) -> str:
    diferencia = diferencia_con_referencia(resultado.solucion, referencia)
    diferencia_txt = formatear_numero(diferencia) if diferencia is not None else "no disponible"
    return (
        f"{resultado.nombre}: {resultado.estado_legible}. "
        f"Iteraciones realizadas: {resultado.iteraciones}. "
        f"Error final: {formatear_numero(resultado.error_final)}. "
        f"Residuo final: {formatear_numero(resultado.residuo_final)}. "
        f"Diferencia con la referencia (norma infinito): {diferencia_txt}. "
        f"Último vector: {vector_a_texto(resultado.solucion)}."
    )


def interpretar_caso(
    caso: dict,
    preparacion: Preparacion,
    jacobi: ResultadoMetodo,
    gauss: ResultadoMetodo,
    referencia: SolucionReferencia,
    comparacion: dict,
) -> str:
    """Arma la lectura del caso con los valores que realmente se calcularon."""
    lineas = [
        caso["objetivo"],
        "",
        f"Dominancia diagonal estricta de la matriz utilizada: "
        f"{'sí' if preparacion.diagonal_dominante else 'no'}.",
        f"¿Se reordenaron las ecuaciones?: {'sí' if preparacion.reordenado else 'no'}.",
    ]
    if preparacion.reordenado:
        lineas.append(preparacion.descripcion_permutacion)
    lineas.append("")
    lineas.extend(preparacion.mensajes)
    lineas.append("")
    lineas.append(_linea_metodo(jacobi, referencia))
    lineas.append(_linea_metodo(gauss, referencia))
    lineas.append(comparacion["texto"])
    if comparacion["diferencia"] is not None:
        lineas.append(
            "Diferencia entre las dos aproximaciones, en norma infinito: "
            f"{formatear_numero(comparacion['diferencia'])}."
        )
    lineas.append("")

    if caso["clave"] == "convergencia":
        if jacobi.convergio and gauss.convergio:
            lineas.append(
                "Los dos métodos alcanzaron la tolerancia. Esto coincide con lo que "
                "se espera de una matriz estrictamente diagonal dominante. Aun así, "
                "los vectores son aproximaciones: el error pedido compara iteraciones "
                "consecutivas, no afirma que el resultado sea exacto."
            )
        else:
            lineas.append(
                "En esta ejecución alguno de los métodos no alcanzó la tolerancia. "
                "Revise el máximo de iteraciones y la tabla antes de concluir."
            )
    elif caso["clave"] == "reordenamiento":
        lineas.append(
            "Sin reordenar, la primera ecuación tiene cero en la diagonal y el "
            "algoritmo no puede dividir. El intercambio de ecuaciones no cambia la "
            "solución del sistema lineal, pero sí cambia qué coeficiente queda en "
            "la diagonal y, por tanto, cómo iteran Jacobi y Gauss-Seidel."
        )
        if referencia.disponible and referencia.solucion is not None:
            lineas.append(
                "La solución de referencia del sistema reordenado es "
                f"{vector_a_texto(referencia.solucion)}."
            )
    else:
        lineas.append(
            "No cumplir la dominancia diagonal no demuestra, por sí solo, que un "
            "método diverja. Aquí la evidencia es otra: el error calculado crece y "
            "la tolerancia no se alcanza. Por eso el estado es una advertencia de "
            "posible divergencia y el programa no se cierra."
        )
        if referencia.disponible:
            lineas.append(
                "El sistema sí tiene solución única, porque el método directo de "
                "NumPy la encuentra. Esa referencia muestra que el problema no es la "
                "existencia de la solución, sino que estas iteraciones no se acercan "
                "a ella con el orden actual. "
                f"Referencia: {vector_a_texto(referencia.solucion)}. "
                f"Residuo de la referencia: {formatear_numero(referencia.residuo)}."
            )
        if jacobi.residuo_final is not None and gauss.residuo_final is not None:
            lineas.append(
                "Los residuos iterativos finales son grandes en comparación con esa "
                "referencia, así que las últimas aproximaciones no se aceptan como "
                "solución del sistema."
            )

    if referencia.disponible and jacobi.convergio:
        distancia = diferencia_con_referencia(jacobi.solucion, referencia)
        lineas.append(
            "Comprobación de Jacobi contra la referencia: diferencia "
            f"{formatear_numero(distancia)} en norma infinito y residuo "
            f"{formatear_numero(jacobi.residuo_final)}."
        )
    if referencia.disponible and gauss.convergio:
        distancia = diferencia_con_referencia(gauss.solucion, referencia)
        lineas.append(
            "Comprobación de Gauss-Seidel contra la referencia: diferencia "
            f"{formatear_numero(distancia)} en norma infinito y residuo "
            f"{formatear_numero(gauss.residuo_final)}."
        )
    return "\n".join(lineas)


def matrices_del_caso(caso: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.array(caso["A"], dtype=float),
        np.array(caso["b"], dtype=float),
        np.array(caso["x0"], dtype=float),
    )
