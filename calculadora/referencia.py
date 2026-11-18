"""Solución de referencia para comprobar los métodos iterativos.

numpy.linalg.solve se usa solo aquí. No forma parte de Jacobi ni de Gauss-Seidel.
Si el sistema no tiene solución única, se informa y la calculadora sigue abierta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calculadora.metodos_iterativos import norma_inf
from calculadora.validaciones import ErrorValidacion


@dataclass
class SolucionReferencia:
    disponible: bool
    solucion: np.ndarray | None
    residuo: float | None
    mensaje: str


def solucion_referencia(A: np.ndarray, b: np.ndarray) -> SolucionReferencia:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or b.shape != (A.shape[0],):
        raise ErrorValidacion(
            "No se puede validar: A debe ser cuadrada y b debe tener n componentes."
        )
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ErrorValidacion("A o b contienen valores no finitos.")

    try:
        # Comprobación directa. No sustituye la iteración de Jacobi o Gauss-Seidel.
        solucion = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return SolucionReferencia(
            disponible=False,
            solucion=None,
            residuo=None,
            mensaje=(
                "numpy.linalg.solve no obtuvo una solución única. El sistema puede "
                "ser singular o estar mal condicionado. Esto no cambia el resultado "
                "iterativo: solo indica que no hay una referencia confiable."
            ),
        )

    if not np.all(np.isfinite(solucion)):
        return SolucionReferencia(
            disponible=False,
            solucion=None,
            residuo=None,
            mensaje="La solución de referencia no tiene valores finitos.",
        )

    residuo = norma_inf(A @ solucion - b)
    return SolucionReferencia(
        disponible=True,
        solucion=solucion,
        residuo=residuo,
        mensaje=(
            "Solución de referencia obtenida con numpy.linalg.solve. Sirve para "
            "comparar; no reemplaza a Jacobi ni a Gauss-Seidel."
        ),
    )


def diferencia_con_referencia(
    aproximacion: np.ndarray | None, referencia: SolucionReferencia
) -> float | None:
    if aproximacion is None or not referencia.disponible or referencia.solucion is None:
        return None
    return norma_inf(aproximacion - referencia.solucion)
