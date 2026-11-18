"""Jacobi y Gauss-Seidel, con implementaciones independientes.

Jacobi calcula todo el vector nuevo usando solo la iteración anterior.
Gauss-Seidel sustituye cada componente en cuanto la calcula y la usa dentro
de la misma iteración.

Ninguno de los dos llama a numpy.linalg.solve. El criterio de parada sí es
común, porque la comparación entre métodos debe usar la misma medida de error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from calculadora.validaciones import TOL_CERO, ErrorValidacion, Preparacion


def norma_inf(vector: np.ndarray) -> float:
    """Norma infinito: el mayor valor absoluto de las componentes."""
    return float(np.max(np.abs(vector)))


def residuo_infinito(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """||Ax - b||∞. Un residuo pequeño significa que x cumple bien el sistema."""
    return norma_inf(A @ x - b)


@dataclass
class ResultadoMetodo:
    nombre: str
    estado: str
    mensaje: str
    solucion: np.ndarray | None
    iteraciones: int
    error_final: float | None
    residuo_final: float | None
    historial: list[dict] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    reordenado: bool = False
    diagonal_dominante: bool = False
    A: np.ndarray | None = None
    b: np.ndarray | None = None

    @property
    def convergio(self) -> bool:
        return self.estado == "tolerancia_alcanzada"

    @property
    def estado_legible(self) -> str:
        return {
            "tolerancia_alcanzada": "Tolerancia alcanzada",
            "maximo_iteraciones": "Máximo de iteraciones alcanzado",
            "posible_divergencia": "Posible divergencia",
            "error_validacion": "Error de validación",
        }.get(self.estado, self.estado)

    def errores_registrados(self) -> tuple[list[int], list[float]]:
        iteraciones = []
        errores = []
        for fila in self.historial:
            if fila["error"] is None:
                continue
            iteraciones.append(int(fila["iteracion"]))
            errores.append(float(fila["error"]))
        return iteraciones, errores


def paso_jacobi(A: np.ndarray, b: np.ndarray, x_anterior: np.ndarray) -> np.ndarray:
    """Una iteración de Jacobi.

    x_i^(k+1) = (b_i - suma_{j distinto de i} a_ij x_j^(k)) / a_ii

    Todas las componentes se calculan con x^(k). No se usa ningún valor nuevo.
    """
    n = len(b)
    x_nuevo = np.zeros(n, dtype=float)
    for i in range(n):
        if abs(A[i, i]) <= TOL_CERO:
            raise ErrorValidacion(
                f"No se puede dividir: a_{i + 1}{i + 1} es cero."
            )
        suma_otros = 0.0
        for j in range(n):
            if j != i:
                suma_otros += A[i, j] * x_anterior[j]
        x_nuevo[i] = (b[i] - suma_otros) / A[i, i]
    return x_nuevo


def paso_gauss_seidel(A: np.ndarray, b: np.ndarray, x_anterior: np.ndarray) -> np.ndarray:
    """Una iteración de Gauss-Seidel.

    x_i^(k+1) = (b_i - suma_{j < i} a_ij x_j^(k+1) - suma_{j > i} a_ij x_j^(k)) / a_ii

    Los componentes ya calculados en esta misma pasada entran de inmediato.
    """
    n = len(b)
    x_nuevo = np.zeros(n, dtype=float)
    for i in range(n):
        if abs(A[i, i]) <= TOL_CERO:
            raise ErrorValidacion(
                f"No se puede dividir: a_{i + 1}{i + 1} es cero."
            )
        suma_nuevos = 0.0
        for j in range(i):
            suma_nuevos += A[i, j] * x_nuevo[j]
        suma_anteriores = 0.0
        for j in range(i + 1, n):
            suma_anteriores += A[i, j] * x_anterior[j]
        x_nuevo[i] = (b[i] - suma_nuevos - suma_anteriores) / A[i, i]
    return x_nuevo


def _parece_divergente(errores: list[float]) -> bool:
    """Hay evidencia numérica de que el error no se está reduciendo.

    No se usa la dominancia diagonal para declarar divergencia. Solo se mira
    el comportamiento del error calculado.
    """
    if len(errores) < 2:
        return False
    if errores[-1] > errores[0]:
        return True
    if len(errores) >= 6:
        mitad = len(errores) // 2
        if min(errores[mitad:]) > min(errores[:mitad]):
            return True
    return False


def _ejecutar(preparacion: Preparacion, paso, nombre: str) -> ResultadoMetodo:
    A = np.array(preparacion.A, dtype=float, copy=True)
    b = np.array(preparacion.b, dtype=float, copy=True)
    x = np.array(preparacion.x0, dtype=float, copy=True)
    tolerancia = preparacion.tolerancia
    historial = [
        {
            "iteracion": 0,
            "x": x.copy(),
            "error": None,
            "residuo": residuo_infinito(A, x, b),
        }
    ]
    errores: list[float] = []
    estado = "maximo_iteraciones"
    extra = ""

    for k in range(1, preparacion.max_iteraciones + 1):
        x_nuevo = paso(A, b, x)
        if not np.all(np.isfinite(x_nuevo)):
            estado = "posible_divergencia"
            extra = (
                " Aparecieron valores no finitos y el proceso se detuvo "
                "para no mostrar resultados inválidos."
            )
            break

        error = norma_inf(x_nuevo - x)
        x = x_nuevo
        errores.append(error)
        historial.append(
            {
                "iteracion": k,
                "x": x.copy(),
                "error": error,
                "residuo": residuo_infinito(A, x, b),
            }
        )

        if error <= tolerancia:
            estado = "tolerancia_alcanzada"
            break
        if error > 1e12:
            estado = "posible_divergencia"
            extra = (
                " El error superó 10^12 y el proceso se detuvo de forma controlada."
            )
            break
    else:
        if _parece_divergente(errores):
            estado = "posible_divergencia"

    if estado == "maximo_iteraciones" and _parece_divergente(errores):
        estado = "posible_divergencia"

    mensajes = {
        "tolerancia_alcanzada": (
            "Se alcanzó la tolerancia: el error entre las dos últimas aproximaciones "
            "es menor o igual que ε. El vector obtenido es una aproximación, no una "
            "solución exacta."
        ),
        "maximo_iteraciones": (
            "Se llegó al máximo de iteraciones sin cumplir la tolerancia. El error "
            "disminuyó respecto al inicio, así que no hay evidencia de divergencia. "
            "Puede repetir el cálculo permitiendo más iteraciones. El resultado sigue "
            "siendo aproximado."
        ),
        "posible_divergencia": (
            "Advertencia: no se alcanzó la tolerancia y el error no muestra un "
            "acercamiento. Hay una posible falta de convergencia. La última fila es "
            "un registro del proceso, no una solución aceptada."
        ),
    }

    solucion = x.copy() if historial else None
    error_final = errores[-1] if errores else None
    residuo_final = historial[-1]["residuo"] if historial else None
    if estado == "posible_divergencia" and extra == "" and not errores:
        solucion = None

    return ResultadoMetodo(
        nombre=nombre,
        estado=estado,
        mensaje=mensajes[estado] + extra,
        solucion=solucion,
        iteraciones=len(errores),
        error_final=error_final,
        residuo_final=residuo_final,
        historial=historial,
        advertencias=list(preparacion.mensajes),
        reordenado=preparacion.reordenado,
        diagonal_dominante=preparacion.diagonal_dominante,
        A=A,
        b=b,
    )


def ejecutar_jacobi(preparacion: Preparacion) -> ResultadoMetodo:
    return _ejecutar(preparacion, paso_jacobi, "Jacobi")


def ejecutar_gauss_seidel(preparacion: Preparacion) -> ResultadoMetodo:
    return _ejecutar(preparacion, paso_gauss_seidel, "Gauss-Seidel")


def comparar_resultados(jacobi: ResultadoMetodo, gauss: ResultadoMetodo) -> dict:
    """Compara dos ejecuciones hechas con los mismos datos."""
    diferencia = None
    if jacobi.solucion is not None and gauss.solucion is not None:
        diferencia = norma_inf(jacobi.solucion - gauss.solucion)

    menos = None
    if jacobi.convergio and gauss.convergio:
        if jacobi.iteraciones < gauss.iteraciones:
            menos = "Jacobi"
        elif gauss.iteraciones < jacobi.iteraciones:
            menos = "Gauss-Seidel"
        else:
            menos = "empate"

    if menos == "Jacobi":
        texto_menos = (
            f"Con estos datos, Jacobi alcanzó la tolerancia en menos iteraciones "
            f"({jacobi.iteraciones} frente a {gauss.iteraciones})."
        )
    elif menos == "Gauss-Seidel":
        texto_menos = (
            f"Con estos datos, Gauss-Seidel alcanzó la tolerancia en menos iteraciones "
            f"({gauss.iteraciones} frente a {jacobi.iteraciones})."
        )
    elif menos == "empate":
        texto_menos = (
            f"Ambos métodos alcanzaron la tolerancia en {jacobi.iteraciones} iteraciones."
        )
    else:
        texto_menos = (
            "No se declara un método más rápido: la comparación del número de "
            "iteraciones solo se hace cuando los dos alcanzan la misma tolerancia."
        )
    if diferencia is not None and not (jacobi.convergio and gauss.convergio):
        texto_menos += (
            " La diferencia entre vectores, si se muestra, corresponde a los últimos "
            "registros y no a dos soluciones aceptadas."
        )

    return {
        "diferencia": diferencia,
        "menos_iteraciones": menos,
        "texto": texto_menos,
    }
