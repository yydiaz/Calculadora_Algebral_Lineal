"""Validaciones previas a Jacobi y Gauss-Seidel.

Comprueba dimensiones, valores numéricos, ceros en la diagonal y dominancia
diagonal estricta por filas. Si la diagonal tiene ceros, busca un
reordenamiento de ecuaciones. Reordenar filas de A exige reordenar b de la
misma manera; el orden de las incógnitas no cambia.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Por debajo de este valor el coeficiente se trata como cero.
TOL_CERO = 1e-12
MAX_ECUACIONES = 8
MAX_ITERACIONES = 500


class ErrorValidacion(Exception):
    """Entrada o estructura del sistema que impide ejecutar el método."""

    def __init__(self, mensaje: str) -> None:
        super().__init__(mensaje)
        self.mensaje = mensaje


@dataclass
class Preparacion:
    """Sistema listo para iterar, ya validado y, si hizo falta, reordenado."""

    A: np.ndarray
    b: np.ndarray
    x0: np.ndarray
    tolerancia: float
    max_iteraciones: int
    A_original: np.ndarray
    b_original: np.ndarray
    reordenado: bool
    permutacion: list[int] | None
    diagonal_dominante: bool
    mensajes: list[str]

    @property
    def descripcion_permutacion(self) -> str:
        if not self.permutacion:
            return "No se reordenaron las ecuaciones."
        movimientos = []
        for nueva, vieja in enumerate(self.permutacion):
            movimientos.append(
                f"la ecuación {vieja + 1} quedó en la posición {nueva + 1}"
            )
        return "Reordenamiento de filas: " + "; ".join(movimientos) + "."


def parsear_numero(texto: str) -> float:
    """Acepta punto o coma decimal, que es lo habitual al escribir en clase."""
    limpio = texto.strip().replace(" ", "")
    if limpio == "":
        raise ErrorValidacion(
            "Hay casillas vacías. Complete la matriz, los vectores, la tolerancia "
            "y el número de iteraciones."
        )
    if "," in limpio and "." in limpio:
        raise ErrorValidacion(
            f"No se entiende el número '{texto}'. Use solo punto o solo coma decimal."
        )
    limpio = limpio.replace(",", ".")
    try:
        valor = float(limpio)
    except ValueError as exc:
        raise ErrorValidacion(f"'{texto}' no es un número válido.") from exc
    if not np.isfinite(valor):
        raise ErrorValidacion(f"'{texto}' no es un valor finito.")
    return valor


def parsear_entero_positivo(texto: str, etiqueta: str, maximo: int) -> int:
    limpio = texto.strip().replace(" ", "")
    if limpio == "":
        raise ErrorValidacion(f"Indique {etiqueta}.")
    if "," in limpio or "." in limpio:
        raise ErrorValidacion(f"{etiqueta.capitalize()} debe ser un entero positivo.")
    try:
        valor = int(limpio)
    except ValueError as exc:
        raise ErrorValidacion(f"{etiqueta.capitalize()} debe ser un entero positivo.") from exc
    if valor < 1:
        raise ErrorValidacion(f"{etiqueta.capitalize()} debe ser un entero positivo.")
    if valor > maximo:
        raise ErrorValidacion(
            f"{etiqueta.capitalize()} no puede ser mayor que {maximo} "
            "para que la tabla de resultados siga siendo legible."
        )
    return valor


def matriz_a_texto(A: np.ndarray, b: np.ndarray | None = None) -> str:
    """Texto alineado para mostrar el sistema en la interfaz o en el informe."""
    A = np.asarray(A, dtype=float)
    lineas = []
    for i in range(A.shape[0]):
        coeficientes = "  ".join(f"{valor:10.4f}" for valor in A[i])
        if b is None:
            lineas.append(f"[ {coeficientes} ]")
        else:
            lineas.append(f"[ {coeficientes} ]  [ {b[i]:10.4f} ]")
    return "\n".join(lineas)


def tiene_cero_en_diagonal(A: np.ndarray) -> bool:
    diagonal = np.diag(np.asarray(A, dtype=float))
    return bool(np.any(np.abs(diagonal) <= TOL_CERO))


def posiciones_cero_diagonal(A: np.ndarray) -> list[int]:
    diagonal = np.diag(np.asarray(A, dtype=float))
    return [i + 1 for i, valor in enumerate(diagonal) if abs(valor) <= TOL_CERO]


def es_diagonal_dominante(A: np.ndarray) -> bool:
    """Dominancia estricta por filas: |a_ii| > suma de |a_ij| con j distinto de i."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    for i in range(n):
        absolutos = np.abs(A[i])
        diagonal = round(float(absolutos[i]), 12)
        suma_resto = round(float(np.sum(absolutos) - absolutos[i]), 12)
        if diagonal <= suma_resto:
            return False
    return True


def _columnas_utiles(fila: np.ndarray, solo_dominantes: bool) -> set[int]:
    absolutos = np.abs(np.asarray(fila, dtype=float))
    total = float(np.sum(absolutos))
    columnas = set()
    for j, valor in enumerate(absolutos):
        diagonal = round(float(valor), 12)
        if diagonal <= TOL_CERO:
            continue
        if not solo_dominantes:
            columnas.add(j)
            continue
        suma_resto = round(total - float(valor), 12)
        if diagonal > suma_resto:
            columnas.add(j)
    return columnas


def buscar_permutacion(A: np.ndarray, solo_dominantes: bool) -> list[int] | None:
    """Asigna cada posición diagonal a una fila distinta.

    El resultado `perm` cumple perm[posicion_nueva] = fila_original.
    Si solo_dominantes es verdadero, cada fila queda estrictamente dominante
    en su nueva diagonal. Si es falso, basta con que la diagonal no sea cero.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    opciones = [_columnas_utiles(A[i], solo_dominantes) for i in range(n)]
    orden = [-1] * n
    usado = [False] * n

    def buscar(columna: int) -> bool:
        if columna == n:
            return True
        for fila in range(n):
            if usado[fila] or columna not in opciones[fila]:
                continue
            usado[fila] = True
            orden[columna] = fila
            if buscar(columna + 1):
                return True
            usado[fila] = False
            orden[columna] = -1
        return False

    if not buscar(0):
        return None
    return orden[:]


def aplicar_permutacion(
    A: np.ndarray, b: np.ndarray, permutacion: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    if sorted(permutacion) != list(range(len(permutacion))):
        raise ErrorValidacion("El reordenamiento obtenido no es una permutación válida.")
    indices = np.array(permutacion, dtype=int)
    return A[indices].copy(), b[indices].copy()


def evaluar_reordenamiento(A: np.ndarray) -> dict:
    """Informa si conviene reordenar, sin modificar la matriz del usuario."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ErrorValidacion("Solo se puede reordenar una matriz cuadrada.")
    if not np.all(np.isfinite(A)):
        raise ErrorValidacion("La matriz contiene valores no finitos.")

    dominante_actual = es_diagonal_dominante(A)
    ceros = posiciones_cero_diagonal(A)
    perm_dominante = buscar_permutacion(A, solo_dominantes=True)
    perm_sin_ceros = buscar_permutacion(A, solo_dominantes=False)
    mensajes = []

    if ceros:
        posiciones = ", ".join(str(p) for p in ceros)
        mensajes.append(
            f"La diagonal actual tiene cero en la posición: {posiciones}. "
            "No se puede dividir entre ese coeficiente."
        )
    if dominante_actual:
        mensajes.append(
            "El orden actual ya es estrictamente diagonal dominante por filas. "
            "No hace falta reordenar."
        )
    elif perm_dominante is not None:
        mensajes.append(
            "Existe un reordenamiento de filas que deja la matriz estrictamente "
            "diagonal dominante. Puede aplicarlo si desea usar esa condición suficiente."
        )
    else:
        mensajes.append(
            "No se encontró un reordenamiento de filas que logre dominancia diagonal estricta."
        )

    if ceros and perm_sin_ceros is None:
        mensajes.append(
            "Tampoco existe un reordenamiento que deje la diagonal sin ceros. "
            "Jacobi y Gauss-Seidel no pueden ejecutarse con este sistema."
        )
    elif ceros and perm_sin_ceros is not None:
        mensajes.append(
            "Sí existe un reordenamiento que elimina los ceros de la diagonal."
        )

    return {
        "ceros": ceros,
        "diagonal_dominante": dominante_actual,
        "permutacion_dominante": perm_dominante,
        "permutacion_sin_ceros": perm_sin_ceros,
        "mensajes": mensajes,
    }


def _leer_matriz(filas: list[list[str]]) -> np.ndarray:
    if not filas:
        raise ErrorValidacion("Indique el tamaño del sistema.")
    n = len(filas)
    if n > MAX_ECUACIONES:
        raise ErrorValidacion(
            f"Esta calculadora admite como máximo {MAX_ECUACIONES} ecuaciones."
        )
    matriz = []
    for i, fila in enumerate(filas):
        if len(fila) != n:
            raise ErrorValidacion(
                f"La fila {i + 1} no tiene {n} coeficientes. A debe ser cuadrada."
            )
        matriz.append([parsear_numero(valor) for valor in fila])
    return np.array(matriz, dtype=float)


def _leer_vector(valores: list[str], nombre: str, n: int) -> np.ndarray:
    if len(valores) != n:
        raise ErrorValidacion(f"El vector {nombre} debe tener {n} componentes.")
    return np.array([parsear_numero(valor) for valor in valores], dtype=float)


def _matriz_desde_entrada(filas_a: list | np.ndarray) -> np.ndarray:
    if isinstance(filas_a, np.ndarray):
        A = np.array(filas_a, dtype=float, copy=True)
    elif filas_a and isinstance(filas_a[0], (list, tuple)) and filas_a[0] and isinstance(filas_a[0][0], str):
        return _leer_matriz(filas_a)
    else:
        A = np.array(filas_a, dtype=float, copy=True)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ErrorValidacion("La matriz A debe ser cuadrada.")
    if A.shape[0] < 1 or A.shape[0] > MAX_ECUACIONES:
        raise ErrorValidacion(f"El tamaño n debe estar entre 1 y {MAX_ECUACIONES}.")
    if not np.all(np.isfinite(A)):
        raise ErrorValidacion("La matriz A contiene valores no finitos.")
    return A


def _vector_desde_entrada(vector: list | np.ndarray, nombre: str, n: int) -> np.ndarray:
    if isinstance(vector, np.ndarray) or (len(vector) > 0 and not isinstance(vector[0], str)):
        arreglo = np.array(vector, dtype=float).reshape(-1)
        if arreglo.shape != (n,) or not np.all(np.isfinite(arreglo)):
            raise ErrorValidacion(f"El vector {nombre} debe tener {n} componentes finitos.")
        return arreglo
    return _leer_vector(list(vector), nombre, n)


def preparar_sistema(
    filas_a: list | np.ndarray,
    vector_b: list | np.ndarray,
    vector_x0: list | np.ndarray,
    tolerancia: str | float,
    max_iteraciones: str | int,
) -> Preparacion:
    """Valida la entrada y reordena solo cuando la diagonal tiene ceros.

    Si no hay ceros, se respeta el orden escrito por el usuario aunque la matriz
    no sea diagonal dominante. Esa condición es suficiente, no necesaria.
    """
    A = _matriz_desde_entrada(filas_a)
    n = A.shape[0]
    b = _vector_desde_entrada(vector_b, "b", n)
    x0 = _vector_desde_entrada(vector_x0, "inicial x0", n)

    if isinstance(tolerancia, str):
        tol = parsear_numero(tolerancia)
    else:
        tol = float(tolerancia)
        if not np.isfinite(tol):
            raise ErrorValidacion("La tolerancia debe ser un número finito.")
    if tol <= 0:
        raise ErrorValidacion("La tolerancia ε debe ser un decimal positivo.")

    if isinstance(max_iteraciones, str):
        max_iter = parsear_entero_positivo(
            max_iteraciones, "el máximo de iteraciones", MAX_ITERACIONES
        )
    else:
        max_iter = int(max_iteraciones)
        if max_iter < 1 or max_iter > MAX_ITERACIONES:
            raise ErrorValidacion(
                f"El máximo de iteraciones debe ser un entero entre 1 y {MAX_ITERACIONES}."
            )

    A_original = A.copy()
    b_original = b.copy()
    mensajes: list[str] = []
    permutacion = None
    reordenado = False

    if tiene_cero_en_diagonal(A):
        info = evaluar_reordenamiento(A)
        elegida = info["permutacion_dominante"] or info["permutacion_sin_ceros"]
        if elegida is None:
            raise ErrorValidacion(
                "La diagonal tiene ceros y no existe un reordenamiento de filas que "
                "los elimine. Jacobi y Gauss-Seidel necesitan dividir entre a_ii, "
                "así que el cálculo se detuvo."
            )
        A, b = aplicar_permutacion(A, b, elegida)
        if tiene_cero_en_diagonal(A):
            raise ErrorValidacion(
                "El reordenamiento no logró quitar los ceros de la diagonal."
            )
        reordenado = True
        permutacion = elegida
        mensajes.append(
            "La diagonal original tenía ceros. Se reordenaron las ecuaciones "
            "y también el vector b, con la misma permutación. "
            + _describir_permutacion(elegida)
        )
        if info["permutacion_dominante"] is not None:
            mensajes.append(
                "El orden nuevo es estrictamente diagonal dominante por filas."
            )
        else:
            mensajes.append(
                "El orden nuevo ya no tiene ceros en la diagonal, pero no es "
                "estrictamente diagonal dominante. El cálculo puede continuar."
            )

    dominante = es_diagonal_dominante(A)
    if dominante:
        mensajes.append(
            "La matriz utilizada es estrictamente diagonal dominante por filas. "
            "Esa condición es suficiente para que Jacobi y Gauss-Seidel converjan, "
            "aunque no es la única situación en la que pueden hacerlo."
        )
    else:
        mensajes.append(
            "La matriz utilizada no es estrictamente diagonal dominante por filas. "
            "Eso no demuestra que el método diverja: la dominancia es una condición "
            "suficiente, no necesaria. El cálculo se ejecutará para observar el error."
        )
        if not reordenado:
            info = evaluar_reordenamiento(A_original)
            if info["permutacion_dominante"] is not None:
                mensajes.append(
                    "Existe otro orden de las ecuaciones que sí es diagonal dominante. "
                    "Puede revisarlo con 'Evaluar reordenamiento'. Esta ejecución respeta "
                    "el orden que está escrito en la pantalla."
                )

    return Preparacion(
        A=A,
        b=b,
        x0=x0.copy(),
        tolerancia=tol,
        max_iteraciones=max_iter,
        A_original=A_original,
        b_original=b_original,
        reordenado=reordenado,
        permutacion=permutacion,
        diagonal_dominante=dominante,
        mensajes=mensajes,
    )


def _describir_permutacion(permutacion: list[int]) -> str:
    movimientos = [
        f"la ecuación {vieja + 1} pasó a la posición {nueva + 1}"
        for nueva, vieja in enumerate(permutacion)
    ]
    return "Detalle: " + "; ".join(movimientos) + "."
