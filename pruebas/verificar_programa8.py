"""Comprueba los tres casos, las fórmulas y que la ventana pueda abrirse.

Se ejecuta con: python pruebas/verificar_programa8.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np

from calculadora.casos import CASOS, interpretar_caso
from calculadora.graficas import crear_figura_errores, guardar_figura
from calculadora.metodos_iterativos import (
    comparar_resultados,
    ejecutar_gauss_seidel,
    ejecutar_jacobi,
    paso_gauss_seidel,
    paso_jacobi,
)
from calculadora.referencia import solucion_referencia
from calculadora.tablas import guardar_csv
from calculadora.validaciones import ErrorValidacion, evaluar_reordenamiento, preparar_sistema

EVIDENCIAS = RAIZ / "evidencias"
SALIDA = []


def anotar(texto: str):
    print(texto)
    SALIDA.append(texto)


def exigir(condicion: bool, mensaje: str):
    if not condicion:
        raise AssertionError(mensaje)
    anotar("OK  " + mensaje)


def probar_formulas():
    A = np.array([[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]])
    b = np.array([6.0, 25.0, -11.0])
    x0 = np.zeros(3)
    jacobi = paso_jacobi(A, b, x0)
    gauss = paso_gauss_seidel(A, b, x0)
    exigir(np.allclose(jacobi, b / np.diag(A)), "Jacobi desde cero despeja b_i / a_ii")
    exigir(np.isclose(gauss[0], jacobi[0]), "La primera incógnita de Gauss-Seidel coincide al partir de cero")
    exigir(not np.allclose(gauss, jacobi), "Gauss-Seidel no repite el vector de Jacobi")
    manual_x2 = (25.0 - (-1.0) * gauss[0]) / 11.0
    exigir(np.isclose(gauss[1], manual_x2), "Gauss-Seidel usa el componente recién calculado")


def ejecutar_caso(caso):
    preparacion = preparar_sistema(
        caso["A"], caso["b"], caso["x0"], caso["tol"], caso["max_iter"]
    )
    jacobi = ejecutar_jacobi(preparacion)
    gauss = ejecutar_gauss_seidel(preparacion)
    referencia = solucion_referencia(preparacion.A, preparacion.b)
    comparacion = comparar_resultados(jacobi, gauss)
    return preparacion, jacobi, gauss, referencia, comparacion


def guardar_caso(caso, jacobi, gauss):
    EVIDENCIAS.mkdir(exist_ok=True)
    clave = caso["clave"]
    guardar_csv(jacobi, str(EVIDENCIAS / f"{clave}_jacobi.csv"))
    guardar_csv(gauss, str(EVIDENCIAS / f"{clave}_gauss_seidel.csv"))
    figura = crear_figura_errores([jacobi, gauss], caso["titulo"])
    guardar_figura(figura, str(EVIDENCIAS / f"{clave}_errores.png"))


def probar_casos():
    convergencia = CASOS[0]
    prep, jacobi, gauss, referencia, comparacion = ejecutar_caso(convergencia)
    exigir(prep.diagonal_dominante and not prep.reordenado, "El caso 1 ya es diagonal dominante y no se reordena")
    exigir(jacobi.convergio and gauss.convergio, "En el caso 1 los dos métodos alcanzan la tolerancia")
    exigir(referencia.disponible, "El caso 1 tiene solución de referencia")
    objetivo = np.array([3.0, -2.5, 7.0])
    exigir(np.allclose(referencia.solucion, objetivo, atol=1e-8), "La referencia del caso 1 es [3, -2.5, 7]")
    exigir(np.allclose(jacobi.solucion, objetivo, atol=5e-4), "Jacobi del caso 1 se acerca a la referencia")
    exigir(np.allclose(gauss.solucion, objetivo, atol=5e-4), "Gauss-Seidel del caso 1 se acerca a la referencia")
    exigir(jacobi.residuo_final < 1e-3 and gauss.residuo_final < 1e-3, "Los residuos del caso 1 quedan pequeños")
    exigir("linalg.solve" not in Path(RAIZ, "calculadora", "metodos_iterativos.py").read_text(encoding="utf-8"), "Los métodos iterativos no llaman a linalg.solve")
    guardar_caso(convergencia, jacobi, gauss)
    anotar(interpretar_caso(convergencia, prep, jacobi, gauss, referencia, comparacion))

    reorden = CASOS[1]
    prep, jacobi, gauss, referencia, comparacion = ejecutar_caso(reorden)
    exigir(prep.reordenado and prep.permutacion == [1, 0, 2], "El caso 2 reordena las filas 2, 1 y 3")
    exigir(prep.diagonal_dominante, "Después de reordenar, el caso 2 es diagonal dominante")
    exigir(abs(prep.A_original[0, 0]) == 0, "La diagonal original del caso 2 empieza en cero")
    exigir(jacobi.convergio and gauss.convergio, "Tras reordenar, los dos métodos convergen")
    exigir(np.allclose(referencia.solucion, [1.0, 2.0, 3.0], atol=1e-8), "La referencia del caso 2 es [1, 2, 3]")
    exigir(np.allclose(jacobi.solucion, [1, 2, 3], atol=5e-4), "Jacobi del caso 2 se acerca a [1, 2, 3]")
    exigir(np.allclose(gauss.solucion, [1, 2, 3], atol=5e-4), "Gauss-Seidel del caso 2 se acerca a [1, 2, 3]")
    info = evaluar_reordenamiento(np.array(reorden["A"], dtype=float))
    exigir(info["permutacion_dominante"] == [1, 0, 2], "La búsqueda de reordenamiento encuentra la permutación [1, 0, 2]")
    guardar_caso(reorden, jacobi, gauss)
    anotar(interpretar_caso(reorden, prep, jacobi, gauss, referencia, comparacion))

    divergencia = CASOS[2]
    prep, jacobi, gauss, referencia, comparacion = ejecutar_caso(divergencia)
    exigir(not prep.diagonal_dominante and not prep.reordenado, "El caso 3 no es dominante y no se reordena solo")
    info = evaluar_reordenamiento(np.array(divergencia["A"], dtype=float))
    exigir(info["permutacion_dominante"] is None, "El caso 3 no tiene un orden de filas diagonal dominante")
    exigir(jacobi.estado == "posible_divergencia", "Jacobi advierte posible divergencia en el caso 3")
    exigir(gauss.estado == "posible_divergencia", "Gauss-Seidel advierte posible divergencia en el caso 3")
    exigir(jacobi.error_final > jacobi.historial[1]["error"], "El error de Jacobi crece en el caso 3")
    exigir(gauss.error_final > gauss.historial[1]["error"], "El error de Gauss-Seidel crece en el caso 3")
    exigir(referencia.disponible, "El caso 3 sí tiene solución única de referencia")
    exigir(np.allclose(referencia.solucion, [1.0, 1.0], atol=1e-8), "La referencia del caso 3 es [1, 1]")
    exigir(np.linalg.norm(jacobi.solucion - referencia.solucion, ord=np.inf) > 0.5, "La última aproximación de Jacobi queda lejos de la referencia")
    guardar_caso(divergencia, jacobi, gauss)
    anotar(interpretar_caso(divergencia, prep, jacobi, gauss, referencia, comparacion))


def probar_validaciones():
    try:
        preparar_sistema([[0, 0, 1], [0, 0, 2], [1, 2, 3]], [1, 2, 3], [0, 0, 0], 0.001, 5)
        raise AssertionError("Debió rechazar una diagonal que no se puede reordenar")
    except ErrorValidacion:
        exigir(True, "Una diagonal irreparable detiene el cálculo con un mensaje")

    try:
        preparar_sistema([[1, 2], [3, 4]], [1], [0, 0], 0.001, 5)
        raise AssertionError("Debió rechazar dimensiones distintas")
    except ErrorValidacion:
        exigir(True, "Un vector b de otra dimensión se rechaza")

    try:
        preparar_sistema([[1, 2], [3, 4]], [1, 2], [0, 0], -0.1, 5)
        raise AssertionError("Debió rechazar la tolerancia")
    except ErrorValidacion:
        exigir(True, "Una tolerancia negativa se rechaza")

    try:
        preparar_sistema([["a", "1"], ["0", "1"]], ["1", "1"], ["0", "0"], "0,001", "4")
        raise AssertionError("Debió rechazar texto")
    except ErrorValidacion:
        exigir(True, "Un dato no numérico se rechaza y la coma decimal es válida en otros casos")

    preparacion = preparar_sistema([["4", "1"], ["1", "3"]], ["1", "2"], ["0", "0"], "0,001", "10")
    exigir(preparacion.tolerancia == 0.001, "La tolerancia acepta coma decimal")


def probar_ventana():
    import matplotlib

    matplotlib.use("TkAgg", force=True)
    from calculadora import interfaz as modulo_interfaz

    modulo_interfaz.messagebox.showinfo = lambda *args, **kwargs: None
    modulo_interfaz.messagebox.showwarning = lambda *args, **kwargs: None
    modulo_interfaz.messagebox.showerror = lambda *args, **kwargs: None
    Aplicacion = modulo_interfaz.Aplicacion

    app = Aplicacion()
    app.update_idletasks()
    app.update()
    for clave in ("inicio", "jacobi", "gauss", "comparacion", "referencia", "casos", "ayuda"):
        app.mostrar(clave)
        app.update()
    app.vistas["casos"].seleccionar(CASOS[0])
    app.vistas["casos"].ejecutar()
    app.update()
    app.vistas["comparacion"].cargar_caso(CASOS[0])
    app.vistas["comparacion"].calcular()
    app.update()
    contenido = app.vistas["comparacion"].texto_j.get("1.0", "end")
    exigir("Tolerancia alcanzada" in contenido or "Estado:" in contenido, "La comparación muestra el resultado en la ventana")
    exigir(app.vistas["comparacion"].lienzo.figura is not None, "La comparación dibuja la gráfica con Matplotlib")
    app.destroy()
    anotar("OK  La ventana abre el menú, ejecuta un caso y dibuja la gráfica")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    EVIDENCIAS.mkdir(exist_ok=True)
    try:
        probar_formulas()
        probar_validaciones()
        probar_casos()
        try:
            probar_ventana()
        except Exception as exc:
            anotar("NO SE PUDO COMPROBAR LA VENTANA: " + str(exc))
            anotar(traceback.format_exc())
            raise
    except Exception:
        (EVIDENCIAS / "resumen_verificacion.txt").write_text("\n".join(SALIDA), encoding="utf-8")
        raise
    (EVIDENCIAS / "resumen_verificacion.txt").write_text("\n".join(SALIDA), encoding="utf-8")
    anotar("Verificación terminada.")


if __name__ == "__main__":
    main()
