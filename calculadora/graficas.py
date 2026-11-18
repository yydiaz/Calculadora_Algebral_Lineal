"""Gráficas del error con los valores calculados por los métodos.

No se dibujan curvas de ejemplo. Si un método no produjo errores, su serie
no aparece.
"""

from __future__ import annotations

from matplotlib.figure import Figure

from calculadora.metodos_iterativos import ResultadoMetodo

COLORES = {
    "Jacobi": "#1B6CA8",
    "Gauss-Seidel": "#0E8A6A",
}


def crear_figura_errores(
    resultados: list[ResultadoMetodo],
    titulo: str = "Error de los métodos iterativos",
) -> Figure:
    figura = Figure(figsize=(8.2, 3.8), dpi=100, facecolor="white")
    ejes = figura.add_subplot(111)
    errores_positivos = []
    hay_datos = False

    for resultado in resultados:
        iteraciones, errores = resultado.errores_registrados()
        if not iteraciones:
            continue
        hay_datos = True
        errores_positivos.extend(valor for valor in errores if valor > 0)
        ejes.plot(
            iteraciones,
            errores,
            marker="o",
            linewidth=2,
            markersize=4.5,
            label=resultado.nombre,
            color=COLORES.get(resultado.nombre, "#333333"),
        )

    ejes.set_title(titulo)
    ejes.set_xlabel("Número de iteración")
    ejes.set_ylabel("Error  ||x(k) − x(k−1)||∞")
    ejes.grid(True, linestyle="--", alpha=0.65)

    if not hay_datos:
        ejes.text(
            0.5,
            0.5,
            "No hay errores calculados para graficar",
            ha="center",
            va="center",
            transform=ejes.transAxes,
        )
    else:
        if errores_positivos:
            menor = min(errores_positivos)
            mayor = max(errores_positivos)
            if menor > 0 and mayor / menor >= 50:
                ejes.set_yscale("log")
                ejes.set_ylabel("Error (escala logarítmica)")
        ejes.legend(loc="best")

    figura.tight_layout()
    return figura


def guardar_figura(figura: Figure, ruta: str) -> None:
    figura.savefig(ruta, dpi=140, bbox_inches="tight", facecolor="white")
