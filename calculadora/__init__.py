"""Calculadora de Álgebra Lineal, Programa 8.

Este paquete resuelve sistemas lineales con los métodos iterativos de Jacobi
y de Gauss-Seidel. La solución de referencia, cuando existe, se obtiene con
numpy.linalg.solve y no reemplaza a los métodos iterativos.

La separación en módulos permite incorporar después otros programas de la
calculadora sin mezclarlos con estos algoritmos.
"""

__version__ = "1.0.0"
