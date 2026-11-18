"""Punto de entrada de la calculadora de Álgebra Lineal, Programa 8."""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from calculadora.interfaz import main


if __name__ == "__main__":
    main()
