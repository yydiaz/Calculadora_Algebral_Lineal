"""Interfaz gráfica de la calculadora.

La ventana solo recoge datos, llama a los módulos numéricos y muestra tablas,
mensajes y gráficas. Las fórmulas no están en esta capa.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np
import matplotlib

try:
    matplotlib.use("TkAgg")
except Exception:
    pass

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from calculadora.ayuda_texto import AYUDA
from calculadora.casos import CASOS, interpretar_caso
from calculadora.graficas import crear_figura_errores
from calculadora.metodos_iterativos import (
    comparar_resultados,
    ejecutar_gauss_seidel,
    ejecutar_jacobi,
)
from calculadora.referencia import diferencia_con_referencia, solucion_referencia
from calculadora.tablas import formatear_numero, guardar_csv, tabla_desde_resultado, vector_a_texto
from calculadora.validaciones import (
    ErrorValidacion,
    aplicar_permutacion,
    evaluar_reordenamiento,
    matriz_a_texto,
    parsear_numero,
    preparar_sistema,
)

NAVY = "#0D2C54"
AZUL = "#1B6CA8"
AZUL_ACTIVO = "#15588A"
FONDO = "#E8EEF5"
TARJETA = "#FFFFFF"
TEXTO = "#1C2833"
VERDE = "#1E7A46"
NARANJA = "#A15C00"
ROJO = "#A32020"
GRIS = "#5C6B7A"


def _boton(master, texto, comando, color=AZUL):
    return tk.Button(
        master,
        text=texto,
        command=comando,
        bg=color,
        fg="white",
        activebackground=AZUL_ACTIVO,
        activeforeground="white",
        relief="flat",
        bd=0,
        padx=12,
        pady=7,
        font=("Segoe UI", 10, "bold"),
        cursor="hand2",
    )


def _tarjeta(master, titulo):
    marco = tk.LabelFrame(
        master,
        text=titulo,
        bg=TARJETA,
        fg=NAVY,
        font=("Segoe UI", 11, "bold"),
        padx=12,
        pady=10,
        labelanchor="nw",
    )
    return marco


def _etiqueta(master, texto, fondo=TARJETA, negrita=False):
    return tk.Label(
        master,
        text=texto,
        bg=fondo,
        fg=TEXTO,
        font=("Segoe UI", 10, "bold" if negrita else "normal"),
        justify="left",
        anchor="w",
    )


class MarcoDesplazable(tk.Frame):
    """Contenedor con scroll vertical para que la ventana pueda achicarse."""

    def __init__(self, master):
        super().__init__(master, bg=FONDO)
        self.canvas = tk.Canvas(self, bg=FONDO, highlightthickness=0)
        barra = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.interior = tk.Frame(self.canvas, bg=FONDO)
        self.ventana = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")
        self.canvas.configure(yscrollcommand=barra.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        barra.pack(side="right", fill="y")
        self.interior.bind("<Configure>", self._al_cambiar_interior)
        self.canvas.bind("<Configure>", self._al_cambiar_canvas)
        self.canvas.bind("<Enter>", self._activar_rueda)
        self.canvas.bind("<Leave>", self._desactivar_rueda)

    def _al_cambiar_interior(self, _evento):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _al_cambiar_canvas(self, evento):
        self.canvas.itemconfigure(self.ventana, width=evento.width)

    def _activar_rueda(self, _evento):
        self.canvas.bind_all("<MouseWheel>", self._rueda)

    def _desactivar_rueda(self, _evento):
        self.canvas.unbind_all("<MouseWheel>")

    def _rueda(self, evento):
        self.canvas.yview_scroll(int(-evento.delta / 120), "units")


class LienzoGrafica(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=TARJETA, height=400)
        self.pack_propagate(False)
        self._canvas = None
        self.figura = None

    def mostrar(self, figura):
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
        self.figura = figura
        self._canvas = FigureCanvasTkAgg(figura, master=self)
        self._canvas.draw()
        widget = self._canvas.get_tk_widget()
        widget.pack(fill="both", expand=True)

    def limpiar(self):
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        self.figura = None


class TablaResultados(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=TARJETA)
        self.tree = ttk.Treeview(self, show="headings", height=11)
        vertical = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        horizontal = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.tree.tag_configure("par", background="#F7FAFC")
        self.tree.tag_configure("impar", background="#FFFFFF")
        self.dataframe = None

    def mostrar_dataframe(self, dataframe):
        self.dataframe = dataframe
        self.tree.delete(*self.tree.get_children())
        columnas = list(dataframe.columns)
        self.tree["columns"] = columnas
        for columna in columnas:
            self.tree.heading(columna, text=columna)
            ancho = 90 if columna == "Iteración" else 120
            self.tree.column(columna, width=ancho, anchor="center", minwidth=80)
        for indice, fila in dataframe.iterrows():
            valores = []
            for columna in columnas:
                if columna == "Iteración":
                    valores.append(str(int(fila[columna])))
                else:
                    valores.append(formatear_numero(fila[columna]))
            etiqueta = "par" if int(indice) % 2 == 0 else "impar"
            self.tree.insert("", "end", values=valores, tags=(etiqueta,))

    def limpiar(self):
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = ()
        self.dataframe = None


class PanelSistema(tk.Frame):
    """Campos de A, b y, si corresponde, x0, tolerancia e iteraciones."""

    def __init__(self, master, incluir_iteracion=True):
        super().__init__(master, bg=TARJETA)
        self.incluir_iteracion = incluir_iteracion
        self.n_var = tk.StringVar(value="3")
        self.tol = tk.StringVar(value="0.0001")
        self.max_iter = tk.StringVar(value="30")
        self.aviso = tk.StringVar(value="")
        self.vars_a: list[list[tk.StringVar]] = []
        self.vars_b: list[tk.StringVar] = []
        self.vars_x: list[tk.StringVar] = []
        self._ultimo_reorden = None
        self._bloqueado = False
        self._n_actual = 0

        controles = tk.Frame(self, bg=TARJETA)
        controles.pack(fill="x", pady=(0, 8))
        _etiqueta(controles, "Tamaño n del sistema:").pack(side="left")
        self.spin = tk.Spinbox(
            controles,
            from_=1,
            to=8,
            width=5,
            textvariable=self.n_var,
            font=("Segoe UI", 10),
            command=self._cambio_n,
        )
        self.spin.pack(side="left", padx=(8, 16))
        _boton(controles, "Evaluar reordenamiento", self.evaluar_reordenamiento).pack(
            side="left", padx=(0, 8)
        )
        _boton(controles, "Aplicar reordenamiento", self.aplicar_reordenamiento, VERDE).pack(
            side="left"
        )

        self.lbl_aviso = tk.Label(
            self,
            textvariable=self.aviso,
            bg=TARJETA,
            fg=NAVY,
            justify="left",
            anchor="w",
            wraplength=860,
            font=("Segoe UI", 10),
        )
        self.lbl_aviso.pack(fill="x", pady=(0, 8))

        self.rejilla = tk.Frame(self, bg=TARJETA)
        self.rejilla.pack(fill="x")

        if incluir_iteracion:
            parametros = tk.Frame(self, bg=TARJETA)
            parametros.pack(fill="x", pady=(10, 0))
            _etiqueta(parametros, "Tolerancia ε:").pack(side="left")
            tk.Entry(parametros, textvariable=self.tol, width=12, font=("Segoe UI", 10)).pack(
                side="left", padx=(6, 18)
            )
            _etiqueta(parametros, "Máximo de iteraciones:").pack(side="left")
            tk.Entry(
                parametros, textvariable=self.max_iter, width=8, font=("Segoe UI", 10)
            ).pack(side="left", padx=(6, 0))

        self.n_var.trace_add("write", lambda *_: self._cambio_n())
        self._cambio_n()

    def _cambio_n(self):
        if self._bloqueado:
            return
        try:
            n = int(self.n_var.get())
        except ValueError:
            return
        if n < 1 or n > 8 or n == self._n_actual:
            return
        self._n_actual = n
        self._construir(n)

    def _construir(self, n: int):
        for hijo in self.rejilla.winfo_children():
            hijo.destroy()
        self.vars_a = []
        self.vars_b = []
        self.vars_x = []
        self._ultimo_reorden = None

        tk.Label(self.rejilla, text="Matriz A", bg=TARJETA, fg=NAVY, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=n, pady=(0, 4)
        )
        tk.Label(self.rejilla, text="b", bg=TARJETA, fg=NAVY, font=("Segoe UI", 10, "bold")).grid(
            row=0, column=n, padx=(12, 0)
        )
        if self.incluir_iteracion:
            tk.Label(
                self.rejilla, text="x0", bg=TARJETA, fg=NAVY, font=("Segoe UI", 10, "bold")
            ).grid(row=0, column=n + 1, padx=(12, 0))

        for i in range(n):
            fila_vars = []
            for j in range(n):
                variable = tk.StringVar()
                entrada = tk.Entry(self.rejilla, textvariable=variable, width=8, font=("Consolas", 11))
                entrada.grid(row=i + 1, column=j, padx=2, pady=2)
                fila_vars.append(variable)
            self.vars_a.append(fila_vars)
            variable_b = tk.StringVar()
            tk.Entry(self.rejilla, textvariable=variable_b, width=8, font=("Consolas", 11)).grid(
                row=i + 1, column=n, padx=(12, 2), pady=2
            )
            self.vars_b.append(variable_b)
            if self.incluir_iteracion:
                variable_x = tk.StringVar()
                tk.Entry(
                    self.rejilla, textvariable=variable_x, width=8, font=("Consolas", 11)
                ).grid(row=i + 1, column=n + 1, padx=(12, 2), pady=2)
                self.vars_x.append(variable_x)

    def limpiar(self):
        for fila in self.vars_a:
            for variable in fila:
                variable.set("")
        for variable in self.vars_b:
            variable.set("")
        for variable in self.vars_x:
            variable.set("")
        self.tol.set("0.0001")
        self.max_iter.set("30")
        self.aviso.set("Se limpiaron la matriz y los vectores. La tolerancia volvió a 0.0001.")
        self._ultimo_reorden = None

    def cargar_caso(self, caso: dict):
        n = len(caso["A"])
        self._bloqueado = True
        self.n_var.set(str(n))
        self._bloqueado = False
        self._n_actual = 0
        self._cambio_n()
        for i in range(n):
            for j in range(n):
                self.vars_a[i][j].set(str(caso["A"][i][j]))
            self.vars_b[i].set(str(caso["b"][i]))
            if self.incluir_iteracion:
                self.vars_x[i].set(str(caso["x0"][i]))
        if self.incluir_iteracion:
            self.tol.set(str(caso["tol"]))
            self.max_iter.set(str(caso["max_iter"]))
        self.aviso.set(f"Datos cargados: {caso['titulo']}.")
        self._ultimo_reorden = None

    def _matriz_y_b(self):
        n = len(self.vars_a)
        if n == 0:
            raise ErrorValidacion("Indique el tamaño del sistema.")
        filas = []
        for i in range(n):
            filas.append([parsear_numero(self.vars_a[i][j].get()) for j in range(n)])
        b = [parsear_numero(variable.get()) for variable in self.vars_b]
        return np.array(filas, dtype=float), np.array(b, dtype=float)

    def preparar(self):
        n = len(self.vars_a)
        filas = [[self.vars_a[i][j].get() for j in range(n)] for i in range(n)]
        b = [variable.get() for variable in self.vars_b]
        x0 = [variable.get() for variable in self.vars_x]
        return preparar_sistema(filas, b, x0, self.tol.get(), self.max_iter.get())

    def evaluar_reordenamiento(self):
        try:
            A, _b = self._matriz_y_b()
            info = evaluar_reordenamiento(A)
        except ErrorValidacion as exc:
            self.aviso.set(exc.mensaje)
            messagebox.showwarning("Reordenamiento", exc.mensaje)
            return
        self._ultimo_reorden = info
        self.aviso.set(" ".join(info["mensajes"]))

    def aplicar_reordenamiento(self):
        try:
            A, b = self._matriz_y_b()
            info = evaluar_reordenamiento(A)
        except ErrorValidacion as exc:
            self.aviso.set(exc.mensaje)
            messagebox.showwarning("Reordenamiento", exc.mensaje)
            return
        self._ultimo_reorden = info
        if info["diagonal_dominante"]:
            self.aviso.set("El orden actual ya es estrictamente diagonal dominante. No se cambió nada.")
            return
        permutacion = info["permutacion_dominante"] or (
            info["permutacion_sin_ceros"] if info["ceros"] else None
        )
        if permutacion is None:
            mensaje = " ".join(info["mensajes"])
            self.aviso.set(mensaje)
            messagebox.showinfo("Reordenamiento", mensaje)
            return
        n = len(permutacion)
        filas = [[self.vars_a[i][j].get() for j in range(n)] for i in range(n)]
        valores_b = [variable.get() for variable in self.vars_b]
        for nueva, vieja in enumerate(permutacion):
            for columna in range(n):
                self.vars_a[nueva][columna].set(filas[vieja][columna])
            self.vars_b[nueva].set(valores_b[vieja])
        A2, b2 = aplicar_permutacion(A, b, permutacion)
        self.aviso.set(
            "Se aplicó el reordenamiento a A y a b. El vector inicial no cambia, "
            "porque las incógnitas siguen en el mismo orden.\n"
            + matriz_a_texto(A2, b2)
        )


def _caja_texto(master, alto=12):
    caja = scrolledtext.ScrolledText(
        master,
        height=alto,
        wrap="word",
        font=("Segoe UI", 10),
        bg="#F8FBFE",
        fg=TEXTO,
        relief="flat",
        padx=8,
        pady=8,
    )
    caja.configure(state="disabled")
    return caja


def _escribir(caja: scrolledtext.ScrolledText, texto: str):
    caja.configure(state="normal")
    caja.delete("1.0", "end")
    caja.insert("1.0", texto)
    caja.configure(state="disabled")


def _color_estado(estado: str) -> str:
    if estado == "tolerancia_alcanzada":
        return VERDE
    if estado == "maximo_iteraciones":
        return NARANJA
    return ROJO


def texto_de_metodo(preparacion, resultado, referencia) -> str:
    lineas = list(resultado.advertencias)
    lineas.append("")
    lineas.append(f"Estado: {resultado.estado_legible}.")
    lineas.append(resultado.mensaje)
    lineas.append("")
    lineas.append(
        "El error es ||x(k) − x(k−1)||∞: mide cuánto cambió la aproximación "
        "respecto de la iteración anterior. No es, por sí solo, la distancia a la "
        "solución verdadera."
    )
    lineas.append(
        "El residuo es ||A x − b||∞: mide cuánto falla el sistema al sustituir "
        "la aproximación."
    )
    lineas.append("")
    lineas.append(f"Iteraciones realizadas: {resultado.iteraciones}.")
    lineas.append(f"Error final: {formatear_numero(resultado.error_final)}.")
    lineas.append(f"Residuo final: {formatear_numero(resultado.residuo_final)}.")
    if resultado.convergio:
        lineas.append(f"Solución aproximada: {vector_a_texto(resultado.solucion)}.")
    else:
        lineas.append(
            "No se presenta una solución aceptada. Último vector registrado: "
            f"{vector_a_texto(resultado.solucion)}."
        )
    lineas.append("")
    lineas.append(referencia.mensaje)
    if referencia.disponible:
        lineas.append(f"Solución de referencia: {vector_a_texto(referencia.solucion)}.")
        lineas.append(f"Residuo de la referencia: {formatear_numero(referencia.residuo)}.")
        distancia = diferencia_con_referencia(resultado.solucion, referencia)
        lineas.append(
            "Diferencia entre el último vector y la referencia, en norma infinito: "
            f"{formatear_numero(distancia)}."
        )
        if not resultado.convergio:
            lineas.append(
                "Como el método no alcanzó la tolerancia, esa diferencia no valida "
                "la aproximación."
            )
    if preparacion.reordenado:
        lineas.append("")
        lineas.append("Sistema original:")
        lineas.append(matriz_a_texto(preparacion.A_original, preparacion.b_original))
        lineas.append("")
        lineas.append("Sistema utilizado después del reordenamiento:")
        lineas.append(matriz_a_texto(preparacion.A, preparacion.b))
    return "\n".join(lineas)


class VistaInicio(MarcoDesplazable):
    def __init__(self, master, app):
        super().__init__(master)
        marco = self.interior
        intro = _tarjeta(marco, "Programa 8")
        intro.pack(fill="x", padx=18, pady=16)
        texto = (
            "Esta calculadora resuelve sistemas de ecuaciones lineales con los "
            "métodos iterativos de Jacobi y de Gauss-Seidel. Desde el menú se entra "
            "a cada método, a la comparación, a la validación con numpy.linalg.solve "
            "y a los tres casos de prueba.\n\n"
            "El método directo solo comprueba resultados. No sustituye las fórmulas "
            "iterativas. La estructura queda separada para poder agregar después otros "
            "programas de la asignatura, sin mezclarlos con este módulo."
        )
        tk.Label(
            intro, text=texto, bg=TARJETA, fg=TEXTO, justify="left", wraplength=860, font=("Segoe UI", 11)
        ).pack(anchor="w")

        guia = _tarjeta(marco, "Recorrido sugerido para la exposición")
        guia.pack(fill="x", padx=18, pady=(0, 16))
        pasos = (
            "1. Abra el Caso 1 y muestre que ambos métodos convergen.\n"
            "2. Abra el Caso 2 y explique el cero de la diagonal y el reordenamiento.\n"
            "3. Abra el Caso 3 y muestre la advertencia de posible divergencia, sin cerrar el programa.\n"
            "4. En Comparación, interprete la tabla, el residuo y la gráfica de errores.\n"
            "5. En Validación de referencia, compare la aproximación con la solución de NumPy."
        )
        tk.Label(
            guia, text=pasos, bg=TARJETA, fg=TEXTO, justify="left", font=("Segoe UI", 11)
        ).pack(anchor="w")
        _boton(guia, "Ir a los casos de prueba", lambda: app.mostrar("casos")).pack(anchor="w", pady=(10, 0))


class VistaMetodo(MarcoDesplazable):
    def __init__(self, master, app, titulo, ejecutor):
        super().__init__(master)
        self.app = app
        self.ejecutor = ejecutor
        self.resultado = None
        marco = self.interior

        barra = tk.Frame(marco, bg=FONDO)
        barra.pack(fill="x", padx=18, pady=(16, 8))
        self.combo = ttk.Combobox(
            barra,
            state="readonly",
            width=32,
            values=[caso["titulo"] for caso in CASOS],
            font=("Segoe UI", 10),
        )
        self.combo.pack(side="left")
        _boton(barra, "Cargar ejemplo", self.cargar_ejemplo).pack(side="left", padx=8)
        _boton(barra, "Limpiar", self.limpiar, GRIS).pack(side="left", padx=(0, 8))
        _boton(barra, "Calcular", self.calcular, VERDE).pack(side="left")
        _boton(barra, "Inicio", lambda: app.mostrar("inicio"), NAVY).pack(side="right")

        tarjeta = _tarjeta(marco, "Datos de entrada")
        tarjeta.pack(fill="x", padx=18, pady=8)
        self.panel = PanelSistema(tarjeta, incluir_iteracion=True)
        self.panel.pack(fill="x")

        self.estado = tk.Label(marco, text="", bg=FONDO, fg=NAVY, font=("Segoe UI", 12, "bold"))
        self.estado.pack(anchor="w", padx=22, pady=(8, 0))

        caja = _tarjeta(marco, "Resultado")
        caja.pack(fill="x", padx=18, pady=8)
        self.resumen = _caja_texto(caja, 14)
        self.resumen.pack(fill="x")

        tabla_marco = _tarjeta(marco, "Tabla de iteraciones")
        tabla_marco.pack(fill="x", padx=18, pady=8)
        self.tabla = TablaResultados(tabla_marco)
        self.tabla.pack(fill="x")
        _boton(tabla_marco, "Guardar tabla CSV", self.guardar_tabla).pack(anchor="w", pady=(8, 0))

        grafica_marco = _tarjeta(marco, "Error por iteración")
        grafica_marco.pack(fill="x", padx=18, pady=(8, 18))
        self.lienzo = LienzoGrafica(grafica_marco)
        self.lienzo.pack(fill="x")
        _boton(grafica_marco, "Guardar gráfica", self.guardar_grafica).pack(anchor="w", pady=(8, 0))
        self.titulo = titulo

    def cargar_ejemplo(self):
        indice = self.combo.current()
        if indice < 0:
            messagebox.showinfo("Ejemplo", "Elija un caso en la lista.")
            return
        self.panel.cargar_caso(CASOS[indice])

    def limpiar(self):
        self.panel.limpiar()
        self.estado.configure(text="")
        _escribir(self.resumen, "")
        self.tabla.limpiar()
        self.lienzo.limpiar()
        self.resultado = None

    def calcular(self):
        try:
            preparacion = self.panel.preparar()
            resultado = self.ejecutor(preparacion)
            referencia = solucion_referencia(preparacion.A, preparacion.b)
        except ErrorValidacion as exc:
            self._mostrar_error(exc.mensaje)
            return
        except Exception as exc:
            self._mostrar_error(
                "Ocurrió un problema inesperado y el cálculo se detuvo sin cerrar "
                f"la calculadora.\n\nDetalle: {exc}"
            )
            return

        self.resultado = resultado
        self.estado.configure(text=resultado.estado_legible, fg=_color_estado(resultado.estado))
        _escribir(self.resumen, texto_de_metodo(preparacion, resultado, referencia))
        self.tabla.mostrar_dataframe(tabla_desde_resultado(resultado))
        self.lienzo.mostrar(
            crear_figura_errores([resultado], f"Error de {resultado.nombre}")
        )
        if resultado.estado == "posible_divergencia":
            messagebox.showwarning("Posible divergencia", resultado.mensaje)
        elif resultado.estado == "tolerancia_alcanzada":
            messagebox.showinfo("Cálculo terminado", "Se alcanzó la tolerancia. La solución es aproximada.")

    def _mostrar_error(self, mensaje: str):
        self.resultado = None
        self.estado.configure(text="Error de validación", fg=ROJO)
        _escribir(self.resumen, mensaje)
        self.tabla.limpiar()
        self.lienzo.limpiar()
        messagebox.showerror("No se pudo calcular", mensaje)

    def guardar_tabla(self):
        if self.resultado is None:
            messagebox.showinfo("Tabla", "Primero ejecute un cálculo.")
            return
        ruta = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="Guardar tabla de iteraciones",
        )
        if not ruta:
            return
        guardar_csv(self.resultado, ruta)
        messagebox.showinfo("Tabla", "La tabla se guardó correctamente.")

    def guardar_grafica(self):
        if self.lienzo.figura is None:
            messagebox.showinfo("Gráfica", "Primero ejecute un cálculo.")
            return
        ruta = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png")],
            title="Guardar gráfica",
        )
        if not ruta:
            return
        self.lienzo.figura.savefig(ruta, dpi=140, bbox_inches="tight", facecolor="white")
        messagebox.showinfo("Gráfica", "La gráfica se guardó correctamente.")


class VistaComparacion(MarcoDesplazable):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.jacobi = None
        self.gauss = None
        marco = self.interior

        barra = tk.Frame(marco, bg=FONDO)
        barra.pack(fill="x", padx=18, pady=(16, 8))
        self.combo = ttk.Combobox(
            barra,
            state="readonly",
            width=32,
            values=[caso["titulo"] for caso in CASOS],
            font=("Segoe UI", 10),
        )
        self.combo.pack(side="left")
        _boton(barra, "Cargar ejemplo", self.cargar_ejemplo).pack(side="left", padx=8)
        _boton(barra, "Limpiar", self.limpiar, GRIS).pack(side="left")
        _boton(barra, "Comparar", self.calcular, VERDE).pack(side="left", padx=8)
        _boton(barra, "Inicio", lambda: app.mostrar("inicio"), NAVY).pack(side="right")

        tarjeta = _tarjeta(marco, "Mismos datos para los dos métodos")
        tarjeta.pack(fill="x", padx=18, pady=8)
        self.panel = PanelSistema(tarjeta, incluir_iteracion=True)
        self.panel.pack(fill="x")

        self.conclusion = tk.Label(
            marco, text="", bg=FONDO, fg=NAVY, justify="left", wraplength=900, font=("Segoe UI", 11, "bold")
        )
        self.conclusion.pack(anchor="w", padx=22, pady=(6, 0))

        columnas = tk.Frame(marco, bg=FONDO)
        columnas.pack(fill="x", padx=18, pady=8)
        columnas.columnconfigure(0, weight=1)
        columnas.columnconfigure(1, weight=1)
        caja_j = _tarjeta(columnas, "Jacobi")
        caja_g = _tarjeta(columnas, "Gauss-Seidel")
        caja_j.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        caja_g.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.texto_j = _caja_texto(caja_j, 16)
        self.texto_g = _caja_texto(caja_g, 16)
        self.texto_j.pack(fill="both", expand=True)
        self.texto_g.pack(fill="both", expand=True)

        self.estado_j = tk.Label(caja_j, text="", bg=TARJETA, font=("Segoe UI", 10, "bold"))
        self.estado_g = tk.Label(caja_g, text="", bg=TARJETA, font=("Segoe UI", 10, "bold"))
        self.estado_j.pack(anchor="w", pady=(6, 0))
        self.estado_g.pack(anchor="w", pady=(6, 0))

        tab_j = _tarjeta(marco, "Tabla de Jacobi")
        tab_j.pack(fill="x", padx=18, pady=8)
        self.tabla_j = TablaResultados(tab_j)
        self.tabla_j.pack(fill="x")
        tab_g = _tarjeta(marco, "Tabla de Gauss-Seidel")
        tab_g.pack(fill="x", padx=18, pady=8)
        self.tabla_g = TablaResultados(tab_g)
        self.tabla_g.pack(fill="x")

        grafica = _tarjeta(marco, "Gráfica comparativa del error")
        grafica.pack(fill="x", padx=18, pady=(8, 18))
        self.lienzo = LienzoGrafica(grafica)
        self.lienzo.pack(fill="x")
        _boton(grafica, "Guardar gráfica", self.guardar_grafica).pack(anchor="w", pady=(8, 0))

    def cargar_ejemplo(self):
        indice = self.combo.current()
        if indice < 0:
            messagebox.showinfo("Ejemplo", "Elija un caso en la lista.")
            return
        self.panel.cargar_caso(CASOS[indice])

    def cargar_caso(self, caso: dict):
        for i, item in enumerate(CASOS):
            if item["clave"] == caso["clave"]:
                self.combo.current(i)
                break
        self.panel.cargar_caso(caso)

    def limpiar(self):
        self.panel.limpiar()
        self.conclusion.configure(text="")
        _escribir(self.texto_j, "")
        _escribir(self.texto_g, "")
        self.estado_j.configure(text="")
        self.estado_g.configure(text="")
        self.tabla_j.limpiar()
        self.tabla_g.limpiar()
        self.lienzo.limpiar()
        self.jacobi = None
        self.gauss = None

    def calcular(self):
        try:
            preparacion = self.panel.preparar()
            jacobi = ejecutar_jacobi(preparacion)
            gauss = ejecutar_gauss_seidel(preparacion)
            referencia = solucion_referencia(preparacion.A, preparacion.b)
            comparacion = comparar_resultados(jacobi, gauss)
        except ErrorValidacion as exc:
            self._error(exc.mensaje)
            return
        except Exception as exc:
            self._error(
                "Ocurrió un problema inesperado y la comparación se detuvo sin cerrar "
                f"la calculadora.\n\nDetalle: {exc}"
            )
            return

        self.jacobi = jacobi
        self.gauss = gauss
        _escribir(self.texto_j, texto_de_metodo(preparacion, jacobi, referencia))
        _escribir(self.texto_g, texto_de_metodo(preparacion, gauss, referencia))
        self.estado_j.configure(text=jacobi.estado_legible, fg=_color_estado(jacobi.estado))
        self.estado_g.configure(text=gauss.estado_legible, fg=_color_estado(gauss.estado))
        self.tabla_j.mostrar_dataframe(tabla_desde_resultado(jacobi))
        self.tabla_g.mostrar_dataframe(tabla_desde_resultado(gauss))
        diferencia = comparacion["diferencia"]
        texto = comparacion["texto"]
        if diferencia is not None:
            texto += f" Diferencia entre vectores: {formatear_numero(diferencia)}."
        self.conclusion.configure(text=texto)
        self.lienzo.mostrar(
            crear_figura_errores(
                [jacobi, gauss],
                "Comparación del error: Jacobi y Gauss-Seidel",
            )
        )
        if jacobi.estado == "posible_divergencia" or gauss.estado == "posible_divergencia":
            messagebox.showwarning(
                "Advertencia de convergencia",
                "Al menos un método no mostró acercamiento. Revise el estado de cada uno. "
                "La calculadora permanece abierta.",
            )

    def _error(self, mensaje: str):
        self.jacobi = None
        self.gauss = None
        self.conclusion.configure(text="Error de validación", fg=ROJO)
        _escribir(self.texto_j, mensaje)
        _escribir(self.texto_g, mensaje)
        self.tabla_j.limpiar()
        self.tabla_g.limpiar()
        self.lienzo.limpiar()
        messagebox.showerror("No se pudo comparar", mensaje)

    def guardar_grafica(self):
        if self.lienzo.figura is None:
            messagebox.showinfo("Gráfica", "Primero ejecute la comparación.")
            return
        ruta = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png")], title="Guardar gráfica"
        )
        if ruta:
            self.lienzo.figura.savefig(ruta, dpi=140, bbox_inches="tight", facecolor="white")
            messagebox.showinfo("Gráfica", "La gráfica se guardó correctamente.")


class VistaReferencia(MarcoDesplazable):
    def __init__(self, master, app):
        super().__init__(master)
        marco = self.interior
        barra = tk.Frame(marco, bg=FONDO)
        barra.pack(fill="x", padx=18, pady=(16, 8))
        self.combo = ttk.Combobox(
            barra,
            state="readonly",
            width=32,
            values=[caso["titulo"] for caso in CASOS],
            font=("Segoe UI", 10),
        )
        self.combo.pack(side="left")
        _boton(barra, "Cargar ejemplo", self.cargar_ejemplo).pack(side="left", padx=8)
        _boton(barra, "Limpiar", self.limpiar, GRIS).pack(side="left")
        _boton(barra, "Validar sistema", self.calcular, VERDE).pack(side="left", padx=8)
        _boton(barra, "Inicio", lambda: app.mostrar("inicio"), NAVY).pack(side="right")

        nota = _tarjeta(marco, "Para qué sirve esta pantalla")
        nota.pack(fill="x", padx=18, pady=8)
        tk.Label(
            nota,
            text=(
                "Aquí se obtiene una solución de referencia con numpy.linalg.solve, "
                "solo cuando el sistema tiene solución única. No es el método de Jacobi "
                "ni el de Gauss-Seidel, y no debe presentarse como si los reemplazara."
            ),
            bg=TARJETA,
            fg=TEXTO,
            wraplength=860,
            justify="left",
            font=("Segoe UI", 11),
        ).pack(anchor="w")

        tarjeta = _tarjeta(marco, "Sistema lineal")
        tarjeta.pack(fill="x", padx=18, pady=8)
        self.panel = PanelSistema(tarjeta, incluir_iteracion=False)
        self.panel.pack(fill="x")

        caja = _tarjeta(marco, "Referencia")
        caja.pack(fill="x", padx=18, pady=(8, 18))
        self.resumen = _caja_texto(caja, 16)
        self.resumen.pack(fill="x")

    def cargar_ejemplo(self):
        indice = self.combo.current()
        if indice < 0:
            messagebox.showinfo("Ejemplo", "Elija un caso en la lista.")
            return
        self.panel.cargar_caso(CASOS[indice])

    def limpiar(self):
        self.panel.limpiar()
        _escribir(self.resumen, "")

    def calcular(self):
        try:
            A, b = self.panel._matriz_y_b()
            referencia = solucion_referencia(A, b)
        except ErrorValidacion as exc:
            _escribir(self.resumen, exc.mensaje)
            messagebox.showerror("Validación", exc.mensaje)
            return
        except Exception as exc:
            mensaje = f"No se pudo validar el sistema.\n\nDetalle: {exc}"
            _escribir(self.resumen, mensaje)
            messagebox.showerror("Validación", mensaje)
            return
        lineas = [
            referencia.mensaje,
            "",
            "Sistema ingresado:",
            matriz_a_texto(A, b),
            "",
        ]
        if referencia.disponible:
            lineas.append(f"Solución de referencia: {vector_a_texto(referencia.solucion)}.")
            lineas.append(f"Residuo ||A x − b||∞: {formatear_numero(referencia.residuo)}.")
            lineas.append(
                "Un residuo muy pequeño indica que el vector satisface el sistema "
                "dentro de la precisión numérica. Sigue siendo una comprobación directa, "
                "no una iteración."
            )
        _escribir(self.resumen, "\n".join(lineas))
        messagebox.showinfo("Validación", "La comprobación terminó. Revise el recuadro de referencia.")


class VistaCasos(MarcoDesplazable):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.caso = CASOS[0]
        marco = self.interior

        selector = tk.Frame(marco, bg=FONDO)
        selector.pack(fill="x", padx=18, pady=(16, 8))
        for caso in CASOS:
            _boton(selector, caso["titulo"], lambda c=caso: self.seleccionar(c)).pack(
                side="left", padx=(0, 8)
            )

        self.titulo = tk.Label(marco, text="", bg=FONDO, fg=NAVY, font=("Segoe UI", 14, "bold"))
        self.titulo.pack(anchor="w", padx=22)

        datos = _tarjeta(marco, "Datos y objetivo")
        datos.pack(fill="x", padx=18, pady=8)
        self.detalle = _caja_texto(datos, 12)
        self.detalle.pack(fill="x")

        acciones = tk.Frame(marco, bg=FONDO)
        acciones.pack(fill="x", padx=18, pady=4)
        _boton(acciones, "Ejecutar caso", self.ejecutar, VERDE).pack(side="left")
        _boton(acciones, "Llevar a comparación", self.llevar_comparacion).pack(side="left", padx=8)
        _boton(acciones, "Inicio", lambda: app.mostrar("inicio"), NAVY).pack(side="right")

        resultado = _tarjeta(marco, "Interpretación con los resultados calculados")
        resultado.pack(fill="x", padx=18, pady=8)
        self.interpretacion = _caja_texto(resultado, 16)
        self.interpretacion.pack(fill="x")

        grafica = _tarjeta(marco, "Gráfica del caso")
        grafica.pack(fill="x", padx=18, pady=(8, 18))
        self.lienzo = LienzoGrafica(grafica)
        self.lienzo.pack(fill="x")
        self.seleccionar(CASOS[0])

    def seleccionar(self, caso: dict):
        self.caso = caso
        self.titulo.configure(text=caso["titulo"])
        A = np.array(caso["A"], dtype=float)
        b = np.array(caso["b"], dtype=float)
        x0 = np.array(caso["x0"], dtype=float)
        texto = (
            f"Objetivo: {caso['objetivo']}\n\n"
            f"Tolerancia: {caso['tol']}\n"
            f"Máximo de iteraciones: {caso['max_iter']}\n"
            f"Vector inicial x0: {vector_a_texto(x0)}\n\n"
            "Sistema:\n"
            f"{matriz_a_texto(A, b)}"
        )
        _escribir(self.detalle, texto)

    def llevar_comparacion(self):
        self.app.vistas["comparacion"].cargar_caso(self.caso)
        self.app.mostrar("comparacion")

    def ejecutar(self):
        caso = self.caso
        try:
            preparacion = preparar_sistema(
                caso["A"], caso["b"], caso["x0"], caso["tol"], caso["max_iter"]
            )
            jacobi = ejecutar_jacobi(preparacion)
            gauss = ejecutar_gauss_seidel(preparacion)
            referencia = solucion_referencia(preparacion.A, preparacion.b)
            comparacion = comparar_resultados(jacobi, gauss)
            texto = interpretar_caso(caso, preparacion, jacobi, gauss, referencia, comparacion)
        except ErrorValidacion as exc:
            _escribir(self.interpretacion, exc.mensaje)
            self.lienzo.limpiar()
            messagebox.showerror("Caso de prueba", exc.mensaje)
            return
        except Exception as exc:
            mensaje = f"El caso se detuvo por un error inesperado.\n\nDetalle: {exc}"
            _escribir(self.interpretacion, mensaje)
            messagebox.showerror("Caso de prueba", mensaje)
            return
        _escribir(self.interpretacion, texto)
        self.lienzo.mostrar(
            crear_figura_errores([jacobi, gauss], caso["titulo"] + ": error por iteración")
        )
        if jacobi.estado == "posible_divergencia" or gauss.estado == "posible_divergencia":
            messagebox.showwarning(
                "Advertencia",
                "El caso terminó de forma controlada. Hay una advertencia de convergencia; "
                "revise la interpretación.",
            )
        else:
            messagebox.showinfo("Caso ejecutado", "El caso se calculó. Revise la interpretación y la gráfica.")


class VistaAyuda(MarcoDesplazable):
    def __init__(self, master, app):
        super().__init__(master)
        marco = self.interior
        caja = _tarjeta(marco, "Guía para usar y defender el programa")
        caja.pack(fill="both", expand=True, padx=18, pady=16)
        texto = scrolledtext.ScrolledText(
            caja, wrap="word", font=("Segoe UI", 11), bg=TARJETA, fg=TEXTO, relief="flat", height=32
        )
        texto.pack(fill="both", expand=True)
        texto.insert("1.0", AYUDA.strip())
        texto.configure(state="disabled")
        _boton(marco, "Inicio", lambda: app.mostrar("inicio"), NAVY).pack(anchor="w", padx=22, pady=(0, 16))


class Aplicacion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calculadora de Álgebra Lineal")
        self.geometry("1180x760")
        self.minsize(1000, 660)
        self.configure(bg=FONDO)
        self._estilos()

        encabezado = tk.Frame(self, bg=NAVY, height=78)
        encabezado.pack(fill="x")
        encabezado.pack_propagate(False)
        tk.Label(
            encabezado,
            text="Calculadora de Álgebra Lineal",
            bg=NAVY,
            fg="white",
            font=("Segoe UI", 20, "bold"),
        ).pack(anchor="w", padx=18, pady=(10, 0))
        tk.Label(
            encabezado,
            text="Métodos Numéricos – Programa 8",
            bg=NAVY,
            fg="#D6E4F0",
            font=("Segoe UI", 12),
        ).pack(anchor="w", padx=18)

        cuerpo = tk.Frame(self, bg=FONDO)
        cuerpo.pack(fill="both", expand=True)
        self._menu(cuerpo)
        self.contenedor = tk.Frame(cuerpo, bg=FONDO)
        self.contenedor.pack(side="left", fill="both", expand=True)

        self.vistas = {
            "inicio": VistaInicio(self.contenedor, self),
            "jacobi": VistaMetodo(self.contenedor, self, "Jacobi", ejecutar_jacobi),
            "gauss": VistaMetodo(self.contenedor, self, "Gauss-Seidel", ejecutar_gauss_seidel),
            "comparacion": VistaComparacion(self.contenedor, self),
            "referencia": VistaReferencia(self.contenedor, self),
            "casos": VistaCasos(self.contenedor, self),
            "ayuda": VistaAyuda(self.contenedor, self),
        }
        self.botones_menu = {}
        self._crear_botones_menu()
        self.mostrar("inicio")

        pie = tk.Frame(self, bg=NAVY, height=28)
        pie.pack(fill="x")
        pie.pack_propagate(False)
        tk.Label(
            pie,
            text="Universidad Americana  ·  Álgebra Lineal (MTM0120)  ·  Jacobi y Gauss-Seidel",
            bg=NAVY,
            fg="#D6E4F0",
            font=("Segoe UI", 9),
        ).pack(side="left", padx=12)

    def _estilos(self):
        estilo = ttk.Style(self)
        try:
            estilo.theme_use("clam")
        except tk.TclError:
            pass
        estilo.configure("Treeview", font=("Segoe UI", 10), rowheight=26, background="white", fieldbackground="white")
        estilo.configure(
            "Treeview.Heading",
            font=("Segoe UI", 10, "bold"),
            background=NAVY,
            foreground="white",
        )
        estilo.map("Treeview", background=[("selected", "#D6E6F5")])

    def _menu(self, cuerpo):
        self.menu = tk.Frame(cuerpo, bg=NAVY, width=250)
        self.menu.pack(side="left", fill="y")
        self.menu.pack_propagate(False)
        tk.Label(
            self.menu,
            text="MENÚ",
            bg=NAVY,
            fg="#9FB4C9",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=16, pady=(16, 8))
        self._opciones = [
            ("inicio", "Inicio"),
            ("jacobi", "Método de Jacobi"),
            ("gauss", "Método de Gauss-Seidel"),
            ("comparacion", "Comparación de métodos"),
            ("referencia", "Validación de referencia"),
            ("casos", "Casos de prueba"),
            ("ayuda", "Información y ayuda"),
        ]

    def _crear_botones_menu(self):
        for clave, titulo in self._opciones:
            boton = tk.Button(
                self.menu,
                text=titulo,
                command=lambda c=clave: self.mostrar(c),
                bg=NAVY,
                fg="white",
                activebackground=AZUL,
                activeforeground="white",
                relief="flat",
                anchor="w",
                padx=16,
                pady=9,
                font=("Segoe UI", 10),
                cursor="hand2",
                bd=0,
            )
            boton.pack(fill="x")
            self.botones_menu[clave] = boton

    def mostrar(self, clave: str):
        for vista in self.vistas.values():
            vista.pack_forget()
        self.vistas[clave].pack(fill="both", expand=True)
        for nombre, boton in self.botones_menu.items():
            if nombre == clave:
                boton.configure(bg=AZUL, font=("Segoe UI", 10, "bold"))
            else:
                boton.configure(bg=NAVY, font=("Segoe UI", 10))


def main():
    app = Aplicacion()
    app.mainloop()
