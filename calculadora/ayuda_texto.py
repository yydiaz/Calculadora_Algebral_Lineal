"""Texto de la pantalla de ayuda, en el mismo lenguaje de la exposición."""

AYUDA = """
CÓMO EJECUTAR LA CALCULADORA

1. Instale las librerías con: pip install -r requirements.txt
2. En la carpeta del proyecto ejecute: python main.py
3. Use el menú de la izquierda. No hace falta escribir comandos durante la exposición.


QUÉ HACE CADA OPCIÓN

Inicio
Resume el Programa 8 y recuerda que esta versión contiene Jacobi y Gauss-Seidel.

Método de Jacobi
Pide el sistema, el vector inicial, la tolerancia y el máximo de iteraciones.
Calcula cada vector nuevo usando solamente la aproximación anterior.

Método de Gauss-Seidel
Pide los mismos datos, pero dentro de una iteración usa de inmediato los
componentes que acaba de calcular.

Comparación
Ejecuta los dos métodos con exactamente los mismos datos y grafica el error.

Validación de referencia
Resuelve el sistema con numpy.linalg.solve cuando hay solución única. Esa
respuesta solo sirve para comprobar. No es Jacobi ni Gauss-Seidel.

Casos de prueba
Trae los tres ejemplos obligatorios, ya con su objetivo y su interpretación.

Información y ayuda
Esta pantalla.


CÓMO FUNCIONA JACOBI

Para cada incógnita se despeja la diagonal:

x_i nuevo = (b_i − suma de a_ij * x_j anterior, con j distinto de i) / a_ii

La idea importante: aunque ya se haya calculado x1 nuevo, x2 todavía se calcula
con x1 anterior. Por eso se dice que la actualización es simultánea. En el
código eso está en paso_jacobi: el vector de entrada no se modifica mientras
se llena el vector de salida.


CÓMO FUNCIONA GAUSS-SEIDEL

La fórmula también despeja la diagonal, pero la suma se parte en dos:

- las incógnitas anteriores a i usan el valor que acaba de calcularse;
- las incógnitas posteriores a i siguen usando la iteración anterior.

Esa es la única diferencia matemática esencial. El sistema, la tolerancia y la
norma del error son los mismos. En el código está en paso_gauss_seidel.


TOLERANCIA, ERROR Y RESIDUO

La tolerancia ε es un decimal positivo elegido por quien usa la calculadora.
Es el error máximo que se está dispuesto a aceptar entre dos iteraciones
seguidas. No es el error exacto respecto de la solución verdadera, porque esa
solución normalmente no se conoce antes de calcular.

Error = norma infinito de (x(k) − x(k−1))

La norma infinito es el mayor valor absoluto entre las componentes. Se detiene
cuando ese error es menor o igual que ε, o cuando se llega al máximo de
iteraciones.

Residuo = norma infinito de (A x − b)

El residuo dice qué tanto falla el sistema al sustituir la aproximación.
Un residuo pequeño apoya que la aproximación es buena. Un error pequeño entre
iteraciones y un residuo grande serían una señal de revisar el resultado.


DIAGONAL DOMINANTE

A es estrictamente diagonal dominante por filas cuando, en cada fila, el valor
absoluto de la diagonal es mayor que la suma de los valores absolutos del resto
de la fila.

Si eso ocurre, Jacobi y Gauss-Seidel convergen. Es una condición suficiente.
No es necesaria: hay sistemas que no la cumplen y aun así pueden converger.
Por eso la calculadora nunca dice "diverge" solamente porque falte esta
condición. La advertencia de posible divergencia sale del error calculado:
si no se cumple la tolerancia y el error final es mayor que el error inicial,
o si la segunda mitad de los errores queda por encima de la primera.


CUÁNDO SE REORDENA

Solo se reordenan filas de forma automática cuando algún coeficiente de la
diagonal es cero. En ese caso no se puede dividir. Si existe otro orden de las
ecuaciones con diagonal distinta de cero, se aplica ese orden a A y a b.
Si además ese orden queda diagonal dominante, se informa.

El botón "Evaluar reordenamiento" permite revisar un orden mejor sin calcular
todavía. Si no existe un orden adecuado, el programa lo dice y no inventa un
intercambio.

Reordenar ecuaciones no cambia la solución del sistema, pero sí puede cambiar
si el método iterativo converge, porque cambia el coeficiente que queda en la
diagonal.


POR QUÉ PUEDE NO CONVERGER

La iteración es de la forma x nuevo = operador(x anterior). Si ese operador
aleja los vectores en lugar de acercarlos, el error crece. Puede pasar cuando
los coeficientes fuera de la diagonal pesan más que la diagonal.

También puede detenerse al máximo de iteraciones con un error que sí baja.
Eso no es divergencia: faltaron iteraciones o la tolerancia es muy exigente.

Si aparecen valores no finitos, el proceso se detiene y muestra la advertencia.
Una entrada incorrecta tampoco cierra la ventana: se muestra el motivo.


CÓMO LEER LA TABLA Y LA GRÁFICA

La fila 0 es el vector inicial. Su error se deja en blanco porque no hay una
iteración anterior.

Desde la fila 1, la columna Error es la norma infinito de la diferencia con la
fila previa. La columna Residuo es ||A x − b||∞ de esa aproximación.

En la gráfica, el eje X es el número de iteración y el eje Y es el error.
Jacobi va en azul y Gauss-Seidel en verde. Si los errores cambian varios
órdenes de magnitud, el eje Y usa escala logarítmica para que las dos curvas
se lean; los puntos siguen siendo los calculados.

Una curva que baja hacia la tolerancia indica acercamiento. Una curva que sube
acompaña la advertencia de posible divergencia.


LOS TRES CASOS

Caso 1. La matriz ya es diagonal dominante. Sirve para ver convergencia y para
comparar cuántas iteraciones usa cada método. La referencia de NumPy debe
quedar cerca de las dos aproximaciones.

Caso 2. La primera diagonal es cero. Antes de iterar, la calculadora pasa la
segunda ecuación al primer lugar, mueve la primera al segundo y deja la tercera
en su sitio. También reordena b. Después los métodos sí pueden ejecutarse.

Caso 3. Ningún orden de filas logra dominancia estricta. Con el orden escrito,
el error crece en los dos métodos. NumPy igual puede mostrar la solución de
referencia: eso demuestra que el sistema tiene solución, pero que estas
iteraciones no la alcanzan. No se debe decir que el último vector es la
respuesta del sistema.


PREGUNTAS QUE EL DOCENTE PUEDE HACER

¿Jacobi usa valores nuevos dentro de la misma iteración?
No. Todos los componentes nuevos salen del vector anterior.

¿Gauss-Seidel espera a terminar la iteración para actualizar?
No. Al calcular x2 ya utiliza el x1 nuevo, y así con las demás.

¿La tolerancia garantiza que la solución es exacta?
No. Garantiza que dos aproximaciones consecutivas se parecen al menos en esa
medida. Por eso también se muestra el residuo y, cuando existe, la referencia.

¿Si no es diagonal dominante, siempre diverge?
No. La dominancia estricta alcanza para asegurar convergencia, pero no es el
único caso posible. La calculadora observa el error antes de advertir.

¿Por qué no se usó numpy.linalg.solve para Jacobi?
Porque solve es un método directo. El Programa 8 pide programar las fórmulas
iterativas. solve solo aparece en la validación de referencia.

¿Qué pasa si la diagonal tiene un cero y no hay reordenamiento posible?
Se muestra el mensaje y no se inicia la división. La ventana permanece abierta.

¿Qué se compara en la gráfica?
El error entre iteraciones, no el valor de las incógnitas. Así se ve cuál
método reduce ese error más pronto, cuando ambos producen datos.
"""
