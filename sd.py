import itertools

def evaluate_expression(x_5_1, x_1_3, x_4_1, x_1_1, x_5_4, x_4_2, x_5_2):
    return x_5_1 or x_1_3 or x_4_1 or (not x_1_1) or (not x_5_4) or (not x_4_2) or (not x_5_2)

# Generar todas las combinaciones posibles de valores para las variables
variables = [0, 1]
combinaciones = list(itertools.product(variables, repeat=8))

# Calcular la tabla de verdad para la expresión lógica
tabla_verdad = []
for combinacion in combinaciones:
    resultado = evaluate_expression(*combinacion)
    tabla_verdad.append(combinacion + (resultado,))

# Imprimir la tabla de verdad
print("|x_5_1|x_1_3|x_4_1|neg_x_1_1|neg_x_5_4|neg_x_4_2|neg_x_5_2|Resultado|")
print("|------|------|------|---------|---------|---------|---------|---------|")
for fila in tabla_verdad:
    print("|{}     |{}     |{}     |{}        |{}        |{}        |{}        |{}        |".format(*fila))
