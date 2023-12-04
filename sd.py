import pandas as pd

# Crear el DataFrame con los datos proporcionados
data = {
    'x_1': [1, 0, 1, 0, 1, 1],
    'x_2': [0, 0, 1, 0, 0, 1],
    'x_3': [1, 0, 1, 0, 0, 1],
    'x_4': [0, 1, 1, 0, 0, 0],
    'target': [0, 0, 0, 0, 0, 0]
}
df = pd.DataFrame(data)

# Supongamos que tenemos la expresión de disyunción 'neg_x_1 v x_2'
expresion_disyuncion = "x_2 ^ x_4 ^ "
# expresion_disyuncion = "x_2 v x_4 v "

# Función para evaluar la expresión de disyunción en el DataFrame
def evaluar_expresion_disyuncion(fila, expresion):
    expresion_components = expresion.split(' v ')[:-1]
    for componente in expresion_components:
        negacion = False
        col_name = componente

        if componente.startswith('neg_'):
            col_name = componente[4:]
            negacion = True

        col_value = fila[col_name]
        if (not negacion and col_value == '1') or (negacion and col_value == '0'):
            return True
    
    return False

def evaluar_expresion_conjuncion(fila, expresion):
    expresion_components = expresion.split(' ^ ')[:-1]
    for componente in expresion_components:
        negacion = False
        col_name = componente

        if componente.startswith('neg_'):
            col_name = componente[4:]
            negacion = True

        col_value = fila[col_name]
        if (not negacion and col_value != '1') or (negacion and col_value != '0'):
            return False
    
    return True

print(df)
df = df.astype(str)

# Filtrar el DataFrame aplicando la función de expresión de disyunción a cada fila
df_no_cumple_disyuncion = df[~df.apply(lambda x: evaluar_expresion_conjuncion(x, expresion_disyuncion), axis=1)]

# Mostrar el DataFrame resultante
print(df_no_cumple_disyuncion)
