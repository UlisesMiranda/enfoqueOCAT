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

# Supongamos que tienes una expresión de disyunción como un string
expresion_disyuncion = "x_2 v x_4"

# Convertir la expresión de disyunción a una lista de componentes
componentes_disyuncion = expresion_disyuncion.split(' v ')

# Función para evaluar la expresión de disyunción en el DataFrame
def evaluar_expresion_disyuncion(fila, componentes_disyuncion):
    for componente in componentes_disyuncion:
        negacion = componente.startswith('neg_')
        col = componente[4:] if negacion else componente
        
        if negacion:
            if fila[col] != 1:
                return True  # Si la negación no se cumple, retorna True
        else:
            if fila[col] == 1:
                return False  # Si alguna columna normal cumple, retorna False
    
    return True  # Si ninguna condición se cumple, retorna True (no se cumple la disyunción)

# Filtrar el DataFrame aplicando la función de expresión de disyunción a cada fila
df_no_cumple_disyuncion = df[~df.apply(evaluar_expresion_disyuncion, axis=1, componentes_disyuncion=componentes_disyuncion)]

# Mostrar el DataFrame resultante
print(df_no_cumple_disyuncion)
