import pandas as pd

num_columnas = 6
nombres_columnas = ["target"]

for i in range(1, num_columnas + 1):
    nombres_columnas.append(f"x_{i}")

df = pd.read_csv('monk+s+problems/monks-2.train', header=None, sep='\s+', names=nombres_columnas, index_col=False)
df.columns = nombres_columnas

# Mostrar el DataFrame resultante
print(df)
df.drop(df.columns[0], axis=1)

df.to_csv('monk-2-train.csv')

# Obtener el nombre de la primera columna
primera_columna = df.columns[0]

print(df.columns)


# Reorganizar las columnas: la primera columna al final
columnas_reordenadas = list(df.columns[1:]) + [primera_columna]

# Crear un nuevo DataFrame con las columnas reordenadas
df_reordenado = df[columnas_reordenadas]

# Guardar el DataFrame resultante como un nuevo archivo CSV
df_reordenado.to_csv('monk-2-train.csv', index=False)