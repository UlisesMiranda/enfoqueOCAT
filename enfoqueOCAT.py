import random
import pandas as pd


def ordenarAscendentementeSinRepeticion(listaNumeros: list):
    return set(sorted(listaNumeros))

def codificarBinario(atributo, listaValores):
    codificacionAtributo = ''
    for valor in listaValores:
        if atributo >= valor:
            codificacionAtributo += '1'
        else:
            codificacionAtributo += '0'
    
    return codificacionAtributo

def dividirEnColumnas(df, columna):
    nueva_columna = df[columna].apply(lambda x: pd.Series(list(x)))  # Divide cada valor en dígitos
    nueva_columna = nueva_columna.add_prefix(f"{columna}_")  # Agrega un prefijo al nombre de las nuevas columnas
    return nueva_columna

def binarizacion(df: pd.DataFrame):
    columnas = df.columns
    dfBinarizado = df.copy()   
    for i, col in enumerate(columnas):
        atributosColumna = df[col].values
        listaValores = ordenarAscendentementeSinRepeticion(atributosColumna)
        
        for j, atributo in enumerate(atributosColumna):
            dfBinarizado.at[j, col] = codificarBinario(atributo, listaValores)
    
    nuevasColumnas = []
    for col in columnas:
        nuevasColumnas.append(dividirEnColumnas(dfBinarizado, col))
    
    dfBinarizado = pd.concat(nuevasColumnas, axis=1)
    
    # dfBinarizado.drop(columnas, axis=1, inplace=True)
    
    print(dfBinarizado)
    print(dfBinarizado.columns)
        

noFilas = 10

columna_1 = [str(random.uniform(0, 3)) for _ in range(noFilas)]
columna_2 = [str(random.uniform(0, 3)) for _ in range(noFilas)]

data = {"col_1": columna_1, "col_2": columna_2}
df = pd.DataFrame(data)

binarizacion(df)



    
