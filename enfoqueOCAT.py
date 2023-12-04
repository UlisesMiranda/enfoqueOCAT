import math
import random
import pandas as pd
import numpy as np


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
    nueva_columna = df[columna].apply(lambda x: pd.Series(list(x))) 
    nueva_columna = nueva_columna.add_prefix(f"{columna}_") 
    return nueva_columna

def binarizacion(df: pd.DataFrame, columnTarget):
    columnas = list(df.columns)
    columnas.remove(columnTarget)
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
    dfBinarizado[columnTarget] = df[columnTarget]
    
    return dfBinarizado

def oneHot(df):
    last_column_name = df.columns[-1]  # Obtener el nombre de la última columna
    columns_to_encode = df.columns[:-1]  # Seleccionar todas las columnas excepto la última
    new_df = pd.DataFrame()  # Crear un nuevo DataFrame para almacenar las columnas codificadas

    for idx, columna in enumerate(columns_to_encode):
        valores_unicos = df[columna].unique()

        for valor in valores_unicos:
            nueva_columna = []
            for fila in df[columna]:
                if fila == valor:
                    nueva_columna.append(1)
                else:
                    nueva_columna.append(0)
            new_df[f'{columna}_{valor}'] = nueva_columna  # Agregar nuevas columnas con nombres 'x_1_1', 'x_1_2', etc.
    
    # new_df[last_column_name] = df[last_column_name]  # Agregar la última columna original al nuevo DataFrame
    
    return new_df

def indicesPositivosAndNegativos(columna: np.array, target):
    pos = np.where(columna == target)
    neg = np.where(columna != target)
    
    return pos, neg

def obtenerIndices(espacio: pd.DataFrame, columnas, target):
    indicesPositivosXcolumna = []
    indicesNegativosXcolumna = []
    for col in columnas:
        columnaActual = espacio[col].values
        indicesPositivos, indicesNegativos = indicesPositivosAndNegativos(columnaActual, target)
        indicesPositivosXcolumna.append([col, indicesPositivos[0]])  
        indicesNegativosXcolumna.append([col, indicesNegativos[0]])
        
    return indicesPositivosXcolumna, indicesNegativosXcolumna

def obtenerMagnitud(lista):
    return len(lista)

def aptitud(numeroPositivos, numeroNegativos):
    if (numeroNegativos != 0):
        return numeroPositivos / numeroNegativos
    else:
        return math.inf

def calcularMagnitud(listaIndices):
    magnitudesIndices = []
    for indices in listaIndices:
        col = indices[0]
        idsIndices: np.array = indices[1]
        magnitudesIndices.append((col, len(idsIndices)))
        
    return magnitudesIndices

def calculoAptitudes(positivosAceptadosMagnitud: list, positivosAceptadosNegadosMagnitud: list,
                     negativosAceptadosMagnitud: list, negativosAceptadosNegadosMagnitud: list, columnas: list) -> list:
    
    aptitudesResultantes = []
    for i, col in enumerate(columnas):
        magnitudPositivosAceptados = positivosAceptadosMagnitud[i][1]
        magnitudNegativosAceptados = negativosAceptadosMagnitud[i][1]
        
        aptitudValor = aptitud(magnitudPositivosAceptados, magnitudNegativosAceptados)
        aptitudesResultantes.append((col, aptitudValor))
    
    for i, col in enumerate(columnas):
        magnitudPositivosNegadosAceptados = positivosAceptadosNegadosMagnitud[i][1]
        magnitudNegativosNegadosAceptados = negativosAceptadosNegadosMagnitud[i][1]
        
        aptitudValor = aptitud(magnitudPositivosNegadosAceptados, magnitudNegativosNegadosAceptados)
        aptitudesResultantes.append(('neg_' + col, aptitudValor))
        
    return sorted(aptitudesResultantes, key=lambda x: x[1], reverse=True)

def obtenerMejoresCandidatos(aptitudesResultantes: list):
    aptitudMasAlta = aptitudesResultantes[0][1]

    # Filtrar las tuplas que tienen el número más alto del segundo valor
    mejorSubconjunto = [tupla[0] for tupla in aptitudesResultantes if tupla[1] == aptitudMasAlta]
    
    return mejorSubconjunto

def obtenerElementoRandom(lista: list):
    return random.choice(lista)

def actualizarEspacioPositivo(espacioPositivo: pd.DataFrame, columna, target, esNegado: bool):
    if esNegado:
        return espacioPositivo[espacioPositivo[columna] == target]
    else:
        return espacioPositivo[espacioPositivo[columna] != target]

def evaluar_expresion_disyuncion(fila, expresion):
    expresion_components = expresion.split(' V ')[:-1]
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
        if(col_value) != '0':
            print("")
        if (not negacion and col_value != '1') or (negacion and col_value != '0'):
            return False
    
    return True


def enfoqueOCAT(df: pd.DataFrame, columnTarget, target):
    columnas = list(df.columns)
    columnas.remove(columnTarget)
    
    conceptosPositivos = df[df[columnTarget] == target]
    conceptosNegativos =  df[df[columnTarget] != target]
    
    conceptosPositivosActual = conceptosPositivos.copy()
    conceptosNegativosActual = conceptosNegativos.copy()

    positivosAceptados = []
    positivosAceptadosNegados = []
    negativosAceptados = []
    negativosAceptadosNegados = []
    
    clausulaGeneral = []
    while not conceptosNegativosActual.empty:
        clausula = ''
        isListaInicial = True
        terminoRandom = ''
        columnasEvaluacion = columnas.copy()
        while not conceptosPositivosActual.empty:                
            
            if (isListaInicial):
                positivosAceptados, positivosAceptadosNegados = obtenerIndices(conceptosPositivosActual, columnasEvaluacion, target)
                negativosAceptados, negativosAceptadosNegados = obtenerIndices(conceptosNegativosActual, columnasEvaluacion, target)
                
                positivosAceptadosMagnitud = calcularMagnitud(positivosAceptados)
                positivosAceptadosNegadosMagnitud = calcularMagnitud(positivosAceptadosNegados)
                negativosAceptadosMagnitud = calcularMagnitud(negativosAceptados)
                negativosAceptadosNegadosMagnitud = calcularMagnitud(negativosAceptadosNegados)
                
                isListaInicial = False
                aptitudesResultantes: list = calculoAptitudes(positivosAceptadosMagnitud, positivosAceptadosNegadosMagnitud, negativosAceptadosMagnitud, negativosAceptadosNegadosMagnitud, columnasEvaluacion)
                
                if not aptitudesResultantes:
                    print("que shoe")
                mejorSubconjunto = obtenerMejoresCandidatos(aptitudesResultantes)
            else:
                if terminoRandom != '':
                    terminoRandomCopy = terminoRandom
                    if terminoRandom.startswith('neg_'):
                        terminoRandomCopy = terminoRandom[4:]
                        
                    positivosAceptados = list(filter(lambda tupla: tupla[0] != terminoRandomCopy, positivosAceptados))      
                    positivosAceptadosNegados = list(filter(lambda tupla: tupla[0] != terminoRandomCopy, positivosAceptadosNegados))      
                    negativosAceptados = list(filter(lambda tupla: tupla[0] != terminoRandomCopy, negativosAceptados))      
                    negativosAceptadosNegados = list(filter(lambda tupla: tupla[0] != terminoRandomCopy, negativosAceptadosNegados))
                    
                    columnasEvaluacion.remove(terminoRandomCopy) 
                    
                    positivosAceptadosMagnitud = calcularMagnitud(positivosAceptados)
                    positivosAceptadosNegadosMagnitud = calcularMagnitud(positivosAceptadosNegados)
                    negativosAceptadosMagnitud = calcularMagnitud(negativosAceptados)
                    negativosAceptadosNegadosMagnitud = calcularMagnitud(negativosAceptadosNegados)     
            
            if mejorSubconjunto:
                terminoRandom = obtenerElementoRandom(mejorSubconjunto)
                mejorSubconjunto.remove(terminoRandom)
            else:
                aptitudesResultantes: list = calculoAptitudes(positivosAceptadosMagnitud, positivosAceptadosNegadosMagnitud, negativosAceptadosMagnitud, negativosAceptadosNegadosMagnitud, columnasEvaluacion)
                if not aptitudesResultantes:
                    print("que shoe")
                mejorSubconjunto = obtenerMejoresCandidatos(aptitudesResultantes)
                # mejorSubconjunto.remove(terminoRandom)
                terminoRandom = obtenerElementoRandom(mejorSubconjunto)
                mejorSubconjunto.remove(terminoRandom)
                
            
            if not (conceptosPositivosActual.empty):
                clausula += terminoRandom + ' v '
            else:
                clausula += terminoRandom 
            
            if "neg" in terminoRandom:
                terminoOriginal = terminoRandom.replace('neg_', '')
                conceptosPositivosActual = actualizarEspacioPositivo(conceptosPositivosActual, terminoOriginal, target, True)
            else:
                conceptosPositivosActual = actualizarEspacioPositivo(conceptosPositivosActual, terminoRandom, target, False)  

        print("Conceptos positivios")
        print(conceptosPositivos)
        
        print("Conceptos neg ac")
        print(conceptosNegativosActual)

        conceptosNegativosActual = conceptosNegativosActual[conceptosNegativosActual.apply(lambda x: evaluar_expresion_disyuncion(x, clausula), axis=1)]
        # conceptosNegativosActual = conceptosNegativosActual[~conceptosNegativosActual.apply(lambda x: evaluar_expresion_conjuncion(x, clausula), axis=1)]
        
        print("Conceptos neg nuevs")
        print(conceptosNegativosActual)
            
        conceptosPositivosActual = conceptosPositivos
        clausulaGeneral.append(clausula[:-3])
        # print(clausulaGeneral)
        
    print(clausulaGeneral)


# # PRUEBA DE DATOS
# columna_1 = [0,1,0,1,1,0,1,0,1,1]
# columna_2 = [1,1,0,0,0,0,1,0,0,1]
# columna_3 = [0,0,1,0,1,0,1,0,0,1]
# columna_4 = [0,0,1,1,0,1,1,0,0,0]
# columna_5 = [1,1,1,1,0,0,0,0,0,0]
# # # Convertir las listas de enteros a listas de strings
# columna_1_str = [str(x) for x in columna_1]
# columna_2_str = [str(x) for x in columna_2]
# columna_3_str = [str(x) for x in columna_3]
# columna_4_str = [str(x) for x in columna_4]
# columna_5_str = [str(x) for x in columna_5]

# # Crear un DataFrame con esas columnas como strings
# dfBinarizado = pd.DataFrame({
#     'x_1': columna_1_str,
#     'x_2': columna_2_str,
#     'x_3': columna_3_str,
#     'x_4': columna_4_str,
#     'target': columna_5_str
# })



# PRACTICA
# Crear un DataFrame vacío con 6 columnas
# num_columnas = 6
# nombres_columnas = ["target"]

# for i in range(1, num_columnas + 1):
#     nombres_columnas.append(f"x_{i}")

# df = pd.read_csv('monk+s+problems/monks-1.train', header=None, sep='\s+', names=nombres_columnas, index_col=False)
# df.columns = nombres_columnas

# # Mostrar el DataFrame resultante
# print(df)
# df.drop(df.columns[0], axis=1)

# df.to_csv('monk-1-train.csv')

# # Obtener el nombre de la primera columna
# primera_columna = df.columns[0]

# print(df.columns)

# # Reorganizar las columnas: la primera columna al final
# columnas_reordenadas = list(df.columns[1:]) + [primera_columna]

# # Crear un nuevo DataFrame con las columnas reordenadas
# df_reordenado = df[columnas_reordenadas]

# Guardar el DataFrame resultante como un nuevo archivo CSV
# df.to_csv('monk-1-train.csv', index=False)

# PROBANDO

df = pd.read_csv('monk-1-train.csv')
df = df.astype(str)
print(df)

atributos = df.columns[:-1]
lista_dataframes = []

for atributo in atributos:
    df_por_atributo = df[[atributo]].copy()
    df_por_atributo[df.columns[-1]] = df[[df.columns[-1]]].copy()
    
    # print(df_por_atributo)
    
    df_bin_atributo = oneHot(df_por_atributo)
    # print(df_bin_atributo)
    
    lista_dataframes.append(df_bin_atributo)
    
print(lista_dataframes)
lista_dataframes.append(df[[df.columns[-1]]])
    
dfBinarizado =pd.concat(lista_dataframes, axis=1)
print(dfBinarizado)

# dfBinarizado = oneHot(df)
# dfBinarizado = dfBinarizado.astype(str)

columnTarget = dfBinarizado.columns[-1]
target = '1'
dfBinarizado = dfBinarizado.astype(str)

# print(dfBinarizado.dtypes)
# dfBinarizado.to_csv('monk-1-train.csv')
enfoqueOCAT(dfBinarizado, columnTarget, target)

# enfoqueOCAT(df, columnTarget, target)



    
