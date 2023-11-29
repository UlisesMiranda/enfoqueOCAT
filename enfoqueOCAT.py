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
    
    print(dfBinarizado)
    nuevasColumnas = []
    for col in columnas:
        nuevasColumnas.append(dividirEnColumnas(dfBinarizado, col))
    
    dfBinarizado = pd.concat(nuevasColumnas, axis=1)
    dfBinarizado[columnTarget] = df[columnTarget]
    
    print(dfBinarizado)
    
    return dfBinarizado

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

def evaluar_expresion_disyuncion(fila, componentes_disyuncion):
    for componente in componentes_disyuncion:
        negacion = componente.startswith('neg_')
        col = componente[4:] if negacion else componente
        
        if negacion:
            if fila[col] != '1':
                return True  # Si la negación no se cumple, retorna True
        else:
            if fila[col] == '1':
                return False  # Si alguna columna normal cumple, retorna False
    
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
        while not conceptosPositivosActual.empty:
            positivosAceptados, positivosAceptadosNegados = obtenerIndices(conceptosPositivosActual, columnas, target)
            negativosAceptados, negativosAceptadosNegados = obtenerIndices(conceptosNegativosActual, columnas, target)
                        
            positivosAceptadosMagnitud = calcularMagnitud(positivosAceptados)
            positivosAceptadosNegadosMagnitud = calcularMagnitud(positivosAceptadosNegados)
            negativosAceptadosMagnitud = calcularMagnitud(negativosAceptados)
            negativosAceptadosNegadosMagnitud = calcularMagnitud(negativosAceptadosNegados)
            
            aptitudesResultantes: list = calculoAptitudes(positivosAceptadosMagnitud, positivosAceptadosNegadosMagnitud, 
                                                    negativosAceptadosMagnitud, negativosAceptadosNegadosMagnitud, columnas)
            
            mejorSubconjunto = obtenerMejoresCandidatos(aptitudesResultantes)
            terminoRandom = obtenerElementoRandom(mejorSubconjunto)
            
            if not (conceptosPositivosActual.empty):
                clausula += terminoRandom + ' V '
            else:
                clausula += terminoRandom 
            
            if "neg" in terminoRandom:
                terminoOriginal = terminoRandom.replace('neg_', '')
                conceptosPositivosActual = actualizarEspacioPositivo(conceptosPositivosActual, terminoOriginal, target, True)
            else:
                conceptosPositivosActual = actualizarEspacioPositivo(conceptosPositivosActual, terminoRandom, target, False)  

        columnasHipotesis = clausula.split(' V ')[:-1]
        
        print(conceptosNegativosActual)
        conceptosNegativosActual = conceptosNegativosActual[~conceptosNegativosActual.apply(evaluar_expresion_disyuncion, axis=1, componentes_disyuncion=columnasHipotesis)]
        print(conceptosNegativosActual)
            
        conceptosPositivosActual = conceptosPositivos
        clausulaGeneral.append(clausula)
        
    print(clausulaGeneral)

noFilas = 10

columna_1 = [str(random.uniform(0, 3)) for _ in range(noFilas)]
columna_2 = [str(random.uniform(0, 3)) for _ in range(noFilas)]
columna_3 = [str(random.randint(0, 1)) for _ in range(noFilas)]

columna_1 = [0,1,0,1,1,0,1,0,1,1]
columna_2 = [1,1,0,0,0,0,1,0,0,1]
columna_3 = [0,0,1,0,1,0,1,0,0,1]
columna_4 = [0,0,1,1,0,1,1,0,0,0]
columna_5 = [1,1,1,1,0,0,0,0,0,0]

# data = {"x_1": columna_1, "x_2": columna_2, "target": columna_3}
# df = pd.DataFrame(data)

# Convertir las listas de enteros a listas de strings
columna_1_str = [str(x) for x in columna_1]
columna_2_str = [str(x) for x in columna_2]
columna_3_str = [str(x) for x in columna_3]
columna_4_str = [str(x) for x in columna_4]
columna_5_str = [str(x) for x in columna_5]

# Crear un DataFrame con esas columnas como strings
df = pd.DataFrame({
    'x_1': columna_1_str,
    'x_2': columna_2_str,
    'x_3': columna_3_str,
    'x_4': columna_4_str,
    'target': columna_5_str
})

# Mostrar el DataFrame resultante
print(df)

columnTarget = 'target'
target = '1'

# dfBinarizado = binarizacion(df, columnTarget)
enfoqueOCAT(df, columnTarget, target)



    
