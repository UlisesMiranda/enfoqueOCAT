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
        valores_unicos = np.sort(df[columna].unique())
        print(valores_unicos)

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
        if(col_value) != '0':
            print("")
        if (not negacion and col_value != '1') or (negacion and col_value != '0'):
            return False
    
    return True


def filter_non_conforming_rows_disyuncion(rules, df):
    non_conforming_rows = []

    for index, row in df.iterrows():
        satisfies_all_rules = True  # Para la conjunción de reglas
        for rule in rules:
            # Separar la regla en elementos
            elements = rule.split(' v ')
            satisfies_any_rule = False  # Para la disyunción de elementos dentro de una regla
            for term in elements:
                # Verificar si al menos un elemento de la regla se cumple en la fila
                element_satisfied = eval(term, row)
                if element_satisfied:
                    satisfies_any_rule = True
                    break
            
            # Si alguno de los elementos de la regla no se cumple, la regla completa no se cumple
            if not satisfies_any_rule:
                satisfies_all_rules = False
                break
        
        # Si la fila cumple con todas las reglas, agregarla a la lista de filas no conformes
        if satisfies_all_rules:
            non_conforming_rows.append(row)

    non_conforming_df = pd.DataFrame(non_conforming_rows)
    return non_conforming_df

def evaluate_rules(rules, df_test:pd.DataFrame, columnTarget):
    predictions = []
    y_true = []

    for index, row in df_test.iterrows():
        satisfies_all_rules = True  # Para la conjunción de reglas
        
        for rule in rules:
            # Separar la regla en elementos
            elements = rule.split(' v ')
            satisfies_any_rule = False  # Para la disyunción de elementos dentro de una regla
            for term in elements:
                # Verificar si al menos un elemento de la regla se cumple en la fila
                element_satisfied = eval(term, row)
                if element_satisfied:
                    satisfies_any_rule = True
                    break
            
            # Si alguno de los elementos de la regla no se cumple, la regla completa no se cumple
            if not satisfies_any_rule:
                satisfies_all_rules = False
                break
        
        # Generar predicción y y_true
        prediction = '1' if satisfies_all_rules else '0'
        predictions.append(prediction)
        y_true.append(row[columnTarget])  # Reemplaza 'target_column' con tu columna objetivo

    return predictions, y_true

# Función para evaluar cada término de la regla en la fila correspondiente
def eval(term, row):
    # Split para obtener las columnas y su negación
    columns = term.split(' ')
    for col in columns:
        # Verificar si la columna está negada
        if col.startswith('neg_'):
            col = col[4:]  # Eliminar el prefijo 'neg_'
            if row[col] == '1':
                return False
        else:
            if row[col] == '0':
                return False
    return True

# CONJUNCION
def extract_non_conforming_rows_conjuncion(rules, df):
    non_conforming_rows = []

    for _, row in df.iterrows():
        satisfies_any_rule = False

        for rule in rules:
            # Separar la regla en elementos de la conjunción
            elements = rule.split(' ^ ')
            satisfies_all_elements = True

            for element in elements:
                # Verificar si cada elemento de la conjunción se cumple en la fila
                if not eval_element_conjuncion(element, row):
                    satisfies_all_elements = False
                    break
            
            # Si todos los elementos de la conjunción se cumplen, la regla se cumple
            if satisfies_all_elements:
                satisfies_any_rule = True
                break

        # Si ninguna regla se cumple para la fila, agregarla a las no conformes
        if satisfies_any_rule:
            non_conforming_rows.append(row)

    non_conforming_df = pd.DataFrame(non_conforming_rows)
    return non_conforming_df

# Función para evaluar cada elemento de la conjunción en la fila correspondiente
def eval_element_conjuncion(element, row):
    # Separar los términos de la disyunción
    terms = element.split(' v ')

    # Verificar si al menos uno de los términos se cumple en la fila
    for term in terms:
        if eval_conjuncion(term, row):
            return True
    
    return False

# Función para evaluar cada término de la regla en la fila correspondiente
def eval_conjuncion(term, row):
    # Si el término comienza con 'neg_', considera la negación
    if term.startswith('neg_'):
        col_name = term[4:]
        return not bool(int(row[col_name]))
    else:
        return bool(int(row[term]))

def evaluate_rules_conjuncion(rules, df, columnTarget):
    predictions = []
    y_true = []

    for _, row in df.iterrows():
        satisfies_any_rule = False

        for rule in rules:
            # Separar la regla en elementos de la conjunción
            elements = rule.split(' ^ ')
            satisfies_all_elements = True

            for element in elements:
                # Verificar si cada elemento de la conjunción se cumple en la fila
                if not eval_element_conjuncion(element, row):
                    satisfies_all_elements = False
                    break
            
            # Si todos los elementos de la conjunción se cumplen, la regla se cumple
            if satisfies_all_elements:
                satisfies_any_rule = True
                break

        # Asignar valor a la predicción
        prediction = '1' if satisfies_any_rule else '0'
        predictions.append(prediction)
        
        y_true.append(row[columnTarget])

    return predictions, y_true


def generarNuevaRepresentacionDf(df):
    atributos = df.columns[:-1]
    lista_dataframes = []

    for atributo in atributos:
        df_por_atributo = df[[atributo]].copy()
        df_por_atributo[df.columns[-1]] = df[[df.columns[-1]]].copy()
        df_bin_atributo = oneHot(df_por_atributo)
        lista_dataframes.append(df_bin_atributo)
        
    lista_dataframes.append(df[[df.columns[-1]]])
    dfBinarizado =pd.concat(lista_dataframes, axis=1)
    print(dfBinarizado)
    
    return dfBinarizado

def calcularMatrizConfusion(y_true, y_pred, targetClass):
    verdaderos_positivos = 0
    verdaderos_negativos = 0
    falsos_positivos = 0
    falsos_negativos = 0

    for i in range(len(y_true)):
        if y_true[i] == targetClass:
            if y_pred[i] == targetClass:
                verdaderos_positivos += 1
            else:
                falsos_negativos += 1
        else:
            if y_pred[i] != targetClass:
                verdaderos_negativos += 1
            else:
                falsos_positivos += 1

    accuracy = (verdaderos_positivos + verdaderos_negativos) / len(y_true)
    recall = verdaderos_positivos / (verdaderos_positivos + falsos_negativos + 0.000001)
    precision = verdaderos_positivos / (verdaderos_positivos + falsos_positivos + 0.0001)
    f1_score = 2 * (precision * recall) / (precision + recall + 0.000001)

    print("\nMatriz de Confusión:")
    print("Verdaderos Positivos:", verdaderos_positivos)
    print("Verdaderos Negativos:", verdaderos_negativos)
    print("Falsos Positivos:", falsos_positivos)
    print("Falsos Negativos:", falsos_negativos)
    print("\nAccuracy:", accuracy)
    print("Sensibilidad (Recall):", recall)
    print("Precisión (Precision):", precision)
    print("Puntuación F1 (F1-score):", f1_score)

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
                # clausula += terminoRandom + ' ^ '
            else:
                clausula += terminoRandom 
            
            if "neg" in terminoRandom:
                terminoOriginal = terminoRandom.replace('neg_', '')
                conceptosPositivosActual = actualizarEspacioPositivo(conceptosPositivosActual, terminoOriginal, target, True)
            else:
                conceptosPositivosActual = actualizarEspacioPositivo(conceptosPositivosActual, terminoRandom, target, False)  

        # print("Conceptos positivios")
        # print(conceptosPositivos)
        
        # print("Conceptos neg ac")
        # print(conceptosNegativosActual)
        queshow = conceptosNegativosActual.copy() 

        # conceptosNegativosActual = conceptosNegativosActual[conceptosNegativosActual.apply(lambda x: evaluar_expresion_disyuncion(x, clausula), axis=1)]
        # conceptosNegativosActual = conceptosNegativosActual[~conceptosNegativosActual.apply(lambda x: evaluar_expresion_conjuncion(x, clausula), axis=1)]
        
        conceptosNegativosActual = filter_non_conforming_rows_disyuncion([clausula[:-3]], conceptosNegativosActual)
        # conceptosNegativosActual = extract_non_conforming_rows_conjuncion([clausula[:-3]], conceptosNegativosActual)
        
        # print("Conceptos neg nuevs")
        # print(conceptosNegativosActual)
            
        conceptosPositivosActual = conceptosPositivos
        clausulaGeneral.append(clausula[:-3])
        
        
        if (queshow.equals(conceptosNegativosActual)):
                # print("")
            conceptosNegativosActual = conceptosNegativosActual.drop(conceptosNegativosActual.index)
            # print(clausulaGeneral)
        
    print("\nHipotesis: ", clausulaGeneral)

    return clausulaGeneral


# PRUEBA DE DATOS
columna_1 = [0,1,0,1,1,0,1,0,1,1]
columna_2 = [1,1,0,0,0,0,1,0,0,1]
columna_3 = [0,0,1,0,1,0,1,0,0,1]
columna_4 = [0,0,1,1,0,1,1,0,0,0]
columna_5 = [1,1,1,1,0,0,0,0,0,0]
# # Convertir las listas de enteros a listas de strings
columna_1_str = [str(x) for x in columna_1]
columna_2_str = [str(x) for x in columna_2]
columna_3_str = [str(x) for x in columna_3]
columna_4_str = [str(x) for x in columna_4]
columna_5_str = [str(x) for x in columna_5]

# Crear un DataFrame con esas columnas como strings
dfBinarizado = pd.DataFrame({
    'x_1': columna_1_str,
    'x_2': columna_2_str,
    'x_3': columna_3_str,
    'x_4': columna_4_str,
    'target': columna_5_str
})
columnTarget = 'target'
target = '1'

print("EJERCICIO DE EJEMPLO\n", dfBinarizado)

hipotesis = enfoqueOCAT(dfBinarizado, columnTarget, target)

y_pred, y_true = evaluate_rules(hipotesis, dfBinarizado, columnTarget)
# y_pred, y_true = evaluate_rules_conjuncion(hipotesis, df_test_binarizado, columnTarget)

calcularMatrizConfusion(y_true, y_pred, target)

# PROBANDO

for i in range(1, 4):
    print(f"\n----DATASET MONK {i}:")

    df_train = pd.read_csv(f'monk-{i}-train.csv')
    df_test = pd.read_csv(f'monk-{i}-test.csv') 
    columnTarget = df_train.columns[-1]
    target = '1'

    df_train = df_train.astype(str)
    df_test = df_test.astype(str)

    df_combinado = pd.concat([df_train, df_test], ignore_index=True)

    # Codificar columnas categóricas en one-hot
    df_encoded = pd.get_dummies(df_combinado, columns=df_train.columns[:-1])

    columnaRemoved = df_encoded.pop(columnTarget)
    df_encoded[columnTarget] = columnaRemoved


    # Filtrar las columnas con el formato x_1_1, x_1_2, x_2_1, etc.
    columnas_binarias = [col for col in df_encoded.columns if col.startswith('x_')]

    # Crear nuevas columnas binarias combinando los valores
    for col in columnas_binarias:
        df_encoded[col] = df_encoded[col].apply(lambda x: '1' if x > 0 else '0')


    # Dividir nuevamente en df_train y df_test
    df_train_binarizado = df_encoded.iloc[:len(df_train)].astype(str)
    df_test_binarizado = df_encoded.iloc[len(df_train):].astype(str)

    print("TRAIN\n", df_train)
    print("TEST\n", df_test)

    # df_train_binarizado = generarNuevaRepresentacionDf(df_train).astype(str)
    # df_test_binarizado = generarNuevaRepresentacionDf(df_test).astype(str)

    print("TRAIN bin\n", df_train_binarizado)
    print("TEST bin\n", df_test_binarizado)

    hipotesis = enfoqueOCAT(df_train_binarizado, columnTarget, target)

    y_pred, y_true = evaluate_rules(hipotesis, df_test_binarizado, columnTarget)
    # y_pred, y_true = evaluate_rules_conjuncion(hipotesis, df_test_binarizado, columnTarget)

    calcularMatrizConfusion(y_true, y_pred, target)





        
