def evaluate_rules(rules, df_test):
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
        prediction = 1 if satisfies_all_rules else 0
        predictions.append(prediction)
        y_true.append(int(row['target_column']))  # Reemplaza 'target_column' con tu columna objetivo

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
                if not eval_element(element, row):
                    satisfies_all_elements = False
                    break
            
            # Si todos los elementos de la conjunción se cumplen, la regla se cumple
            if satisfies_all_elements:
                satisfies_any_rule = True
                break

        # Si ninguna regla se cumple para la fila, agregarla a las no conformes
        if not satisfies_any_rule:
            non_conforming_rows.append(row)

    non_conforming_df = pd.DataFrame(non_conforming_rows)
    return non_conforming_df

# Función para evaluar cada elemento de la conjunción en la fila correspondiente
def eval_element(element, row):
    # Separar los términos de la disyunción
    terms = element.split(' v ')

    # Verificar si al menos uno de los términos se cumple en la fila
    for term in terms:
        if eval(term, row):
            return True
    
    return False

# Función para evaluar cada término de la regla en la fila correspondiente
def eval(term, row):
    # Si el término comienza con 'neg_', considera la negación
    if term.startswith('neg_'):
        col_name = term[4:]
        return not bool(int(row[col_name]))
    else:
        return bool(int(row[term]))

def evaluate_rules_conjuncion(rules, df):
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
                if not eval_element(element, row):
                    satisfies_all_elements = False
                    break
            
            # Si todos los elementos de la conjunción se cumplen, la regla se cumple
            if satisfies_all_elements:
                satisfies_any_rule = True
                break

        # Asignar valor a la predicción
        prediction = 1 if satisfies_any_rule else 0
        predictions.append(prediction)
        
        y_true.append(int(row['target_column']))

    return predictions, y_true

# Ejemplo de uso:
import pandas as pd

# Suponiendo que tienes un DataFrame de prueba llamado df_test
data = {
    'x_1_1': ['0', '1', '1', '0', '0'],
    'x_1_2': ['1', '0', '1', '0', '1'],
    'x_2_1': ['1', '0', '1', '0', '1'],
    'x_2_2': ['1', '0', '1', '0', '1'],
    'x_3_1': ['1', '0', '1', '0', '1'],
    'x_3_2': ['0', '1', '0', '1', '0'],
    'x_4_1': ['1', '0', '1', '0', '1'],
    'x_4_2': ['0', '1', '0', '1', '0'],
    'target_column': ['0', '1', '1', '1', '0']
}  # Columna objetivo de ejemplo

df_test = pd.DataFrame(data)

print(df_test)

rules = [
    'x_1_1 ^ x_1_2', 'neg_x_2_1'
]

# Evaluación de reglas en el DataFrame de prueba
predictions, y_true = evaluate_rules_conjuncion(rules, df_test)

matriz = extract_non_conforming_rows_conjuncion(rules, df_test)
print(matriz)

# Cálculo de la matriz de confusión y accuracy
confusion_matrix = [[0, 0], [0, 0]]
for pred, true in zip(predictions, y_true):
    confusion_matrix[pred][true] += 1

accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / sum(sum(confusion_matrix, []))

print("Matriz de Confusión:")
for row in confusion_matrix:
    print(row)

print("Accuracy:", accuracy)
