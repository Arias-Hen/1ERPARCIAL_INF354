import numpy as np

# Usamos los datos ya preparados
X = final_table2.drop(columns=['popularity_discretized'])  # Usamos las características filtradas
y = final_table2['vote_average_discretized']  # Usamos la variable objetivo

# Dividir los datos en entrenamiento y prueba manualmente (por ejemplo, el 70% para entrenamiento, 30% para prueba)
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Función para entrenar y predecir con regularización L1 o L2 sin usar librerías
def train_predict(X_train, X_test, y_train, y_test, regularization='l1', lambda_value=0.01):
    # Inicializar los pesos
    n_features = X_train.shape[1]
    weights = np.random.randn(n_features) * 0.01  # Pesos iniciales pequeños

    # Parámetros de entrenamiento
    learning_rate = 0.001
    n_iterations = 1000
    
    # Entrenar con regularización L1 o L2
    for i in range(n_iterations):
        predictions = np.dot(X_train, weights)
        error = predictions - y_train
        gradient = np.dot(X_train.T, error) / len(y_train)
        
        if regularization == 'l1':
            l1_gradient = lambda_value * np.sign(weights)
            l2_gradient = 0
        elif regularization == 'l2':
            l1_gradient = 0
            l2_gradient = lambda_value * weights
        
        # Actualizar los pesos
        weights -= learning_rate * (gradient + l1_gradient + l2_gradient)

    # Predicciones en el conjunto de prueba
    y_pred = np.dot(X_test, weights)
    y_pred_class = np.round(y_pred)  # Redondear para obtener clases

    # Asegurar que las predicciones estén dentro de los límites posibles de las clases
    y_pred_class = np.clip(y_pred_class, y_train.min(), y_train.max())
    
    # Calcular exactitud manualmente
    correct_predictions = np.sum(y_pred_class == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions

    return accuracy * 100, y_pred_class  # Retorna el porcentaje de exactitud y las predicciones

# Función para calcular la matriz de confusión manualmente
def confusion_matrix_manual(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    
    return matrix

# Calcular exactitud y predicciones con penalización L1 (Lasso)
accuracy_l1, y_pred_l1 = train_predict(X_train, X_test, y_train, y_test, regularization='l1')

# Calcular exactitud y predicciones con penalización L2 (Ridge)
accuracy_l2, y_pred_l2 = train_predict(X_train, X_test, y_train, y_test, regularization='l2')

# Obtener número de clases (únicos valores en y_train)
num_classes = len(np.unique(y_train))

# Calcular matrices de confusión para L1 y L2
conf_matrix_l1 = confusion_matrix_manual(y_test, y_pred_l1, num_classes)
conf_matrix_l2 = confusion_matrix_manual(y_test, y_pred_l2, num_classes)

# Mostrar los resultados de exactitud
print(f"Exactitud con L1 (Lasso): {accuracy_l1:.2f}%")
print(f"Matriz de Confusión (L1):\n{conf_matrix_l1}\n")

print(f"Exactitud con L2 (Ridge): {accuracy_l2:.2f}%")
print(f"Matriz de Confusión (L2):\n{conf_matrix_l2}")
