import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 7. Preparar los datos para el modelo
X = final_table.drop(columns=['vote_average_discretized'])
y = final_table['vote_average_discretized'].astype(int)  # Variable objetivo

# 8. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 9. Balancear las clases con RandomOverSampler
ros = RandomOverSampler(random_state=42)
X_train_balanced, y_train_balanced = ros.fit_resample(X_train, y_train)

# 10. Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# 11. Regresión logística con penalización L1 (Lasso)
logreg_l1 = LogisticRegression(penalty='l1', solver='liblinear')
logreg_l1.fit(X_train_scaled, y_train_balanced)
y_pred_l1 = logreg_l1.predict(X_test_scaled)

# 12. Regresión logística con penalización L2 (Ridge)
logreg_l2 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=1000)
logreg_l2.fit(X_train_scaled, y_train_balanced)
y_pred_l2 = logreg_l2.predict(X_test_scaled)

# 13. Evaluación: Exactitud
accuracy_l1 = accuracy_score(y_test, y_pred_l1)
accuracy_l2 = accuracy_score(y_test, y_pred_l2)

print(f'Exactitud con L1 (Lasso): {accuracy_l1 * 100:.2f}%')
print(f'Exactitud con L2 (Ridge): {accuracy_l2 * 100:.2f}%')

# 14. Matriz de confusión y reporte de clasificación
conf_matrix_l1 = confusion_matrix(y_test, y_pred_l1)
conf_matrix_l2 = confusion_matrix(y_test, y_pred_l2)

print("\nMatriz de Confusión (L1):")
print(conf_matrix_l1)
print("\nMatriz de Confusión (L2):")
print(conf_matrix_l2)

print("\nReporte de Clasificación (L1):")
print(classification_report(y_test, y_pred_l1))
print("\nReporte de Clasificación (L2):")
print(classification_report(y_test, y_pred_l2))
