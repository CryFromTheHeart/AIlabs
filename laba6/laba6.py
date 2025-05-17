# Импорт всех необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                            confusion_matrix, roc_curve, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt


data = pd.read_csv('../train.csv')

median_price = data['SalePrice'].median()
data['Price_Category'] = (data['SalePrice'] > median_price).astype(int)

X = data.drop(['SalePrice', 'Price_Category', 'Id'], axis=1)
y = data['Price_Category']

categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'),
    "Linear SVM": SVC(kernel='linear', random_state=42),
    "Perceptron": Perceptron(random_state=42)
}

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"\n{name} Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.3f}")

# Настройка гиперпараметров для SVM
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__degree': [2, 3]  # Для полиномиального ядра
}

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42))
])

grid_search = GridSearchCV(svm_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest SVM parameters:", grid_search.best_params_)
print("Best ROC AUC:", grid_search.best_score_)

# 4. Оценка лучшей модели
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)
y_proba = best_svm.decision_function(X_test)

print("\nBest SVM Performance:")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")

# 5. Визуализация результатов
# Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC-кривая
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'SVM (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 6. Сравнение всех моделей
results = []
for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred)
    })

# Добавляем лучшую SVM модель
results.append({
    'Model': 'SVM (Optimized)',
    'Accuracy': accuracy_score(y_test, best_svm.predict(X_test)),
    'F1-score': f1_score(y_test, best_svm.predict(X_test)),
    'ROC AUC': roc_auc_score(y_test, best_svm.decision_function(X_test))
})

results_df = pd.DataFrame(results)
print("\nComparison of all models:")
print(results_df.to_markdown(index=False))

C_values = [0.001, 0.01, 0.1, 1, 10, 100]
accuracies = []

for C in C_values:
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=C, penalty='l2', solver='liblinear'))
    ])
    model.fit(X_train, y_train)
    accuracies.append(accuracy_score(y_test, model.predict(X_test)))

plt.plot(C_values, accuracies, marker='o')
plt.xscale('log')  # Логарифмическая шкала для C
plt.xlabel('C (Сила регуляризации)')
plt.ylabel('Accuracy')
plt.title('Влияние параметра C на точность')
plt.show()