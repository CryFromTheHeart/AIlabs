import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve, classification_report, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

data = pd.read_csv('../train.csv')

median_price = data['SalePrice'].median()
y = (data['SalePrice'] > median_price).astype(int)
X = data.drop('SalePrice', axis=1)

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

params = {
    'max_depth': [3, 5, 7, None],
    'max_features': ['sqrt', 'log2', None]
}

# Обучение Decision Tree
dt = DecisionTreeClassifier(criterion='gini')
dt_cv = GridSearchCV(dt, params, cv=5, scoring='accuracy')
dt_cv.fit(X_train, y_train)

# Обучение Random Forest
rf = RandomForestClassifier(criterion='gini')
rf_cv = GridSearchCV(rf, params, cv=5, scoring='accuracy')
rf_cv.fit(X_train, y_train)

# Обучение Gradient Boosting
gb = GradientBoostingClassifier()
gb_cv = GridSearchCV(gb, params, cv=5, scoring='accuracy')
gb_cv.fit(X_train, y_train)

# Оценка моделей
models = {
    'Decision Tree': dt_cv,
    'Random Forest': rf_cv,
    'Gradient Boosting': gb_cv
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred))
    #print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    #print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")

# Визуализация дерева (для Decision Tree)
plt.figure(figsize=(15, 10))
plot_tree(dt_cv.best_estimator_, max_depth=3, filled=True, feature_names=X.columns, class_names=['Недорогие', 'Дорогие'])
plt.show()

plt.figure(figsize=(20, 12))
plot_tree(rf_cv.best_estimator_.estimators_[0], max_depth=3, filled=True,
          feature_names=X.columns,
          class_names=['Недорогие', 'Дорогие'],
          rounded=True)
plt.title("Одно из деревьев Random Forest", fontsize=14)
plt.show()

plt.figure(figsize=(20, 12))
plot_tree(gb_cv.best_estimator_.estimators_[0, 0], max_depth=3, filled=True,
         feature_names=X.columns,
         class_names=['Недорогие', 'Дорогие'],
         rounded=True)
plt.title("Одно из деревьев Gradient Boosting", fontsize=14)
plt.show()

print("Лучшие параметры для Decision Tree:")
print(dt_cv.best_params_)

print("\nЛучшие параметры для Random Forest:")
print(rf_cv.best_params_)

print("\nЛучшие параметры для Gradient Boosting:")
print(gb_cv.best_params_)

# Функция для построения матрицы ошибок
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Недорогие', 'Дорогие'], 
                yticklabels=['Недорогие', 'Дорогие'])
    plt.title(f'Confusion Matrix ({model_name})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Построение для каждой модели
for name, model in models.items():
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, name)

plt.figure(figsize=(8, 6))
colors = ['blue', 'green', 'red']

for (name, model), color in zip(models.items(), colors):
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:  # Если нет predict_proba, используем decision_function
        y_proba = model.decision_function(X_test)
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=color,
             label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-кривые всех моделей')
plt.legend(loc="lower right")
plt.show()

dt_importances = pd.Series(
    dt_cv.best_estimator_.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(10)

print("Топ-10 важных признаков Decision Tree:")
print(dt_importances.to_string())

rf_importances = pd.Series(
    rf_cv.best_estimator_.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(10)

print("\nТоп-10 важных признаков Random Forest:")
print(rf_importances.to_string())

gb_importances = pd.Series(
    gb_cv.best_estimator_.feature_importances_,
    index=X.columns
).sort_values(ascending=False).head(10)

print("\nТоп-10 важных признаков Gradient Boosting:")
print(gb_importances.to_string())