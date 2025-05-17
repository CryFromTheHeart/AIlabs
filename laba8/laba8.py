import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
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

# Загрузка данных
data = pd.read_csv('../train.csv')

# Преобразование в бинарную классификацию
median_price = data['SalePrice'].median()
y = (data['SalePrice'] > median_price).astype(int)  # 1 - дорогие, 0 - недорогие
X = data.drop(['SalePrice', 'Id'], axis=1)

# Обработка пропусков
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Числовые признаки
num_imputer = SimpleImputer(strategy='median')
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])

# Категориальные признаки
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])

# Кодирование категориальных признаков
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# k-NN
knn = KNeighborsClassifier()
params_knn = {
    'n_neighbors': [3, 5, 10, 20],
    'metric': ['euclidean', 'manhattan']
}
knn_cv = GridSearchCV(knn, params_knn, cv=5, scoring='accuracy')
knn_cv.fit(X_train, y_train)
print("Лучшие параметры k-NN:", knn_cv.best_params_)

# k-WNN (взвешенный k-NN)
kwnn = KNeighborsClassifier(weights='distance')
params_kwnn = {
    'n_neighbors': [3, 5, 10, 20],
    'metric': ['euclidean', 'manhattan']
}
kwnn_cv = GridSearchCV(kwnn, params_kwnn, cv=5, scoring='accuracy')
kwnn_cv.fit(X_train, y_train)
print("Лучшие параметры k-WNN:", kwnn_cv.best_params_)

# Парзеновское окно 
def parzen_predict(X_train, y_train, X_test, h=1.0, kernel='gaussian'):
    y_pred = []
    for x in X_test:
        distances = np.linalg.norm(X_train - x, axis=1)
        if kernel == 'gaussian':
            weights = np.exp(-(distances**2) / (2 * h**2))
        else:  # rectangular
            weights = (distances <= h).astype(float)
        # Выбор класса с максимальным суммарным весом
        y_pred.append(np.argmax([np.sum(weights[y_train == c]) for c in [0, 1]]))
    return np.array(y_pred)

# Подбор h
best_h, best_acc = 0, 0
for h in [0.5, 1.0, 2.0, 5.0, 10, 50]:
    y_pred = parzen_predict(X_train, y_train, X_test, h=h)
    acc = accuracy_score(y_test, y_pred)
    if acc > best_acc:
        best_h, best_acc = h, acc
print(f"Лучшее h: {best_h}, Accuracy: {best_acc:.3f}")

models = {
    'k-NN': knn_cv.best_estimator_,
    'k-WNN': kwnn_cv.best_estimator_,
    'Parzen': lambda X: parzen_predict(X_train, y_train, X, h=best_h)
}

for name, model in models.items():
    y_pred = model.predict(X_test) if name != 'Parzen' else model(X_test)
    print(f"\n{name}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Визуализация результатов
n_values = params_knn['n_neighbors']

# Средняя accuracy для k-NN
knn_acc = [
    np.mean([knn_cv.cv_results_[f'split{i}_test_score'][j] for i in range(5)])
    for j in range(len(n_values))
]

# Средняя accuracy для k-WNN
kwnn_acc = [
    np.mean([kwnn_cv.cv_results_[f'split{i}_test_score'][j] for i in range(5)])
    for j in range(len(params_kwnn['n_neighbors']))
]

plt.figure(figsize=(10, 5))
plt.plot(n_values, knn_acc, 'o-', label='k-NN')
plt.plot(params_kwnn['n_neighbors'], kwnn_acc, 's-', label='k-WNN')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Зависимость Accuracy от количества соседей')
plt.grid(True)
plt.show()

# Визуализация ядер
d = np.linspace(0, 3, 100)
h = 1.0
plt.figure(figsize=(10, 5))
plt.plot(d, np.exp(-d**2 / (2 * h**2)), label='Гауссовское')
plt.plot(d, (d <= h).astype(float), label='Прямоугольное')
plt.plot(d, np.where(d <= h, 0.75 * (1 - d**2), 0), label='Епанечникова')
plt.xlabel('Расстояние')
plt.ylabel('Вес')
plt.legend()
plt.title('Ядра Парзеновского окна (h=1.0)')
plt.show()