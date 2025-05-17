import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# 1. Подготовка данных
data = pd.read_csv('../train.csv')
target = data['SalePrice']
features = data.drop(['Id', 'SalePrice'], axis=1)

# Предварительная обработка данных
def preprocess_data(df):
    # Заполнение пропусков
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('None')
        else:
            df[col] = df[col].fillna(df[col].median())
    
    # Кодирование категориальных переменных
    df = pd.get_dummies(df, drop_first=True)
    return df

processed_features = preprocess_data(features.copy())

# Выбор 40 наиболее важных признаков (как в предыдущей работе)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(processed_features, target)
feature_importance = pd.DataFrame({
    'Feature': processed_features.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
top_40_features = feature_importance.head(15)['Feature'].values
X = processed_features[top_40_features]
y = target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Линейная регрессия
print("\n" + "="*50)
print("Линейная регрессия")
print("="*50)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Прогнозирование и оценка
y_pred_lr = lr.predict(X_test_scaled)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"MSE: {mse_lr:.2f}")
print(f"R2: {r2_lr:.3f}")

# График зависимости предсказанных значений от реальных
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Линейная регрессия: реальные vs предсказанные значения')
plt.grid()
plt.show()

# График остатков
residuals = y_test - y_pred_lr
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_lr, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('График остатков линейной регрессии')
plt.grid()
plt.show()

# 3. Регуляризованные методы регрессии
print("\n" + "="*50)
print("Регуляризованные методы регрессии")
print("="*50)

# a. Гребневая регрессия (Ridge)
ridge = Ridge()
param_grid = {'alpha': np.logspace(-3, 3, 100)}
ridge_grid = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train_scaled, y_train)

best_ridge = ridge_grid.best_estimator_
y_pred_ridge = best_ridge.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nГребневая регрессия (Ridge):")
print(f"Лучший alpha: {ridge_grid.best_params_['alpha']:.4f}")
print(f"MSE: {mse_ridge:.2f}")
print(f"R2: {r2_ridge:.3f}")

# b. Lasso-регрессия
lasso = Lasso()
lasso_grid = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train_scaled, y_train)

best_lasso = lasso_grid.best_estimator_
y_pred_lasso = best_lasso.predict(X_test_scaled)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nLasso-регрессия:")
print(f"Лучший alpha: {lasso_grid.best_params_['alpha']:.4f}")
print(f"MSE: {mse_lasso:.2f}")
print(f"R2: {r2_lasso:.3f}")

# c. Эластичная сеть (ElasticNet)
elastic = ElasticNet(max_iter=10000)
param_grid_en = {
    'alpha': np.logspace(-3, 3, 20),
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
elastic_grid = GridSearchCV(elastic, param_grid_en, cv=5, scoring='neg_mean_squared_error')
elastic_grid.fit(X_train_scaled, y_train)

best_elastic = elastic_grid.best_estimator_
y_pred_elastic = best_elastic.predict(X_test_scaled)
mse_elastic = mean_squared_error(y_test, y_pred_elastic)
r2_elastic = r2_score(y_test, y_pred_elastic)

print("\nЭластичная сеть (ElasticNet):")
print(f"Лучшие параметры: alpha={elastic_grid.best_params_['alpha']:.4f}, l1_ratio={elastic_grid.best_params_['l1_ratio']}")
print(f"MSE: {mse_elastic:.2f}")
print(f"R2: {r2_elastic:.3f}")

# Графики зависимости коэффициентов от alpha
alphas = np.logspace(-3, 3, 100)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_)
    
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# График для Ridge регрессии
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ridge, alpha=0.5, color='orange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Ridge регрессия: реальные vs предсказанные значения')
plt.grid()
plt.show()

# График остатков для Ridge
residuals_ridge = y_test - y_pred_ridge
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_ridge, residuals_ridge, alpha=0.5, color='orange')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('График остатков Ridge регрессии')
plt.grid()
plt.show()

# График для Lasso регрессии
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lasso, alpha=0.5, color='green')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Lasso регрессия: реальные vs предсказанные значения')
plt.grid()
plt.show()

# График остатков для Lasso
residuals_lasso = y_test - y_pred_lasso
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_lasso, residuals_lasso, alpha=0.5, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('График остатков Lasso регрессии')
plt.grid()
plt.show()

# График для ElasticNet регрессии
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_elastic, alpha=0.5, color='purple')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('ElasticNet регрессия: реальные vs предсказанные значения')
plt.grid()
plt.show()

# График остатков для ElasticNet
residuals_elastic = y_test - y_pred_elastic
plt.figure(figsize=(8, 6))
plt.scatter(y_pred_elastic, residuals_elastic, alpha=0.5, color='purple')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('График остатков ElasticNet регрессии')
plt.grid()
plt.show()

# Сравнительный график всех методов
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred_lr, alpha=0.3, color='blue', label='Линейная')
plt.scatter(y_test, y_pred_ridge, alpha=0.3, color='orange', label='Ridge')
plt.scatter(y_test, y_pred_lasso, alpha=0.3, color='green', label='Lasso')
plt.scatter(y_test, y_pred_elastic, alpha=0.3, color='purple', label='ElasticNet')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение всех методов: реальные vs предсказанные значения')
plt.legend()
plt.grid()
plt.show()

# 4. Полиномиальная регрессия
print("\n" + "="*50)
print("Полиномиальная регрессия")
print("="*50)

# Выберем один наиболее важный признак для визуализации
best_feature = feature_importance.iloc[0]['Feature']
X_single = X[[best_feature]].values
y_single = y.values

# Разделение на обучающую и тестовую выборки
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_single, y_single, test_size=0.3, random_state=42)

# Создаем конвейер для полиномиальной регрессии
degrees = [1, 2, 3, 4, 5]
results_poly = []

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    model.fit(X_train_s, y_train_s)
    y_pred_poly = model.predict(X_test_s)
    mse_poly = mean_squared_error(y_test_s, y_pred_poly)
    r2_poly = r2_score(y_test_s, y_pred_poly)
    results_poly.append({'degree': degree, 'MSE': mse_poly, 'R2': r2_poly})

results_poly_df = pd.DataFrame(results_poly)
print("\nРезультаты полиномиальной регрессии:")
print(results_poly_df)

# Визуализация полиномиальной регрессии
plt.figure(figsize=(12, 6))

# Сортируем данные для гладкого графика
sort_idx = X_test_s.flatten().argsort()
X_test_s_sorted = X_test_s[sort_idx]
y_test_s_sorted = y_test_s[sort_idx]

# Линейная регрессия (degree=1)
model_lin = Pipeline([
    ('poly', PolynomialFeatures(degree=1)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
model_lin.fit(X_train_s, y_train_s)
y_pred_lin = model_lin.predict(X_test_s_sorted)
plt.scatter(X_test_s, y_test_s, color='blue', alpha=0.3, label='Реальные значения')
plt.plot(X_test_s_sorted, y_pred_lin, color='red', linewidth=2, label='Линейная регрессия')

# Полиномиальная регрессия (degree=3)
model_poly = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
model_poly.fit(X_train_s, y_train_s)
y_pred_poly = model_poly.predict(X_test_s_sorted)
plt.plot(X_test_s_sorted, y_pred_poly, color='green', linewidth=2, label='Полиномиальная регрессия (degree=3)')

plt.xlabel(best_feature)
plt.ylabel('SalePrice')
plt.title('Сравнение линейной и полиномиальной регрессии')
plt.legend()
plt.grid()
plt.show()

# 5. Сравнение всех моделей
print("\n" + "="*50)
print("Сравнение всех моделей")
print("="*50)

results_comparison = pd.DataFrame({
    'Model': ['Линейная', 'Ridge', 'Lasso', 'ElasticNet', 'Полиномиальная (degree=3)'],
    'MSE': [mse_lr, mse_ridge, mse_lasso, mse_elastic, results_poly_df.loc[2, 'MSE']],
    'R2': [r2_lr, r2_ridge, r2_lasso, r2_elastic, results_poly_df.loc[2, 'R2']]
})

print(results_comparison)

# Визуализация сравнения
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.barplot(x='Model', y='MSE', data=results_comparison)
plt.title('Сравнение MSE')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x='Model', y='R2', data=results_comparison)
plt.title('Сравнение R2')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 6. Выводы
print("\n" + "="*50)
print("Выводы")
print("="*50)
print("1. Лучшая модель:", results_comparison.loc[results_comparison['R2'].idxmax(), 'Model'])
print("2. Наилучшие параметры:")
print(f"   - Ridge: alpha={ridge_grid.best_params_['alpha']:.4f}")
print(f"   - Lasso: alpha={lasso_grid.best_params_['alpha']:.4f}")
print(f"   - ElasticNet: alpha={elastic_grid.best_params_['alpha']:.4f}, l1_ratio={elastic_grid.best_params_['l1_ratio']}")
print("3. Рекомендации:")
print("   - Для данного набора данных лучше всего показала себя ElasticNet регрессия")
print("   - Полиномиальная регрессия дает хорошие результаты, но склонна к переобучению")
print("   - Регуляризованные методы помогают улучшить обобщающую способность модели")