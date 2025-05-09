import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('../train.csv')

data.fillna(data.mean(numeric_only=True), inplace=True)
data.fillna('Unknown', inplace=True)

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column].astype(str))
    label_encoders[column] = le

X = data.drop(['Id', 'SalePrice'], axis=1)
y = data['SalePrice']

y_binned = pd.cut(y, bins=5, labels=False)

X_train, X_test, y_train, y_test, y_train_binned, y_test_binned = train_test_split(
    X, y, y_binned, test_size=0.2, random_state=42)

model_all = RandomForestRegressor(random_state=42)
model_all.fit(X_train, y_train)
y_pred = model_all.predict(X_test)
mse_all = mean_squared_error(y_test, y_pred)
print(f'MSE на всех признаках: {mse_all:.2f}')

scoring_funcs = {
    'f_classif': f_classif,
    'chi2': chi2
}

results = {}

for method_name, scoring_func in scoring_funcs.items():
    mse_scores = []
    k_values = range(1, X_train.shape[1] + 1)
    
    for k in k_values:
        try:
            selector = SelectKBest(scoring_func, k=k)
            X_train_selected = selector.fit_transform(X_train, y_train_binned)
            X_test_selected = selector.transform(X_test)
            
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train_selected, y_train)
            
            y_pred = model.predict(X_test_selected)
            mse = mean_squared_error(y_test, y_pred)
            mse_scores.append(mse)
        except ValueError as e:
            print(f"Ошибка при k={k} для {method_name}: {str(e)}")
            mse_scores.append(np.nan)
    
    results[method_name] = mse_scores

plt.figure(figsize=(12, 6))
for method_name, mse_scores in results.items():
    plt.plot(range(1, X_train.shape[1] + 1), mse_scores, label=method_name)

plt.axhline(y=mse_all, color='r', linestyle='--', label='Все признаки')
plt.xlabel('Количество выбранных признаков')
plt.ylabel('MSE')
plt.title('Зависимость MSE от количества выбранных признаков (с биннингом SalePrice)')
plt.legend()
plt.grid()
plt.show()

best_k = {}
for method_name, mse_scores in results.items():
    valid_scores = [x for x in mse_scores if not np.isnan(x)]
    if valid_scores:
        best_k[method_name] = np.argmin(valid_scores) + 1
        print(f'Лучшее количество признаков для {method_name}: {best_k[method_name]} с MSE = {min(valid_scores):.2f}')
    else:
        print(f'Метод {method_name} не дал валидных результатов')

print(f'MSE на всех признаках: {mse_all:.2f}')