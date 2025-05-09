import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# Функция для вывода топ влияющих характеристик
def print_top_features(components, n_top_features=5):
    for i in range(components.shape[0]):
        # Получаем абсолютные значения компонентов
        component_abs = components.iloc[i].abs()
        # Сортируем по убыванию
        sorted_features = component_abs.sort_values(ascending=False)
        # Получаем топ характеристик
        top_features = sorted_features.head(n_top_features)
        print(f"ТОп для компонента {i + 1}:")
        print(top_features)
        print("\n")

data = pd.read_csv('../preprocessed_data.csv')
X = data.drop('SalePrice', axis=1)  
y = data['SalePrice']               

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
mse_before = mean_squared_error(y_test, y_pred)
print(f'MSE до PCA: {mse_before:.2f}')

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

components = pd.DataFrame(pca.components_, columns=X_train.columns)
print(components.head(2))
print_top_features(components)

components = pd.DataFrame(
    pca.components_, 
    columns=X_train.columns,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)

print(f"Объясненная дисперсия: {pca.explained_variance_ratio_}")

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis')
plt.xlabel('PC1 (Главная компонента 1)')
plt.ylabel('PC2 (Главная компонента 2)')
plt.colorbar(label='Цена дома')
plt.show()

knn_pca = KNeighborsRegressor(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

y_pred_pca = knn_pca.predict(X_test_pca)
mse_after = mean_squared_error(y_test, y_pred_pca)
print(f'MSE после PCA: {mse_after:.2f}')