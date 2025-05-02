import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

num_colums = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt']
categorical_colums = [
    'MSZoning',       # Тип зонирования (жилая, коммерческая и т.д.)
    'Neighborhood',   # Район
    'HouseStyle',     # Стиль дома (1 этаж, 2 этажа и т.д.)
    'RoofStyle',      # Тип крыши
    'CentralAir'      # Наличие кондиционера (Y/N)
]

data = pd.read_csv('../train.csv')
print(data.info())

for col in num_colums:
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col], kde=True)
    plt.title(f'Распределение {col}')
    plt.show()

sns.boxplot(data[num_colums])
plt.xticks(rotation=45)
plt.show()

for col in categorical_colums:
    plt.figure(figsize=(10, 5))
    sns.countplot(data[col])
    plt.title(f'Распределение {col}')
    plt.show()

corr_matrix = data[num_colums].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Тепловая карта корреляций')
plt.show()

sns.scatterplot(data=data, x='GrLivArea', y='SalePrice')
plt.title('Зависимость цены от площади')
plt.show()