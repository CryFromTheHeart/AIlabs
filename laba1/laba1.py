import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder, MinMaxScaler
from pathlib import Path

stdScaler = StandardScaler()
minMaxScaler = MinMaxScaler()

skip_line = '\n----------------------------------------------------------------'

num_colums = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt']

categorical_colums = [
    'MSZoning',       # Тип зонирования (жилая, коммерческая и т.д.)
    'Neighborhood',   # Район
    'HouseStyle',     # Стиль дома (1 этаж, 2 этажа и т.д.)
    'RoofStyle',      # Тип крыши
    'CentralAir'      # Наличие кондиционера (Y/N)
]

selected_columns = [*num_colums, *categorical_colums]

data = pd.read_csv('../train.csv', usecols=selected_columns)
#data = pd.read_csv('titanic.csv', usecols=selected_columns)

data = data.drop_duplicates()

print("Проверка первых пяти полей")
print(data.head())

print(skip_line)

std_num_colums_scale_colums = ['SalePrice', 'OverallQual', 'GrLivArea', 'YearBuilt']
nm_num_colums_scale_colums = ['GarageCars']

num_data = data[num_colums]

print("\nКол-во пропусков")
print(num_data.isnull().sum())

# 1 поле, но можно описать все (если есть пропуски в столбцах)
num_data.loc[:, 'SalePrice'] = num_data.loc[:, 'SalePrice'].fillna(num_data['SalePrice'].median())

print()
print(num_data.isnull().sum())

print(skip_line)

print("Масштабируем")
std_num_data_scaled = stdScaler.fit_transform(num_data[std_num_colums_scale_colums])
std_num_data_scaled = pd.DataFrame(std_num_data_scaled, columns=std_num_colums_scale_colums)
print(std_num_data_scaled)


nm_num_data_scaled = minMaxScaler.fit_transform(num_data[nm_num_colums_scale_colums])
nm_num_data_scaled = pd.DataFrame(nm_num_data_scaled, columns=nm_num_colums_scale_colums)
print(nm_num_data_scaled)

print(skip_line)

print("Обработка категориальных признаков")

categorical_data = data[categorical_colums]

data_encoded = pd.get_dummies(categorical_data, columns=['MSZoning', 'RoofStyle'])

neighborhood_mean_price = data.groupby('Neighborhood')['SalePrice'].mean().to_dict()
data_encoded['Neighborhood_encoded'] = categorical_data['Neighborhood'].map(neighborhood_mean_price)
neighborhood_scaled = stdScaler.fit_transform(data_encoded[['Neighborhood_encoded']])
data_encoded['Neighborhood_encoded'] = neighborhood_scaled

house_style_order = {
    '1Story': 1,
    '1.5Fin': 2,
    '1.5Unf': 3,
    '2Story': 4,
    '2.5Fin': 5,
    '2.5Unf': 6,
    'SFoyer': 7,
    'SLvl': 8
}
data_encoded['HouseStyle_encoded'] = categorical_data['HouseStyle'].map(house_style_order)

le = LabelEncoder()
data_encoded['CentralAir_encoded'] = le.fit_transform(categorical_data['CentralAir'])

print("После endcod-а")
print(data_encoded)

print(skip_line)

print("Финальная выборка")
data_final = pd.concat([std_num_data_scaled, nm_num_data_scaled, data_encoded], axis=1)

existing_cat_cols = [col for col in categorical_colums if col in data_final.columns]
data_final = data_final.drop(existing_cat_cols, axis=1)

print(data_final)

print("Итоговые столбцы:", list(data_final.columns))
print(skip_line)

# Путь к корневой папке (на один уровень выше текущей)
root_path = Path(__file__).parent.parent

# Сохраняем data_final в корень
output_path = root_path / "preprocessed_data.csv"
data_final.to_csv(output_path, index=False)

print(f"Файл с предобработанными данными сохранен по пути: {output_path}")