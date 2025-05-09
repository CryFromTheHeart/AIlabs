import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from pathlib import Path

stdScaler = StandardScaler()
minMaxScaler = MinMaxScaler()

skip_line = '\n----------------------------------------------------------------'

# Измененный список числовых столбцов (без SalePrice)
num_columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'YearBuilt']

categorical_columns = [
    'MSZoning',
    'Neighborhood',
    'HouseStyle',
    'RoofStyle',
    'CentralAir'
]

selected_columns = num_columns + categorical_columns

# Загрузка тестовых данных
data = pd.read_csv('test.csv', usecols=selected_columns)
data = data.drop_duplicates()

print("Проверка первых пяти полей:")
print(data.head())
print(skip_line)

# Масштабируемые столбцы (без SalePrice)
std_num_columns_scale = ['OverallQual', 'GrLivArea', 'YearBuilt']
nm_num_columns_scale = ['GarageCars']

# Обработка числовых данных
num_data = data[num_columns]

print("Кол-во пропусков:")
print(num_data.isnull().sum())

# Заполнение пропусков медианами (для всех числовых столбцов)
for col in num_columns:
    num_data[col] = num_data[col].fillna(num_data[col].median())

print("\nПосле заполнения пропусков:")
print(num_data.isnull().sum())
print(skip_line)

# Масштабирование
print("Масштабируем числовые признаки:")
std_num_data_scaled = stdScaler.fit_transform(num_data[std_num_columns_scale])
std_num_data_scaled = pd.DataFrame(std_num_data_scaled, columns=std_num_columns_scale)
print(std_num_data_scaled.head())

nm_num_data_scaled = minMaxScaler.fit_transform(num_data[nm_num_columns_scale])
nm_num_data_scaled = pd.DataFrame(nm_num_data_scaled, columns=nm_num_columns_scale)
print(nm_num_data_scaled.head())
print(skip_line)

# Обработка категориальных признаков
print("Обработка категориальных признаков:")
categorical_data = data[categorical_columns]

# One-Hot Encoding для MSZoning и RoofStyle
data_encoded = pd.get_dummies(categorical_data, columns=['MSZoning', 'RoofStyle'])

# Для Neighborhood используем фиксированное среднее значение (так как нет SalePrice)
neighborhood_mean_fixed = 180000  # Примерное среднее значение
data_encoded['Neighborhood_encoded'] = neighborhood_mean_fixed
neighborhood_scaled = stdScaler.fit_transform(data_encoded[['Neighborhood_encoded']])
data_encoded['Neighborhood_encoded'] = neighborhood_scaled

# Ordinal Encoding для HouseStyle
house_style_order = {
    '1Story': 1, '1.5Fin': 2, '1.5Unf': 3, '2Story': 4,
    '2.5Fin': 5, '2.5Unf': 6, 'SFoyer': 7, 'SLvl': 8
}
data_encoded['HouseStyle_encoded'] = categorical_data['HouseStyle'].map(house_style_order)

# Label Encoding для CentralAir
le = LabelEncoder()
data_encoded['CentralAir_encoded'] = le.fit_transform(categorical_data['CentralAir'])

print("После encoding:")
print(data_encoded.head())
print(skip_line)

# Сбор финального датасета
print("Финальная выборка:")
data_final = pd.concat([std_num_data_scaled, nm_num_data_scaled, data_encoded], axis=1)

# Удаление исходных категориальных столбцов
existing_cat_cols = [col for col in categorical_columns if col in data_final.columns]
data_final = data_final.drop(existing_cat_cols, axis=1, errors='ignore')

print(data_final.head())
print("\nИтоговые столбцы:", list(data_final.columns))
print(skip_line)

# Сохранение результатов
root_path = Path(__file__).parent.parent
output_path = root_path / "preprocessed_test_data.csv"
data_final.to_csv(output_path, index=False)

print(f"Файл с предобработанными тестовыми данными сохранен по пути: {output_path}")