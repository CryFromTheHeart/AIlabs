import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
data = pd.read_csv('../train.csv')
target = data['SalePrice']
data = data.drop(['Id', 'SalePrice'], axis=1)

# Предварительная обработка данных
# Заполнение пропущенных значений и кодирование категориальных переменных
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

processed_data = preprocess_data(data.copy())

# Масштабирование данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(processed_data)

# 1. K-Means++ кластеризация
# Метод локтя для определения оптимального числа кластеров
wcss = []
for i in range(1, 30):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 30), wcss, marker='o')
plt.title('Метод локтя для K-Means')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
plt.show()

# Выбираем оптимальное количество кластеров
optimal_k = 8
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

# 2. Агломеративная кластеризация
# Построение дендрограммы
plt.figure(figsize=(15, 7))
dendrogram(linkage(scaled_data, method='ward'))
plt.title('Дендрограмма')
plt.xlabel('Образцы')
plt.ylabel('Расстояние')
plt.show()

# Выполнение агломеративной кластеризации
agg_clustering = AgglomerativeClustering(n_clusters=optimal_k, metric='euclidean', linkage='ward')
agg_labels = agg_clustering.fit_predict(scaled_data)

# 3. DBSCAN кластеризация
# Подбор параметров для DBSCAN
# Выбираем MinPts 
min_pts = 8 

# Вычисляем расстояния до k ближайших соседей
nn = NearestNeighbors(n_neighbors=min_pts).fit(scaled_data)
distances, _ = nn.kneighbors(scaled_data)
k_distances = np.sort(distances[:, -1])  # Берем k-го соседа

# Строим график
plt.figure(figsize=(10, 6))
plt.plot(k_distances)
plt.title(f'k-distance graph (MinPts={min_pts})')
plt.ylabel(f'Расстояние до {min_pts}-го соседа')
plt.xlabel('Точки, отсортированные по расстоянию')
plt.show()

dbscan = DBSCAN(eps=26, min_samples=min_pts)
dbscan_labels = dbscan.fit_predict(scaled_data)

# Оценка качества кластеризации
def evaluate_clustering(labels, data):
    if len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        calinski = calinski_harabasz_score(data, labels)
        davies = davies_bouldin_score(data, labels)

        print(f"""
        Silhouette: {silhouette:.3f} (чем ближе к 1, тем лучше)
        Calinski-Harabasz: {calinski:.0f} (чем выше, тем лучше)
        Davies-Bouldin: {davies:.3f} (чем ближе к 0, тем лучше)
        """)

        return silhouette, calinski, davies
    return None, None, None

kmeans_scores = evaluate_clustering(kmeans_labels, scaled_data)
agg_scores = evaluate_clustering(agg_labels, scaled_data)
dbscan_scores = evaluate_clustering(dbscan_labels, scaled_data)

# Визуализация с помощью PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')

plt.subplot(1, 3, 2)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=agg_labels, cmap='viridis')
plt.title('Agglomerative Clustering')

plt.subplot(1, 3, 3)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=dbscan_labels, cmap='viridis')
plt.show()

# Анализ связи кластеров с целевой переменной
def analyze_clusters(labels, target, algorithm_name):
    cluster_data = pd.DataFrame({'Cluster': labels, 'SalePrice': target})
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Cluster', y='SalePrice', data=cluster_data)
    plt.title(f'Распределение цен по кластерам ({algorithm_name})')
    plt.show()
    
    print(f"\nСтатистика по кластерам ({algorithm_name}):")
    print(cluster_data.groupby('Cluster')['SalePrice'].describe())

analyze_clusters(kmeans_labels, target, 'K-Means')
analyze_clusters(agg_labels, target, 'Agglomerative')
analyze_clusters(dbscan_labels, target, 'DBSCAN')