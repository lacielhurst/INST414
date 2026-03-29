import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
countries = pd.read_csv("C:\\Users\\lacie\\Lacie Files\\INST 414\\Country-data.csv")

# Select features for clustering and scale them
features = ['Gdpp', 'total_fer']
X = countries[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use the elbow method to find the optimal number of clusters
inertia_values = []
for k in range(2,31):
    cluster_model = KMeans(n_clusters=k, random_state=42)
    cluster_model.fit(X_scaled)
    inertia_values.append(cluster_model.inertia_)

plt.plot(range(2,31), inertia_values)
plt.xticks(range(2,31))
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method to Find Optimal k")
plt.show()

# Use KMeans to cluster the countries using k=7 
cluster_model = KMeans(n_clusters=7, random_state=42)
cluster_model.fit(X_scaled)
countries['cluster'] = cluster_model.predict(X_scaled)

# Get an idea of what the clusters look like
print(countries['cluster'].value_counts())
print(countries[['country', 'cluster']].sort_values('cluster')[0:20])

# Visualize the clusters
sns.scatterplot(data=countries, x='Gdpp', y='total_fer', hue='cluster')
plt.xlabel("GDP per Capita")
plt.ylabel("Total Fertility Rate")
plt.title("K-Means Clustering of Countries by GDP Per Capita and Total Fertility Rate")
plt.show()