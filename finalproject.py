import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
countries = pd.read_csv("C:\\Users\\lacie\\Lacie Files\\INST 414\\Country-data.csv")

# Part One: Clustering by GDP per capita and total fertility
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
countries['cluster'].value_counts()
countries[['country', 'cluster']].sort_values('cluster')[0:20]

# Visualize the clusters
sns.scatterplot(data=countries, x='Gdpp', y='total_fer', hue='cluster')
plt.xlabel("GDP per Capita")
plt.ylabel("Total Fertility Rate")
plt.title("K-Means Clustering of Countries by GDP Per Capita and Total Fertility Rate")
plt.show()

# Connects clusters to all variables included 
cluster_summary = countries.groupby('cluster')[['Gdpp','total_fer','child_mort','health','life_expec']].mean().round(2)
print(cluster_summary)

# Part Two: Predicting life expectancy
# Creates features and target
features = ['Gdpp', 'total_fer','child_mort','health']
X = countries[features]
y = countries['life_expec']

# Scales the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splits data into test and train
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
test_indices = y_test.index

# Determines the best k value
rmse_values = []

for k in range(1, 21):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_values.append(rmse)

plt.plot(range(1,21), rmse_values)
plt.xlabel("K Value")
plt.ylabel("RMSE")
plt.title("Choosing Optimal K for KNN")
plt.show()

# Train the model
knn = KNeighborsRegressor(n_neighbors=9)
knn.fit(X_train, y_train)

# Predict the model
y_pred = knn.predict(X_test)

# Evaluate the model 
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R² Score:", r2)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs Predicted Life Expectancy")
plt.show()

results = pd.DataFrame({
    'Country': countries.loc[test_indices, 'country'],
    'Actual Life Expectancy': y_test,
    'Predicted Life Expectancy': y_pred
})

results['Error'] = abs(results['Actual Life Expectancy'] - results['Predicted Life Expectancy'])
predictions = results.sort_values(by='Error', ascending=True)

print("Best Predictions:")
print(predictions.head(10))

print("Worst Predictions:")
print(predictions.tail(10))

results['cluster'] = countries.loc[test_indices, 'cluster']
cluster_pred = results.groupby('cluster')[['Actual Life Expectancy', 'Predicted Life Expectancy']].mean().round(2)
print(cluster_pred)