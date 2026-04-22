# Required packages
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load the dataset
students = pd.read_csv("C:\\Users\\lacie\\Lacie Files\\INST 414\\StudentPerformanceFactors.csv")

# Function to create a new variable that categorizes exam score
def categorize(score):
    if score < 68:
        return 0
    else:
        return 1

# Create the new variable using the function
students["Exam_Status"] = students["Exam_Score"].apply(categorize)

# Subsets the dataset to only include desired variables
students = students[["Hours_Studied","Attendance","Sleep_Hours","Previous_Scores","Physical_Activity","Tutoring_Sessions","Exam_Status"]]

# Creates features and target. Creates "X_original" to refer back to for interpretation purposes
X_original = students[["Hours_Studied","Attendance","Sleep_Hours","Previous_Scores","Physical_Activity","Tutoring_Sessions"]]
X = X_original.copy()
y = students["Exam_Status"]

# Splits data into test and train. Creates "X_test_original" to refer back to for interpretation purposes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
X_test_original = X_test.copy()

# Scales the features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Determines best k value
k_values = range(1, 41)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracies.append(knn.score(X_test, y_test))

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("K vs Accuracy")
plt.show()

# Fits model
knn = KNeighborsClassifier(n_neighbors=27, metric='euclidean')
knn.fit(X_train, y_train)

# Predicts model
y_pred = knn.predict(X_test)

# Prints actual and predicted values
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})
print(results.head(10))

# Accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report 
print(classification_report(y_test, y_pred))

# Prints incorrect predictions including their features 
results = X_test_original.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred

incorrect = results[results["Actual"] != results["Predicted"]]
print(incorrect.head(5))