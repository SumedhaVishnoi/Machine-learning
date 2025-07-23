#Linear regression 
#predicting continous values 

from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
import pandas as pd 

#sample dataset
df= pd.DataFrame({
    "hours":[1,2,3,4,5],
    "marks":[20,40,50,65,85]
})

x= df[["hours"]] #features 
y= df["marks"] #target

#train-test split
x_train, x_test,y_train , y_test = train_test_split(x,y,test_size=0.2)

#train model 
model= LinearRegression()
model.fit(x_train, y_train)

# predict 
predictions = model.predict(x_test)
print("predicted marks: ", predictions )

#evaluate 
print("MSE:", mean_squared_error(y_test, predictions))
print("Score (R²):", model.score(x_test, y_test))


# Logistic regression 

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

# Load iris for binary classification
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Convert into binary (Setosa or not)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print(classification_report(y_test, y_pred))


## advanced supervised learning 


# decision trees 
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X_train, y_train)

# Predict
y_pred = tree.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(tree, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()


# random forest 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("Accuracy (RF):", accuracy_score(y_test, y_pred))
import pandas as pd
feature_importances = pd.Series(rf.feature_importances_, index=iris.feature_names)
print(feature_importances.sort_values(ascending=False))


# SVM 
from sklearn.svm import SVC
svm = SVC(kernel='linear')  # Also try: 'rbf', 'poly'
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("Accuracy (SVM):", accuracy_score(y_test, y_pred))



