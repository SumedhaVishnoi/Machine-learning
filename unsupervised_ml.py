# clustering with k- means 
from sklearn.cluster import KMeans
import seaborn as sns 
import matplotlib.pyplot as plt 
#loading the dataset
df= sns.load_dataset("iris ")
x = df.drop("species",axis=1)
#kmeans clustering 
kmeans = KMeans(n_clusters=3,random_state=42)
df["cluster"] = kmeans.fit_predict(x)
#visualize character 
sns.scatterplot(x="sepal length",y="petal length ",hue="cluster",data=df,palette="deep")
plt.title("kmeans clustering on iris ")
plt.show

#PCA- principal component analysis 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]
sns.scatterplot(x="pca1", y="pca2", hue="species", data=df)
plt.title("PCA on Iris Dataset")
plt.show()


