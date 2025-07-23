name = "sumi"
age= 21
height=5.6
is_student = true 

print(type(name))
print(type(age))
print(type(height))
print(type(is_student))

#if-else
age = 18 
if age>=18 
   print("you are eligible to vote")
else:
    print("not eligible yet ")
    
#loops
#for loop 
for i in range(5):
    print("hello",i)
    
#while loop 
count = 0 
while count<3:
    print("counting:",count)
    count +=1
    
#lists 
fruits=["apple","banana","mango"]
#access 
print(fruits[0])
#add 
fruits.append("orange")
#remove
fruits.remove("banana")
#loop 
for fruit in fruits:
    print(fruits)
    
#Dictionaries
student={"name": Sumi,
        "age":21,
        "branch":"ece"} 
#access
print(student["name"])

#add new key
student["cgpa"]=8.6
#loop throgh keys and values 
for key, value in student.items():
    print(key,"->",value)
    

#Numpy- used for : fast numerical operations 
multi dimensional arrays 
linear algebra , math functions 

impot numpy as np 
a= np.array([1,2,3])
b= np.array([[1,2],[3,4]])
print(a.shape)
print(b.shape)

zeroes = np.zeroes((2,3))
ones = np.ones((3,3))
rand = np.random.rand(2,2)
eye = np.eye(3)


#Pandas- used: handle tabular data 
               analyze dataset 
               
import panda as pd  
#to read a csv file 
df=pd.read_csv()
#to explore about the data 
print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

df["species"]                   # Single column
df[["sepal_length", "species"]] # Multiple columns

df[df["species"] == "setosa"]   # Filter rows
df[df["sepal_length"] > 5.0]    # Filter by condition

df.dropna()              # Drop missing values
df.fillna(0)             # Fill missing with 0
df.rename(columns={"sepal_length": "SepalLength"}, inplace=True)

#DATA VISUALIZATION WITH MATPLOTLIB AND SEABORN 
#MATPLOTLIB 
import matplotlib.pyplot as plt 
x= [1,2,3,4]
y=[10,20,30,40]
plt.plot(x,y)
plt.title("line plot")
plt.xlabel("x axis")
plt.ylabel("y axis ")
plt.show 
         
# bar plot and histogram 
# Bar Chart
names = ['A', 'B', 'C']
scores = [65, 70, 85]

plt.bar(names, scores)
plt.title("Student Scores")
plt.show()

# Histogram
import numpy as np
data = np.random.randn(1000)
plt.hist(data, bins=30)
plt.title("Histogram")
plt.show()


#seaborn 
import seaborn as sns
import pandas as pd
# Load iris dataset
df = sns.load_dataset("iris")
df.head()


#scatter plot 
sns.scatterplot(x="sepal_length", y="petal_length", data=df, hue="species")
plt.title("Sepal vs Petal Length")
plt.show()

#box plot and violin plot
sns.boxplot(x="species", y="sepal_length", data=df)
plt.title("Boxplot of Sepal Length by Species")
plt.show()
sns.violinplot(x="species", y="petal_width", data=df)

#heatmap- correlation matrix 
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# descriptive statistics 
imprt pandas as pd 
improt seaborn as sns 
df= sns.load_dataset('iris')
print("mean :", df.mean(numeric_only= true ))
print("median :", df.median(numeric_only= true ))
print("standdard deviation  :", df.std(numeric_only= true ))
print("min/max :", df.min(numeric_only= true ),df.max(numeric_only=True))

# plot normal distribution 
import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(loc=50, scale=10, size=1000)
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
plt.title("Normal Distribution")
plt.show()

# correlation and covarience 
print(df.corr(numeric_only=True))

#hypthesis testing 
from scipy.stats import ttest_ind
setosa = df[df["species"] == "setosa"]["sepal_length"]
virginica = df[df["species"] == "virginica"]["sepal_length"]
t_stat, p_val = ttest_ind(setosa, virginica)
print("T-Stat:", t_stat)
print("P-Value:", p_val)




  