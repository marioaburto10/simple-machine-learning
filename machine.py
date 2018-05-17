# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset using iris file from https://archive.ics.uci.edu
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Load dataset using local iris file, using pandas to load the data
url = "./iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# specifying the names of each column when loading the data. This will help later when we explore the data.
dataset = pandas.read_csv(url, names=names)
print(dataset)


# Dimensions of Dataset
# shape
print(dataset.shape) # this prints (150, 5): 150 instances and 5 attributes or 150 rows, 5 columns

# Peek at the Data
# head
print(dataset.head(20)) # first 20 rows of the data

# Statistical Summary
# descriptions
# taking a look at a summary of each attribute. This includes the count, mean, the min and max values as well as some percentiles.
print(dataset.describe()) # all of the numerical values have the same scale (centimeters) and similar ranges between 0 and 8 centimeters.

# Class Distribution
# number of instances (rows) that belong to each class. We can view this as an absolute count
print(dataset.groupby('class').size()) # each class has the same number of instances (50 or 33% of the dataset).

# Data Visualization
# univariate plots: plots of each individual variable.
# Given that the input variables are numeric, we can create box and whisker plots of each.
# This gives us a much clearer idea of the distribution of the input attributes:
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show() 

# histogram of each input variable to get an idea of the distribution.
dataset.hist()
plt.show() # two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.
