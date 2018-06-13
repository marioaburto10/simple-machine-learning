# 1. Load libraries/Dependencies
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


# 2. Load dataset using local iris file, using pandas to load the data
# Load dataset using iris file from https://archive.ics.uci.edu
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
url = "./iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# specifying the names of each column when loading the data. This will help later when we explore the data.
dataset = pandas.read_csv(url, names=names)
print(dataset)




# 3. Summarize the Dataset
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




# 4. Data Visualization
# 4.1 univariate plots: plots of each individual variable.
# Given that the input variables are numeric, we can create box and whisker plots of each.
# This gives us a much clearer idea of the distribution of the input attributes:
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show() 

# histogram of each input variable to get an idea of the distribution.
dataset.hist()
plt.show() # two of the input variables have a Gaussian distribution. This is useful to note as we can use algorithms that can exploit this assumption.

# 4.2 Multivariate Plots
# interactions between the variables
# these are scatterplots of all pairs of attributes. This can be helpful to spot structured relationships between input variables
scatter_matrix(dataset) # scatter plot matrix
plt.show() # Note the diagonal grouping of some pairs of attributes. This suggests a high correlation and a predictable relationship.




# 5. Evaluate Some Algorithms
# create some models of the data and estimate their accuracy on unseen data.

# 5.1 Separate out a validation dataset.
# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed) # now have training data in the X_train and Y_train for preparing models and a X_validation and Y_validation sets that we can use later.

# 5.2 Set-up the test harness to use 10-fold cross validation.
# use 10-fold cross validation to estimate accuracy.
# will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits
seed = 7 # The specific random seed does not matter
scoring = 'accuracy' # using the metric of ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.

# 5.3 Build 5 different models to predict species from flower measurements
# We don’t know which algorithms would be good on this problem or what configurations to use.
# evaluate 6 different algorithms: Logistic Regression (LR), Linear Discriminant Analysis (LDA), K-Nearest Neighbors (KNN), Classification and Regression Trees (CART), Gaussian Naive Bayes (NB), Support Vector Machines (SVM).
# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.
# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg) # LR: 0.966667 (0.040825), LDA: 0.975000 (0.038188), KNN: 0.983333 (0.033333), CART: 0.975000 (0.038188), NB: 0.975000 (0.053359), SVM: 0.981667 (0.025000)


# 5.4 Select the best model.
# We now have 6 models and accuracy estimations for each. We need to compare the models to each other and select the most accurate.
# also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show() # can see that the box and whisker plots are squashed at the top of the range, with many samples achieving 100% accuracy.




# 6. Make Predictions
# The KNN algorithm was the most accurate model that we tested. Now we want to get an idea of the accuracy of the model on our validation set.
# This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak
# run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report.
# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions)) #  the accuracy is 0.9 or 90%. The confusion matrix provides an indication of the three errors made. Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support showing excellent results (granted the validation dataset was small).
