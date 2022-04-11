from enum import auto
from tkinter import Y
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal_length','sepal_width','petal_length','petal-width','class']
dataset = read_csv(url, names=names)
last_row = dataset.iloc[-1:]#getting last row output

#slope
# print(dataset.shape)
# print(dataset.head(50))
# print(dataset.describe())
# print(last_row)
# print(dataset.groupby('class').size())

#Plotting
# dataset.plot(kind='box',subplots=True, layout=(2,2), sharex=False, sharey=False)
# dataset.hist()
# scatter_matrix(dataset)
# pyplot.show()

#Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]

X_train, X_validation, Y_train, Y_validation = train_test_split(X,y,test_size=0.20,random_state=1,shuffle=True)


#Spot check algorithm
models = []
models.append(('LR', LogisticRegression(solver='libinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma=auto)))