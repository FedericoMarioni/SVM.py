import pandas as pd  # pandas is used to load and manipulate data
import numpy as np  # data manipulation
import matplotlib.pyplot as plt  # drawing graphs
import matplotlib.colors as colors
import sklearn.metrics
from sklearn.utils import resample  # down sample the dataset
from sklearn.model_selection import train_test_split  # split data into training and testing sets
from sklearn.preprocessing import scale  # scale and center data
from sklearn.svm import SVC  # this is the algorithm for classification
from sklearn.model_selection import GridSearchCV  # this will do cross validation
from sklearn.metrics import confusion_matrix  # this will give us a confusion matrix for evaluation
from sklearn.metrics import plot_confusion_matrix  # draws a confusion matrix
from sklearn.decomposition import PCA  # to perform PCA for reducing the dimensions of the data. to plot the data

df = pd.read_excel('default of credit card clients.xls', header=1)
df.rename({'default payment next month': 'Default'}, axis='columns', inplace=True)
df.drop('ID', axis=1, inplace=True)

# Identifying missing data
# print(len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)]))

df_no_missing = df.loc[(df['EDUCATION'] != 0) & (df['MARRIAGE'] != 0)]

# print(len(df_no_missing.loc[(df['EDUCATION'] == 0) | (df_no_missing['MARRIAGE'] == 0)]))
# print(len(df_no_missing)) print(len(df))

# Down sampling the data set to 1000 Y = 1, 1000 Y = 0, N= 2000.

df_no_default = df_no_missing[df_no_missing['Default'] == 0]
df_default = df_no_missing[df_no_missing['Default'] == 1]

df_no_default_down_sampled = resample(df_no_default, replace=False, n_samples=1000, random_state=42)
df_default_down_sampled = resample(df_default, replace=False, n_samples=1000, random_state=42)

# print(len(df_default_down_sampled))
# print(len(df_no_default_down_sampled))

df_down_sampled = pd.concat([df_default_down_sampled, df_no_default_down_sampled])
# print(len(df_down_sampled))

# Splitting the data into two parts: Independent variables and dependent variable

X = df_down_sampled.drop('Default', axis=1).copy()
Y = df_down_sampled['Default'].copy()

# One-Hot encoding: We cant use categorical data, we need to transform the categorical data variables to binaries.

X_encoded = pd.get_dummies(X, columns=['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5',
                                       'PAY_6'])

# Centering and Scaling the data: Normalization of the data. SVM assumes that each column has a mean of 0
# and a standard deviation of 1. So we need to the this to both training and testing data sets.

# First, split the data into training and testing data set

X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, random_state=42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

# Build a preliminary SVM

clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scaled, Y_train)

# See how it performs on the testing data set and see a confusion matrix

plot_confusion_matrix(clf_svm, X_test_scaled, Y_test, values_format='d', display_labels=["Did not default",
                      "Defaulted"])

# plt.show() We have 79 % proportions of 0s correctly predicted and 61% proportion of 1s correctly predicted
# lets run a second model and evaluate and see if we can do better
# second model: optimize parameters with cross validation and GirdSearchCV()
# in SVM its all about finding the best value for gamma and potentially C [regularization parameter]
# so if we do that we see that the optimal values are C=100 and gamma=0.001

clf_svm_optimized = SVC(random_state=42, C=100, gamma=0.001)
clf_svm_optimized.fit(X_train_scaled, Y_train)
plot_confusion_matrix(clf_svm_optimized, X_test_scaled, Y_test, values_format='d', display_labels=
                      ["Did not default", "Defaulted"])
plt.show()
# proportion of 0s predicted as 0s: 76%, proportion of 1s predicted as 1s: 65%
# the thing with SVM is that they do well out of the box, you cant improve much their first accuracy

# Since we have 24 features it is impossible to graph the decision boundary, so we are going to use
# principal component analysis (PCA) to do dimensionality reduction and see if we can plot a 2 dimensional
# graph of the decision boundary

pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)

per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = [str(x) for x in range (1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height=per_var)
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False
)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal components')
plt.title('Scree plot')
plt.show()

# the scree plot tells us how good this approx. of the decision boundary truly is
# the first 2 X should explain much more variance than all of the other






















