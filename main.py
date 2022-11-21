from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

# A. Load dataset
df = pd.read_csv("winequality.csv")


# C. Prepare dataset for training
# splitting data into 70% training and 30% test
#define X and y
X = df.iloc[:, :11].values
y = df.iloc[:, -1].values


#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, 
                                                    test_size = 0.3, random_state = 100)

# D. Build an SVM model
# create an instance of the SVM classifier with a Linear kernel
model = svm.SVC(kernel='linear')

# train the model using the training set
model.fit(X_train, y_train)

# predict the classes in the test set
y_pred = model.predict(X_test)


#Evaluate the model
# accuracy
print("Accuracy: %.3f%%" % (metrics.accuracy_score(y_test, y_pred)*100))

# precision
# print("Precision: %.3f " % metrics.precision_score(y_test, y_pred, average = 'weighted'))

# recall
print("Recall: %.3f" % metrics.recall_score(y_test, y_pred, average = 'weighted'))


# F1 (F-Measure)
print("F1: %.3f" % metrics.f1_score(y_test, y_pred, average = 'weighted'))


# get feature names
f_names = df.columns.tolist()


# get weight for each feature
f_weights = model.coef_[0].tolist()


# removing  minus from the weight
f_weights_abs = [abs(x) for x in f_weights]


# match the weights and the names of the feature and sort it in decending order
srt_list = list(zip(f_weights_abs, f_names))
srt_list.sort(reverse=True)

# print
for e in srt_list:
    print(e[1], ": ", e[0])

