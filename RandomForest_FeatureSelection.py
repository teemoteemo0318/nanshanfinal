import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
np.random.seed(1)

#Function to evaluate model performance

def getAccuracy(pre,ytest): 
    count = 0
    for i in range(len(ytest)):
        if ytest[i]==pre[i]: 
            count+=1
            acc = float(count)/len(ytest)

    return acc



#Load dataset as pandas data frame
df1 = pd.read_excel('training_data.xlsx',index_col=0)
df2 = pd.read_excel('testing_data.xlsx',index_col=0)
X_train = df1.iloc[:,:-1]
y_train = df1.iloc[:,-1]
X_test = df2.iloc[:,:-1]
y_test = df2.iloc[:,-1]

Xtraining = np.array(X_train)
ytraining = np.array(y_train)
Xtest = np.array(X_test)
ytest = np.array(y_test)

#Extract attribute names from the data frame
feat = df1.keys()
feat_labels = feat.get_values()

#Print the size of Data in MBs
#print("Size of Data set before feature selection: %.2f MB"%(Xtrain.nbytes/1e6))


#Create a random forest classifier with the following Parameters
trees = 250
max_feat = 17
max_depth = 5
min_sample = 2

clf = RandomForestClassifier(n_estimators=trees, max_features=max_feat, max_depth=max_depth, min_samples_split= min_sample, random_state=0, n_jobs=-1)

#Train the classifier and calculate the training time
import time
start = time.time()
clf.fit(Xtrain, ytrain)
end = time.time()
print("Execution time for building the Tree is: %f"%(float(end)- float(start)))
pre = clf.predict(Xtest)


#Evaluate the model performance for the test data
acc = getAccuracy(pre, ytest)

print("Accuracy of model before feature selection is %.2f"%(100*acc))
print(feature)

max_feat = 5
sfm = SelectFromModel(clf, threshold=0.01)
sfm.fit(Xtrain,ytrain)

#Transform input dataset
Xtrain_2 = sfm.transform(Xtrain)
Xtest_2 = sfm.transform(Xtest)

#see the size and shape of new dataset print("Size of Data set before feature selection: %.2f MB"%(Xtrain_1.nbytes/1e6))
shape = np.shape(Xtrain_2)
print("Shape of the dataset ",shape)

#Model training time
start = time.time()
clf.fit(Xtrain_2, ytrain)
end = time.time()
print("Execution time for building the Tree is: %f"%(float(end)- float(start)))

#evaluate the model on test data
pre = clf.predict(Xtest_2)
count = 0
acc2 = getAccuracy(pre, ytest)
print("Accuracy after feature selection %.2f"%(100*acc2))
