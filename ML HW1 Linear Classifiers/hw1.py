# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 13:52:58 2022

@author: darkb
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

dataset=pd.read_csv('bank-additional-full.csv',delimiter=';')

print(dataset.info())

print(dataset.describe(include='all').T)

job_quantity=dataset.job.value_counts()

plt.figure(figsize=(15,10))
plt.pie(x=job_quantity, labels=job_quantity.index, autopct='%1.1f%%')
plt.title('Ratio of the Jobs',color = 'red',fontsize = 35)
plt.show()

plt.figure(figsize = (15,15))
sns.displot(dataset["age"])
plt.show()

plt.figure(figsize = (15,15))
sns.displot(dataset["marital"])
plt.show()

education_quantity=dataset.education.value_counts()

plt.figure(figsize=(15,10))
plt.pie(x=education_quantity, labels=education_quantity.index, autopct='%1.1f%%')
plt.title('Ratio of the Customer Education Levels',color = 'red',fontsize = 35)
plt.show()

plt.figure(figsize=(15,10))
sns.countplot(dataset['housing'])
plt.show()

plt.figure(figsize=(15,10))
sns.countplot(dataset['loan'])
plt.show()

plt.figure(figsize=(15,10))
sns.countplot(dataset['y'])
plt.show()



x_numeric_data=dataset.select_dtypes(include=['int64']).copy()
plt.figure(figsize=(15,10))
sns.pairplot(x_numeric_data)
plt.show()

plt.figure(figsize=(15,10))
sns.boxplot(x='job', y='duration',hue='y',data=dataset)
plt.show()





X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1].values

x_numeric_data=X.select_dtypes(include=['int64']).copy()
x_categorical_data=X.select_dtypes(include=['object']).copy()
x_float_data=X.select_dtypes(include=['float64']).copy()

column_names=list()
for names in x_categorical_data.columns:
    column_names.append(names)
    
print(column_names)

x_categorical_data=pd.get_dummies(data=x_categorical_data, columns=column_names,drop_first=True)
print(x_categorical_data)

x_categorical_data.info()

print(x_categorical_data.info())

X=pd.concat([x_categorical_data,x_numeric_data,x_float_data],axis=1)
print(X)

'''Simdi yes no yani output kismini sayi haline getirelim'''
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(y)


X=X.values #Turn X into Numpy array






# '''Simdi Training ve Test kumelerini olusturmamiz lazim'''
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=1,shuffle=True)



# '''Feature Scaling'''
# from sklearn.preprocessing import StandardScaler
# sc=StandardScaler()
# X_train[:,43:] = sc.fit_transform(X_train[:,43:])

# X_test[:,43:]=sc.transform(X_test[:,43:])



class myKNN:
    __distance=None
    __neighbors=None
    __givenX=None
    __givenY=None
    def __init__(self,distance,neighbors):
        self.__distance=distance
        self.__neighbors=neighbors
    def fit(self,givenX,givenY):
        self.__givenX=givenX
        self.__givenY=givenY
    def returnDistanceType(self):
        return self.__distance
    def returnNeighborNumber(self):
        return self.__neighbors
        
    def __calculateDistance(self,givenInput):
        if self.__distance=="euclidean":
            distanceMatrix=np.square(np.subtract(self.__givenX,givenInput))
            distanceMatrix=distanceMatrix.sum(axis=1)
            distanceMatrix=np.sqrt(distanceMatrix)
    
            return distanceMatrix
        elif self.__distance=="manhattan":
            distanceMatrix=np.absolute(np.subtract(self.__givenX,givenInput))
            distanceMatrix=distanceMatrix.sum(axis=1)
            return distanceMatrix
        else:
            print(self.__distance," is not a proper measure parameter (should be manhattan or euclidean)")
    def __singlePredict(self,singleTestData):
        distanceMatrix=self.__calculateDistance(singleTestData) #Distance Matrix
        closestNeighbors=np.argpartition(distanceMatrix, self.__neighbors) #rows of k closest neighbors
        closestNeighbors=distanceMatrix[closestNeighbors[:self.__neighbors]] #Indices of k closest neighbors
        print(closestNeighbors,"*********",len(closestNeighbors))
        neighborLabels=np.zeros(len(closestNeighbors)) #y values of closest neighbors
        for i in range(len(closestNeighbors)):
            realIndex=np.where(distanceMatrix==closestNeighbors[i])
            print("hello ***",realIndex[0])
            neighborLabels[i]=self.__givenY[np.amin(realIndex).astype(int)]
            
        for finalNeighbors in neighborLabels:
            if len(neighborLabels)%2==1: #odd number of neighbors case
                if np.count_nonzero(neighborLabels==finalNeighbors)>(len(neighborLabels)/2):
                    print(finalNeighbors)
                    return finalNeighbors
            else: #even number of neighbors case
                if np.count_nonzero(neighborLabels==finalNeighbors)>=(len(neighborLabels)/2):
                    return finalNeighbors
    def predict(self,X_test):
        y_pred=np.zeros(len(X_test))
        for i in range(len(X_test)):
            singleResult=self.__singlePredict(X_test[i])
            #if singleResult==1:
            y_pred[i]=singleResult
        return y_pred
    



from datetime import datetime



from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report

kf = KFold(n_splits=5,shuffle=True)

kf.get_n_splits(X)

sc=StandardScaler()

myKnnClassifier=myKNN("euclidean",4) ### Euclidean Classifier

knnReport=[]

from sklearn.metrics import confusion_matrix

cf_list=[]

steps=0

for train_index, test_index in kf.split(X):
    start = datetime.now()
 
    X_train, X_test = X[train_index], X[test_index]
 
    y_train, y_test = y[train_index], y[test_index]
    
    
    X_train[:,43:] = sc.fit_transform(X_train[:,43:])

    X_test[:,43:]=sc.transform(X_test[:,43:])
 
    myKnnClassifier.fit(X_train,y_train)
 
    y_pred=myKnnClassifier.predict(X_test)
    report=classification_report(y_test, y_pred)
    
    cf_list.append(confusion_matrix(y_test, y_pred))
    
    
    knnReport.append(('****************************************************','KNN Distance:',myKnnClassifier.returnDistanceType(),' Neighbors:',myKnnClassifier.returnNeighborNumber(), 'Accuracy:% ',(metrics.accuracy_score(y_test, y_pred)*100),str(report)))
    
   
    end = datetime.now()
    print("Step ",steps,"KNN(Euclidean) elapsed time is: ",end-start)
    steps=steps+1



for a,b,c,d,e,f,g,h in knnReport:
    print(a)
    print(b,c,d,e,f,g,h)
    

knnReport.clear()

plt.figure(figsize=(15,15))

ax = sns.heatmap(cf_list[0], annot=True, cmap='jet',fmt='10.0f')

ax.set_title('Confusion Matrix with labels (KNN Euclidean) \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show() 

plt.figure(figsize=(15,15))

ax = sns.heatmap(cf_list[1], annot=True, cmap='jet',fmt='10.0f')

ax.set_title('Confusion Matrix with labels (KNN Euclidean) \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()   

plt.figure(figsize=(15,15))

ax = sns.heatmap(cf_list[2], annot=True, cmap='jet',fmt='10.0f')

ax.set_title('Confusion Matrix with labels (KNN Euclidean) \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()   

plt.figure(figsize=(15,15))

ax = sns.heatmap(cf_list[3], annot=True, cmap='jet',fmt='10.0f')

ax.set_title('Confusion Matrix with labels (KNN Euclidean) \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()   

plt.figure(figsize=(15,15))

ax = sns.heatmap(cf_list[4], annot=True, cmap='jet',fmt='10.0f')

ax.set_title('Confusion Matrix with labels (KNN Euclidean) \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()   


cf_list.clear() #Clearing the list




myKnnClassifier=myKNN("manhattan",4) ### Euclidean Classifier



cf_list=[]

steps=0

for train_index, test_index in kf.split(X):
    start = datetime.now()
 
    X_train, X_test = X[train_index], X[test_index]
 
    y_train, y_test = y[train_index], y[test_index]
    
    
    X_train[:,43:] = sc.fit_transform(X_train[:,43:])

    X_test[:,43:]=sc.transform(X_test[:,43:])
 
    myKnnClassifier.fit(X_train,y_train)
 
    y_pred=myKnnClassifier.predict(X_test)
    report=classification_report(y_test, y_pred)
    
    cf_list.append(confusion_matrix(y_test, y_pred))
    
    
    knnReport.append(('****************************************************','KNN Distance:',myKnnClassifier.returnDistanceType(),' Neighbors:',myKnnClassifier.returnNeighborNumber(), 'Accuracy:% ',(metrics.accuracy_score(y_test, y_pred)*100),str(report)))
    
   
    end = datetime.now()
    print("Step ",steps,"KNN(Manhattan) elapsed time is: ",end-start)
    steps=steps+1


for a,b,c,d,e,f,g,h in knnReport:
    print(a)
    print(b,c,d,e,f,g,h)
    

knnReport.clear()

plt.figure(figsize=(15,15))

ax = sns.heatmap(cf_list[0], annot=True, cmap='jet',fmt='10.0f')

ax.set_title('Confusion Matrix with labels (KNN Manhattan) \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show() 

plt.figure(figsize=(15,15))

ax = sns.heatmap(cf_list[1], annot=True, cmap='jet',fmt='10.0f')

ax.set_title('Confusion Matrix with labels (KNN Manhattan) \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()   

plt.figure(figsize=(15,15))

ax = sns.heatmap(cf_list[2], annot=True, cmap='jet',fmt='10.0f')

ax.set_title('Confusion Matrix with labels (KNN Manhattan) \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()   

plt.figure(figsize=(15,15))

ax = sns.heatmap(cf_list[3], annot=True, cmap='jet',fmt='10.0f')

ax.set_title('Confusion Matrix with labels (KNN Manhattan) \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()   

plt.figure(figsize=(15,15))

ax = sns.heatmap(cf_list[4], annot=True, cmap='jet',fmt='10.0f')

ax.set_title('Confusion Matrix with labels (KNN Manhattan) \n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])

## Display the visualization of the Confusion Matrix.
plt.show()   


cf_list.clear() #Clearing the list



                
    
# classifier=myKNN("euclidean",4)
# classifier.fit(X_train,y_train)
# y_pred=classifier.predict(X_test)
                

# print("KNN --> Accuracy: ",(metrics.accuracy_score(y_test, y_pred)*100),"%")
# report = classification_report(y_test, y_pred)
# print(report)



'''Part 3 Linear SVM'''
from sklearn.svm import SVC
linear_classifier = SVC(kernel = 'linear', random_state = 42,probability=True)


#Roc curve
from sklearn.metrics import roc_curve,auc


def assess_SVM_Model(model,kernel,threshold):
    cf_list=[]
    fpr_tpr_list=[]
    auc_list=[]
    report_list=[]
    steps=0

    for train_index, test_index in kf.split(X):
        start = datetime.now()
     
        X_train, X_test = X[train_index], X[test_index]
     
        y_train, y_test = y[train_index], y[test_index]
        
        
        X_train[:,43:] = sc.fit_transform(X_train[:,43:])

        X_test[:,43:]=sc.transform(X_test[:,43:])
     
        model.fit(X_train,y_train)
     
        y_pred = (model.predict_proba(X_test)[:,1] >= threshold).astype(bool) # set threshold
        report=classification_report(y_test, y_pred)
        cf_list.append(confusion_matrix(y_test, y_pred))
        
        fpr,tpr,ThreshHold=roc_curve(y_test,y_pred)
        aucValue=auc(fpr,tpr)
        fpr_tpr_list.append((fpr,tpr))
        auc_list.append(aucValue)
        
        
        report_list.append(('****************************************************\n'+kernel+' SVM'+'Accuracy:% '+str((metrics.accuracy_score(y_test, y_pred)*100))+'\n'+str(report)))
        
       
        end = datetime.now()
        print("Step ",steps," ",kernel,"SVM elapsed time is: ",end-start)
        
        steps=steps+1


    for report in knnReport:
        print(report)
    for i in range(len(auc_list)):
        print("Step ",i," Auc: ",auc_list[i])
    print("Mean AUC Value for ",kernel," SVM with ",threshold," is: ",sum(auc_list)/len(auc_list))
    
    #Roc curve

    
    plt.figure(figsize=(15,10))
    plt.plot(fpr_tpr_list[0][0],fpr_tpr_list[0][1],label='AUC %0.2f'% auc_list[0])
    plt.plot(fpr_tpr_list[1][0],fpr_tpr_list[1][1],label='AUC %0.2f'% auc_list[1])
    plt.plot(fpr_tpr_list[2][0],fpr_tpr_list[2][1],label='AUC %0.2f'% auc_list[2])
    plt.plot(fpr_tpr_list[3][0],fpr_tpr_list[3][1],label='AUC %0.2f'% auc_list[3])
    plt.plot(fpr_tpr_list[4][0],fpr_tpr_list[4][1],label='AUC %0.2f'% auc_list[4])
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves of Different KFold Steps')
    plt.legend(loc='best')
    plt.show()


assess_SVM_Model(linear_classifier, 'linear', 0.5)
            
            
        
            
            
            
        
        
                
            
            
            
            









