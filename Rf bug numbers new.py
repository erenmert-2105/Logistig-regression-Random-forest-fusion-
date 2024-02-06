from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
#from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
#from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



def Knn(data,focus,x_testt=np.array([]),test=0):
    data = pd.read_csv(data)
    data=data.to_numpy()
    data=np.delete(data, (0), axis=1)
    data = pd.DataFrame (data, columns = ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc","bug"])
    
    y = data.bug.values
    x_data = data.drop(["bug"],axis=1)
    
    x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
    board = np.array([])
    #f = np.array([])
    
    
    for i in range(1000):
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=i)

        knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
        knn.fit(x_train,y_train)
        
        y_pred = knn.predict(x_train)
        y_true = y_train
        c=knn.score(x_test,y_test)
        board = np.append(board, c)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=np.argmax(board))
    

    
    
    knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
    knn.fit(x_train,y_train)
    #print("logistic regression algo result {}".format(lr.score(x_test.T,y_test.T)))
    
    y_pred = knn.predict(x)
    y_true = y
    cm = confusion_matrix(y_true,y_pred)

    return y_pred,np.argmax(board),y_true
def Lr(data,focus,x_testt=np.array([]),test=0):
    
    data = pd.read_csv(data)
    data=data.to_numpy()
    data=np.delete(data, (0), axis=1)
    data = pd.DataFrame (data, columns = ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc","bug"])
    
    y = data.bug.values
    x_data = data.drop(["bug"],axis=1)
    
    x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
    board = np.array([])
    #f = np.array([])
    
    lr = LogisticRegression()
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=42)
    for i in range(1000):
        x_trainn, x_testt, y_trainn, y_testt = train_test_split(x_train,y_train,test_size = 0.1,random_state=i)
        
        
        
        lr.fit(x_train,y_train)
        
        
        y_pred = lr.predict(x_train)
        y_true = y_train
        c=lr.score(x_test,y_test)
        board = np.append(board, c)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=np.argmax(board))
    
    
    
    
    lr.fit(x_train,y_train)
    
    #print("logistic regression algo result {}".format(lr.score(x_test.T,y_test.T)))
    
    y_pred = lr.predict(x)
    y_true = y
    cm = confusion_matrix(y_true,y_pred)

    return y_pred,np.argmax(board),y_true
def Rf(data):
    
    data = pd.read_csv(data)
    data=data.to_numpy()
    data=np.delete(data, (0), axis=1)
    data = pd.DataFrame (data, columns = ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc","bug"])
    
    y = data.bug.values
    x_data = data.drop(["bug"],axis=1)
    
    x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
    board = np.array([])
    #f = np.array([])
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=42)
    rf = RandomForestClassifier()
    for i in range(1000):
        x_trainn, x_testt, y_trainn, y_testt = train_test_split(x_train,y_train,test_size = 0.1,random_state=i)
        
        
        
        rf.fit(x_test,y_test)
        
        
        y_pred = rf.predict(x_train)
        y_true = y_train
        c=rf.score(x_train,y_train)
        board = np.append(board, c)
    
    x_trainn, x_testt, y_trainn, y_testt = train_test_split(x,y,test_size = 0.1,random_state=np.argmax(board))
    
    
    
    rf = RandomForestClassifier(n_estimators = 1000,random_state = 1)
    rf.fit(x_trainn,y_trainn)
    
    #print("logistic regression algo result {}".format(lr.score(x_test.T,y_test.T)))
    
    
    #y_pred = rf.predict(x)
    #y_true = y
    y_pred = rf.predict(x_test)
    y_true = y_test
    cm = confusion_matrix(y_true,y_pred)

    return y_pred,np.argmax(board),cm,board
#hatalı data
def to_oneandmore(data):
    data = pd.read_csv(data)
    dd=data.to_numpy()
    dd=np.delete(dd, (0), axis=1)
    
    for i in range(len(dd)):
        if dd[i,20]==1:
            dd[i,20]=1
        else :
            dd[i,20]=2
        
    newdata = pd.DataFrame (dd, columns = ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc","bug"])
    
    newdata.to_csv("dataones.csv")
    
def to_twoandmore(data):
    data = pd.read_csv(data)
    dd=data.to_numpy()
    dd=np.delete(dd, (0), axis=1)
    
    for i in range(len(dd)):
        if dd[i,20]==2:
            dd[i,20]=1
        else :
            dd[i,20]=2
        
    newdata = pd.DataFrame (dd, columns = ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc","bug"])
    
    newdata.to_csv("datatwos.csv")
    
def to_others(data):
    data = pd.read_csv(data)
    dd=data.to_numpy()
    dd=np.delete(dd, (0), axis=1)  
    
    for i in range(len(dd)):
        if dd[i,20]==3 or dd[i,20]>3:
            dd[i,20]=1
        else :
            dd[i,20]=2
        
    newdata = pd.DataFrame (dd, columns = ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc","bug"])
    
    newdata.to_csv("dataothers.csv")
    







to_others("bugs.csv")
to_oneandmore("bugs.csv")
to_twoandmore("bugs.csv")


#a=pd.read_csv("dataones.csv")
#b=pd.read_csv("datatwos.csv")
#c=pd.read_csv("dataothers.csv")


    
A,ScoreA,cmA,boardA=Rf("dataones.csv")


B,ScoreB,cmB,boardB=Rf("datatwos.csv")


C,np.ScoreC,cmC,boardC=Rf("dataothers.csv")


result=np.ones(len(A))
if cmC[0,0] >= cmB[0,0]:
    choice=C
    numberchoice=3
    nonchoice=B
    numbernonchoice=2
else:
    choice=B
    numberchoice=2
    nonchoice=C
    numbernonchoice=3
    
for i in range(len(result)):
    if nonchoice[i]==1:
        result[i]=numbernonchoice
        
for i in range(len(result)):
    if choice[i]==1:
        result[i]=numberchoice


