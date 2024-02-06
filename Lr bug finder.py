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
import random

def out(data):
    data = pd.read_csv(data)
    data.drop(["name","version","name.1"],axis=1,inplace=True)
    
    dt=data.to_numpy()
    bugs=np.array([])
    counter=0
    for i in range(len(dt)):
        if dt[i,20]>0:
            counter=counter+1
            bugs = np.append(bugs,dt[i,:])
    bugs = bugs.reshape(counter, 21)        
    data = pd.DataFrame (bugs, columns = ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc","bug"])
    data.to_csv('bugs.csv')
def Lr(data):
    
    n=len(data.columns)-1
    y = data.iloc[:,n].values
    x_data = data.drop(data.iloc[:,n], axis=1)
    
    x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

    
    lr = LogisticRegression()
    #board = np.array([])

    #for i in range(1000):
    #    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=i)
        
    #    lr = LogisticRegression()
    #    lr.fit(x_train,y_train)
        
    #    c=lr.score(x_test,y_test)
    #    board = np.append(board, c)
    #x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=np.argmax(board))
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=40)
    
    
    
    lr.fit(x_train,y_train)
    
    #print("logistic regression algo result {}".format(lr.score(x_test.T,y_test.T)))
    
    y_pred = lr.predict(x)
    y_true = y
    cm = confusion_matrix(y_true,y_pred)

    #return y_pred,np.argmax(board),cm
    return y_pred,cm

def onesvszeros(data):
    out(data)
    data = pd.read_csv(data)
    data.drop(["name","version","name.1"],axis=1,inplace=True)
    
    dt=data.to_numpy()
    
    
    for i in range(len(dt)):
        if dt[i,20]>1:
            dt[i,20]=1
    data = pd.DataFrame (dt, columns = ["wmc","dit","noc","cbo","rfc","lcom","ca","ce","npm","lcom3","loc","dam","moa","mfa","cam","ic","cbm","amc","max_cc","avg_cc","bug"])
    
    
    
    ones=np.array([])
    zeros=np.array([])
    counterones=0
    counterzeros=0
    for i in range(len(dt)):
        if dt[i,20]==1:
            ones = np.append(ones, dt[i,:])
            counterones=counterones+1
        elif dt[i,20]==0:
            zeros = np.append(zeros, dt[i,:])
            counterzeros=counterzeros+1
    
    ones = ones.reshape(counterones, 21)
    zeros = zeros.reshape(counterzeros, 21)
    
    onesmax = np.amax(ones,axis=0)
    onesmin = np.amin(ones,axis=0)
    
    zerosmax = np.amax(zeros,axis=0)
    zerosmin = np.amin(zeros,axis=0)
    
    farkmax=onesmax-zerosmax
    farkmin=onesmin-zerosmin
    return farkmax,farkmin,ones,zeros

def gaper(data):
    farkmax,farkmin,ones,zeros=onesvszeros(data)
    quaranteed=np.zeros(20)
    for i in range(len(quaranteed)):
        if farkmax[i]>0:
            quaranteed[i]=1
        else:
            if farkmin[i]<0:
                quaranteed[i]=0
    gap=np.zeros((len(quaranteed),2))
    ones=np.delete(ones,(20),axis=1)
    zeros=np.delete(zeros,(20),axis=1)
    for i in range(19):
        if quaranteed[i]==1:
            
            gap[i,0]=np.max(zeros[i],axis=0)
            gap[i,1]=farkmax[i]+np.max(zeros[i],axis=0)
        else:
            if quaranteed[i]==2:
            
                gap[i,0]=np.min(zeros[i],axis=0)
                gap[i,1]=np.min(zeros[i],axis=0)-farkmax[i]
    
    return gap,ones,zeros

def newdataoptimazer(data):
    gap,ones,zeros=gaper(data)
    
    newones=np.array([])
    newzeros=np.array([])
    counter=0
    for i in range(len(gap)):
        if gap[i,0]!=0:
            counter=counter+1
            newones= np.append(newones, ones[:,i])
            newzeros=np.append(newzeros, zeros[:,i])
            
    newones = newones.reshape(len(ones), counter)
    newzeros = newzeros.reshape(len(zeros), counter)
    
    needed=len(newzeros)-len(newones)
    
    addones=np.zeros((needed,counter))
    delete=np.array([])
    for i in range(len(gap)):
        if gap[i,0]==0:
            delete=np.append(delete, i)
    counterr=0
    for i in range((len(gap)-counter)):
        gap=np.delete(gap, (int(delete[i])+counterr), axis=0)
        counterr=counterr-1
    
    for i in range(len(addones)):
        for f in range(counter):
            addones[i,f]=random.uniform(gap[f,0],gap[f,1])
            
    #newones=addones+newones
    #newones=np.concatenate(addones,newones,axis=None)
    
    newones = np.vstack((newones, addones))
    
    
    lastones=np.ones((len(newones),counter+1))
    lastzeros=np.zeros((len(newzeros),counter+1))
    
    
    for i in range(len(lastones)):
        for f in range(counter):
                lastones[i,f]=newones[i,f]
                lastzeros[i,f]=newzeros[i,f]
    
    
    df1 = pd.DataFrame (lastones) 
    df2 = pd.DataFrame (lastzeros)
    df3 = pd.concat([df1, df2])           
    
    return df3


#y_pred,value,cm=Lr(newdataoptimazer("4.csv"))
y_pred,cm=Lr(newdataoptimazer("4.csv"))


       
   