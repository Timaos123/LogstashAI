#coding:utf8
'''
Created on 2018年9月20日

@author: Administrator
'''

import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import A0_getTfidfMat
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.externals import joblib
if __name__ == '__main__':
    
    print("loading data ...")
    with open("datas/CVData.pkl","rb") as dataPkl:
        myDataList=pkl.load(dataPkl)
    
    print("training every fold ...")
    
    recallList=[]
    precisionList=[]
    accList=[]
    
    for sampleItem in myDataList:
        
        print("getting Tf-Idf Matrix ...")
        trainSampleList=sampleItem[0]
        testSampleList=sampleItem[1]
        trainXList=[row for row in trainSampleList]
        trainTypeArr=np.array([row[1] for row in trainSampleList])
        testXList=[row for row in testSampleList]
        testTypeArr=np.array([row[1] for row in testSampleList])
        trainTfidfMat=A0_getTfidfMat.getTfIdfMat(np.array(trainXList))
        countMTModel=joblib.load("countMat/TMCountMat.model")
        testCountMat=countMTModel.transform(np.array(testXList)[:,0])
        tfidfMTModel=joblib.load("countMat/TMTfidfMat.model")
        testTfidfMat=tfidfMTModel.transform(testCountMat)
        
        print("developing the model ...")
        myModel=DecisionTreeClassifier()
        myModel.fit(trainTfidfMat,trainTypeArr)
         
        print("predicting ...")
        yPre=myModel.predict(testTfidfMat)
        
        print("evaluating ...")
        recallList.append(recall_score(yPre,testTypeArr,labels=1))
        precisionList.append(precision_score(yPre,testTypeArr,labels=1))
        accList.append(accuracy_score(yPre,testTypeArr))
        
    print("recall:",np.mean(recallList))
    print("precision:",np.mean(precisionList))
    print("acc:",np.mean(accList))