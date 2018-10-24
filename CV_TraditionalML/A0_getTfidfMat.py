#coding:utf8
'''
Created on 2018年4月24日

@author: 20143
'''
import csv
import pandas as pd
import numpy as np
import scipy as sc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.externals import joblib

def getTfIdfMat(X):
    '''
    X=np.array([[x1,y1],[x2,y2],...])
    '''
    X=[messageItem.tolist()[0] for messageItem in X]
    countModel= CountVectorizer(stop_words = 'english', decode_error="replace",max_df = 0.5)
    xCount = countModel.fit_transform(X)
    joblib.dump(countModel,"countMat/TMCountMat.model")
    tfidfModel = TfidfTransformer().fit(xCount)
    joblib.dump(tfidfModel,"countMat/TMTfidfMat.model")
    xTfidf=tfidfModel.transform(xCount)
    return xTfidf

def deleteWordNotInTrain(sentence,vocabList):
    sentenceList=sentence.split(" ")
    sentence=" ".join([word for word in sentenceList if word in vocabList])
    return sentence
    
if __name__ == '__main__':
    dataDf=pd.read_csv("wcData85_1.csv").sample(600000,replace=True)
    try:
        dataDf.loc[dataDf.severity=="err","severity"]=1
        dataDf.loc[dataDf.severity=="crit","severity"]=1
        dataDf.loc[dataDf.severity!=1,"severity"]=0
    except:
        pass
    xData=np.array(dataDf.loc[:,["message"]])
    yData=np.array(dataDf.loc[:,["severity"]])
    
    xTfidf=getTfIdfMat(xData).toarray()
    print("deleting elements smaller than downFDiv...")
    downFDiv=1/4*(xTfidf.max()+xTfidf.min())
    print("downFDiv:",downFDiv)
    xTfidf[xTfidf<downFDiv]=0
    xTfidf=sc.sparse.csr_matrix(xTfidf)
    
    X_train,X_test, y_train, y_test =train_test_split(xTfidf,yData,test_size=0.4, random_state=0)
    featureLen=len([sumItem for sumItem in X_train.sum(0).tolist()[0] if sumItem>0])
    
    print("number of corpus in train data:",X_train.shape[0])
    print("number of words in train data:",featureLen)
    print("number of corpus in test data:",X_test.shape[0])
    print("number of words in test data:",featureLen)
    
    #save training data
    print("saving train data...")
    X_train=X_train.toarray().tolist()
    y_train=y_train.tolist()
    trainData=np.array([row[0]+row[1] for row in list(zip(X_train,y_train))])
    print("trainDataShape:",trainData.shape)
    
    print("building trainDataFile...")
    with open("trainDataTM.csv","w+",encoding="utf8") as trainDataFile:
        myWriter=csv.writer(trainDataFile)
        for i in range(len(trainData)):
            myWriter.writerow(trainData[i])
            
    #save test data
    print("saving test data...")
    X_test=X_test.toarray().tolist()
    y_test=y_test.tolist()
    testData=np.array([row[0]+row[1] for row in list(zip(X_test,y_test))])
    
    print("building testDataFile...")
    with open("testDataTM.csv","w+",encoding="utf8") as testDataFile:
        myWriter=csv.writer(testDataFile)
        for i in range(len(testData)):
            myWriter.writerow(testData[i])
    print("finished!")