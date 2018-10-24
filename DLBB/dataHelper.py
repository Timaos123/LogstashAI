#coding:utf8
'''
Created on 2018年5月26日

@author: Administrator
'''
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from keras.preprocessing import text
import logging

def oneHotB(item):
    if item==0:
        return [1,0]
    else:
        return [0,1]
    
def loadLogData(fileName="logstash.csv"):
    '''load logstash.csv'''
    lsDf=pd.read_csv(fileName)
    try:
        lsDf.loc[lsDf.severity=="crit","severity"]=1
        lsDf.loc[lsDf.severity=="err","severity"]=1
        lsDf.loc[lsDf.severity!=1,"severity"]=0
    except:
        logging.info("should have been preprocessed ...")
        
    xyDf=lsDf.loc[:,["message","severity"]]
    
    xyDf=pd.DataFrame(np.array(xyDf),columns=["message","type"])
    
    return xyDf

def splitTrainTest(dataDf,testSize=0.4,mode="balanced"):
    '''
    dataDf:["message","type"]
    mode:balanced,random
    '''
    print("mode:",mode)
    if mode=="balanced":
        print("spliting 1 ...")
        oneDataArr=np.array(dataDf.loc[dataDf.type==1,:])
        np.random.shuffle(oneDataArr)
        oneTrainDataArr=oneDataArr[:int((1-testSize)*len(oneDataArr))]
        oneTestDataArr=oneDataArr[int((1-testSize)*len(oneDataArr)):-1]
        
        print("splitting 0 ...")
        zeroDataArr=np.array(dataDf.loc[dataDf.type==0,:])
        np.random.shuffle(zeroDataArr)
        zeroTrainDataArr=zeroDataArr[:int((1-testSize)*len(zeroDataArr))]
        zeroTestDataArr=zeroDataArr[int((1-testSize)*len(zeroDataArr)):-1]
        
        print("combining 0 and 1 ...")
        xTrain=np.array(oneTrainDataArr.tolist()+zeroTrainDataArr.tolist())[:,0]
        yTrain=np.array(oneTrainDataArr.tolist()+zeroTrainDataArr.tolist())[:,1]
        xTest=np.array(oneTestDataArr.tolist()+zeroTestDataArr.tolist())[:,0]
        yTest=np.array(oneTestDataArr.tolist()+zeroTestDataArr.tolist())[:,1]
        
        return xTrain,xTest,yTrain,yTest
    elif mode=="random":
        print("splitting ...")
        xTrain,xTest,yTrain,yTest=train_test_split(np.array(dataDf),test_size=testSize)
        return xTrain,xTest,yTrain,yTest
    else:
        raise "wrong mode..."

if __name__ == '__main__':
    pass