#coding:utf8
'''
Created on 2018年5月26日

@author: Administrator
'''

import dataHelper as dh
import numpy as np
import keras
from keras.preprocessing import text,sequence
from keras.layers import Input,Embedding
import os
import pickle as pkl
from sklearn.cross_validation import StratifiedKFold as skf

def preprocessing(testSize=0.4,trainTestSplitMode="balanced",dataSize=2000,fileName="logstash.csv"):
    
    print("loading data ...")
    dataDf=dh.loadLogData(fileName=fileName).loc[:dataSize]
    
    print("splitting data in train and test ...")
    xTrain,xTest,yTrain,yTest=dh.splitTrainTest(dataDf,testSize=testSize,mode=trainTestSplitMode)
    
    print("loading tokenizer ...")
    myTokenizer=text.Tokenizer()
    myTokenizer.fit_on_texts(xTrain)
    vocabSize=max(myTokenizer.word_index.values())
    print("vocab size:",vocabSize)
    
    maxLenOfSent=max([len(row) for row in xTrain])
    print("max len of sentences:",maxLenOfSent)
        
    print("saving tokenizer ...")
    if os.path.exists("tokenizers")==False:
        os.mkdir("tokenizers")
    with open("tokenizers/myTokenizer.model","wb+")as myTokenizerFile:
        pkl.dump(myTokenizer,myTokenizerFile)
        
    print("serializing(to seq) ...")
    xTrain=myTokenizer.texts_to_sequences(xTrain)
    xLanTest=xTest
    xTest=myTokenizer.texts_to_sequences(xTest)

        
    print("padding ...")
    xTrain=sequence.pad_sequences(xTrain,maxLenOfSent)
    xTest=sequence.pad_sequences(xTest,maxLenOfSent)
    
    print("one-hot embedding ...")
    yTrain=keras.utils.to_categorical(yTrain,2)
    yTest=keras.utils.to_categorical(yTest,2)
    
    return xTrain,xTest,yTrain,yTest,vocabSize,maxLenOfSent,xLanTest

def preprocessingCV(foldK=2,dataSize=2000,fileName="logstash.csv"):
        
    print("loading data ...")
    dataDf=dh.loadLogData(fileName=fileName).loc[:dataSize]
    
    print(foldK,"-fold cross validation ...")
    dataArr=np.array(dataDf)
    dataSKF=skf(dataArr[:,1],n_folds=foldK,shuffle=True)
    dataIndexList=list(dataSKF)
    
    print("transforming dataIndexList into dataList ...")
    dataList=[([dataArr[row0Item][0] for row0Item in row[0]],[dataArr[row0Item][1] for row0Item in row[1]]) for row in dataIndexList]
    
    return dataList
    

if __name__ == '__main__':
    processingCV()