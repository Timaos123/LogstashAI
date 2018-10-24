#coding:utf8
'''
Created on 2018年9月17日

@author: Administrator
'''
import os
import numpy as np
import tqdm
import A0_preprocessing as pc
import CNNTrainLoading as CNNTL
import CNNDLBBTrainDeveloping as CNNTD
from keras.callbacks import TensorBoard,EarlyStopping
from sklearn.metrics import recall_score,precision_score,accuracy_score

if __name__ == '__main__':
    
    dataSize=600000
    
    print("just computing vocabSize, maxLenOfSent, xLanTest ...")
    xTrain,xTest,yTrain,yTest,vocabSize,maxLenOfSent,xLanTest=pc.preprocessing(dataSize=dataSize)
    
    print("loading and splitting data ...")
    splittedData=pc.preprocessingCV(foldK=4)
    
    print("x:",splittedData[0][0][0:5])
    print("y:",splittedData[0][1][0:5])
    
    recallList=[]
    precisionList=[]
    accList=[]
    for splittedDataItem in tqdm.tqdm(splittedData):
        print("building CNN model ...")
        myCNN=CNNTD.CNNModel(splittedDataItem[0],splittedDataItem[1],vocabSize,maxLenOfSent)
        
        print("training ...")
        if os.path.exists("./log")==False:
            os.mkdir("./log")
        myCNN.model.fit(xTrain,yTrain,epochs=10,callbacks=[TensorBoard(log_dir="./log"),\
                                     EarlyStopping(monitor="val_loss",\
                                                   patience=0,\
                                                   mode="auto")],\
                                     batch_size=64\
                                                           )
        print("predicting ...")
        preY=np.argmax(myCNN.model.predict(xTest),axis=1)
        testY=np.argmax(yTest,axis=1)
        recallList.append(recall_score(preY,testY,labels=1))
        precisionList.append(precision_score(preY,testY,labels=1))
        accList.append(accuracy_score(preY,testY))
    
    print("acc:",np.mean(accList))
    print("recall:",np.mean(recallList))
    print("precision:",np.mean(precisionList))
    