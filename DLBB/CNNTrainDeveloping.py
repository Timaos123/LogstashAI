#coding:utf8
'''
Created on 2018年5月26日

@author: Administrator
'''
import pandas as pd
import numpy as np
import dataHelper as dh
import os
import A0_preprocessing as pc
from sklearn.metrics import recall_score,precision_score,accuracy_score

import keras
from keras.models import Model
from keras.preprocessing import text,sequence
from keras.layers import Input,Embedding,Conv2D,MaxPooling2D,Dense
from keras.layers.core import Reshape,Activation,Lambda,Flatten
import keras.backend as K
from keras.callbacks import TensorBoard,EarlyStopping
from keras.utils.generic_utils import CustomObjectScope

def getRecall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def getPrecision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

class CNNModel:
    def __init__(self,\
                 xTrain,\
                 yTrain,\
                 vocabSize,\
                 maxLenOfSent,\
                 embeddingSize=100,\
                 filterNum=1,\
                 filterWide=2,\
                 poolSize=4):
        self.xTrain=xTrain
        self.yTrain=yTrain
        self.vocabSize=vocabSize
        self.maxLenOfSent=maxLenOfSent
        self.embeddingSize=embeddingSize
        self.filterNum=filterNum
        self.filterWide=filterWide
        self.poolSize=poolSize
        self.model=self.buildModel()
        
    def buildModel(self):
        
        inputLayer=Input(shape=(self.maxLenOfSent,))
        embeddingLayer=Embedding(input_dim=self.vocabSize+1,\
                                 output_dim=self.embeddingSize\
                                 )(inputLayer)
        reshapeLayer=Reshape((self.filterNum,\
                              self.maxLenOfSent,\
                              self.embeddingSize,\
                              ))(embeddingLayer)
        convLayer=Conv2D(filters=self.filterNum,\
                         kernel_size=(
                             self.filterWide,\
                             self.embeddingSize),\
                         data_format="channels_first"\
                         )(reshapeLayer)
#         reshapeLayer=Reshape((self.filterNum,\
#                               convLayer.shape[3],\
#                               convLayer.shape[2]))(convLayer)
        poolingLayer=MaxPooling2D(pool_size=(self.poolSize,\
                                             1),
                                  data_format="channels_first"\
                                  )(convLayer)
        flattenLayer=Flatten()(poolingLayer)
        denseLayer=Dense(units=2,activation="softmax")(flattenLayer)
        
        myModel=Model(inputs=inputLayer,outputs=denseLayer)
        myModel.compile(optimizer="sgd",\
                           loss="categorical_crossentropy",\
                           metrics=["acc",getRecall,getPrecision])
        return myModel

if __name__ == '__main__':
    dataSize=600000
    epochs=50
    poolSize=4
    fileName="wcData85_1.csv"
    
    print("loading data and preprocessing ...")
    xTrain,xTest,yTrain,yTest,vocabSize,maxLenOfSent,xLanTest=pc.preprocessing(dataSize=dataSize,fileName=fileName)
    
    print("building CNN model ...")
    myCNN=CNNModel(xTrain,yTrain,vocabSize,maxLenOfSent)
    
    print("training ...")
    if os.path.exists("./log")==False:
        os.mkdir("./log")
    myCNN.model.fit(xTrain,yTrain,epochs=epochs,callbacks=[TensorBoard(log_dir="./log"),\
                                 EarlyStopping(monitor="val_loss",\
                                               patience=0,\
                                               mode="auto")],\
                                 batch_size=64\
                                                       )
    
    print("saving model ...")
    if os.path.exists("CNNmodel")==False:
        os.mkdir("CNNmodel")
    myCNN.model.save("CNNmodel/myCNNModel")
    
    print("loading model ...")
    with CustomObjectScope({"getRecall":getRecall,"getPrecision":getPrecision}):
        newCNNModel=keras.models.load_model("CNNmodel/myCNNModel")
    
    print("loading convelutional model ...")
    convModel = Model(inputs=newCNNModel.input,  
                                     outputs=newCNNModel.get_layer('conv2d_1').output)
    
    print("predicting ...")
    preY=np.argmax(newCNNModel.predict(xTest),axis=1)
    testY=np.argmax(yTest,axis=1)
    print("recall:",recall_score(preY,testY,labels=1))
    print("precision:",precision_score(preY,testY,labels=1))
    print("acc:",accuracy_score(preY,testY))
    preConv=convModel.predict(xTest)
    
    print("extracting features ...")
    indexArr=np.array(list(range(preConv.shape[0])))
    np.random.shuffle(indexArr)
    i=0
    for indexItem in range(5):
        i+=1
        print("source",i,":",xLanTest[indexArr[indexItem]].split(" "))
        print("conv",i,":",preConv[indexArr[indexItem]][0].T[0][0:len(xLanTest[indexArr[indexItem]].split(" "))])