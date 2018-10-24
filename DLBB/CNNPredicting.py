#coding:utf8
'''
Created on 2018年7月2日

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


if __name__ == '__main__':
    with CustomObjectScope({"getRecall":getRecall,"getPrecision":getPrecision}):
        newCNNModel=keras.models.load_model("CNNmodel/myCNNModel")