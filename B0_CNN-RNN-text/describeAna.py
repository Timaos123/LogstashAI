#coding:utf8
'''
Created on 2018年3月23日

@author: 20143
'''
import data_helper
import pandas as pd
if __name__ == '__main__':
    input_file = "logstash.csv"
#     x_, y_, vocabulary, vocabulary_inv, df, labels = data_helper.load_data(input_file,200000)
    df = pd.read_csv(input_file)
    print(df.groupby("severity")["_id"].count())
    