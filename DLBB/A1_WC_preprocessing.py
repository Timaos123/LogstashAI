#coding:utf8
'''
Created on 2018年6月15日

@author: Administrator
'''
import numpy as np
import pandas as pd
import tqdm as tqdm

if __name__ == '__main__':
    dataList=[]
    with open("errRecorder.txt","w+",encoding="utf8") as errFile:
        with open("recreate85_1.out","r",encoding="utf8") as wcFile:
            i=1
            for row in tqdm.tqdm(wcFile):
                try:
                    row=np.array(row.strip().split(" "))
                    rowX=" ".join(row[5:8].tolist())
                    rowY=row[8]
                    dataList.append([rowX,rowY])
                except Exception as ex:
                    print("code err",i)
                i+=1
        dataList=np.array(dataList)
        dataDf=pd.DataFrame(dataList,columns=["message","severity"])
        
    print("saving data ...")
    dataDf.to_csv("wcData85_1.csv")
    
    print("loading data ...")
    wcDataDf=pd.read_csv("wcData85_1.csv")
#     print(wcDataDf.groupby("severity")["severity"].count())
    print("deleting wrong data ...")
    wcDataDf=wcDataDf.loc[wcDataDf["severity"]!="-",:]
    
    print("transforming data type ...")
    wcDataDf["severity"]=wcDataDf["severity"].astype("int")
    
    print("label ones ...")
    wcDataDf.loc[wcDataDf.severity<400,"severity"]=0
    
    print("label zeros ...")
    wcDataDf.loc[wcDataDf.severity>=400,"severity"]=1
    
    print("calculating ...")
    print(wcDataDf.groupby("severity")["severity"].count())
    
    print("saving file ...")
    wcDataDf.to_csv("wcData85_1.csv")
    
    print("finished!")