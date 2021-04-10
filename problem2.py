# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 06:34:32 2020

@author: Vipul
"""

import sys
"""import pandas as pd

inp = pd.read_csv(sys.argv[1],names=["feature_1","feature_2","label"])

op=inp.tail()

op.to_csv(sys.argv[2],index=False)"""

import pandas as pd
import csv
import numpy as np

import os
if os.path.exists(sys.argv[2]):
  os.remove(sys.argv[2])

#inp = pd.read_csv("input2.csv",names=["age","weight","height"])
inp = pd.read_csv(sys.argv[1],names=["age","weight","height"])


inp["age_scaled"]=(inp.age-np.mean(inp.age))/np.std(inp.age)
inp["weight_scaled"]=(inp.weight-np.mean(inp.weight))/np.std(inp.weight)
#inp["height_scaled"]=(inp.height-np.mean(inp.height))/np.std(inp.height)

#round(np.mean(inp.age_scaled))
#round(np.mean(inp.weight_scaled))

#inp2=inp.loc[:,["age_scaled","weight_scaled","height"]]
inp2=inp[["age_scaled","weight_scaled","height"]]


weights=[0,0,0]
inp_lst=inp2.values.tolist()

def predict(wt,row):
    return wt[0]+wt[1]*row[0]+wt[2]*row[1]

weight_order=[]
def pla(lr,iterr):
    itr=0
    weights=[0,0,0]
    
    for i in range(0,iterr):
        itr+=1
        err_f0=0
        err_f1=0
        err_f2=0
        labels=[]
        accuracy=0
        weights_bkp=list(weights)
        
        for rw in inp_lst:
            pred=predict(weights,rw)
            labels.append(pred)
            if rw[-1]==pred:
                accuracy+=1
            error= pred-rw[-1]
            #error= pred-rw[-1]
            #sq_err+=error**2
            err_f0+=error
            err_f1=err_f1+(error*rw[0])
            err_f2=err_f2+(error*rw[1])
           
        if err_f0 !=0:
            weights[0]=weights[0]- lr*(err_f0/len(inp_lst))
            weights[1]=weights[1]- lr*(err_f1/len(inp_lst))
            weights[2]=weights[2]- lr*(err_f2/len(inp_lst))
                
        #print("Weight: ")
        """print(itr)
        print(accuracy)
        print(weights)
        print(labels)"""
        weight_order.append([weights[0],weights[1],weights[2]])
        
        
        if err_f0==0:
            
            break
            #pass
            
    with open(sys.argv[2], 'a', newline='') as csvfile:
        fieldnames = ['lr','iterr','w0','w1', 'w2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'lr':lr,'iterr':iterr,'w0': weights[0],'w1': weights[1], 'w2': weights[2] })
        
 
alpha=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10 , .3]
it=[100,100,100,100,100,100,100,100,100,40]

for i in range(len(alpha)):
    pla(alpha[i],it[i])
#pla(.2,120)    

#df_out=pd.DataFrame(weight_order) 
#csvfile.close()