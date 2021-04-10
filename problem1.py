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

#inp = pd.read_csv("input1.csv",names=["feature_1","feature_2","label"])
inp = pd.read_csv(sys.argv[1],names=["feature_1","feature_2","label"])
#weights=[1,0.1,0.1]

weights=[0,0,0]

#weights=[59,-6,-12]

inp_lst=inp.values.tolist()

#act0=[weights[0]+weights[1]*inp_lst[i][0]+weights[2]*inp_lst[i][1] for i in range(len(inp_lst))]
#activation=[1 if (weights[0]+weights[1]*inp_lst[i][0]+weights[2]*inp_lst[i][1]) >0  else -1 for i in range(len(inp_lst))]

def predict(wt,row):
    if (wt[0]+wt[1]*row[0]+wt[2]*row[1]) >0:
        return 1
    else:
        return -1
    
#pred0=[predict(weights,rw) for rw in inp_lst]
weight_order=[]
def pla(lr):
    itr=0
    
    while True:
        itr+=1
        sq_err=0
        labels=[]
        accuracy=0
        weights_bkp=list(weights)
        
        for rw in inp_lst:
            pred=predict(weights,rw)
            labels.append(pred)
            if rw[-1]==pred:
                accuracy+=1
            error= rw[-1]-pred
            #error= pred-rw[-1]
            #sq_err+=error**2
            sq_err+=error
            """weights[0]=weights[0]+sq_err*lr
            weights[1]=weights[1]+sq_err*lr*rw[0]
            weights[2]=weights[2]+sq_err*lr*rw[1]"""
            
            """weights[0]=weights[0]+error
            weights[1]=weights[1]+error*rw[0]
            weights[2]=weights[2]+error*rw[1]"""
            
            """weights[0]=weights[0]+error*lr
            weights[1]=weights[1]+error*lr*rw[0]
            weights[2]=weights[2]+error*lr*rw[1]"""
            if error !=0:
                weights[0]=weights[0]+rw[-1]
                weights[1]=weights[1]+rw[-1]*rw[0]
                weights[2]=weights[2]+rw[-1]*rw[1]
                
        #print("Weight: ")
        """print(itr)
        print(accuracy)
        print(weights)
        print(labels)"""
        weight_order.append([weights[1],weights[2],weights[0]])
        
        with open(sys.argv[2], 'a', newline='') as csvfile:
            fieldnames = ['w1', 'w2', 'w0']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'w1': weights[1], 'w2': weights[2], 'w0': weights[0]})
        if weights_bkp==weights:
            csvfile.close()
            break
            #pass
        
#pla(30,.02)
#pla(.01)  
pla(1)    
df_out=pd.DataFrame(weight_order)      
#df_out.to_csv("output1.csv",index=False, header=False)
#df_out.to_csv(sys.argv[2],index=False, header=False)




