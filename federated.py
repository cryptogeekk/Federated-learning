#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 23:19:09 2020

@author: krishna
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from random import shuffle

header_names = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'success_pred']
column_names=np.array('header names')

def create_category(training_dataset):
    category_type=training_dataset['attack_type'].tolist()  #taking attack_type data from dataframe and converting it into list
    #category=['u2r','r2l','probe','dos','benign']
    
    benign=['normal']
    probe=['nmap', 'ipsweep', 'portsweep', 'satan','mscan', 'saint', 'worm']
    r2l=['ftp_write', 'guess_passwd', 'snmpguess','imap', 'spy', 'warezclient', 'warezmaster','multihop', 'phf', 'imap', 'named', 'sendmail','xlock', 'xsnoop', 'worm']
    u2r=['ps', 'buffer_overflow', 'perl', 'rootkit','loadmodule', 'xterm', 'sqlattack', 'httptunnel']
    dos=['apache2', 'back', 'mailbomb', 'processtable','snmpgetattack', 'teardrop', 'smurf', 'land','neptune', 'pod', 'udpstorm']
    
    for type in range(0,len(training_dataset)):
         if category_type[type] in probe:
             category_type[type]='probe'
         elif category_type[type] in r2l:
             category_type[type]='r2l'
         elif category_type[type] in u2r:
             category_type[type]='u2r'
         elif category_type[type] in dos:
             category_type[type]='dos'
         else:
             category_type[type]='benign'

    category_type_series=pd.Series(category_type)
    training_dataset['attack_category']=category_type_series
    return training_dataset

#Reading the trainning dataset
training_dataset=pd.read_csv("KDDTrain+.csv")
training_dataset.columns=header_names  #Adding a headers to a dataframe.
training_dataset_prepared=create_category(training_dataset)

#handling the categorical columns of service,flag and protocol_type.
    #service
train_service=training_dataset_prepared['service']
train_service_unique=sorted(train_service.unique())

service_columns=['Service_' + x for x in train_service_unique]

train_service_encoded=pd.get_dummies(train_service)
train_service_encoded=pd.DataFrame(train_service_encoded)
train_service_encoded.columns=service_columns

    #flag
train_flag=training_dataset_prepared['flag']
train_flag_unique=sorted(train_flag.unique())

flag_column=['Flag_' + x for x in train_flag_unique]

train_flag_encoded=pd.get_dummies(train_flag)
train_flag_encoded=pd.DataFrame(train_flag_encoded)
train_flag_encoded.columns=flag_column

    #protocol_type
train_protocol=training_dataset_prepared['protocol_type']
train_protocol_unique=sorted(train_protocol.unique())

protocol_columns=['Protocol_' + x for x in train_protocol_unique]

train_protocol_encoded=pd.get_dummies(train_protocol)
train_protocol_encoded=pd.DataFrame(train_protocol_encoded)
train_protocol_encoded.columns=protocol_columns

#removing the service,flag and protocol columns
training_dataset_prepared.drop(['service','protocol_type','flag'], axis=1, inplace=True)

#joining the categorical encoded attribute into main dataframe
frames=[train_service_encoded,train_flag_encoded,train_protocol_encoded]
training_dataset_prepared=pd.concat([training_dataset_prepared,train_service_encoded,train_flag_encoded,train_protocol_encoded], axis=1, sort=False)

#handling the missing and infinite value and deleting unnecessary values
info=training_dataset_prepared.describe()
training_dataset_prepared.drop(['num_outbound_cmds'], axis=1, inplace=True)     #Dropping the num_outbound coumn since it only contains 0 value.

training_dataset_prepared.replace([np.inf,-np.inf],np.nan,inplace=True)                  #handling the infinite value
training_dataset_prepared.fillna(training_dataset_prepared.mean(),inplace=True)

#Doing the feature scaling
# from sklearn.preprocessing import StandardScaler
# sc_x=StandardScaler()


#splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split
train_set,test_set=train_test_split(training_dataset_prepared,test_size=0.2,random_state=42)
    #sorting the train_set and test set
pd.DataFrame.sort_index(train_set,axis=0,ascending=True,inplace=True) 
pd.DataFrame.sort_index(test_set,axis=0,ascending=True,inplace=True) 

#Processing the dataset for federated learning
normal_dataset=train_set[train_set.attack_type.eq('normal')]

attack1=['neptune', 'smurf']
dataset1=train_set[train_set.attack_type.isin(attack1)]
dataset1=pd.concat([dataset1,normal_dataset[:5395]], ignore_index=True)
dataset1=dataset1.sample(frac=1).reset_index(drop=True)
# dataset1['attack_type'].value_counts()


attack2=['land','teardrop']
dataset2=train_set[train_set.attack_type.isin(attack2)]
dataset2=pd.concat([dataset2,normal_dataset[5395*1:5395*2]], ignore_index=True)
dataset2=dataset2.sample(frac=1).reset_index(drop=True)


attack3=['pod','back']
dataset3=train_set[train_set.attack_type.isin(attack3)]
dataset3=pd.concat([dataset3,normal_dataset[5395*2:5395*3]], ignore_index=True)
dataset3=dataset3.sample(frac=1).reset_index(drop=True)

attack4=['portsweep','nmap']
dataset4=train_set[train_set.attack_type.isin(attack4)]
dataset4=pd.concat([dataset4,normal_dataset[5395*3:5395*4]], ignore_index=True)
dataset4=dataset4.sample(frac=1).reset_index(drop=True)

attack5=['ipsweep','satan']
dataset5=train_set[train_set.attack_type.isin(attack5)]
dataset5=pd.concat([dataset5,normal_dataset[5395*4:5395*5]], ignore_index=True)
dataset5=dataset5.sample(frac=1).reset_index(drop=True)

attack6=['imap','warezmaster']
dataset6=train_set[train_set.attack_type.isin(attack6)]
dataset6=pd.concat([dataset6,normal_dataset[5395*5:5395*6]], ignore_index=True)
dataset6=dataset6.sample(frac=1).reset_index(drop=True)

attack7=['ftp_write','guess_passwd']
dataset7=train_set[train_set.attack_type.isin(attack7)]
dataset7=pd.concat([dataset7,normal_dataset[5395*6:5395*7]], ignore_index=True)
dataset7=dataset7.sample(frac=1).reset_index(drop=True)

attack8=['multihop','spy']
dataset8=train_set[train_set.attack_type.isin(attack8)]
dataset8=pd.concat([dataset8,normal_dataset[5395*7:5395*8]], ignore_index=True)
dataset8=dataset8.sample(frac=1).reset_index(drop=True)

attack9=['phf','warezclient','buffer_overflow']
dataset9=train_set[train_set.attack_type.isin(attack9)]
dataset9=pd.concat([dataset9,normal_dataset[5395*8:5395*9]], ignore_index=True)
dataset9=dataset9.sample(frac=1).reset_index(drop=True)

attack10=['loadmodule','perl', 'rootkit']
dataset10=train_set[train_set.attack_type.isin(attack10)]
dataset10=pd.concat([dataset10,normal_dataset[5395*9:5395*10]], ignore_index=True)
dataset10=dataset10.sample(frac=1).reset_index(drop=True)

# dataset1['attack_category'].value_counts()
# dataset2['attack_category'].value_counts()
# dataset3['attack_category'].value_counts()
# dataset4['attack_category'].value_counts()
# dataset5['attack_category'].value_counts()
# dataset6['attack_category'].value_counts()
# dataset7['attack_category'].value_counts()
# dataset8['attack_category'].value_counts()
# dataset9['attack_category'].value_counts()
# dataset10['attack_category'].value_counts()

# train_set['attack_type'].value_counts()

#--------------------------for test set-----------------------

test_y=test_set['attack_category']
test_x=test_set
test_x.drop(['attack_category'], axis=1, inplace=True)
test_x.drop(['attack_type'], axis=1, inplace=True)

#encoding the test_y
test_y=pd.get_dummies(test_y)


#------------------splitting the dataset of train into train_x and train_y------------
from sklearn.model_selection import train_test_split
header_list=['benign', 'dos', 'probe', 'r2l', 'u2r']

dataset1_train_y=dataset1['attack_category']
dataset1.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset1_train_x=dataset1
dataset1_train_y=pd.get_dummies(dataset1_train_y)
header_diff=set(header_list)-set(dataset1_train_y.columns)
for header in header_diff:
    dataset1_train_y[header]=0

dataset2_train_y=dataset2['attack_category']
dataset2.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset2_train_x=dataset2
dataset2_train_y=pd.get_dummies(dataset2_train_y)
header_diff=set(header_list)-set(dataset2_train_y.columns)
for header in header_diff:
    dataset2_train_y[header]=0

dataset3_train_y=dataset3['attack_category']
dataset3.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset3_train_x=dataset3
dataset3_train_y=pd.get_dummies(dataset3_train_y)
header_diff=set(header_list)-set(dataset3_train_y.columns)
for header in header_diff:
    dataset3_train_y[header]=0


dataset4_train_y=dataset4['attack_category']
dataset4.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset4_train_x=dataset4
dataset4_train_y=pd.get_dummies(dataset4_train_y)
header_diff=set(header_list)-set(dataset4_train_y.columns)
for header in header_diff:
    dataset4_train_y[header]=0
    
dataset5_train_y=dataset5['attack_category']
dataset5.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset5_train_x=dataset5
dataset5_train_y=pd.get_dummies(dataset5_train_y)
header_diff=set(header_list)-set(dataset5_train_y.columns)
for header in header_diff:
    dataset5_train_y[header]=0

dataset6_train_y=dataset6['attack_category']
dataset6.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset6_train_x=dataset5
dataset6_train_y=pd.get_dummies(dataset6_train_y)
header_diff=set(header_list)-set(dataset6_train_y.columns)
for header in header_diff:
    dataset6_train_y[header]=0
    
dataset7_train_y=dataset7['attack_category']
dataset7.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset7_train_x=dataset7
dataset7_train_y=pd.get_dummies(dataset7_train_y)
header_diff=set(header_list)-set(dataset7_train_y.columns)
for header in header_diff:
    dataset7_train_y[header]=0

dataset8_train_y=dataset8['attack_category']
dataset8.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset8_train_x=dataset8
dataset8_train_y=pd.get_dummies(dataset8_train_y)
header_diff=set(header_list)-set(dataset8_train_y.columns)
for header in header_diff:
    dataset8_train_y[header]=0

dataset9_train_y=dataset9['attack_category']
dataset9.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset9_train_x=dataset9
dataset9_train_y=pd.get_dummies(dataset9_train_y)
header_diff=set(header_list)-set(dataset9_train_y.columns)
for header in header_diff:
    dataset9_train_y[header]=0
    
dataset10_train_y=dataset10['attack_category']
dataset10.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset10_train_x=dataset10
dataset10_train_y=pd.get_dummies(dataset10_train_y)
header_diff=set(header_list)-set(dataset10_train_y.columns)
for header in header_diff:
    dataset10_train_y[header]=0





#making a global model by intializing parameters to zero
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()

from sklearn.model_selection import GridSearchCV


        

class Server:
    def __init__(self, dataset_test_x,dataset_test_y,global_model):
        self.dataset_test_x=dataset_test_x
        self.dataset_test_y=dataset_test_y
        self.global_model=global_model
        
    # def parameter(self,param):
    #     self.param=param
    
    # global_model=classifier
    
    def receive_paramter(self, n_estimators1):
        self.n_estimators_list=[]
        self.n_estimators_list.append(n_estimators1)
        
    def sum_estimator(self):
        self.sum_estimator=sum(self.n_estimators_list)
        self.final_estimator=int(self.sum_estimator/10)
        self.n_estimators_list=0
            
        
    def update_model(self):
        self.global_model.set_params(self,n_estimators=self.final_estimator)
            
    def split_datsaet(self):
        self.train_server_train_x=dataset_test_x[:15000]
        self.train_server_train_y=dataset_test_y[:15000]
        self.test_server_test_x=dataset_test_x[15000:]
        self.test_server_test_y=dataset_test_y[15000:]
            
    def test_updated_model(self):
        self.global_model.fit(self.train_server_train_x,self.train_server_train_y)
        self.predicted_value=self.global_model.predict(self.test_server_test_x)
            
        from sklearn.metrics import precision_score
        self.precision_model=precision_score(self.test_server_test_y,self.predicted_value,average='micro')     
        print('The precision of global model is ', self.precision_model)
        # self.precision_list=[]
        # self.precision_list.append(precision_model)
            
        #calculating loss function
        self.param_grid=[{'n_estimators':self.final_estimator}]
        self.global_model.fit(self.train_server_train_x,self.train_server_train_y)
        self.grid_search=GridSearchCV(self.global_model, self.param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
        self.grid_search.fit(self.test_server_test_x,self.test_server_test_y)
        self.best=self.grid_search.best_params_
            
        cvres=grid_search.cv_results_['mean_test_score']
        rmse_score=np.sqrt(cvres)
        print('The rmse score of global model is ', rmse_score)
        # rmse_score_list=[]
        # rmse_score_list.append(rmse_score)

    precision_list=[]
    rmse_score_list=[]

        
    def store_result(self):
        # self.precision_list=[]
        self.precision_list.append(self.precision_model)
        # rmse_score_list=[]
        rmse_score_list.append(rmse_score)
        
    def return_final_scores(self):
        return self.precision_list
        return self.rmse_score_list
    
    def start_server(self):
        self.sum_estimator()
        self.update_model()
        # self.split_datsaet()
        self.test_updated_model()
        self.store_result()
                
                
# x=[1,2]
# y=[3,4]      
# server1=server(x,y,classifier)
# server1.variable
        
#creating a server 1
server1=Server(test_x,test_y,model)
        
class Nodes:
    def __init__(self, dataset_train_x,dataset_train_y,n_estimators,model):
        self.dataset_train_x=dataset_train_x
        self.dataset_train_y=dataset_train_y
        self.n_estimators=n_estimators
        # self.min_sample_split=min_sample_split
        self.model=model
        
        from sklearn.model_selection import GridSearchCV

    
       
    def learn(self):
        self.param_grid=[{'n_estimators':self.n_estimators}]
        self.model.fit(self.dataset_train_x,self.dataset_train_y)
        self.grid_search=GridSearchCV(self.model, self.param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True)
        self.grid_search.fit(self.dataset_train_x,self.dataset_train_y)
        self.best_estimator=self.grid_search.best_params_
        
        # best_estimator=[]   #keeping the record of best estimator in a list
        # best_estimator.append(best['n_estimators'])
        
        # rmse_scores=[]  #keeping the record of rmse score in a list
        
        # cvres=min(grid_search.cv_results_['mean_test_score'])
        # rmse_score.append(cvres)
        
    def send(self):
        # self.server1=Server(test_x,test_y)
        server1.receive_paramter(self.best_estimator)
        
    def start_node(self):
        self.learn()
        self.send()
        
        

#Making the  parameters
parameter1=[5,10,15]
parameter1=[20,25,30]
parameter1=[35,40,45]
parameter1=[50,55,60]
parameter1=[65,70,75]
parameter1=[80,85,90]
parameter1=[95,100,105]
parameter1=[110,115,120]
parameter=[120,125,130]
parameter=[135,150,170]
parameter=[190,250,300]
parameter=[400,500,600]
parameter=[700,800,900]


node1=Nodes(train_x_1,train_y_1,parameter1,model)
node2=Nodes()
node3=Nodes()
node4=Nodes()
node5=Nodes()
node6=Nodes()
node7=Nodes()
node8=Nodes()
node9=Nodes()
node10=Nodes()


        
        
        
        
