#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 18:48:40 2020

@author: krishna
"""

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

#scaling the data
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
# training_dataset_columns=list(training_dataset_prepared.columns)
# training_dataset_columns.remove('attack_category')
# training_dataset_columns.remove('attack_type')
# training_dataset_prepared=sc_x.fit_transform(training_dataset_prepared[training_dataset_columns])
category_dropped=pd.DataFrame()
category_dropped['attack_type']=training_dataset_prepared['attack_type']
category_dropped['attack_category']=training_dataset_prepared['attack_category']
training_dataset_prepared.drop(['attack_type','attack_category'],axis=1,inplace=True)
training_dataset_columns=list(training_dataset_prepared.columns)
training_dataset_prepared=sc_x.fit_transform(training_dataset_prepared)
training_dataset_prepared=pd.DataFrame(training_dataset_prepared)
training_dataset_prepared.columns=training_dataset_columns
training_dataset_prepared=pd.concat([training_dataset_prepared,category_dropped],axis=1)

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

attack9=['phf','warezclient']
dataset9=train_set[train_set.attack_type.isin(attack9)]
dataset9=pd.concat([dataset9,normal_dataset[5395*8:5395*9]], ignore_index=True)
dataset9=dataset9.sample(frac=1).reset_index(drop=True)

attack10=['loadmodule','perl']
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

# test_y=test_set['attack_category']
# test_x=test_set
# test_x.drop(['attack_category'], axis=1, inplace=True)
# test_x.drop(['attack_type'], axis=1, inplace=True)

# #encoding the test_y
# # from sklearn.preprocessing import LabelEncoder
# # encoder=LabelEncoder()
# test_y=pd.get_dummies(test_y)


#------------------splitting the dataset of train into train_x and train_y------------
from sklearn.model_selection import train_test_split
header_list=['benign', 'dos', 'probe', 'r2l', 'u2r']
from sklearn.preprocessing import LabelEncoder
encode=LabelEncoder()

dataset1_train_y=dataset1['attack_category']
dataset1.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset1_train_x=dataset1
dataset1_train_y=encode.fit_transform(dataset1_train_y)
# header_diff=set(header_list)-set(dataset1_train_y.columns)
# for header in header_diff:
#     dataset1_train_y[header]=0


dataset2_train_y=dataset2['attack_category']
dataset2.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset2_train_x=dataset2
dataset2_train_y=encode.fit_transform(dataset2_train_y)
# header_diff=set(header_list)-set(dataset2_train_y.columns)
# for header in header_diff:
#     dataset2_train_y[header]=0

dataset3_train_y=dataset3['attack_category']
dataset3.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset3_train_x=dataset3
dataset3_train_y=encode.fit_transform(dataset3_train_y)
# header_diff=set(header_list)-set(dataset3_train_y.columns)
# for header in header_diff:
#     dataset3_train_y[header]=0


dataset4_train_y=dataset4['attack_category']
dataset4.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset4_train_x=dataset4
dataset4_train_y=encode.fit_transform(dataset4_train_y)
# header_diff=set(header_list)-set(dataset4_train_y.columns)
# for header in header_diff:
#     dataset4_train_y[header]=0
    
dataset5_train_y=dataset5['attack_category']
dataset5.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset5_train_x=dataset5
dataset5_train_y=encode.fit_transform(dataset5_train_y)
# header_diff=set(header_list)-set(dataset5_train_y.columns)
# for header in header_diff:
#     dataset5_train_y[header]=0

dataset6_train_y=dataset6['attack_category']
dataset6.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset6_train_x=dataset6
dataset6_train_y=encode.fit_transform(dataset6_train_y)
# header_diff=set(header_list)-set(dataset6_train_y.columns)
# for header in header_diff:
#     dataset6_train_y[header]=0
    
dataset7_train_y=dataset7['attack_category']
dataset7.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset7_train_x=dataset7
dataset7_train_y=encode.fit_transform(dataset7_train_y)
# header_diff=set(header_list)-set(dataset7_train_y.columns)
# for header in header_diff:
#     dataset7_train_y[header]=0

dataset8_train_y=dataset8['attack_category']
dataset8.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset8_train_x=dataset8
dataset8_train_y=encode.fit_transform(dataset8_train_y)
# header_diff=set(header_list)-set(dataset8_train_y.columns)
# for header in header_diff:
#     dataset8_train_y[header]=0

dataset9_train_y=dataset9['attack_category']
dataset9.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset9_train_x=dataset9
dataset9_train_y=encode.fit_transform(dataset9_train_y)
# header_diff=set(header_list)-set(dataset9_train_y.columns)
# for header in header_diff:
#     dataset9_train_y[header]=0
    
dataset10_train_y=dataset10['attack_category']
dataset10.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset10_train_x=dataset10
dataset10_train_y=encode.fit_transform(dataset10_train_y)
# header_diff=set(header_list)-set(dataset10_train_y.columns)
# for header in header_diff:
#     dataset10_train_y[header]=0


#------------Preparing dataset for prsonalized evaluation
normal_test_dataset=test_set[test_set.attack_type.eq('normal')]     #13386/10 =1338

dataset1_test=test_set[test_set.attack_type.isin(attack1)]
dataset1_test=pd.concat([dataset1_test,normal_test_dataset[:1338]],ignore_index=True)
dataset1_test=dataset1_test.sample(frac=1).reset_index(drop=True)
dataset1_test_y=dataset1_test['attack_category']
dataset1_test_x=dataset1_test
dataset1_test_x.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset1_test_y=encode.fit_transform(dataset1_test_y)


dataset2_test=test_set[test_set.attack_type.isin(attack2)]
dataset2_test=pd.concat([dataset2_test,normal_test_dataset[1338:1338*2]],ignore_index=True)
dataset2_test=dataset2_test.sample(frac=1).reset_index(drop=True)
dataset2_test_y=dataset2_test['attack_category']
dataset2_test_x=dataset2_test
dataset2_test_x.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset2_test_y=encode.fit_transform(dataset2_test_y)

dataset3_test=test_set[test_set.attack_type.isin(attack3)]
dataset3_test=pd.concat([dataset3_test,normal_test_dataset[1338*2:1338*3]],ignore_index=True)
dataset3_test=dataset3_test.sample(frac=1).reset_index(drop=True)
dataset3_test_y=dataset3_test['attack_category']
dataset3_test_x=dataset3_test
dataset3_test_x.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset3_test_y=encode.fit_transform(dataset3_test_y)

dataset4_test=test_set[test_set.attack_type.isin(attack4)]
dataset4_test=pd.concat([dataset4_test,normal_test_dataset[1338*3:1338*4]],ignore_index=True)
dataset4_test=dataset4_test.sample(frac=1).reset_index(drop=True)
dataset4_test_y=dataset4_test['attack_category']
dataset4_test_x=dataset4_test
dataset4_test_x.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset4_test_y=encode.fit_transform(dataset4_test_y)

dataset5_test=test_set[test_set.attack_type.isin(attack5)]
dataset5_test=pd.concat([dataset5_test,normal_test_dataset[1338*4:1338*5]],ignore_index=True)
dataset5_test=dataset5_test.sample(frac=1).reset_index(drop=True)
dataset5_test_y=dataset5_test['attack_category']
dataset5_test_x=dataset5_test
dataset5_test_x.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset5_test_y=encode.fit_transform(dataset5_test_y)

dataset6_test=test_set[test_set.attack_type.isin(attack6)]
dataset6_test=pd.concat([dataset6_test,normal_test_dataset[1338*5:1338*6]],ignore_index=True)
dataset6_test=dataset6_test.sample(frac=1).reset_index(drop=True)
dataset6_test_y=dataset6_test['attack_category']
dataset6_test_x=dataset6_test
dataset6_test_x.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset6_test_y=encode.fit_transform(dataset6_test_y)

dataset7_test=test_set[test_set.attack_type.isin(attack7)]
dataset7_test=pd.concat([dataset7_test,normal_test_dataset[1338*6:1338*7]],ignore_index=True)
dataset7_test=dataset7_test.sample(frac=1).reset_index(drop=True)
dataset7_test_y=dataset7_test['attack_category']
dataset7_test_x=dataset7_test
dataset7_test_x.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset7_test_y=encode.fit_transform(dataset7_test_y)

dataset8_test=test_set[test_set.attack_type.isin(attack8)]
dataset8_test=pd.concat([dataset8_test,normal_test_dataset[1338*7:1338*8]],ignore_index=True)
dataset8_test=dataset8_test.sample(frac=1).reset_index(drop=True)
dataset8_test_y=dataset8_test['attack_category']
dataset8_test_x=dataset8_test
dataset8_test_x.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset8_test_y=encode.fit_transform(dataset8_test_y)

dataset9_test=test_set[test_set.attack_type.isin(attack9)]
dataset9_test=pd.concat([dataset9_test,normal_test_dataset[1338*8:1338*9]],ignore_index=True)
dataset9_test=dataset9_test.sample(frac=1).reset_index(drop=True)
dataset9_test_y=dataset9_test['attack_category']
dataset9_test_x=dataset9_test
dataset9_test_x.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset9_test_y=encode.fit_transform(dataset9_test_y)

dataset10_test=test_set[test_set.attack_type.isin(attack10)]
dataset10_test=pd.concat([dataset10_test,normal_test_dataset[1338*9:1338*10]],ignore_index=True)
dataset10_test=dataset10_test.sample(frac=1).reset_index(drop=True)
dataset10_test_y=dataset10_test['attack_category']
dataset10_test_x=dataset10_test
dataset10_test_x.drop(['attack_category','attack_type'], axis=1, inplace=True)
dataset10_test_y=encode.fit_transform(dataset10_test_y)




    

from sklearn.linear_model import SGDClassifier
model=SGDClassifier()

# n_estimator_list=[]
# print('--------------Calculating ---------------')
# from sklearn.model_selection import GridSearchCV
# n_estimator_list=[]
# param_grid={'alpha':alpha}
# grid_search=GridSearchCV(model,param_grid,cv=5,scoring="neg_mean_squared_error", return_train_score=True)
# grid_search.fit(dataset1_train_x.values,dataset1_train_y)
# n_estimator_list.append(grid_search.best_params_['alpha'])
# print('--------------Calculating for node 1---------------')
# print(grid_search.best_params_['alpha'])

alpha_list=np.linspace(1,0.0001,300)
precision_list=[]
loss_list=[]
learning_list=[]
counter=1

def learn(alpha):
    n_estimator_list=[]
    # print('--------------Calculating ---------------')
    from sklearn.model_selection import GridSearchCV
    n_estimator_list=[]
    param_grid={'alpha':alpha}
    grid_search=GridSearchCV(model,param_grid,cv=5,scoring="neg_mean_squared_error", return_train_score=True)
    grid_search.fit(dataset1_train_x,dataset1_train_y)
    n_estimator_list.append(grid_search.best_params_['alpha'])
    # print('--------------Calculating for node 1---------------')
    # print(grid_search.best_params_['alpha'])
    
    grid_search.fit(dataset2_train_x,dataset2_train_y)
    n_estimator_list.append(grid_search.best_params_['alpha'])
    # print('--------------Calculating for node 2---------------')
    # print(grid_search.best_params_['alpha'])
    
    grid_search.fit(dataset3_train_x,dataset3_train_y)
    n_estimator_list.append(grid_search.best_params_['alpha'])
    # print('--------------Calculating for node 3---------------')
    # print(grid_search.best_params_['alpha'])
    
    grid_search.fit(dataset4_train_x,dataset4_train_y)
    n_estimator_list.append(grid_search.best_params_['alpha'])
    # print('--------------Calculating for node 4---------------')
    # print(grid_search.best_params_['alpha'])
    
    grid_search.fit(dataset5_train_x,dataset5_train_y)
    n_estimator_list.append(grid_search.best_params_['alpha'])
    # print('--------------Calculating for node 5---------------')
    # print(grid_search.best_params_['alpha'])
    
    grid_search.fit(dataset6_train_x,dataset6_train_y)
    n_estimator_list.append(grid_search.best_params_['alpha'])
    # print('--------------Calculating for node 6---------------')
    # print(grid_search.best_params_['alpha'])
    
    grid_search.fit(dataset7_train_x,dataset7_train_y)
    n_estimator_list.append(grid_search.best_params_['alpha'])
    # print('--------------Calculating for node 7---------------')
    # print(grid_search.best_params_['alpha'])
    
    grid_search.fit(dataset8_train_x,dataset8_train_y)
    n_estimator_list.append(grid_search.best_params_['alpha'])
    # print('--------------Calculating for node 8---------------')
    # print(grid_search.best_params_['alpha'])
    
    grid_search.fit(dataset9_train_x,dataset9_train_y)
    n_estimator_list.append(grid_search.best_params_['alpha'])
    # print('--------------Calculating for node 9---------------')
    # print(grid_search.best_params_['alpha'])
    
    grid_search.fit(dataset10_train_x,dataset10_train_y)
    n_estimator_list.append(grid_search.best_params_['alpha'])
    # print('--------------Calculating for node 10---------------')
    # print(grid_search.best_params_['alpha'])
    
    average=(sum(n_estimator_list)/10)
    learning_list.append(average)
    print('learning rate is:',average)
    return average


def set_learning_rate(number1,number2):
    returned_average=learn(alpha_list[number1:number2])
    average_parameter={'alpha':returned_average}
    model.set_params(**average_parameter)

    
    

#--------------------------Calucalte accuracy and loss function-----------------------------
def calculate_accuracy(dataset_train_x,dataset_train_y,dataset_test_x,dataset_test_y): 
    # print('##------Calculating for round------## ', counter)
    # counter=counter+1
    # returned_average=learn(alpha_list[number1:number2])
    # average_parameter={'alpha':returned_average}
    # model.set_params(**average_parameter)
    # precision_list=[]
    # loss_list=[]
    
    loss_list=['benign','dos','probe','u2r','r2l']
    model.fit(dataset_train_x,dataset_train_y)
    model_predicted=model.predict(dataset_test_x)
    model_predicted=pd.DataFrame(model_predicted)
    # model_predicted=model_predicted.rename(columns={'0':'attack_category'})
            
    from sklearn.metrics import precision_score,mean_squared_error,f1_score,hinge_loss
    precision_score=precision_score(dataset_test_y,model_predicted, average='micro')
    # print('The precision at  is ', precision_score)
    class_predicted=model.decision_function(dataset_test_x)
    
    class_predicted=pd.DataFrame(class_predicted)
    
    #doing min_max scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    class_predicted=scaler.fit_transform(class_predicted)
    
    
    loss=(mean_squared_error(dataset_test_y,class_predicted))  
    # print("The average loss is",loss)
    # loss_list.append(loss)
    # precision_list.append(precision_score)
    return precision_score,loss



def launch_round():
    precision1,loss1=calculate_accuracy(dataset1_train_x,dataset1_train_y,dataset1_test_x,dataset1_test_y)

    precision2,loss2=calculate_accuracy(dataset2_train_x,dataset2_train_y,dataset2_test_x,dataset2_test_y)

    precision3,loss3=calculate_accuracy(dataset3_train_x,dataset3_train_y,dataset3_test_x,dataset3_test_y)

    precision4,loss4=calculate_accuracy(dataset4_train_x,dataset4_train_y,dataset4_test_x,dataset4_test_y)

    precision5,loss5=calculate_accuracy(dataset5_train_x,dataset5_train_y,dataset5_test_x,dataset5_test_y)

    precision6,loss6=calculate_accuracy(dataset6_train_x,dataset6_train_y,dataset6_test_x,dataset6_test_y)

    precision7,loss7=calculate_accuracy(dataset7_train_x,dataset7_train_y,dataset7_test_x,dataset7_test_y)

    precision8,loss8=calculate_accuracy(dataset8_train_x,dataset8_train_y,dataset8_test_x,dataset8_test_y)

    precision9,loss9=calculate_accuracy(dataset9_train_x,dataset9_train_y,dataset9_test_x,dataset9_test_y)

    precision10,loss10=calculate_accuracy(dataset10_train_x,dataset10_train_y,dataset10_test_x,dataset10_test_y)
    
    average_precision=((precision1+precision2+precision3+precision4+precision5+precision6+precision7+precision8+precision9+precision10))/10
    average_loss=((loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10))/10
    print('Precision ====>',average_precision)
    print('Loss =========>',average_loss)
    precision_list.append(average_precision)
    loss_list.append(average_loss)



count_number=np.arange(1,300,3)
for i in range(1,101):  
    print('\n')
    print('##-----calculating for round ', i, '-----##')
    set_learning_rate(count_number[i],count_number[i+1])
    launch_round()




    



















































def calculate_accuracy(train_x,train_y,test_x,test_y):
    loss_list=['benign','dos','probe','u2r','r2l']
    model.fit(train_x,train_y)
    model_predicted=model.predict(test_x)
    model_predicted=pd.DataFrame(model_predicted)
    model_predicted.columns=test_y.columns
        
    from sklearn.metrics import precision_score,mean_squared_error,f1_score,hinge_loss
    precision_score=precision_score(test_y,model_predicted, average='micro')  
    print('The precision at  is ', precision_score)
    class_proba=model.predict_proba(test_x)
    class_proba_managed=pd.DataFrame()
    # temp=class_proba[0]
    # [row[1] for row in temp]
    class_proba_managed=class_proba[1]
    class_proba_managed=pd.DataFrame(class_proba_managed)
    class_proba_managed.columns=test_y.columns
    
        
    loss_function=[]
    
    for columns in test_y.columns:
        loss_function.append(hinge_loss(test_y[columns],class_proba_managed[columns]))
        
    average_loss=(sum(loss_function))/2
    print("The average loss is",average_loss)






def calculate(dataset_x,dataset_y):
    loss_list=['benign','dos','probe','u2r','r2l']
    model.fit(dataset_x,dataset_y)
    model_predicted=model.predict(test_x)
    model_predicted=pd.dataframe(model_predicted)
    model_predicted.columns=loss_list
    
    from sklearn.metrics import precision_score,mean_squared_error,f1_score
    precision_score=precision_score(test_y,model_predicted, average='micro')  
    print("The precision at round ", round1, ' is ', precision_score)
    class_proba=model.predict_proba(test_x)
    
    loss_function=[]

    for columns in loss_list:
        loss_function.append(mean_squared_error(test_y[columns],class_proba[columns]))
    
    average_loss=int(sum(loss_function)/5)
    print("The loss in ", round1, ' is average_loss')


def test_result(returned_average,round1):
    average_parameter={'n_estimators':returned_average}
    from sklearn.metrics import precision_score,mean_squared_error,f1_score
    model.set_params(**average_parameter)
    
    calculate(dataset1_train_x,dataset1_train_y)
    
    
returned_average=learn(n_estimator)
test_result(returned_average,1)

 
    
    
 

# def execute(n_estimator):
#     learn(dataset1_train_x,dataset1_train_y,n_estimator)    
#     learn(dataset2_train_x,dataset2_train_y,n_estimator)  
#     learn(dataset3_train_x,dataset3_train_y,n_estimator)  
#     learn(dataset4_train_x,dataset4_train_y,n_estimator)  
#     learn(dataset5_train_x,dataset5_train_y,n_estimator)  
#     learn(dataset6_train_x,dataset6_train_y,n_estimator)  
#     learn(dataset7_train_x,dataset7_train_y,n_estimator)  
#     learn(dataset8_train_x,dataset8_train_y,n_estimator)  
#     learn(dataset9_train_x,dataset9_train_y,n_estimator)  
#     learn(dataset10_train_x,dataset10_train_y,n_estimator)  
    
# execute(n_estimator)
