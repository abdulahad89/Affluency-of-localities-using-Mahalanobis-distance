# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 23:34:50 2019

@author: Abdul Ahad
"""
import pandas as pd
from scipy.spatial.distance import mahalanobis
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_excel('file:///D:/intern/Feature eng/AAP- feature engineeringnew.xlsx')
#df=df.drop(columns=['Unnamed: 0','Unnamed: 0.1'])
d=pd.read_excel('file:///D:/intern/Feature eng/data.xlsx')

corr_d=d.drop(columns=['Part no.','distance','weight','Unnamed: 0'])   ###With jj colony

total = df.isnull().sum().sort_values(ascending=False)
d=df.dropna()
d.to_excel('D:\intern\Feature eng\data.xlsx')



##########   Mahanolobis distance formula   ##########
x = pd.read_excel('D:\intern\Feature eng\data.xlsx')
x = x.iloc[:,[5,6,7]]

Sx = x.cov().values
Sx = sp.linalg.inv(Sx)

mean = x.mean().values

def mahalanobisR(X,meanCol,IC):
    m = []
    for i in range(X.shape[0]):
        m.append(mahalanobis(X.iloc[i,:],meanCol,IC) )
    return(m)

mR = mahalanobisR(x,mean,Sx)
d['distance']=mR



# giving Relative weights for different data points 
weight=[]
for i in d['Buy']:
    if (i>=18000):                    #If propety rate are greater than 7000 than it will be positive
        weight.append(1)
    else:
        weight.append(-1)           #If propety rate are less than 7000 than it will be negative
        
d['weight']=weight
d['Relative dist']=d['distance']*d['weight']

#Female to male Ratio
d['ratio']=d['Female']/d['Male']

#Making a final dataset
d.to_excel('D:\intern\Feature eng\data.xlsx')

####Working on Rajinder Nagar constituency #########
#Young Female who are married#
#Reading from main datatset and finding young married

dt=pd.read_excel('file:///D:/intern/election/Rajinder_nagar_.xlsx')
young = dt['Age'] <= 35
F_married=dt['G_Relation']=='H'
young_Married=dt[young & F_married]

##Adding young married female into our features dataset
Y_Female_M=[]
for i in range(1,178):
    x = young_Married[young_Married['Ward Num'] == i]['Age'].count()
    Y_Female_M.append(x)   

df.to_excel('D:\intern\Feature eng\AAP- feature engineeringnew.xlsx')

###Plots
sns.scatterplot(x='Relative dist',y='Y_M_%',data=d)
sns.scatterplot(x='Relative dist',y='Young Female %',data=d)
sns.scatterplot(x='ratio',y='Relative dist',data=d)
sns.scatterplot(x='Young Female %',y='Relative dist',data=d)
sns.scatterplot(x='ratio',y='Young Female %',data=d)

sns.scatterplot(x='Salon',y='R',data=d)


#########    Heat map to check correlation between different variables 
corr_mat=corr_d.corr()                                   ##Calculating the correlation matrix
f,ax=plt.subplots(figsize=(14,11))
sns.heatmap(corr_mat,square=True,annot=True,fmt='.2f')   ##Using the corr matrix to plot heatmap


##Scatterplots showing relationship with different variables###
#          Pairplot for variables    ########
sns.set()
cols = ['Male','Female','Coffee Shops','Salon','Buy','Young Female %','Married Female %',
        'Y_Married','Y_M_%','ratio','Relative dist']
sns.pairplot(d[cols], size = 2.5)
plt.show()
y=sns.scatterplot(x='Part no.',y='Buy',data=d)
y 
sns.scatterplot(x='Relative dist',y='Buy',data=d)

#### TO predict missing Values ###########
####Extension of project ###
############ Predicting Housing Rates  ############################################
rk=pd.read_excel('file:///D:/intern/Feature eng/data123.xlsx')
corr_t_rk=rk.drop(d.index[[2,3,4,18,19,20,21,22,26,27,28,29,30,31,32,38,39,40,41,]])     ##Without jj colony
corr_t_rk=corr_t_rk.drop(columns=['Unnamed: 0'])
x_train_rk=corr_t_rk.iloc[:,[4,11]]
y_train_rk=corr_t_rk['Buy/per sqft'].values

corr_t=d.drop(d.index[[43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,
                       71,72,73,74,75,76,77]])     ##Withput jj colony

from sklearn.linear_model import LinearRegression
x_train=corr_t.iloc[:,[4,12]].values
y_train=corr_t['Buy'].values
test=d[d['Localities']=='j j colony inderpuri']
x_test=test.iloc[:,[6,15]].values

lin=LinearRegression()          
lin.fit(x_train,y_train)        ####TRAIN MODEL##########

y_test=lin.predict(x_test)

from sklearn.metrics import r2_score
print(r2_score(y_train,lin.predict(x_train)))

y_predict_rk=lin.predict(x_train_rk)
print(r2_score(y_train_rk,y_predict_rk))