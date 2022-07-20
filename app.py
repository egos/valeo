import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import itertools 
import math
from math import factorial as f
from datetime import timedelta
from streamlit import session_state
import matplotlib.pyplot as plt
import json
# $C^{k-1}_{n+k-1} = \frac{(n+k-1)!}{n! (k-1)!}$
st.set_page_config(page_title = "_IHM", layout="wide")
print('BEGIN')


with open('VALEO_1.tmj', 'r') as f:
  data = json.load(f)
  
dfs = pd.DataFrame(data['layers'][1]['objects']).drop(columns= ['rotation','width','height','visible','gid']).rename(columns = {'class' : 'Class', 'name' : 'Name'})
dfs.x = (dfs.x/16).astype(int)
dfs.y = (dfs.y/16).astype(int) - 1
dfs['Color'] = dfs['Class'].map({'C':10,'EV':20,'TP':30})


dfline  = pd.DataFrame(data['layers'][2]['objects']).drop(columns=['rotation','width','name','height','visible']).rename(columns = {'class' : 'Class'})

for idx, row in dfline.iterrows():
    properties = row.properties
    dfline.loc[idx, ['end','long','start']] = [d['value'] for d in properties]    
    polyline = row.polyline
    x = row.x
    y = row.y
    polyline = [(p['x'] + x, p['y'] + y) for p in polyline]
    dfline.at[idx , 'polyline'] = polyline
dfline.start = pd.to_numeric(dfline.start)
dfline.end = pd.to_numeric(dfline.end)
dfline = dfline.drop(columns= ['properties','x','y'])

D = dfs.Class.value_counts().to_dict()
R = np.random.randint(0,D['EV'],D['C'])


dfline['Select'] = False
dfs['Connect'] = 0
dfs.loc[dfs.Class == 'C', 'Connect']= 1
for i in range(D['C']):
    j = R[i]
    mask = (dfline.Class == 'C-E') & (dfline.start == i) & (dfline.end == j)
    # print(i,j, mask.sum())
    dfline.loc[mask, 'Select'] = True 
    
    mask = (dfs.Class == 'EV') & (dfs.Name == str(j))
    dfs.loc[mask, 'Connect'] +=  + 1
    
for i in np.unique(R):
    j = np.random.randint(0,D['TP'])
    mask = (dfline.Class == 'P-E') & (dfline.start == j) & (dfline.end == i)
    # print(i,j, mask.sum())
    dfline.loc[mask, 'Select'] = True
    
    mask = (dfs.Class == 'TP') & (dfs.Name == str(j))
    dfs.loc[mask, 'Connect'] +=  + 1
    
c1, c2 = st.columns([0.3,0.7])      
if c1.checkbox('ALL') : 
    dfline['Select'] = True
    dfs['Connect'] = 1
    
if c2.button('RANDOMIZE') : 
    pass
    
    
 
c2._legacy_dataframe(dfline)  
c2._legacy_dataframe(dfs) 



N = data['height']
A0 = data['layers'][0]['data']
A0 = np.array(A0).reshape(10,7)
pas = 16
unique = np.unique(A0)
A0[A0 == unique[0]] = 0
A0[A0 == unique[1]] = 1
for idx, row in dfs.iterrows():
    if row.Connect > 0:
        A0[row.y, row.x] = row.Color + int(row.Name)
        
A = np.kron(A0, np.ones((16,16), dtype=int))
fig, ax = plt.subplots(figsize = (10,10))
ax.imshow(A)
for idx, row in dfline.iterrows():
    if row.Select: 
        p = np.array(row.polyline)
        f = ax.plot(p[:,0],p[:,1],'k')

style = dict(size=10, color='black') 
for idx, row in dfs.iterrows():
    if row.Connect > 0:
        x = row.x*16
        y = row.y*16
        text = row.Class + str(row.Name)
        f = ax.text(row.x*16+8, row.y*16+8,text , **style,  ha='center', weight='bold') 
        
c1.pyplot(fig)