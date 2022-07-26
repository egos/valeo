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
import collections
from utils import *

# $C^{k-1}_{n+k-1} = \frac{(n+k-1)!}{n! (k-1)!}$
st.set_page_config(page_title = "_IHM", layout="wide")
print('BEGIN')

data, dfs, dfline , df1 = load_data(500)
st._legacy_dataframe(df1)
ListSelectbox = df1.index
index = st.selectbox('pop',ListSelectbox)
row = df1.loc[index]

print(row)
st._legacy_dataframe(row.to_frame().T)
ElemsList = ['Clist','Elist','Plist']
Elems = ['C','E','P']
IDSelects = []
ID_CtoE = row.ID_CtoE
ID_EtoP = row.ID_EtoP
for n in range(3):
    IDSelects+= ['{}{}'.format(Elems[n],i) for i in row[ElemsList[n]]]

dflineSelect = dfline[dfline.ID.isin(ID_CtoE + ID_EtoP)].copy()
dfsSelect = dfs[dfs.ID.isin(IDSelects)].copy()

c1, c2 = st.columns([0.3,0.7])      
if c2.checkbox('ALL combinaison') : 
    dflineSelect = dfline.copy()
    dfsSelect = dfs.copy()


N = data['height']
A0 = data['layers'][0]['data']
A0 = np.array(A0).reshape(10,7)
pas = 16
unique = np.unique(A0)
A0[A0 == unique[0]] = 0
A0[A0 == unique[1]] = 1
for idx, row in dfsSelect.iterrows():
    A0[row.y, row.x] = row.Color + int(row.Name)

A = np.kron(A0, np.ones((16,16), dtype=int))
fig, ax = plt.subplots(figsize = (10,10))
ax.imshow(A)
for idx, row in dflineSelect.iterrows():
    p = row.polyline
    f = ax.plot(p[:,0],p[:,1],'k')

style = dict(size=10, color='black') 
for idx, row in dfsSelect.iterrows():
    x = row.x*16
    y = row.y*16
    text = row.Class + str(row.Name)
    f = ax.text(row.x*16+8, row.y*16+8,text , **style,  ha='center', weight='bold') 
     
c2._legacy_dataframe(dflineSelect)  
c2._legacy_dataframe(dfsSelect) 
       
c1.pyplot(fig)