import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import itertools 
import math
import time
from math import factorial as f
from datetime import timedelta
from streamlit import session_state
import matplotlib.pyplot as plt
import json
import collections
import copy
from utils import *

# $C^{k-1}_{n+k-1} = \frac{(n+k-1)!}{n! (k-1)!}$
st.set_page_config(page_title = "_IHM", layout="wide")
print('BEGIN')


data, dfs, dfline , df1 = load_data(10, time.ctime())


if 'df1' not in session_state:

    df1['Epoch'] = df1.index
    session_state['df1'] = df1
else :
    df1 = session_state['df1']

if 'Epoch' not in session_state:
    Epoch = df1.index.max()+1
    session_state['Epoch'] = Epoch
else : 
    Epoch = session_state['Epoch']

c1,c2 = st.columns(2)

if c2.button('RESET'):
    data, dfs, dfline , df1 = load_data(10, time.ctime())
    df1['Epoch'] = df1.index
    session_state['df1'] = df1
    Epoch = df1.index.max()+1
    session_state['Epoch'] = Epoch
    
    
if c1.button('RUN'):
    L = []      
    List = df1[:7].index.values
    np.random.shuffle(List)
    print(List)
    for n in range(3):
        print(n)
        i1 = List[n*2]
        i2 = List[n*2 + 1]
        dfx = df1.loc[[i1,i2]].copy()
        L2 = Reprodution(dfx,dfs, dfline)
        if L2:
            for l in L2:
                l.append(Epoch)
                l.append(dfx.Epoch.tolist())
                Epoch+= 1
                L.append(l)
        
    col = ['Clist', 'CtoE','Econnect','Elist','Ecount', 'EtoP','Pconnect','Plist','Pcount', 'ID_CtoE','ID_EtoP','dist','Epoch','Parents']
    dfx = pd.DataFrame(L, columns= col)
    # dfx
    df1 = df1.append(dfx)
    df1['Name'] = (df1.ID_CtoE + df1.ID_EtoP).str.join(',')
    df1 = df1.drop_duplicates(subset='Name').sort_values('dist').reset_index(drop = True)    
    session_state['df1'] = df1
    session_state['Epoch'] = Epoch

st._legacy_dataframe(df1.drop(columns = ['ID_CtoE','ID_EtoP']))
ListSelectbox = df1.index
index = st.selectbox('individu',ListSelectbox)
row = df1.loc[index]

# print(row)
# st._legacy_dataframe(row.to_frame().T)
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

fig = plot_(data,dflineSelect, dfsSelect)
     
c2._legacy_dataframe(dflineSelect)  
c2._legacy_dataframe(dfsSelect) 
       
c1.pyplot(fig)