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
file = 'VALEO_full.tmj'
Comb = 	{'C': [0, 1, 2, 3], 'E': [0, 1, 2], 'P': [0, 1]}
pop = 200
    
if 'algo' not in session_state: 
    algo = load_data_brut(file)
    df1 = indiv_init(algo, pop)
    session_state['algo'] = algo
else : 
    algo = session_state['algo']
    
      
c1,c2 = st.columns(2)

if c2.button('RESET'):
    algo = load_data_brut(file)
    df1 = indiv_init(algo, pop)
    session_state['algo'] = algo
        
# if c1.button('RUN'):
#     L = []      
#     List = df1[:7].index.values
#     np.random.shuffle(List)
#     print(List)
#     for n in range(3):
#         print(n)
#         i1 = List[n*2]
#         i2 = List[n*2 + 1]
#         dfx = df1.loc[[i1,i2]].copy()
#         L2 = Reprodution(dfx,dfs, dfline)
#         if L2:
#             for l in L2:
#                 l.append(Epoch)
#                 l.append(dfx.Epoch.tolist())
#                 Epoch+= 1
#                 L.append(l)
        
#     col = ['Clist', 'CtoE','Econnect','Elist','Ecount', 'EtoP','Pconnect','Plist','Pcount', 'ID_CtoE','ID_EtoP','dist','Epoch','Parents']
#     dfx = pd.DataFrame(L, columns= col)
#     # dfx
#     df1 = df1.append(dfx)
#     df1['Name'] = (df1.ID_CtoE + df1.ID_EtoP).str.join(',')
#     df1 = df1.drop_duplicates(subset='Name').sort_values('dist').reset_index(drop = True)    
#     session_state['df1'] = df1
#     session_state['Epoch'] = Epoch

# st._legacy_dataframe(df1.drop(columns = ['ID_CtoE','ID_EtoP']))
st.write(str(algo.Comb))
df1 = algo.df
dfs = algo.dfslot
dfline = algo.dfline
st._legacy_dataframe(df1.drop(columns= ['D','Name', 'Name_txt']).astype(str))


ListSelectbox = df1.index
index = st.selectbox('individu',ListSelectbox)
row = df1.loc[index]

print(row)
# print(dfline)
# st._legacy_dataframe(row.to_frame().T)
ElemsList = ['Clist','Elist','Plist']
Elems = ['C','E','P']
IDSelects = []
List_EtoC = row.List_EtoC
List_PtoE = row.List_PtoE
for n in range(3):
    IDSelects+= ['{}{}'.format(Elems[n],i) for i in row[ElemsList[n]]]

dflineSelect = dfline[dfline.ID.isin(row.Name)].copy()
dfsSelect = dfs[dfs.ID.isin(IDSelects)].copy()

c1, c2, c3 = st.columns([0.3,0.3,0.4])      
if c2.checkbox('ALL combinaison') : 
    dflineSelect = dfline.copy()
    dfsSelect = dfs.copy()

fig = plot_(algo,dflineSelect, dfsSelect, str(row.name) + ' : ' + row.Name_txt + ' / '+ str(row.dist))

     
c3.table(dflineSelect.astype('string').drop(columns = ['polyline','long']))  
c3.table(dfsSelect.astype('string'))
c2.table(row.astype('string'))
       
c1.pyplot(fig)