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

@st.cache(allow_output_mutation=True)
def load_data(Size):
    with open('VALEO_1.tmj', 'r') as f:
      data = json.load(f)

    dfs = pd.DataFrame(data['layers'][1]['objects']).drop(columns= ['rotation','width','height','visible','gid']).rename(columns = {'class' : 'Class', 'name' : 'Name'})
    dfs.x = (dfs.x/16).astype(int)
    dfs.y = (dfs.y/16).astype(int) - 1
    dfs.Name = pd.to_numeric(dfs.Name)
    dfs['ID'] = dfs.Class + dfs.Name.astype(str)
    dfs['Color'] = dfs['Class'].map({'C':10,'E':20,'P':30})


    dfline  = pd.DataFrame(data['layers'][2]['objects']).drop(columns=['rotation','width','name','height','visible']).rename(columns = {'class' : 'Class'})
    for idx, row in dfline.iterrows():
        properties = row.properties
        dfline.loc[idx, ['end','long','start']] = [d['value'] for d in properties]    
        polyline = row.polyline
        x = row.x
        y = row.y
        polyline = [(p['x'] + x, p['y'] + y) for p in polyline]
        p = np.array(polyline)
        dfline.at[idx , 'polyline'] = p   
        dfline.loc[idx ,'dist'] = np.abs(np.diff(p.T)).sum()
    dfline.start = pd.to_numeric(dfline.start)
    dfline.end = pd.to_numeric(dfline.end)
    dfline.dist = dfline.dist.astype(int)
    dfline = dfline.drop(columns= ['properties','x','y'])
    dfline['ID'] = dfline.Class + dfline.start.astype(str) + dfline.end.astype(str)

    L = []
    D = dfs.Class.value_counts().to_dict()

    for n in range(Size): 
        Clist = list(range(D['C']))

        CtoE = np.random.randint(0,D['E'],D['C'])
        Econnect = dict(collections.Counter(CtoE))
        Elist = sorted(Econnect)
        Ecount = len(Elist)

        EtoP = np.random.randint(0,D['P'],Ecount)
        Pconnect = dict(collections.Counter(EtoP))
        Plist = sorted(Pconnect)
        Pcount = len(Plist)

        ID_CtoE = ['C-E{}{}'.format(i, CtoE[i]) for i in Clist]
        ID_EtoP = ['E-P{}{}'.format(Elist[i]  , EtoP[i]) for i in range(Ecount)]

        dist = dfline.loc[dfline.ID.isin(ID_CtoE + ID_EtoP), 'dist'].sum()  

        l = [Clist, CtoE,Econnect,Elist,Ecount, EtoP,Pconnect,Plist,Pcount, ID_CtoE,ID_EtoP,dist]
        L.append(l)  

    col = ['Clist', 'CtoE','Econnect','Elist','Ecount', 'EtoP','Pconnect','Plist','Pcount', 'ID_CtoE','ID_EtoP','dist']
    df = pd.DataFrame(L, columns= col)

    df['Name'] = (df.ID_CtoE + df.ID_EtoP).str.join(',')
    df1 = df.drop_duplicates(subset='Name').sort_values('dist')
    return data, dfs, dfline , df1.drop(columns = 'Name').reset_index(drop= True)