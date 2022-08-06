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
import copy

# @st.cache(allow_output_mutation=True)
def load_data(Size, time):
    return load_data_brut(Size)

def load_data_brut(Size):
    with open('VALEO_full.tmj', 'r') as f:
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
        # properties = properties[:0] + properties[-2:]
        dfline.loc[idx, ['end','long','start']] = [d['value'] for d in properties]    
        polyline = row.polyline
        x,y = row.x,row.y
        polyline = [(p['x'] + x, p['y'] + y) for p in polyline]
        p = np.array(polyline)
        dfline.at[idx , 'polyline'] = p   
        dfline.loc[idx ,'dist'] = np.abs(np.diff(p.T)).sum()
    dfline.start = pd.to_numeric(dfline.start)
    dfline.end = pd.to_numeric(dfline.end)
    dfline.dist = dfline.dist.astype(int)
    dfline = dfline.drop(columns= ['properties','x','y'])
    t = dfline.Class.str.split('-')
    dfline['ID'] = t.str[0] + dfline.start.astype(str) + '-' + t.str[1] + dfline.end.astype(str)
    # dfline['ID'] = dfline.Class + dfline.start.astype(str) + dfline.end.astype(str)

    L = []
    D = dfs.Class.value_counts().to_dict()
    Clist = dfs[dfs.Class =='C'].Name.sort_values().unique().tolist()

    for n in range(Size): 
        # Clist = list(range(D['C']))        

        CtoE = np.random.randint(0,D['E'],D['C'])
        Econnect = dict(collections.Counter(CtoE))
        Elist = sorted(Econnect)
        Ecount = len(Elist)

        EtoP = np.random.randint(0,D['P'],Ecount)
        Pconnect = dict(collections.Counter(EtoP))
        Plist = sorted(Pconnect)
        Pcount = len(Plist)       
        

        ID_CtoE = ['C{}-E{}'.format(C, CtoE[i]) for i, C in enumerate(Clist)]
        ID_EtoP = ['E{}-P{}'.format(Elist[i]  , EtoP[i]) for i in range(Ecount)]

        dist = dfline.loc[dfline.ID.isin(ID_CtoE + ID_EtoP), 'dist'].sum()  

        l = [Clist, CtoE,Econnect,Elist,Ecount, EtoP,Pconnect,Plist,Pcount, ID_CtoE,ID_EtoP,dist]
        L.append(l)  

    col = ['Clist', 'CtoE','Econnect','Elist','Ecount', 'EtoP','Pconnect','Plist','Pcount', 'ID_CtoE','ID_EtoP','dist']
    df = pd.DataFrame(L, columns= col)

    df['Name'] = (df.ID_CtoE + df.ID_EtoP).str.join(',')
    df1 = df.drop_duplicates(subset='Name').sort_values('dist')
    return data, dfs, dfline , df1.drop(columns = 'Name').reset_index(drop= True)

def plot_(data,dflineSelect, dfsSelect): 
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
    return fig

def indiv_verif(row, NewCtoE, dfs, dfline): 
    D = dfs.Class.value_counts().to_dict()
    Clist = list(range(D['C']))
    Clist = dfs[dfs.Class =='C'].Name.sort_values().unique().tolist()
    
    CtoE = NewCtoE
    Econnect = dict(collections.Counter(CtoE))
    Elist = sorted(Econnect)
    Ecount = len(Elist)   
    
    if Ecount > row.Ecount:
        x = row.EtoP
        y = np.random.randint(0,D['P'],Ecount - row.Ecount)
        EtoP = np.append(x, y)
    elif Ecount < row.Ecount:
        # print(Elist,'avant',row.EtoP,row.Ecount, Ecount)
        EtoP = np.random.choice(row.EtoP,Ecount)
        # print("Ecount",EtoP)
    else : 
        EtoP = row.EtoP
    # print(Elist,'apres',EtoP,row.Ecount, Ecount)
    Pconnect = dict(collections.Counter(EtoP))
    Plist = sorted(Pconnect)
    Pcount = len(Plist)    

    # ID_CtoE = ['C-E{}{}'.format(i, CtoE[i]) for i in Clist]
    # ID_EtoP = ['E-P{}{}'.format(Elist[i]  , EtoP[i]) for i in range(Ecount)]
    ID_CtoE = ['C{}-E{}'.format(C, CtoE[i]) for i, C in enumerate(Clist)]
    ID_EtoP = ['E{}-P{}'.format(Elist[i]  , EtoP[i]) for i in range(Ecount)]

    dist = dfline.loc[dfline.ID.isin(ID_CtoE + ID_EtoP), 'dist'].sum()  
    l = [Clist, CtoE,Econnect,Elist,Ecount, EtoP,Pconnect,Plist,Pcount, ID_CtoE,ID_EtoP,dist]
    return l 

def Reprodution(dfx,dfs, dfline):
    c1, c2 = copy.deepcopy(dfx.CtoE.values)
    
    if (c1 != c2).any():        
        m = c1 != c2
        index = np.where(m)[0]
        
        if len(index) > 1:    
            idx = np.random.choice(index)
        elif len(index) ==1:
            idx = index             
        c1[idx] , c2[idx] = c2[idx] , c1[idx]
        NewCtoE = c1 , c2
        L = []

        for i in range(2):
            row = dfx.iloc[i]
            l = indiv_verif(row,NewCtoE[i],dfs, dfline) 
            L.append(l)
        return L