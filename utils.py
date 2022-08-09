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
from types import SimpleNamespace

# @st.cache(allow_output_mutation=True)
# def load_data(Size, time):
#     return load_data_brut(Size)


def load_data_brut(file):
    print("begin")
    
    with open(file, 'r') as f:
      data = json.load(f)

    dfslot = pd.DataFrame(data['layers'][1]['objects']).drop(columns= ['rotation','width','height','visible','gid']).rename(columns = {'class' : 'Class', 'name' : 'Name'})
    dfslot.x = (dfslot.x/16).astype(int)
    dfslot.y = (dfslot.y/16).astype(int) - 1
    dfslot.Name = pd.to_numeric(dfslot.Name)
    dfslot['ID'] = dfslot.Class + dfslot.Name.astype(str)
    dfslot['Color'] = dfslot['Class'].map({'C':10,'E':20,'P':30})


    dfline  = pd.DataFrame(data['layers'][2]['objects']).drop(columns=['rotation','width','name','height','visible','id']).rename(columns = {'class' : 'Class'})
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
    dfline['ID'] = t.str[1] + dfline.end.astype(str) + '-' + t.str[0] + dfline.start.astype(str)
    dfline = dfline.sort_values('ID').reset_index(drop = True)
    # dfline['ID'] = dfline.Class + dfline.start.astype(str) + dfline.end.astype(str)

    # L = []
    # D = dfs.Class.value_counts().to_dict()
    # Clist = dfs[dfs.Class =='C'].Name.sort_values().unique().tolist()
    DropList = ['C0','E2','P0']
    dfline = dfline[~dfline.ID.str.contains('|'.join(DropList))]
    dfslot = dfslot[~dfslot.ID.isin(DropList)]
    
    algo = dict(
        pop = 50,
        dfslot = dfslot,
        dfline = dfline,
        epoch = 0,
        indivs = [],
        df = [],
        Comb = dfslot.groupby('Class').Name.unique().apply(list).apply(sorted).to_dict(),
        height = data['height'],
        A0 = data['layers'][0]['data'],
    )
    algo = SimpleNamespace(**algo)
    return algo

def indiv_create(algo):
    dfline = algo.dfline
    D = algo.Comb
    Clist = D['C']
    Ccount = len(D['C'])
    CtoE = list(np.random.choice(D['E'],Ccount))
    # CtoE = [1,0,1]
    d = collections.defaultdict(list)
    for i in range(Ccount):      d[CtoE[i]].append(D['C'][i])
    Econnect = dict(sorted(d.items()))
    Elist = sorted(Econnect)
    Ecount = len(Elist)

    EtoP = list(np.random.choice(D['P'],Ecount))
    d = collections.defaultdict(list)
    for i in range(Ecount):      d[EtoP[i]].append(Elist[i])
    Pconnect = dict(sorted(d.items()))
    Plist = sorted(Pconnect)
    Pcount = len(Plist)

    List_EtoC = [['E{}-C{}'.format(start, end) for end in List] for start , List in Econnect.items()]
    List_PtoE = [['P{}-E{}'.format(start, end) for end in List] for start , List in Pconnect.items()]
    Name = list(itertools.chain.from_iterable(List_EtoC + List_PtoE))

    dist = dfline.loc[dfline.ID.isin(Name), 'dist'].sum()*2
    col = ['D', 'Clist','CtoE','Econnect','Elist','Ecount', 'EtoP','Pconnect','Plist','Pcount', 'List_EtoC','List_PtoE', 'dist', 'Name']
    l = [D,Clist, CtoE,Econnect,Elist,Ecount, EtoP,Pconnect,Plist,Pcount, List_EtoC,List_PtoE, dist, Name]
    indiv = SimpleNamespace(**dict(zip(col,l)))
    indiv = dict(zip(col,l))

    return indiv
# SAVE
    # for n in range(Size): 
    #     # Clist = list(range(D['C']))        

    #     CtoE = np.random.randint(0,D['E'],D['C'])
    #     Econnect = dict(collections.Counter(CtoE))
    #     Elist = sorted(Econnect)
    #     Ecount = len(Elist)

    #     EtoP = np.random.randint(0,D['P'],Ecount)
    #     Pconnect = dict(collections.Counter(EtoP))
    #     Plist = sorted(Pconnect)
    #     Pcount = len(Plist)       
        

    #     ID_CtoE = ['C{}-E{}'.format(C, CtoE[i]) for i, C in enumerate(Clist)]
    #     ID_EtoP = ['E{}-P{}'.format(Elist[i]  , EtoP[i]) for i in range(Ecount)]

    #     dist = dfline.loc[dfline.ID.isin(ID_CtoE + ID_EtoP), 'dist'].sum()  

    #     l = [Clist, CtoE,Econnect,Elist,Ecount, EtoP,Pconnect,Plist,Pcount, ID_CtoE,ID_EtoP,dist]
    #     L.append(l)  

    # col = ['Clist', 'CtoE','Econnect','Elist','Ecount', 'EtoP','Pconnect','Plist','Pcount', 'ID_CtoE','ID_EtoP','dist']
    # df = pd.DataFrame(L, columns= col)

    # df['Name'] = (df.ID_CtoE + df.ID_EtoP).str.join(',')
    # df1 = df.drop_duplicates(subset='Name').sort_values('dist')
    # return data, dfs, dfline , df1.drop(columns = 'Name').reset_index(drop= True)

def indiv_init(algo, pop):
    L = []
    for i in range(pop):
        l = indiv_create(algo)
        algo.indivs.append(l)
        algo.epoch += 1
        L.append(l)
    # col = ['D', 'CtoE','Econnect','Elist','Ecount', 'EtoP','Pconnect','Plist','Pcount', 'List_EtoC','List_PtoC', 'dist', 'Name']
    # df = pd.DataFrame([vars(i) for i in indivs])
    df = pd.DataFrame(L)
    df['Name_txt'] = df.Name.str.join(',')
    df = df.drop_duplicates(subset='Name_txt').sort_values('dist').reset_index(drop = True)
    algo.df = df
    return df

def plot_(algo,dflineSelect, dfsSelect, name): 
    # N = data['height']
    # A0 = data['layers'][0]['data']
    height = algo.height
    A00 = algo.A0
    A0 = np.array(A00).reshape(10,7)
    pas = 16
    unique = np.unique(A0)
    A0[A0 == unique[0]] = 0
    A0[A0 == unique[1]] = 1
    for idx, row in dfsSelect.iterrows():
        A0[row.y, row.x] = row.Color + int(row.Name)   
        
    A = np.kron(A0, np.ones((16,16), dtype=int))
    fig, ax = plt.subplots(figsize = (10,10))
    ax.set_title(name)
    plt.xticks([])
    plt.yticks([])
    ax.imshow(A)
    for idx, row in dflineSelect.iterrows():
        p = row.polyline
        f = ax.plot(p[:,0],p[:,1],'k', linewidth=4)

    style = dict(size=15, color='black') 
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
        NewEtoP = np.random.randint(0,D['P'],Ecount - row.Ecount)
        EtoP = np.append(row.EtoP, NewEtoP)
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
        # search for gen diff between indivs = index        
        if len(index) > 1:    
            idx = np.random.choice(index)
        elif len(index) == 1:
            idx = index             
        c1[idx] , c2[idx] = c2[idx] , c1[idx]
        NewCtoE = c1 , c2
        L = []
        # verif each children
        for i in range(2):
            row = dfx.iloc[i]
            l = indiv_verif(row,NewCtoE[i],dfs, dfline) 
            L.append(l)
        return L