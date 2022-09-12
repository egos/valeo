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
    dfline.dist = (dfline.dist.astype(int)*2/100).round(2)
    dfline = dfline.drop(columns= ['properties','x','y'])
    t = dfline.Class.str.split('-')
    dfline['ID'] = t.str[1] + dfline.end.astype(str) + '-' + t.str[0] + dfline.start.astype(str)
    dfline = dfline.sort_values('ID').reset_index(drop = True)
    
    Comb_All = dfslot.groupby('Class').Name.unique().apply(list).apply(sorted).to_dict()

    DropList = ['C0','E2']
    # DropList = []
    if len(DropList) > 0 : 
        dfline = dfline[~dfline.ID.str.contains('|'.join(DropList))]
        dfslot = dfslot[~dfslot.ID.isin(DropList)]
    
    A0 = data['layers'][0]['data']
    height = data['height']
    A0 = np.array(A0).reshape(10,7)
    pas = 16
    unique = np.unique(A0)
    A0[A0 == unique[0]] = 0
    A0[A0 == unique[1]] = 1
    
    confs = pd.read_excel('test.xlsx')
    
    algo = dict(
        pop = 50,
        fitness = 'dist',
        dfslot = dfslot,
        dfline = dfline,
        epoch = 0,
        Nindiv = 0,
        Nrepro = 0,
        indivs = [],
        df = [],
        Tuyau = 'T1',
        Pompe = 'P1',
        EV = 'E1',        
        confs = confs,
        Comb = dfslot.groupby('Class').Name.unique().apply(list).apply(sorted).to_dict(),
        CombAll = Comb_All,
        dist = dfline.set_index('ID').dist.to_dict(),
        height = data['height'],
        A0 = A0,
    )
    algo = SimpleNamespace(**algo)
    return algo

def indiv_create(algo, row = None, NewCtoE = None): 
     
    dfline = algo.dfline
    D = algo.Comb    
    Clist = D['C']
    Ccount = len(D['C'])
    
    if NewCtoE is not None : CtoE = NewCtoE
    else : CtoE = np.random.choice(D['E'],Ccount)
    
    d = collections.defaultdict(list)
    for i in range(Ccount): 
        d[CtoE[i]].append(D['C'][i])
    Econnect = dict(sorted(d.items()))
    Edist = dict(sorted(d.items()))
    # Econnect = dict(collections.Counter(CtoE))
    Elist = sorted(Econnect)
    Ecount = len(Elist)      
        
    if row is not None :  
        if Ecount > row.Ecount:
            # print(D['P'],Ecount - row.Ecount)
            NewEtoP = np.random.randint(0,len(D['P']),Ecount - row.Ecount)
            EtoP = np.append(row.EtoP, NewEtoP)
        elif Ecount < row.Ecount : 
            EtoP = np.random.choice(row.EtoP,Ecount)
            # print("Ecount",EtoP)
        else :
            EtoP = row.EtoP
        # print(NewCtoE , 'avant',row.Elist,row.EtoP,'apres',Elist,EtoP,row.Ecount, Ecount)
    else : EtoP = np.random.choice(D['P'],Ecount)
    
    d = collections.defaultdict(list)
    for i in range(Ecount):      d[EtoP[i]].append(Elist[i]) 
    Pconnect = dict(sorted(d.items()))   
    # Pconnect = dict(collections.Counter(EtoP))
    Plist = sorted(Pconnect)
    Pcount = len(Plist)    
    
    List_EtoC = [['E{}-C{}'.format(start, end) for end in List] for start , List in Econnect.items()]
    List_PtoE = [['P{}-E{}'.format(start, end) for end in List] for start , List in Pconnect.items()]
        
    Name = list(itertools.chain.from_iterable(List_EtoC + List_PtoE))
    dist_Connect = (dfline.loc[dfline.ID.isin(Name), ['ID','dist']].set_index('ID').dist).to_dict()
    dist = dfline.loc[dfline.ID.isin(Name), 'dist'].sum()
    dist = round(dist,2)
    Name_txt = ','.join(Name)
    
    algo.Nindiv += 1
    col = ['D', 'Clist','CtoE','Econnect','Elist','Ecount', 'EtoP',
           'Pconnect','Plist','Pcount', 'List_EtoC','List_PtoE','dist_Connect', 'dist', 'Name','ID', 'Name_txt']
    l = [D, Clist, CtoE,Econnect,Elist,Ecount, EtoP,
         Pconnect,Plist,Pcount, List_EtoC,List_PtoE,dist_Connect, dist, Name,algo.Nindiv, Name_txt]
    # indiv = SimpleNamespace(**dict(zip(col,l)))
    indiv = dict(zip(col,l))
    algo.indivs.append(indiv)
    algo.Nrepro +=1
    
    # calcul debit
    d =  Calcul_Debit(algo ,indiv, False)
    col  = ['Pression', 'Debit','SumDebit']
    indiv.update({(c +'_s'): d[c] for c in col})
    d =  Calcul_Debit(algo ,indiv, True)
    indiv.update({(c +'_g'): d[c] for c in col})
    
    info , d = calcul_Masse_cout(indiv, algo)
    indiv.update(d)

        
    return indiv

def calcul_Masse_cout(indiv, algo): 
    dmasse = {}
    dcout = {}
    confs = algo.confs

    masse, cout = confs[confs.Name == algo.Pompe][['Masse','Cout']].values[0]
    masse, cout 
    dmasse['Pmasse'] = indiv['Ecount']*masse
    dcout['Pcout']   = indiv['Ecount']*cout

    masse, cout = confs[confs.Name == algo.Tuyau][['Masse','Cout']].values[0]
    masse, cout
    dmasse['Tmasse'] = indiv['dist']*masse
    dcout['Tcout']   = indiv['dist']*cout


    Ccount = len(algo.Comb['C'])
    masse, cout = confs[confs.Name == algo.EV][['Masse','Cout']].values[0]
    masse, cout
    dmasse['Emasse'] = Ccount*masse
    dcout['Ecout']   = Ccount*cout

    dmasse['Reservoir'] = 600
    dcout['Reservoir']  = 30    
    
    info = [dmasse, dcout]
    Masse = round(sum(dmasse.values()),2)
    Cout = round(sum(dcout.values()),2)
    
    return  info, { 'Masse' : Masse, 'Cout' : Cout}

def Reprodution(dfx, algo):
    
    c1, c2 = copy.deepcopy(dfx.CtoE.values)
    
    if (c1 != c2).any():        
        m = c1 != c2
        index = np.where(m)[0]
        
        # search for gen diff between indivs = index        
        if len(index) > 1:    
            idx = np.random.choice(index)
        elif len(index) == 1:
            idx = index  
            
        # print(c1, c2,'crossover',index, c1[idx] , c2[idx])           
        c1[idx] , c2[idx] = c2[idx] , c1[idx]
        NewCtoE = c1 , c2
        # print(NewCtoE)
        L = []
        parents = dfx.ID.tolist()
        for i in range(2):
            row = dfx.iloc[i]            
            indiv = indiv_create(algo, row,NewCtoE[i]) 
            indiv['parent'] =  parents
            L.append(indiv)
        return L 
 
def indiv_init(algo, pop):
    algo.Nindiv = 0 
    algo.indivs = []
    algo.epoch = 0
    algo.Nrepro = 0
    L = []
    for i in range(pop):        
        indiv = indiv_create(algo)        
        L.append(indiv)
    df = pd.DataFrame(L)
    df = df.drop_duplicates(subset='Name_txt')
    df = df.reset_index(drop = True)
    
    return df

def plot_(algo,dflineSelect, dfsSelect, name): 
    A0 = algo.A0.copy()
    
    for idx, row in dfsSelect.iterrows():
        A0[row.y, row.x] = row.Color + int(row.Name)   
        
    A = np.kron(A0, np.ones((16,16), dtype=int))
    fig, ax = plt.subplots(figsize = (8,8))
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

def debit(Dict_dist, group = True):
    d_EtoC = Dict_dist['EtoC']
    d_PtoE = Dict_dist['PtoE']
    p  = [-5.16e-04, -1.54e-02, 4.87]
    coef_E  = 7.64e-04
    coef_C  = 0.036
    coef_d  = 2.35e-04    
    
    A = coef_E + d_EtoC * coef_d + coef_C 
    Z =( A**-0.5).sum() if group else A**-0.5
    As , Bs, Cs = p[0] - (coef_d * d_PtoE) - 1/(Z**2), p[1] , p[2]
    delta = (Bs**2) - (4 * As * Cs)
    Qt  = np.array((- Bs - delta**0.5)/(2*As))
    Pt = np.array(Qt**2 / Z**2)
    a0 = p[0] * (Qt**2) + p[1] * Qt + p[2] - Pt
    Qi = (Pt / A)**0.5
    Pi = coef_C * (Qi**2)
    key = ['Qt','Pt','Qi','Pi']
    val = [Qt, Pt, Qi, Pi]
    val = [v.round(1) for v in val]
    return dict(zip(key,val))

def Calcul_Debit(algo ,indiv, group):
    Econnect = indiv['Econnect']
    Pconnect = indiv['Pconnect']
    EtoP = indiv['EtoP']
    Pression = []
    Debit = []
    Data = {}
    # on loop sur chaque EV pour remonter pompe EV capteur 
    for i, (e,Clist) in enumerate(Econnect.items()):
        p = EtoP[i]
        name = 'P{}-E{}'.format(p,e)
        dc = np.array([algo.dist['E{}-C{}'.format(e,c)] for c in Clist])
        dp = algo.dist['P{}-E{}'.format(p,e)]
        info = [i,e,Clist, p, dc, dp]
        Dict_dist = {'EtoC': dc,'PtoE':dp }
        res = debit(Dict_dist, group)
        Data[name] = res
        Pression = Pression + list(res['Pi'])
        Debit = Debit + list(res['Qi'])
    SumDebit = round(sum(Debit),1)
    # keys = ['info','Data','Pression','Debit','SumDebit']
    # vals = [info, Data,Pression, Debit, SumDebit] 
    keys = ['Pression','Debit','SumDebit']
    vals = [Pression, Debit, SumDebit] 
    return dict(zip(keys,vals))



