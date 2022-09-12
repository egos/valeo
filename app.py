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
pop = 10
        
menu = st.sidebar.radio("MENU", ['Input','Algo'], index  = 1)

if 'algo' not in session_state: 
    algo = load_data_brut(file)
    algo.df = indiv_init(algo, pop)
    session_state['algo'] = algo
else : 
    algo = session_state['algo']
    
if menu == 'Input':
    st.header('INPUT')
    
    c1, c2, c3 = st.columns([0.3,0.3,0.4])  
    dflineSelect = algo.dfline.copy()
    dfsSelect = algo.dfslot.copy()

    fig = plot_(algo,dflineSelect, dfsSelect, 'Input')     
    c2.table(dflineSelect.astype('string').drop(columns = ['polyline','long']))  
    c3.table(dfsSelect.astype('string'))
    c1.pyplot(fig)
    st.table(algo.confs)
      
if menu == 'Algo':       
    with st.expander("Params", True):
        c1,c2,c3 = st.columns(3)
        algo.pop   = c1.number_input(label  = 'indiv pop init',value = 10, min_value = 1,  max_value  = 1000,step = 10)
        algo.epoch = c2.number_input(label  = 'Iteration / run',value = 1, min_value = 1,  max_value  = 10,step = 1)
        algo.fitness = c3.selectbox('fitness',['dist','Masse','Cout'])
        session_state['algo'] = algo
        
        c1,c2 = st.columns(2)        
        if c1.button('RESET'):
            # algo = load_data_brut(file)
            algo.df = indiv_init(algo, algo.pop)
            session_state['algo'] = algo    
      
        if c2.button('RUN'):
            for i in range(algo.epoch):
                df1 = algo.df
                df1 = df1.sort_values(algo.fitness).reset_index(drop = True)
                L = []      
                List = df1[:7].index.values
                np.random.shuffle(List)
                # print(List)
                L = [] 
                for n in range(3):    
                    i1 , i2 = List[n*2] , List[n*2 + 1]
                    # print(n,i1,i2)
                    dfx = df1.loc[[i1,i2]].copy()
                    L2 = Reprodution(dfx, algo)
                    if L2 is not None :  
                        L += L2   
                        # algo.Nrepro +=1
                dfx = pd.DataFrame(L)
                algo.df = pd.concat([df1, dfx]).drop_duplicates(subset='Name_txt').reset_index(drop = True)
                session_state['algo'] = algo 

        df1 = algo.df
        df1 = df1.sort_values(algo.fitness).reset_index(drop = True)
        dfs = algo.dfslot
        dfline = algo.dfline

        Col_drop_1 = ['Clist','D','Name','Name_txt','dist_Connect','List_EtoC','List_PtoE']
        Col_drop_2 = ['Pression_s', 'Debit_s','SumDebit_s'] + ['Pression', 'Debit','SumDebit']
        Col_drop_2 = ['Pression_s', 'Debit_s'] + ['Pression_g', 'Debit_g']
        Col_drop   = Col_drop_1 + Col_drop_2
        # Col_drop = Col_drop_1

    st.write('Pattern : ',str(algo.Comb) ,' ---------- N indivs Total : ',  str(algo.Nrepro), ' ---- indivs : ' , df1.shape)
    with st.expander("Dataframe", True):
        # st._legacy_dataframe(df1.drop(columns= Col_drop).astype(str), height  = 800)
        st.dataframe(df1.drop(columns= Col_drop).astype(str))
                
    with st.expander("Plot", False):    
        c1, c2 = st.columns([0.3,0.7])   
        ListSelectbox = df1.index
        index = st.selectbox('individu',ListSelectbox)
        row = df1.loc[index]
        
        
        ElemsList = ['Clist','Elist','Plist']
        Elems = ['C','E','P']
        IDSelects = []
        List_EtoC = row.List_EtoC
        List_PtoE = row.List_PtoE
        for n in range(3):
            IDSelects+= ['{}{}'.format(Elems[n],i) for i in row[ElemsList[n]]]

        dflineSelect = dfline[dfline.ID.isin(row.Name)].copy()
        dfsSelect    = dfs[dfs.ID.isin(IDSelects)].copy()

        
        if c2.checkbox('ALL combinaison') : 
            dflineSelect = dfline.copy()
            dfsSelect = dfs.copy()

        fig = plot_(algo,dflineSelect, dfsSelect, str(row.name) + ' : ' + row.Name_txt + ' / '+ str(row.dist))     
        c2.table(row.astype('string'))
            
        c1.pyplot(fig)