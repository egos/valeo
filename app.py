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

Col_drop_1 = ['Clist','D','Name','Name_txt','dist_Connect','List_EtoC','List_PtoE']
Col_drop_2 = ['Pression_s', 'Debit_s','SumDebit_s'] + ['Pression', 'Debit','SumDebit']
Col_drop_2 = ['Pression_s', 'Debit_s'] + ['Pression_g', 'Debit_g']
Col_drop_2 = ['Debit_s','SumDebit_s'] + ['Debit_g','SumDebit_g']
Col_drop_2 = ['Debit_s'] + ['Debit_g']
Col_drop   = Col_drop_1 + Col_drop_2

keydrop= ["confs", "dfslot","dfline","indivs","df",'A0']

if 'algo' not in session_state: 
    print('init')
    algo = load_data_brut(file)
    algo.df = indiv_init(algo, pop)
    session_state['algo'] = algo
else : 
    algo = session_state['algo']
with st.expander('Options : 🖱️ press submit for change take effect', True):
    with st.form("Pattern"):  
        select = st.multiselect('Pattern',algo.CombAll, algo.CombAll)
        select = [s for s in algo.CombAll if s not in select]
        if select == [] : select = None
        submitted = st.form_submit_button("Submit & Reset")        
        if submitted:
            algo = load_data_brut(file, select)
            algo.df = indiv_init(algo, pop)
            session_state['algo'] = algo
            print('submitted : Pattern')
    with st.form("Elements Type"):         
        Clist = algo.Clist     
        Ctype = algo.DataCategorie['Nozzle']['Unique']        
        # ListCategorie = ['Pompe', 'Nozzle']
        col = st.columns(len(Clist) + 1)
        Nozzles = []
        for i in range(len(Clist)):            
            c = Clist[i]
            # print(Ctype, i, c)
            Nozzle =  col[i].selectbox('C' + str(c),Ctype, index = 0)
            Nozzles.append(Nozzle)
        Pompe = "Pa" if not col[len(Clist)].checkbox('pompe 3') else "Pc"
            
        submitted = st.form_submit_button("Submit & Reset")      
        if submitted:
            algo = load_data_brut(file, select)
            algo.Nozzle = Nozzles
            algo.Pompe  = [Pompe] * len(Comb['P'])
            algo.df = indiv_init(algo, pop)
            session_state['algo'] = algo
            print('submitted : Elements Type')

    
menu = st.sidebar.radio("MENU", ['Input','Algo'], index  = 1)

    
if menu == 'Input':
    st.subheader('INPUT')
    
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
        c1,c2,c3,c4,c5 = st.columns(5)
        algo.pop   = c1.number_input(label  = 'indiv pop init',value = 10, min_value = 1,  max_value  = 1000,step = 10)
        iterations = c2.number_input(label  = 'iterations / run',value = 1, min_value = 1,  max_value  = 10,step = 1)
        algo.fitness = c3.selectbox('fitness',['dist','Masse','Cout'])
        txt = "pourcentage d'indiv selectionné avec la meilleur fitness pour crossover => donne 2 enfants"
        algo.crossover = c4.number_input(label  = 'Crossover % ',value = int(algo.crossover*100), min_value = 0,  max_value  = 100,step = 10, help =txt)/100
        txt = "pourcentage d'indiv selectionné avec la meilleur fitness pour mutation  => donne 1 enfants"
        algo.mutation = c5.number_input(label  = 'Mutation % ',value = int(algo.mutation*100), min_value = 0,  max_value  = 100,step = 10, help =txt)/100
        session_state['algo'] = algo
        
        c1,c2,c3 = st.columns(3)  
            
        if c1.button('RESET'):
            print('Params : RESET')
            # algo = load_data_brut(file)
            algo.df = indiv_init(algo, algo.pop)
            session_state['algo'] = algo    
      
        if c2.button('RUN'):
            print("Params : RUN")
            L = [] 
            #Crossover
            for i in range(iterations):
                algo.epoch +=1
                df1 = algo.df
                df1 = df1.sort_values(algo.fitness).reset_index(drop = True)
                idxmaxCross = int(df1.shape[0]*algo.crossover)
                idxmaxMuta  = int(df1.shape[0]*algo.mutation)
                if idxmaxCross <  2 : idxmaxCross = 2            
                if idxmaxMuta ==  0 : idxmaxMuta = 1
                Ncross   = int(idxmaxCross/2)
                Nmuta    = int(idxmaxMuta)
                # if Ncross == 0 : st.warning("⚠️ crossover %  too low for pop = {}".format(algo.pop))  
                # if Nmuta == 0  : st.warning("⚠️ mutation  %  too low for pop = {}".format(algo.pop))                      
                Lcross = df1[:idxmaxCross].index.values
                np.random.shuffle(Lcross)
                Lmuta = df1[:idxmaxMuta].index.values
                np.random.shuffle(Lmuta)                
                print(Lcross,idxmaxCross, Ncross,Lmuta, idxmaxMuta, Nmuta)
                L = [] 
                for n in range(Ncross):    
                    i1 , i2 = Lcross[n*2] , Lcross[n*2 + 1]
                    dfx = df1.loc[[i1,i2]].copy()
                    L2 = AG_CrossOver(dfx, algo)
                    if L2 is not None : L += L2  
                for n in range(Nmuta):
                    row = df1.loc[Lmuta[n]].copy()
                    indiv = Mutation(row, algo)
                    L.append(indiv)
            
                dfx = pd.DataFrame(L)
                algo.df = pd.concat([df1, dfx]).drop_duplicates(subset='Name_txt').reset_index(drop = True)
                session_state['algo'] = algo 

        df1 = algo.df
        df1 = df1.sort_values(algo.fitness).reset_index(drop = True)
        dfs = algo.dfslot
        dfline = algo.dfline


        # Col_drop = Col_drop_1

    st.write('Pattern : ',str(algo.Comb) ,' ---------- indivs Total : ',
             str(algo.Nrepro), ' ---- indivs  unique: ' , str(df1.shape[0]),
             '-params :',algo.pop,algo.epoch,algo.fitness, algo.crossover, algo.mutation)
    
    if st.sidebar.checkbox("Show Conf files :"):
        
        d = {k : v for k,v in vars(algo).items() if k not in keydrop}
        st.sidebar.json(d, expanded=True) 
    
    with st.expander("Dataframe", True):
        # st._legacy_dataframe(df1.drop(columns= Col_drop).astype(str), height  = 800)
        st.dataframe(df1.drop(columns= Col_drop).astype(str))
                
    with st.expander("Plot", False):    
        c1, c2 = st.columns([0.3,0.7])   
        ListSelectbox = df1.index
        index = c2.selectbox('individu',ListSelectbox)
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

        
        # if c2.checkbox('ALL combinaison') : 
        #     dflineSelect = dfline.copy()
        #     dfsSelect = dfs.copy()

        fig = plot_(algo,dflineSelect, dfsSelect, str(row.name) + ' : ' + row.Name_txt + ' / '+ str(row.dist))     
        c2.table(row.astype('string'))
            
        c1.pyplot(fig)