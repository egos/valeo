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
st.set_page_config(page_title = "VALEO_AG_IHM", layout="wide")
# Comb = 	{'C': [0, 1, 2, 3], 'E': [0, 1, 2], 'P': [0, 1]}
pop = 10      

Col_drop_1 = ['Clist','Name','Name_txt','dist_Connect','List_EtoC','List_PtoE']
Col_drop_2 = ['Pression_s', 'Debit_s','SumDebit_s'] + ['Pression', 'Debit','SumDebit']
Col_drop_2 = ['Pression_s', 'Debit_s'] + ['Pression_g', 'Debit_g']
Col_drop_2 = ['Debit_s','SumDebit_s'] + ['Debit_g','SumDebit_g']
Col_drop_2 = ['CtoE','EtoP']
Col_drop   = []

ColSysteme = ['Clist','Name','Name_txt','dist_Connect','List_EtoC','List_PtoE']
ColAlgo = ['CtoE','EtoP','Econnect','Elist','Ecount','Pconnect','Plist','Pcount']
ColResults = ['dist', 'PressionList','DebitList']
keydrop= ["confs", "dfslot","dfline","indivs","df",'A0','DataCategorie', 'DictLine','DictPos','dist']
ColDfVal = ['Ecount','Pcount', 'dist','ID','SumDebit_s','SumDebit_g','Masse', 'Cout','Alive','Group', 'Vg', 'Vp','Vnp']

menu = st.sidebar.radio("MENU", ['Input','Algo'], index  = 1)

if 'algo' not in session_state: 
    print(' ')
    print('BEGIN')
    file = {'SheetMapName' : 'map', 'uploaded_file' : None}
    
    algo = load_data_brut(file)
    algo.df = indiv_init(algo, pop)
    session_state['algo'] = algo
else : 
    print('reload')
    algo = session_state['algo']
      
with st.expander('Options : 🖱️ press submit for change take effect', True):

    with st.form('Map excel sheet name'):
        c1, c2 = st.columns([0.6,0.4])
        
        uploaded_file = c1.file_uploader('drag & drop excel : confs & map files',type="xlsx") 
        SheetMapName  = c2.text_input(label = "map excel sheet name", value = algo.SheetMapName) 
        file = {'SheetMapName' : SheetMapName, 'uploaded_file' : uploaded_file}
        submitted = st.form_submit_button("Submit & Reset")
        if submitted: 
            print('submitted Map')
            algo = load_data_brut(file)
            algo.df = indiv_init(algo, pop)
            session_state['algo'] = algo
            
    with st.form("Slots"):    
            
        Clist = algo.Clist     
        Nclist = list(range(len(Clist)))
        Ctype = algo.DataCategorie['Nozzle']['Unique']        

        
        c1, c2 = st.columns(2)
        Npa = int(c1.number_input(label= 'Npa',key='Npa' , value= 2))    
        Npc = int(c2.number_input(label= 'Npc',key='Npc' , value= 2))    
        
        col = st.columns(len(Clist))
        Nozzles = []
        d = collections.defaultdict(list)
        
        for i in range(len(Clist)):            
            c = Clist[i]
            Nozzle =  col[i].selectbox(str(c),Ctype, index = 0)            
            Nozzles.append(Nozzle)
            Gr = col[i].selectbox(str(c),Nclist, index = 0, label_visibility  = "hidden") 
            d[Gr].append(i)  
             
            
        # creation DictGroup , les group a 1 elem ==> gr 0
        d = dict(sorted(d.items())) 
        
        # d2 = collections.defaultdict(list)  
        GroupDict = {}    
        for key , val in d.items():
            if len(val) > 1 : 
                # d2[key] = val
                for i in val : 
                    GroupDict[i] = key
            else : 
                # d2[0].append(val[0]) 
                GroupDict[val[0]] = 0
        # d2[0] = sorted(d2[0])     
        submitted = st.form_submit_button("Submit & Reset")      
        GroupDict = dict(sorted(GroupDict.items())) 
        GroupDict = np.array(list(GroupDict.values()))
        if submitted:
            
            print('submitted Slots') 

            algo = load_data_brut(file)
            # algo.GroupDict = dict(sorted(d2.items())) 
            algo.GroupDict = GroupDict
            algo.Group = ~(GroupDict == 0).all()

            algo.Nozzles = Nozzles
            Nvals   = [algo.DataCategorie['Nozzle']['Values'][n]['a'] for n in Nozzles]
            algo.Nvals = dict(zip(Clist, Nvals))
            
            algo.Npa = Npa
            algo.Npc = Npc
            algo.Pmax = Npa + Npc
            algo.PompesSelect = ['Pa'] * algo.Npa + ['Pc'] * algo.Npc
            
            algo.df = indiv_init(algo, pop)
            session_state['algo'] = algo
            print('submitted : Elements Type')
if st.sidebar.checkbox("Show Conf files :"):        
    d = {k : v for k,v in vars(algo).items() if k not in keydrop}
    s = pd.Series(d).rename('Val').astype(str)
    s.index= s.index.astype(str)
    # st.sidebar.json(d, expanded=True) 
    st.sidebar.table(s)
    
if menu == 'Input':
    st.subheader('INPUT')
    
    Col1 = ['a','b','c']
    Format = dict(zip(Col1,["{:.2e}"]))
    Format.update(dict(zip(['Masse','Cout'],["{:.0f}",  "{:,.2f}"])))
    dfInput = algo.confs.copy()  
    dfInput['Actif'] = dfInput['Actif'] ==1

    SelectLine = algo.DictLine.keys()
    SelectSlot = algo.DictPos.keys()
    
    fig = new_plot(algo, SelectLine, SelectSlot)
    c1, c2 = st.columns([0.7,0.3])  
    c1.table(dfInput.style.format(Format, na_rep=' '))
    c2.pyplot(fig) 
    
    dfline = pd.DataFrame(algo.DictLine).T[['dist','path']]
    dfline['path'] = dfline.path.astype(str)
    dfslot = pd.DataFrame(algo.DictPos).T
    dfslot.columns = ('y','x')    
    c1, c2 = st.columns([0.8,0.2])  
    c1.table(dfline.style.format(precision = 2))
    c2.table(dfslot)  
      
if menu == 'Algo':  
    
    with st.expander("Params", True):
        c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
        algo.pop   = c1.number_input(label  = 'indiv pop init',value = 10, min_value = 1,  max_value  = 1000,step = 10)
        iterations = c2.number_input(label  = 'iterations / run',value = 1, min_value = 1,  max_value  = 1000,step = 1)
        algo.fitness = c3.selectbox('fitness',['dist','Masse','Cout'])
        txt = "indivs selectionnés avec la meilleur fitness pour crossover => 2 enfants"
        algo.crossover = c4.number_input(label = 'Crossover',value = int(algo.crossover), min_value = 0, max_value  = 100,step = 10, help =txt)
        txt = "indivs selectionnés avec la meilleur fitness pour mutation  => 1 enfants"
        algo.mutation  = c5.number_input(label = 'Mutation', value = int(algo.mutation), min_value = 0, max_value  = 100,step = 10, help =txt)
        txt = "limite de pression pour nettoyer les capteurs"
        algo.Nlim = c6.number_input(label  = 'Pression limite', value = algo.Nlim, min_value = 0.0, max_value = 5.0, step = 0.1, help =txt)
        txt = "Maximum de pompe disponible"
        options = list(range(1,len(algo.Comb['E']) +1))
        # algo.Pmax = c7.selectbox(label  = 'Pompe limite',options = options,index = len(options)-1,  help =txt)
        session_state['algo'] = algo
        
        txt = 'E1-C0,E1-C1,E1-C2,E1-C3,P1-E1'
        default =  'E1-C0,E1-C1,E1-C2,E1-C3,P1-E1'
        default =  ''
        NameIndiv = st.text_input('E1-C0,E1-C1,E1-C2,E1-C3,P1-E1', default,help = txt)
        
        
        c1,c2,c3,c4, c5, c6  = st.columns(6)              
        if c1.button('RESET'):
            print('Params : RESET')
            algo.df = indiv_init(algo, algo.pop)
            session_state['algo'] = algo             
        if c2.button('RUN'):
            print("Params : RUN")   
            latest_iteration = st.empty()                 
            my_bar = st.empty()     
            for i in range(iterations):
                latest_iteration.text(f'{iterations - i} iterations left')
                my_bar.progress((i+1)/iterations)
                algo.epoch +=1
                df0 = algo.df
                df0 = df0.sort_values(algo.fitness).reset_index(drop = True)
                df1 = df0[df0.Alive].copy()
                idxmaxCross = int(algo.crossover)
                idxmaxMuta  = int(algo.mutation)
                # if idxmaxCross <  2 : idxmaxCross = 2            
                # if idxmaxMuta ==  0 : idxmaxMuta = 1
                if idxmaxCross >  len(df1) : idxmaxCross = len(df1)            
                if idxmaxMuta  >  len(df1) : idxmaxMuta = len(df1)
                Ncross = int(idxmaxCross/2)
                Nmuta  = int(idxmaxMuta)            
                Lcross = df1[:idxmaxCross].index.values
                np.random.shuffle(Lcross)
                Lmuta = df1[:idxmaxMuta].index.values
                np.random.shuffle(Lmuta)           
                # print(len(df1) , Lcross,idxmaxCross, Ncross,Lmuta, idxmaxMuta, Nmuta)
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
                algo.df = pd.concat([df0, dfx]).drop_duplicates(subset='Name_txt').reset_index(drop = True)
                session_state['algo'] = algo 
               
        algo.Plot = c3.checkbox('Show  figure', value = False, help = "desactiver cette option ameliore les performances")
        if c4.checkbox('Hide Algo Columns', value = True, help = str(ColAlgo))       : Col_drop += ColAlgo
        if c5.checkbox('Hide System Columns', value = True, help = str(ColSysteme))  : Col_drop += ColSysteme
        if c6.checkbox('Hide Results Columns', value = True, help = str(ColResults)) : Col_drop += ColResults
                              
        df1 = algo.df
        df1 = df1.sort_values([algo.fitness]).reset_index(drop = True)
        dfline = algo.dfline
        NameIndiv = NameIndiv.replace(" ",'').split(';')
        if NameIndiv != ['']:
            L = []
            for Name in NameIndiv: 
                if Name != '' :
                    indiv = Indiv_reverse(Name,algo)             
                    L.append(indiv)
            df1 = pd.DataFrame(L)
            df1 = df1.drop_duplicates(subset='Name_txt')
            df1 = df1.reset_index(drop = True)

    st.write('Pattern : ',str(algo.Comb) ,' ---------- indivs Total : ',
             str(algo.Nrepro), ' ---- indivs  unique: ' , str(df1.shape[0]),
             '-params :',algo.pop,algo.epoch,algo.fitness, algo.crossover, algo.mutation)
        
    with st.expander("Dataframe", True):
        # st._legacy_dataframe(df1.drop(columns= Col_drop).astype(str), height  = 800)
        dfx = df1.drop(columns= Col_drop)
        for col in dfx.columns:
            if col not in ColDfVal : 
                dfx[col]= dfx[col].astype(str)
            else : 
                if col == 'dist' : dfx[col]= (100*dfx[col]).astype(int)
        st.dataframe(dfx, use_container_width  =True)
                
    with st.expander("Plot", False): 
        pass
        if algo.Plot: 
            c1 , c2 = st.columns(2)
            MinCol = 3 if  len(df1) >= 3 else len(df1)
            Ncol = c1.number_input(label  = 'indiv number',value = MinCol, min_value = 1,  max_value  = len(df1),step = 1)
            # Ncol = 3 if len(df1) >=3 else len(df1)
            Ncolmin  = 4 if Ncol < 4 else Ncol
            col = st.columns(Ncolmin)
                    
            for i in range(Ncol):   
                c1, c2 = st.columns([0.3,0.7])   
                ListSelectbox = df1.index
                index = col[i].selectbox('indiv detail ' + str(i),options = ListSelectbox, index = i)
                row = df1.loc[index]
                        
                ElemsList = ['Clist','Elist','Plist']
                Elems = ['C','E','P']
                SelectSlot = []
                List_EtoC = row.List_EtoC
                List_PtoE = row.List_PtoE
                for n in range(3):
                    SelectSlot+= ['{}{}'.format(Elems[n],i) for i in row[ElemsList[n]]]
                SelectLine = row.Name

                # fig = plot_(algo,dflineSelect, dfsSelect, str(row.name) + ' : ' + row.Name_txt + ' / '+ str(row.dist))     
                col[i].dataframe(row.drop(labels= Col_drop).astype('str'),  use_container_width  =True)                    
                fig = new_plot(algo, SelectLine, SelectSlot)
                col[i].pyplot(fig)

# c3.pyplot(fig)
                
                