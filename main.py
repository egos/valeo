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

#conf file algo keys
keydrop= ['Nvals',"confs","dfcapteur", "dfslot","dfline","indivs",
          "df",'dfmap','A0','DataCategorie', 'DictLine','DictPos','dist','durite',
          'duriteType','duriteVal']

ColSysteme = ['Clist','Name','List_EtoC','List_PtoE','duriteVal']
ColAlgo    = ['CtoE','EtoP','Econnect','Elist','Ecount','Pconnect','Plist','Pcount']
ColResults = ['PressionList','DebitList','dist_Connect']
ColBus     = ['BusName','BusDist']
# col pour astype int
ColDfVal   = ['Ecount','Pcount', 'dist','ID','SumDebit_s','SumDebit_g',
            'Masse', 'Cout','Alive','Group']
ColPompe = ['Ptype0', 'Ptype', 'PtypeCo']
ColBase =  ['ID', 'Option', 'Pompes', 'Esplit', 'Debit','dist', 'Masse', 'Cout',
            'fitness','Epoch', 'Alive','parent','Name_txt']

menu = st.sidebar.radio("MENU", ['Input','Algo'], index  = 1)

if 'algo' not in session_state: 
    print(' ')
    print('BEGIN')
    File = {'SheetMapName' : 'map', 'uploaded_file' : None, 'DistFactor' : 1.0}
    
    algo = load_data_brut(File)
    # algo.df = indiv_init(algo, pop)
    session_state['algo'] = algo
else : 
    print('reload')
    algo = session_state['algo']
      
with st.expander('input & pathfinding : 🖱️ press submit for change take effect', True):

    with st.form('Map excel sheet name'):
        c1, c2 = st.columns([0.6,0.4])
        
        uploaded_file = c1.file_uploader('drag & drop excel : confs & map files',type="xlsx") 
        SheetMapName  = c2.text_input(label = "map excel sheet name", value = algo.SheetMapName) 
        DistFactor = c2.number_input(label = 'DistFactor ==> metre', value = 1.0)
        File = {'SheetMapName' : SheetMapName, 'uploaded_file' : uploaded_file, 'DistFactor' : DistFactor}
        submitted = st.form_submit_button("Submit & Reset")
        if submitted: 
            print('submitted Map')
            # session_state.clear()
            algo = load_data_brut(File)
            # algo.df = indiv_init(algo, pop)
            session_state['algo'] = algo          
    st.download_button(label='📥 download input data template',
                            data= export_excel(algo),
                            file_name= 'input.xlsx') 
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
    
    dfline = pd.DataFrame(algo.dfline)
    print(dfline.columns)
    dfline['path'] = dfline.path.astype(str)
    dfslot = pd.DataFrame(algo.DictPos).T
    dfslot.columns = ('y','x')    
    c1, c2 = st.columns([0.8,0.2])  
    # c1.table(dfline.style.format(precision = 2))
    c1.table(dfline)
    c2.table(dfslot)  
      
if menu == 'Algo': 
    # with st.expander("limite pression par slot pompe", True):

    
    with st.expander("Capteurs", True):    
            
        Clist = algo.Clist     
        Nclist = list(range(len(Clist)))
        Ctype = algo.DataCategorie['Nozzle']['Unique']        

        LenStCol = 5 if len(Clist) > 5 else len(Clist)
        col = st.columns(LenStCol)
        Nozzles = []
        Nozzlelimits = []
        d = collections.defaultdict(list)
        for i in range(len(Clist)):            
            c = Clist[i]
            idx = i%5
            stCol = col[idx]
            Nozzle =  stCol.selectbox('C' + str(c),Ctype, index = algo.Nature0[i])            
            Nozzles.append(Nozzle)
            Gr = stCol.selectbox(str(c),Nclist, index = algo.Group0[i], label_visibility  = "collapsed") 
            d[Gr].append(i)  
            Nozzlelimit = stCol.number_input(str(c),value  = algo.Limit0[i],step =0.1, label_visibility  = "collapsed") 
            Nozzlelimits.append(Nozzlelimit)          
        
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
        GroupDict = dict(sorted(GroupDict.items())) 
        GroupDict = np.array(list(GroupDict.values()))
        # algo = load_data_brut(file)
        # algo.GroupDict = dict(sorted(d2.items())) 
        algo.GroupDict = GroupDict
        algo.Group = ~(GroupDict == 0).all()
        
        algo.Nozzles = Nozzles
        Nvals   = [algo.DataCategorie['Nozzle']['Values'][n]['a'] for n in Nozzles]
        algo.Nvals = dict(zip(Clist, Nvals))
        algo.Nozzlelimits = np.array(Nozzlelimits)
        
        
    with st.expander("Pompe limite & options", True):               
        SplitText = 'si no group = Deactivate'
        c1 ,c2 ,c3 ,c4, c5 = st.columns(5)
        Npa = int(c1.number_input(label= 'Npa',key='Npa' , value= 4))    
        Npc = int(c2.number_input(label= 'Npc',key='Npc' , value= 0))  
        PompeB = c3.checkbox(label= 'Pompe B', help = 'si group = False')
        Split  = c4.selectbox('Split',['Deactivate','Auto','Forced'] , help = 'si no group = Deactivate')
        BusActif  = c5.checkbox(label = 'Bus')      
        
        algo.PompeB = PompeB & (not algo.Group)
        if not algo.Group : Split = 'Deactivate'
        algo.Split  = Split
        algo.BusActif = BusActif & (not algo.Group) & (not algo.PompeB)        
        algo.Npa = Npa
        algo.Npc = Npc
        algo.Pmax = Npa + Npc
        algo.PompesSelect = ['Pa'] * algo.Npa + ['Pc'] * algo.Npc
        
        ListPlimSlot = algo.ListPlimSlot
        stCol = st.columns(len(ListPlimSlot))
        New_ListPlimSlot = []
        for i in range(len(ListPlimSlot)):
            v = stCol[i].number_input(label = 'P{}'.format(i), min_value=0, value = ListPlimSlot[i], key = 'ListPlimSlot' + str(i))
            New_ListPlimSlot.append(v)   
        algo.ListPlimSlot = New_ListPlimSlot    
        
     
     
    with st.expander("indivs params", True):
        c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
        algo.pop   = c1.number_input(label  = 'indiv pop init',value = 10, min_value = 1,  max_value  = 1000,step = 10)
        iterations = c2.number_input(label  = 'iterations / run',value = 1, min_value = 1,  max_value  = 1000,step = 1)
        # algo.fitness = c3.selectbox('fitness',['dist','Masse','Cout'])
        txt = "indivs selectionnés avec la meilleur fitness pour crossover => 2 enfants"
        algo.crossover = c3.number_input(label = 'Crossover',value = int(algo.crossover), min_value = 0, max_value  = 100,step = 10, help =txt)
        txt = "indivs selectionnés avec la meilleur fitness pour mutation  => 1 enfants"
        algo.mutation  = c4.number_input(label = 'Mutation', value = int(algo.mutation), min_value = 0, max_value  = 100,step = 10, help =txt)
        
        txt = "Maximum de pompe disponible"
        options = list(range(1,len(algo.Comb['E']) +1))
        
        
        ListFitness = ['dist','Masse','Cout']
        c = st.columns(3)
        ListRes = []
        for i in range(3): 
            fit = ListFitness[i]
            res  = c[i].number_input(label = fit,value = int(algo.fitnessCompo[i]*100), min_value = 0, max_value  = 100,step = 10, help ="")
            res /= 100
            ListRes.append(res)
        algo.fitnessCompo = np.array(ListRes)
                    
        default =  "E0-C0,E1-C1,E2-C2,E3-C3,P0-E0,P0-E1,P1-E2,P1-E3"
        NameIndiv = st.text_input('reverse name_txt to indiv', default,help = default)
        NameIndiv = NameIndiv.replace(" ",'').split(';')
        
    session_state['algo'] = algo        
    st.write('Group = ',algo.Group, ', Pompe_B = ',algo.PompeB , ', Split = ', algo.Split, ', BUS = ', algo.BusActif)   
    c1,c2,c3,c4 = st.columns(4) 
    
    KeepResults =  c1.checkbox('Keep results') 
            
    if c2.button('RESET'):
        print('Params : RESET')              
        if (NameIndiv != ['']):
            L = []
            for Name in NameIndiv: 
                if Name != '' :
                    indiv = Indiv_reverse(Name,algo)             
                    L.append(indiv)
            df = pd.DataFrame(L)
            df = df.reset_index(drop = True)
        else : 
            df = indiv_init(algo, algo.pop)
        if KeepResults:
            algo.df = pd.concat([df,algo.df]) 
        else :
            algo.df = df.drop_duplicates(subset='Name_txt')
        session_state['algo'] = algo
                        
    if c4.button('RUN'):
        print("Params : RUN")   
        # latest_iteration = st.empty()                 
        my_bar = st.empty()     
        for i in range(iterations):
            # latest_iteration.text(f'{iterations - i} iterations left')
            my_bar.progress((i+1)/iterations)
            algo.epoch +=1
            df0 = algo.df
            df0 = df0.sort_values('fitness').reset_index(drop = True)
            df1 = df0[df0.Alive].copy()
            idxmaxCross = int(algo.crossover)
            idxmaxMuta  = int(algo.mutation)
            # if idxmaxCross <  2 : idxmaxCross = 2            
            # if idxmaxMuta ==  0 : idxmaxMuta = 1
            if idxmaxCross >  len(df1) : idxmaxCross = len(df1)            
            if idxmaxMuta  >  len(df1) : idxmaxMuta  = len(df1)
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
          
    if c3.button('recalculation', help = 'Pompe B , Bus , debit / pression , masse cout , fitness Alive'):
        indivs = []
        for indiv in algo.indivs :
            indiv = Gen_Objectif(algo, indiv)
            indivs.append(indiv)  
        algo.indivs = indivs
        df = pd.DataFrame(indivs) 
        df = df.reset_index(drop = True)
        algo.df = df.drop_duplicates(subset='Name_txt')
        session_state['algo'] = algo   
    df1 = algo.df.copy()
 
    Col_drop = []
    c1,c2,c3,c4, c5, c6  = st.columns(6)   
    algo.Plot = c1.checkbox('Show  figure & details', value = False, help = "desactiver cette option ameliore les performances")
    if not c2.checkbox('Algo'   , value = False, help = str(ColAlgo)) : Col_drop += ColAlgo
    if not c3.checkbox('System' , value = False, help = str(ColSysteme)) : Col_drop += ColSysteme
    if not c4.checkbox('Results', value = False, help = str(ColResults)) : Col_drop += ColResults 
    if not c5.checkbox('BUS'    , value = False, help = str(ColBus)) : Col_drop += ColBus       
    if not c6.checkbox('Pompe'  , value = False, help = str(ColPompe)) : Col_drop += ColPompe  
    
    if len(df1)>0 :
    
        df1 = df1.sort_values(['fitness']).reset_index(drop = True)
        dfline = algo.dfline    
        
        DictParams = dict(
            Pattern = algo.Comb,
            indivs_total = algo.Nrepro,
            indivs_unique = df1.shape[0],
            # pop_init = algo.pop,
            epoch = algo.epoch,
            # fitnessCompo = algo.fitnessCompo*100,
            # crossover = algo.crossover,
            # mutation = algo.mutation,            
        )
        st.write(str(DictParams))
        # st.metric(label="create", value=algo.Nrepro, delta=-0.5,)
        
        col1 = df1.columns[~df1.columns.isin(ColBase)]
        df1 = df1[ColBase + df1.columns[~df1.columns.isin(ColBase)].tolist()]
        dfx = df1.drop(columns= Col_drop)
        for col in dfx.columns:
            if col not in ColDfVal :
                dfx[col]= dfx[col].astype(str)
            else : 
                if col == 'dist' : dfx[col]= (100*dfx[col]).astype(int)    
                
        with st.expander("Dataframe", True):
            st.dataframe(dfx, use_container_width  =True)                    
        with st.expander("Figures", True): 
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
                    if row.Option == 'Bus' :   SelectLine = row.BusName

                    # fig = plot_(algo,dflineSelect, dfsSelect, str(row.name) + ' : ' + row.Name_txt + ' / '+ str(row.dist))     
                    col[i].dataframe(row.drop(labels= Col_drop).astype('str'),  use_container_width  =True)                    
                    fig = new_plot(algo, SelectLine, SelectSlot)
                    col[i].pyplot(fig)

# c3.pyplot(fig)
                
                