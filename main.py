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
import plotly.express as px
import time
import pickle
from types import SimpleNamespace

# $C^{k-1}_{n+k-1} = \frac{(n+k-1)!}{n! (k-1)!}$
st.set_page_config(page_title = "VALEO_AG_IHM", layout="wide")
pop = 10      

#conf file algo keys
keydrop= ['Nvals',"confs","dfcapteur", "dfslot","dfline","indivs",
          "df",'dfmap','A0','DataCategorie', 'DictLine','DictPos','dist','durite',
          'duriteType','duriteVal']

# ColSysteme = ['Clist','Name','List_EtoC','List_PtoE','duriteVal']
# ColAlgo    = ['CtoE','EtoP','Econnect','Elist','Ecount','Pconnect','Plist','Pcount']
# ColResults = ['PressionList','DebitList','dist_Connect','DetailsMasse','DetailsCout']
# ColBus     = ['BusName','BusDist', 'Esplit', 'EvSum']
# # col pour astype int
ColDfVal   = ['Ecount','Pcount', 'dist','ID','SumDebit_s','SumDebit_g',
            'Masse', 'Cout','Alive','Group']
# ColPompe = ['Ptype0', 'Ptype', 'PtypeCo','PompesCo', 'PompeSum']
ColBase =  ['ID','Pconnect','Ptypes' ,'Debit','dist', 'Masse', 'Cout',
            'fitness','Epoch', 'Alive','parent','Name_txt','PressionList','DebitList']

today = time.strftime("%Y%m%d")
print(today)

if 'algo' not in session_state: 
    print(' ')
    print('BEGIN')
    File = {'SheetMapName' : 'map', 'uploaded_file' : None, 'DistFactor' : 0.1}
    algo = load_data_brut(File)
    session_state['algo'] = algo
else : 
    print('reload')
    algo = session_state['algo']


title ='input & pathfinding : 🖱️ press submit for change take effect'
# with st.sidebar:
#     # c1, c2 = st.columns(2)
#     uploaded_file = st.file_uploader('LOAD Save.pickle') 
#     if uploaded_file is not None: 
#         if 'load' not in session_state:
#             print(uploaded_file)
#             SaveAlgo = pickle.load(uploaded_file)
#             algo = SimpleNamespace(**SaveAlgo)
#             session_state['algo'] = algo
#             session_state['load'] = True 
#     PickleDonwload = st.empty()

with st.sidebar.form('Map excel sheet name'):
    st.subheader(title)
    uploaded_file = st.file_uploader('drag & drop excel : confs & map files',type="xlsx") 
    SheetMapName  = st.selectbox(label = "map excel sheet name", options = algo.SheetMapNameList) 
    DistFactor = st.number_input(label = 'DistFactor ==> metre', value = 0.1)
    length  = st.number_input(label = 'length (m)', value = 5.0)
    Width = st.number_input(label = 'Width(m)', value = 2.0)
    File = {
        'SheetMapName' : SheetMapName,
        'uploaded_file' : uploaded_file,
        'DistFactor' : (Width, length)
        }
    
    if st.form_submit_button("Submit & Reset"): 
        print('submitted Map')
        # session_state.clear()
        algo = load_data_brut(File)
        Update_Algo(algo)
        df = indiv_init(algo, 1)
        algo.df = df.drop_duplicates(subset='Name_txt')
        session_state['algo'] = algo
        # fig = new_plot_2(algo, plotedges = False)
        # c3.pyplot(fig) 

with st.expander('Map Config'):
    Col1 = ['a','b','c']
    Format = dict(zip(Col1,["{:.2e}"]))
    Format.update(dict(zip(['Masse','Cout'],["{:.0f}",  "{:,.2f}"])))
    dfInput = algo.confs.copy()  
    dfInput['Actif'] = dfInput['Actif'] ==1

    c1, c2 = st.columns([0.7,0.2])  

    c1.data_editor(dfInput.style.format(Format, na_rep=' '),
                   hide_index= True,
                   num_rows = "dynamic",
                   use_container_width=True,
                   height = int(35.2*(len(dfInput)+2)))
    
    rs = c2.slider('map slot size', 0.2, 1.2, step = 0.1, value = 0.4)
    fig = new_plot_2(algo, plotedges = False , rs= rs)
    c2.pyplot(fig) 
    # c1, c2 = st.columns([0.7,0.2]) 
    c1,c2,c3,c4 = st.columns([3,2,1,1])
    dfline = pd.DataFrame(algo.dfline.drop(columns = ['path','edge']))
    # print(dfline.columns)
    # dfline['path'] = dfline.path.astype(str)
    # dfslot = pd.DataFrame(algo.DictPos).T
    # dfslot.columns = ('y','x')    
    # c1, c2 = st.columns([0.8,0.2])  
    # c1.table(dfline.style.format(precision = 2))
    dfStyler  = dfline.style.set_properties(**{'text-align': 'center'})
    dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    c1.data_editor(dfStyler,
                   use_container_width=True,
                   hide_index= True,) 

# with st.expander("slots properties", False):
    # c1,c2,c3 = st.columns(3)
    # with st.expander("Capteurs", True):
    algo.dfc =  c2.data_editor(
        algo.SlotsDict0['dfc'].copy(),
        column_config={
            "nature" : st.column_config.SelectboxColumn(
                options = algo.DataCategorie['Nozzle']['Unique'],
                required = True, 
                width="small",
            ),
        },
        disabled =['capteur'],
        hide_index=True,            
        # height = 300,
        use_container_width = True,
        )
    algo.dfe =c3.data_editor(
        algo.SlotsDict0['dfe'].copy(),
        disabled =['EVname'],
        hide_index=True,
        use_container_width = True,
        )

    algo.dfp =c4.data_editor(
        algo.SlotsDict0['dfp'].copy(),
        disabled =['pompe'],
        hide_index=True,
        use_container_width = True,
        )

st.sidebar.download_button(label='📥 download input data template',
                        data= export_excel(algo, False),
                        file_name= 'input.xlsx') 
st.sidebar.download_button(label='📥 download input + pathfinding',
                            data= export_excel(algo, True),
                            file_name= 'input.xlsx') 

with st.expander("indivs params", True):

    c1,c2,c3,c4 = st.columns(4)
    algo.pop        = c1.number_input(label  = 'indiv pop init',value = algo.pop, min_value = 1,  max_value  = 100000,step = 10)
    algo.iterations = c1.number_input(label  = 'iterations / run',value = algo.iterations, min_value = 1,  max_value  = 1000,step = 1)
    help = "indivs selectionnés avec la meilleur fitness pour crossover => 2 enfants"
    algo.crossover  = c1.number_input(label = 'Crossover',value = int(algo.crossover), min_value = 0, max_value  = 100,step = 10, help =help)
    help = "indivs selectionnés avec la meilleur fitness pour mutation  => 1 enfants"
    algo.mutation   = c1.number_input(label = 'Mutation', value = int(algo.mutation), min_value = 0, max_value  = 100,step = 10, help =help)
    
    txt = "Maximum de pompe disponible"
    options = list(range(1,len(algo.Comb['E']) +1))        
    
    ListFitness = ['dist','Masse','Cout']
    ListRes = []
    for i in range(3): 
        res = c2.number_input(label = ListFitness[i] + '%',value = int(algo.fitnessCompo[i]*100), min_value = 0, max_value  = 100,step = 10, help ="")
        res /= 100
        ListRes.append(res)
    algo.fitnessCompo = np.array(ListRes)

    SplitText = 'si no group = Deactivate'
    Npa = int(c3.number_input(label= 'Npa',key='Npa' , value= algo.Npa))    
    Npc = int(c3.number_input(label= 'Npc',key='Npc' , value= algo.Npc))  
    PompeB = c4.toggle(label= 'Pompe B', help = 'si group = False',value = algo.PompeB)
    ListSplitName = ['Deactivate','Auto','Forced']
    Split = c4.toggle('Split')
    BusActif = True     
    # algo.PompeB = PompeB & (not algo.Group) & (not BusActif)
    # if not algo.Group : Split = 'Deactivate'
    algo.Split  = Split      
    algo.Npa = Npa
    algo.Npc = Npc
    algo.Pmax = Npa + Npc
    algo.PompesSelect = ['Pa'] * algo.Npa + ['Pc'] * algo.Npc

    algo.Tmode = c4.selectbox(label="Tmode",options= [False,'Bus','T0','Tx'])                     

    default =  "E0-C0,E1-C1,E2-C2,E3-C3,P0-E0,P0-E1,P1-E2,P1-E3"
    default = ''
    NameIndiv = st.text_input('reverse name_txt to indiv', default,help = "E0-C0,E1-C1,E2-C2,E3-C3,P0-E0,P0-E1,P1-E2,P1-E3")
    NameIndiv = NameIndiv.replace(" ",'').replace('"','').split(';')
    
session_state['algo'] = algo        
st.write('Group = ',algo.Group, ', Pompe_B = ',algo.PompeB , ', Split = ', algo.Split, ', BUS = ', algo.BusActif)   
c0,c1,c2,c3,c4 = st.columns(5) 
algo.Plot = c0.checkbox('Show  figure & details', value = False, help = "desactiver cette option ameliore les performances")
KeepResults =  c1.checkbox('Keep results') 
# algo.DebitCalculationNew = c1.checkbox('DebitCalculationNew', value = algo.DebitCalculationNew) 
        
if c2.button('RESET'):
    Update_Algo(algo)
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
    algo.SaveRun = []
    session_state['algo'] = algo
                    
if c4.button('RUN'):
    print("Params : RUN") 
    algo.SaveRun = [] 
    iterations = algo.iterations
    d = dict(
        indivs_total  = algo.Nrepro,
        indivs_unique = algo.df.shape[0],
        indivs_alive  = algo.df.Alive.sum(),)
    algo.SaveRun.append(d)         
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
        
        d = dict(
            indivs_total = algo.Nrepro,
            indivs_unique = algo.df.shape[0],
            indivs_alive = algo.df.Alive.sum(),)
        algo.SaveRun.append(d)    
        
        session_state['algo'] = algo 
        
if c3.button('recalculation', help = 'Pompe B , Bus , debit / pression , masse cout , fitness Alive'):
    indivs = []
    for idx , row  in algo.df.iterrows() :
        indiv = row.to_dict()
        indiv = Gen_Objectif_New(algo, indiv)
        indivs.append(indiv)  
    algo.indivs = indivs
    df = pd.DataFrame(indivs) 
    df = df.reset_index(drop = True)
    algo.df = df
    # algo.df = df.drop_duplicates(subset='Name_txt')
    session_state['algo'] = algo   
df1 = algo.df.copy()

# plot gen run stat
if len(algo.SaveRun)> 1 : 
    with st.expander("Run Stats", True):
        c1, c2 = st.columns([0.4,0.6])
        dfStat = pd.DataFrame(algo.SaveRun)
        c1.dataframe(dfStat, use_container_width  =True)    
        fig = px.line(dfStat)
        fig.update_layout(        
                        yaxis_title ='count',
                        xaxis_title ='epoch',
                        font=dict(size=16,family = "Arial"),
                        margin=dict(l=10, r=10, t=30, b=10),
                        )
        c2.plotly_chart(fig,use_container_width=True) 

# df & plot 
if len(df1)>0 :

    df1 = df1.sort_values(['fitness']).reset_index(drop = True)
    dfline = algo.dfline    

    DictParams = dict(
        Pattern = algo.Comb,
        indivs_total = algo.Nrepro,
        indivs_unique = df1.shape[0],
        indivs_alive = df1.Alive.sum(),
        epoch = algo.epoch,        )
    st.write(str(DictParams))

    # st.metric(label="create", value=algo.Nrepro, delta=-0.5,)

    dfx = df1[ColBase].copy()
    for col in dfx.columns:
        if col not in ColDfVal :
            # print(col)
            dfx[col]= dfx[col].astype(str)

    with st.expander("Dataframe", True):
        dfx.insert(0, "Select", False)
        dfx.loc[:3,'Select'] = True
        edited_df = st.data_editor(
                    dfx,
                    disabled=dfx.columns.drop(['Select']),
                    hide_index=True,
                )      
        dfSelect = df1[edited_df.Select].copy()
        Range = len(dfSelect) 

    with st.expander("Figures", True):
        ListResultsExport = []
        if (Range> 0) & algo.Plot:
            stcol  = st.columns(3)
            hideEtoC = stcol[0].checkbox('hideEtoC',False)
            rs = stcol[1].slider('slot size', 0.2, 1.2, step = 0.1, value = 0.4)
            Empty = stcol[2].empty()

            colSt = st.columns(4) 
            idxcol = 0 
            for i in range(Range): 
                row = dfSelect.iloc[i]
                indiv = row.to_dict()                    
                fig = new_plot_2(algo,indiv, hideEtoC, rs = rs)
                ListResultsExport.append({'row':row, 'fig': fig})
                colSt[idxcol].pyplot(fig)
                idxcol +=1
                if idxcol > 3:
                    idxcol = 0

            Empty.download_button(label ='📥 download results',
                data = export_excel_test(algo, ListResultsExport),
                file_name= 'results.xlsx')          
            
# PickleDonwload.download_button(
#     label="📥 download pickle Save_{}.pickle".format(today), key='pickle_Save_pickle',
#     data=pickle.dumps(vars(algo)),
#     file_name="Save_{}.pickle".format(today)) 

# c3.pyplot(fig)
                
                