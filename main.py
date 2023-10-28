import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time
from streamlit import session_state
import copy
import plotly.express as px
import time
from utils import *

# streamlit app section 

st.set_page_config(page_title = "VALEO_AG_IHM", layout="wide")
pop = 10      

# increase button width
st.markdown("""<style> div.stButton button { width: 200px;}  </style> """,unsafe_allow_html=True)

# feature with numeric / bool values
ColDfVal   = ['Ecount','Pcount', 'dist','ID','Masse', 'Cout','Alive','Group']

today = time.strftime("%Y-%m-%d-%H:%M:%S")
print(today)

# initialize data structure algo on start
if 'algo' not in session_state: 
    print(' ')
    print('BEGIN')
    File = {'SheetMapName' : 'map', 'uploaded_file' : None, 'DistFactor' : [2,5]}
    algo = load_data(File)
    session_state['algo'] = algo
else : 
    print('reload')
    algo = session_state['algo']

# input excel map section
with st.sidebar.form('Map excel sheet name'):
    text ='input & pathfinding : 🖱️ press submit for change take effect'
    st.subheader(text)
    uploaded_file = st.file_uploader('drag & drop excel : confs & map files',type="xlsx") 
    SheetMapName  = st.selectbox(label = "map excel sheet name", options = algo.SheetMapNameList) 
    length     = st.number_input(label = 'length (m)', value = 5.0)
    Width      = st.number_input(label = 'Width(m)'  , value = 2.0)
    File = {
        'SheetMapName' : SheetMapName,
        'uploaded_file': uploaded_file,
        'DistFactor'   : (Width, length)
        }
    
    if st.form_submit_button("Submit & Reset"): 
        print('submitted Map')
        algo = load_data(File)
        df = indiv_init(algo, 1)
        algo.df = df.drop_duplicates(subset='Name_txt')
        session_state['algo'] = algo

# edited map section
if st.toggle('Activate feature edited map'):
    with st.expander('Map custom',True):
        c1 , c2 = st.columns([3,1])

        dfmap = algo.dfmap.copy().astype(str)
        dfmap.columns = dfmap.columns.astype(str)
        dfmap2 = c1.data_editor(dfmap,
                                use_container_width = True,
                                height = int(35.2*(len(dfmap)+1))
                                )
        dfmap2 = dfmap2.replace(['0','1'],[0,1])

        #if map is not functionnal
        try : 
            algo2 = load_data(copy.deepcopy(algo.File),dfmap = dfmap2)
            fig = Plot_indiv(algo2, plotedges = False)
            c2.pyplot(fig)
        except : 
            st.warning("warning map is not correctly designed")

        if st.button("edited map"): algo = copy.copy(algo2)

# map section
with st.expander('Map Config',True):
    Col1 = ['a','b','c']
    Format = dict(zip(Col1,["{:.2e}"]))
    Format.update(dict(zip(['Masse','Cout'],["{:.0f}",  "{:,.2f}"])))

    dfInput = algo.confs.copy()
    # dfInput['Actif'] = dfInput['Actif'] ==1
    algo.confs2 = st.data_editor(
                # dfInput.style.format(Format, na_rep=' '),
                algo.confs.copy(),
                # hide_index= True,
                num_rows = "dynamic",
                use_container_width=True,
                height = int(35.2*(len(dfInput)+2)),
                )
    
    c1,c2,c3,c4 = st.columns([1.5,0.2,1,1])   
    Coldfline = ["ID","Connect", "dist","durite"]
    dflineSlice = c1.data_editor(algo.dfline0[Coldfline].copy(),
                #    column_order = ("ID", "dist","connect","durite"),
                   use_container_width=True,
                   disabled = ['ID','Connect'],
                   hide_index= True
                   ) 
    algo.dfline2[Coldfline] = dflineSlice    

    #slots
    for slot , df in algo.SlotsDict0.items():
        color = algo.PlotColor[slot]
        df = df.style.format()
        df = df.set_properties(**{'background-color': color, 'text-align': 'center'}, subset  =['Slot'])
        edited_df = c3.data_editor(
            df,
            disabled =['Slot'],
            hide_index=True,
            # use_container_width = True,
            )
        algo.SlotsDict[slot] = edited_df
    
    rs = c4.slider('map slot size', 0.2, 1.2, step = 0.1, value = 0.4)
    fig = Plot_indiv(algo, plotedges = False , rs= rs)
    c4.pyplot(fig) 

st.sidebar.download_button(label='📥 download input data template',
                        data= export_excel_input(algo, False),
                        file_name= 'input.xlsx') 
st.sidebar.download_button(label='📥 download input + pathfinding',
                            data= export_excel_input(algo, True),
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

    Npa = int(c3.number_input(label= 'Npa',key='Npa' , value= algo.Npa))    
    Npc = int(c3.number_input(label= 'Npc',key='Npc' , value= algo.Npc)) 
    algo.Npa = Npa
    algo.Npc = Npc
    algo.Pmax = Npa + Npc
    algo.PompesSelect = ['Pa'] * algo.Npa + ['Pc'] * algo.Npc

    algo.Group  = c4.toggle(label= 'Group', help = 'depend of nozzle group number in map config') 
    algo.PompeB = c4.toggle(label= 'Pompe B', help = 'if group = False')
    algo.Split  = c4.toggle('Split', help = 'if Group = True')      
    algo.Tmode = c4.radio(label="BusMode",options= [False,'Bus','T'])                   

    NameIndiv = st.text_input('reverse name_txt to indiv', '',help = "E0-C0,E1-C1,E2-C2,E3-C3,P0-E0,P0-E1,P1-E2,P1-E3")
    NameIndiv = NameIndiv.replace(" ",'').replace('"','').split(';')
    
session_state['algo'] = algo 

# app parameters and actions
c0,c1,c2,c3,c4 = st.columns(5) 
algo.Plot = c0.checkbox('Show  figure & info', value = True)
KeepResults =  c1.checkbox('Keep results') 
        
if c2.button('RESET'):
    Update_Algo(algo)
    DictAlgo = dict(
        Group = algo.Group,
        Pompe_B = algo.PompeB,
        Split =   algo.Split,
        PbusActif =  algo.BusActif,
        BusMode =  algo.Tmode,
            ) 
    if algo.ErrorParams == 2:
        st.error('ERROR , conflicts with group : ' + str(DictAlgo))
    else : 
        if algo.ErrorParams !=0:
            st.warning(algo.ErrorParams + str(DictAlgo))
        else :
            st.success('no conflicts for parameters : ' + str(DictAlgo))
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
else : 
    st.success('')                    

if c4.button('RUN'):
    # append indivs list with genetic generation
    print("Params : RUN") 
    algo.SaveRun = [] 
    iterations = algo.iterations
    d = dict(
        indivs_total  = algo.Nrepro,
        indivs_unique = algo.df.shape[0],
        indivs_alive  = algo.df.Alive.sum(),)
    algo.SaveRun.append(d)                       
    my_bar = st.empty()    
    # select batch of indiv based on fitness and users gen parameters 
    for i in range(iterations):
        my_bar.progress((i+1)/iterations)
        algo.epoch +=1
        df0 = algo.df
        df0 = df0.sort_values('fitness').reset_index(drop = True)
        df1 = df0[df0.Alive].copy()
        idxmaxCross = int(algo.crossover)
        idxmaxMuta  = int(algo.mutation)
        if idxmaxCross >  len(df1) : idxmaxCross = len(df1)            
        if idxmaxMuta  >  len(df1) : idxmaxMuta  = len(df1)
        Ncross = int(idxmaxCross/2)
        Nmuta  = int(idxmaxMuta)            
        Lcross = df1[:idxmaxCross].index.values
        np.random.shuffle(Lcross)
        Lmuta = df1[:idxmaxMuta].index.values
        np.random.shuffle(Lmuta)           
        L = [] 
        for n in range(Ncross):  
            i1 , i2 = Lcross[n*2] , Lcross[n*2 + 1]
            dfx = df1.loc[[i1,i2]].copy()
            L2 = Gen_CrossOver(dfx, algo)
            if L2 is not None : L += L2  
        for n in range(Nmuta):
            row = df1.loc[Lmuta[n]].copy()
            indiv = Gen_Mutation(row, algo)
            L.append(indiv)
    
        dfx = pd.DataFrame(L)
        algo.df = pd.concat([df0, dfx]).drop_duplicates(subset='Name_txt').reset_index(drop = True)
        
        d = dict(
            indivs_total = algo.Nrepro,
            indivs_unique = algo.df.shape[0],
            indivs_alive = algo.df.Alive.sum(),)
        algo.SaveRun.append(d)    
        
        session_state['algo'] = algo 

# begin of analysis part        
df1 = algo.df.copy()

# show unique indiv stats after gen run 
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

# df print & plot fig
if (len(df1)>0) & (algo.ErrorParams !=2) :

    df1 = df1.sort_values(['fitness']).reset_index(drop = True)
    dfline = algo.dfline2    

    DictParams = dict(
        Pattern = algo.Comb,
        indivs_total = algo.Nrepro,
        indivs_unique = df1.shape[0],
        indivs_alive = df1.Alive.sum(),
        epoch = algo.epoch,        )
    st.write(str(DictParams))

    ColexportDefault = ['ID','dist','Debit','Masse','Cout','fitness','Epoch',
                        'Alive','parent','ICount','Name_txt','PressionList','DebitList']
    Colexport = st.multiselect('indiv keys', df1.columns.tolist(),ColexportDefault)

    dfx = df1[Colexport].copy()
    for col in dfx.columns:
        if col not in ColDfVal : dfx[col]= dfx[col].astype(str)

    with st.expander("Results Table", True):
        dfx.insert(0, "Select", False)
        dfx.loc[:3,'Select'] = True
        edited_df = st.data_editor(
                    dfx,
                    disabled=dfx.columns.drop(['Select']),
                    hide_index=True,
                    use_container_width=True,

                )      
        dfSelect = df1[edited_df.Select].copy()
        Range = len(dfSelect) 

    with st.expander("Figures & detailled results", True):
        
        if (Range> 0) & algo.Plot:
            stcol  = st.columns(4)
            hideEtoC = stcol[0].checkbox('hide EtoC',False)
            rs = stcol[2].slider('slot size', 0.2, 1.2, step = 0.1, value = 0.4)
            Empty = stcol[3].empty()
            ListResultsExport = []

            if stcol[1].toggle('Detailled results'):                
                idxcol = 0 
                for i in range(Range):
                    
                    row = dfSelect.iloc[i]
                    st.dataframe(row[Colexport].to_frame().T,hide_index= True)

                    c1, c2, c3 = st.columns([1,2,2])
                    indiv = row.to_dict()
                    fig = Plot_indiv(algo,indiv, hideEtoC, rs = rs)
                    row.name = row.ID

                    c1.pyplot(fig)
                    G = indiv['G']
                    
                    # edges
                    df = nx.to_pandas_edgelist(G).drop(columns = ['path'])
                    c2.dataframe(df, use_container_width=True, hide_index=True)
                    # nodes    
                    df = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient='index')
                    df = df.drop(columns = ['pos','Type','Actif','Categorie'])
                    c3.dataframe(df, use_container_width=True)
                    ListResultsExport.append({'row':row[Colexport], 'fig': fig})

            else :                    
                colSt = st.columns(4) 
                idxcol = 0 
                for i in range(Range): 
                    row = dfSelect.iloc[i]
                    indiv = row.to_dict()                    
                    fig = Plot_indiv(algo,indiv, hideEtoC, rs = rs)
                    row.name = row.ID
                    colSt[idxcol].dataframe(row[Colexport].drop(['ID']).astype(str), use_container_width=True)                  
                    colSt[idxcol].pyplot(fig)
                    idxcol +=1
                    if idxcol > 3:
                        idxcol = 0
                        colSt = st.columns(4) 
                    ListResultsExport.append({'row':row[Colexport], 'fig': fig})

            Empty.download_button(label ='📥 download results',
                data = export_excel_report(algo, ListResultsExport),
                file_name= 'results.xlsx')     
                  



                
                