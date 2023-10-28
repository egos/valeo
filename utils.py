import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import itertools 
from io import BytesIO
import matplotlib.pyplot as plt
import collections
import copy
from types import SimpleNamespace
import matplotlib.patches as mpatch
import xlsxwriter
import time

# calculation function & utilities

# -------------DATA & input----------------------
def export_excel_report(algo, ListResultsExport):
    # generate excel report for indivs & features selected by users 
    Col = 0
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()

    for i in range(len(ListResultsExport)) : 
        row = ListResultsExport[i]['row']
        fig = ListResultsExport[i]['fig']

        for i in range(len(row)):
            worksheet.write(i,0+Col*2, row.index[i])
            worksheet.write(i,1+Col*2, str(row.values[i]))

        imgdata = BytesIO()
        fig.savefig(imgdata, format='png',bbox_inches='tight', dpi=100)
        worksheet.insert_image(i+1,Col*2, '', {'image_data': imgdata})
        Col+=1
    workbook.close()
    return output.getvalue() 
   
def export_excel_input(algo, Addpath):
    # export map config & pathfinding
    confs = algo.confs2
    dfmap = algo.dfmap
    dfline = algo.dfline2.drop(columns= 'path')
    dfcapteur = algo.SlotsDict['C']
    
    output = BytesIO()

    writer = pd.ExcelWriter(output, engine='xlsxwriter')   
    confs.to_excel(writer, sheet_name='confs',index=False)  
    dfmap.to_excel(writer, sheet_name='map') 
    if Addpath : 
        dfline.to_excel(writer, sheet_name='lines') 
        dfcapteur.to_excel(writer, sheet_name='slot') 

    writer.close()
    processed_data = output.getvalue()
    return processed_data

def PathFinding(dfmap, DistFactor = [2,5]):
    # generate dfmap & lines with a pathfinding algorithm

    SlotColor = {'C' : 10, 'E': 20, 'P' : 30,"T":40}
    slots = ['C','P','E',"T"]    
    
    A0 = dfmap.copy().values
    Size = max(A0.shape)
    DistFactor = np.array(DistFactor)/np.array(A0.shape)
    
    Comb = collections.defaultdict(list)
    CombNodes = collections.defaultdict(list)
    DictPos = {}    
    ListWall = (np.argwhere(A0[1:-1,1:-1] == 1)+1).tolist()
        
    slots = ['C','P','E',"T"]  
    slotsN = dict(zip(slots,[0,0,0,0]))
    for s in slots: 
        CombNodes[s] = []

    # list the slot recording from input file 
    for iy, ix in np.ndindex(A0.shape):
        v = A0[iy, ix]
        if type(v) == str: 
            slot = v[0]
            n = slotsN[slot]
            slotsN[slot] = n+1
            v = slot + str(n)
            A0[iy,ix] = SlotColor[slot]*20
            Comb[v[0]].append(int(n))
            CombNodes[slot].append(v)
            DictPos[v] = (iy,ix)

    CombNodes = dict(CombNodes)
    print(CombNodes) 

    # check that border is wall type
    A0 = A0.astype(float)      
    Ax = np.ones((Size,Size))
    Ax[:A0.shape[0],:A0.shape[1]] = A0

    DictLine = {}

    # generate all slot combination
    def it_(L):
        return list(itertools.combinations(L, 2))
    
    ListEtoC = [(n1,n2) for n1 in CombNodes['E'] for n2 in CombNodes['C']]
    ListEdges = it_(CombNodes['P']+CombNodes['T']+CombNodes['E']) + ListEtoC

    # pathfinding with PF_forward & PF_backward
    for begin, end in ListEdges:
        start = DictPos[begin]
        A = Ax.copy()
        A1 = PF_forward(A,start)
        goal = DictPos[end]        
        path = PF_backward(A1.copy() ,start,  goal)
        path = np.array(path)       
        dist = (((np.diff(path.T)==0).T)*DistFactor).sum().round(2)
        ID = begin + '-' + end
        DictLine[ID] = {'path' : path, 'dist' : dist}     
    
    return DictLine, DictPos, A0,dict(Comb), ListWall  

def load_data(File , dfmap = None):
    # generate namespace algo , contains all parameters for runnning the code

    print('Init algo namespace')

    # manage input excel 
    uploaded_file = File['uploaded_file']
    SheetMapName  = File['SheetMapName']
    DistFactor    = File['DistFactor']

    # default input
    uploaded_file = uploaded_file if uploaded_file else 'data.xlsx'

    sheet_names = pd.ExcelFile(uploaded_file).sheet_names
    SheetMapNameList = [n for n in sheet_names if "map" in n]
    print(sheet_names)

    # used if map is modified in app
    if dfmap is None:
        dfmap = pd.read_excel(uploaded_file, sheet_name= SheetMapName, index_col=0) 
        ImportLine = True   
    else : 
        ImportLine = False
 
    DictLine, DictPos, A0, Comb, ListWall = PathFinding(dfmap, DistFactor)

    CombNodes = {slot :  ["{}{}".format(slot,n) for n in nList] for slot , nList  in  Comb.items()}
    CombAll = list(DictPos.keys())       
    
    dfline = pd.DataFrame(DictLine).T
    dfline.index.name = 'ID'
    dfline.reset_index(inplace = True)

    # check if customp line / slot are in inputs excel
    if ("lines" in sheet_names) & ImportLine: 
        print("Load line from excel")
        lines_add = pd.read_excel(uploaded_file, sheet_name= 'lines', index_col=0)
        dfline['dist'] = lines_add['dist'].copy()

    dfline['s'] = dfline.ID.str.split('-').str[0]
    dfline['t'] = dfline.ID.str.split('-').str[1]
    dfline['Connect'] = dfline['s'].str[0] + dfline['t'].str[0]       
    
    dfline['durite'] = 4
    dfline['edge'] = dfline.ID.str.split('-').apply(lambda x : (x[0] , x[1]))
    dfline.columns = dfline.columns.astype(str)    
    
    confs = pd.read_excel(uploaded_file, sheet_name= 'confs')
    confs.reset_index(drop = True, inplace= True) 
    confs.Name = confs.Name.astype(str)    

    # slot dataframe : dfc dfe dfp
    # dfc nozzle
    if ("slot" in sheet_names) & ImportLine:
        dfc = pd.read_excel(uploaded_file, sheet_name= 'slot', index_col=0)
        dfc.limit = dfc.limit.astype(float)
        # print(dfc)
    else : 
        dfc = pd.DataFrame()        
        size = len(CombNodes['C'])
        dfc['Slot'] = CombNodes['C']
        dfc['group']   = [0]*size
        dfc['nature']  = ['F']*size
        dfc['limit']   = [2.0]*size

    # dfe EV        
    size = len(CombNodes['E'])
    dfe = pd.DataFrame()
    dfe['Slot'] = CombNodes['E']
    dfe['Nmax'] = [len(CombNodes['C'])]*size

    # dfp PUMP
    dfp = pd.DataFrame()
    size = len(CombNodes['P'])
    dfp = pd.DataFrame()
    dfp['Slot'] = CombNodes['P']
    dfp['Nmax'] = [len(CombNodes['E'])]*size
    dfp['Bus']  = [False]*size
    
    SlotsDict0 = { 'C' : dfc,'E' : dfe,'P' : dfp}                
    PlotColor = {'C' : "#93c9ee", 'E': '#a2ee93', 'P' : "#c593ee", 'T': '#C0C0C0'}

    algo = dict(
        SheetMapName = SheetMapName,
        sheet_names = sheet_names,
        SheetMapNameList = SheetMapNameList,
        uploaded_file = uploaded_file,
        DistFactor = DistFactor,
        File = File, 
        PlotColor = PlotColor,
        dfmap = dfmap,
        Group = False,
        GroupDict = None,
        DictPos = DictPos,
        gr = {},
        pop = 10,
        fitness = 'dist',
        fitnessCompo = np.array([1,0,0]),
        crossover = 20,
        mutation = 20,
        SlotsDict0 = SlotsDict0,
        SlotsDict = copy.deepcopy(SlotsDict0),  
        Plot = False,
        dfline0 = dfline, 
        dfline2 = dfline.copy(),  
        epoch = 0,
        Nindiv = 0,
        Nrepro = 0,
        indivs = [],
        df = pd.DataFrame(),
        DataCategorie =  None,  
        Npa = 10,
        Npc = 0,
        PompesSelect = ['Pa'] * 10 + ['Pc'] * 0,  
        Pmax = 10,
        PompeB = False,
        BusActif = True, 
        Split = True, 
        EV = ['Ea'],               
        confs  = confs,
        confs2 = confs.copy(),
        Comb = Comb,
        CombAll = CombAll,
        CombNodes = CombNodes,
        A0 = A0,
        ListWall = ListWall,
        SaveRun = [],
        iterations = 1,
        G0 = None,
        Tmode= False,
        ErrorParams = False, 
        NameEV = 'Ea',
        NameReservoir = 'Ra',
        )
    algo = SimpleNamespace(**algo)

    Update_Algo(algo)

    return algo

def Update_Algo(algo):
    # used to update conf, map, graph, group, for taking into account change made by user

    print('Update_Algo')
    
    # GRAPH
    G0 = nx.from_pandas_edgelist(algo.dfline2, 's', 't', ['dist','durite','Connect','path'])
    print(G0)
    SlotList = algo.CombAll
    DictNodeGraph = {k: {'pos' : algo.DictPos[k]} for k in SlotList}
    nx.set_node_attributes(G0, DictNodeGraph)
    algo.G0 = G0
    
    # DataCategorie
    DataCategorie = {}
    confs = algo.confs2
    mask = confs['Actif'] == 1
    df = confs[mask].copy()
    for Categorie in df.Categorie.unique():        
        dfx = df[df.Categorie == Categorie]
        DataCategorie[Categorie] = {
            'Unique' : dfx.Name.unique().tolist(),
            'Values' : dfx.set_index('Name').dropna(axis = 1).to_dict('index')
                    }          
    
    # MODE for graph :  T Bus & group
    algo.ErrorParams = 0
    if np.any(algo.SlotsDict['P'].Bus.values) & (algo.Tmode != False):
        if ('T' not in algo.CombNodes) & (algo.Tmode == 'T'):
            algo.Tmode = 'Bus'
            algo.ErrorParams = 'Warning : no T detected BusMode switch to Bus  :'
        algo.BusActif = True
    else : 
        if algo.Tmode != False : algo.ErrorParams = 'Warning : no BUS Pump slot activated in map config BusMode switch to False  : '
        algo.Tmode  = False 
        algo.SlotsDict['P'].Bus = False
        algo.BusActif = False
    
    GroupDict = algo.SlotsDict['C'].group.values

    # ErrorParams
    if (len(np.unique(GroupDict)) == 1) & algo.Group: 
        algo.ErrorParams = 'Warning : group activated & no group detected for nozzle map config  : ' 

    cond1 = algo.Group & (algo.PompeB | (algo.Tmode == 'T')) 
    cond2 = (not algo.Group) & (algo.Split)
    if cond1 | cond2: algo.ErrorParams = 2
            
    # DataCategorie
    DataCategorie['Pompe']['Values']['Pb']['Masse'] -= DataCategorie['Pompe']['Values']['Pa']['Masse']
    DataCategorie['Pompe']['Values']['Pb']['Cout'] -= DataCategorie['Pompe']['Values']['Pa']['Cout']
    algo.DataCategorie = DataCategorie
    algo.NameEV = confs.loc[(confs.Actif == 1) & (confs.Categorie == 'EV'),'Name'].iloc[0]
    algo.NameReservoir = confs.loc[(confs.Actif == 1) & (confs.Categorie == 'Reservoir'),'Name'].iloc[0]

    #EDGES attributes
    for n0, n1  in algo.G0.edges():
        durite = str(G0[n0][n1]['durite'])
        dist      = G0[n0][n1]['dist']
        duriteVal = DataCategorie['Tuyau']['Values'][durite]['a']
        dmasse    = DataCategorie['Tuyau']['Values'][durite]['Masse']
        dcout     = DataCategorie['Tuyau']['Values'][durite]['Cout']
        
        attrs = dict(
            coeff = round(duriteVal * dist,5),
            Masse = round(dmasse * dist,2),
            Cout  = round(dcout * dist,2),
        )
        nx.set_edge_attributes(G0, {(n0, n1): attrs})

    #NODES attributes
    DictNodeAttr = {}
    for i, n in enumerate(algo.SlotsDict['C'].nature.values):
        c = 'C{}'.format(i)
        DictNodeAttr[c] = DataCategorie['Nozzle']['Values'][n]
    for i, e in enumerate(algo.CombNodes['E']):
        DictNodeAttr[e] = DataCategorie['EV']['Values'][algo.NameEV]
    for i, p in enumerate(algo.CombNodes['P']):
        DictNodeAttr[p] = {'Bus' : algo.SlotsDict['P'].Bus.values[i]}
    nx.set_node_attributes(G0, DictNodeAttr)   
    
    # generate gr dict in format [(group number , [nozzle list]), ...]
    gr = collections.defaultdict(list)
    for i, g in enumerate(GroupDict):
        gr[g].append('C{}'.format(i))
    gr = dict(gr)
    gr2 = []
    for g, Clist in gr.items():
        if g == 0:
            for c in Clist: 
                gr2.append((g,[c]))
        else : 
            gr2.append((g,Clist))
    algo.GroupDict = GroupDict  
    algo.gr = gr2
    print(algo.GroupDict, algo.gr)

def Plot_indiv(algo,indiv = None, hideEtoC = False, plotedges = True, rs = 0.4):
    # plot figure and detail, slot are made with patch 

    # initialisation
    if indiv is not None: 
        G = indiv['G'].copy() 
    else :
        G = algo.G0.copy()       
    edges = G.edges(data = True) 
    nodes = G.nodes(data = True)
    LenPath = len(edges)
    ListWall = algo.ListWall
    A0 = algo.A0
    Ymax , Xmax = A0.shape

    PlotColor = {'C' : "#93c9ee", 'E': '#a2ee93', 'P' : "#c593ee", 'T': '#C0C0C0'}
    fig, ax = plt.subplots(figsize = (8,8))

    # proportionality by distfactor and ticks labels
    width, length = algo.DistFactor
    N = int(width//0.5)
    ticks2 = np.arange(0,(N+1)*0.5,0.5)
    ticks1 = ticks2*(A0.shape[1]-1)/width
    ax.set_xticks(ticks1, ticks2)
    N = int(length//0.5)
    ticks2 = np.arange(0,(N+1)*0.5,0.5)
    ticks1 = ticks2*(A0.shape[0]-1)/length
    ax.set_yticks(ticks1, ticks2[::-1])

    # plot canvas
    ax.imshow(np.zeros(A0.shape), cmap='gray',vmin=0,vmax=1)  
    ax.add_patch(mpatch.Rectangle((0,0), Xmax-1, Ymax-1, color='#d8d8d8'))
    
    # size of slot ajusted by rs value
    lc = len(algo.CombNodes['C'])
    lpe = LenPath - lc
    MinOffset = rs*0.7
    MinOffset = MinOffset if MinOffset < rs else rs
    offset  = np.linspace(-MinOffset,MinOffset,lpe)
    offsetC = np.linspace(-MinOffset,MinOffset,lc)

    i, j  = 0,0
    if plotedges : 
        for n0 , n1, data in edges:

            # offset iterations depend of slot type 
            if  (data['Connect'] == 'EC'): 
                n = offsetC[j] 
                j+=1
            else : 
                n = offset[i]
                i+=1
    
            p = data['path']        
            if  (data['Connect'] == 'EC')  & (not hideEtoC):
                ax.plot(p[:,1]+n,p[:,0]+n,"white", linewidth=1, zorder=1, linestyle ='-')
            elif  (n0[0]  == 'E') & (n1[0]!='C'):
                ax.plot(p[:,1]+n,p[:,0]+n,'#1FA063', linewidth=2, zorder=1, linestyle ='-')
            elif n0[0]  == 'P' : 
                ax.plot(p[:,1]+n,p[:,0]+n,PlotColor['P'], linewidth=2, zorder=1, linestyle ='-')
            elif data['Connect']  != 'EC' : 
                ax.plot(p[:,1]+n,p[:,0]+n,"#3286ff", linewidth=2, zorder=1, linestyle ='-')

    # ListWall
    for x,y in ListWall: 
        ax.add_patch(mpatch.Rectangle((y-0.5,x-0.5), 1, 1, color='black')) 

    # nodes / slots
    style = dict(size = 8*rs/0.4, color='black')
    nature = algo.SlotsDict['C'].nature.values
    for n, data in nodes: 
        x , y = data['pos']
        slot = n[0]
        nslot = int(n[1:])
        if slot =='C' :             
            text = str(nslot) + ':' + nature[nslot] 
        else :
            text = nslot = int(n[1:])
        color = PlotColor[slot]
        ax.add_patch(mpatch.Rectangle((y-rs,x-rs), rs*2, rs*2, color=color))
        ax.add_patch(mpatch.Rectangle((y-rs,x-rs), rs*2, rs*2, color='black', fill = None))
        ax.text(y, x,str(text), **style,  ha='center', weight='bold') 
        if (indiv is not None): 
            if (slot == 'E'):
                if len(G.adj[n].items()) > 1:
                    N = len(indiv['EtoC'][n])
                    for i in range(N):
                        size = rs*2/8
                        circle1 = plt.Circle((y-rs+size+i*size*2,x+rs -size), size, color='k',fill=False)
                        ax.add_patch(circle1)
                else :
                    ax.plot((y-rs, y+rs),(x-rs, x+rs),'k', linewidth=1, zorder=1, linestyle ='-')
            if (slot == 'P'):
                N = len(indiv['Ptypes'][n])
                for i in range(N):
                    size = rs*2/8
                    circle1 = plt.Circle((y-rs+size+i*size*2,x+rs -size), size, color='k',fill=False)
                    ax.add_patch(circle1)        
    return fig

# -------------INDIV func-------------
def indiv_create(algo, row = None, NewCtoE = None, IniEtoP = None): 
    # generate connexion between slot C -> E -> P  
    # all other attribute of indiv are generated with Gen_Objectif
    # args : row , NewCtoE, IniEtoP used in case of genetic operations

    D = algo.Comb    
    Ccount = len(D['C'])
    PompesSelect = algo.PompesSelect
    PompesSelect = ['Pa'] * algo.Npa + ['Pc'] * algo.Npc
        
    ElistMax = np.random.choice(D['E'],algo.Pmax) if len(D['E']) >  algo.Pmax  else D['E']

    # check if new values for CtoE    
    if NewCtoE is not None : 
        # check number slot EV :  P  < Pmax and random choice the P list
        Pmax = algo.Pmax         
        Elist = np.unique(NewCtoE)
        Ecount = len(Elist)
        n = Ecount -  Pmax 
        if n > 0:
            Edrop = np.random.choice(Elist,n, replace=False)
            Edispo = Elist[~np.isin(Elist, Edrop)]
            mask = np.isin(NewCtoE,Edrop)            
            NewCtoE[mask] = np.random.choice(Edispo,mask.sum())        
        CtoE = NewCtoE        
    else : 
        CtoE = np.random.choice(ElistMax,Ccount)

    EtoC = collections.defaultdict(list)
    d = collections.defaultdict(list)
    for i in range(Ccount): 
        d[CtoE[i]].append(D['C'][i])
        EtoC['E{}'.format(CtoE[i])].append('C{}'.format(D['C'][i]))
    EtoC = dict(sorted(EtoC.items()))
    Econnect = dict(sorted(d.items()))
    Elist = sorted(Econnect)
    Ecount = len(Elist)      
    
    #check the new E slot list
    if row is not None :          
        if Ecount > row.Ecount:            
            NewEtoP = np.random.choice(D['P'],Ecount - row.Ecount)
            EtoP = np.append(row.EtoP, NewEtoP)
            if not algo.BusActif : 
                # if bus actif consider only 1 pump
                for pt in row.Ptype:  PompesSelect.remove(pt)   
            NewPtype = np.random.choice(PompesSelect,Ecount - row.Ecount, replace=False)
            Ptype = np.append(row.Ptype, NewPtype)
        
        # check E slot is different from parents
        elif Ecount < row.Ecount : 
            EtoP = np.random.choice(row.EtoP,Ecount)
            Ptype = np.random.choice(row.Ptype,Ecount, replace=False)
        else :
            EtoP  = row.EtoP
            Ptype = row.Ptype
    else : 
        EtoP = np.random.choice(D['P'],Ecount)
        Ptype = np.random.choice(PompesSelect,Ecount, replace=False)
    # used only by reverse indiv
    if IniEtoP is not None : 
        EtoP = IniEtoP
        Ptype = np.random.choice(PompesSelect,Ecount, replace=False)

    # generate PtoE PtoC and PtypeCo
    PtoE = collections.defaultdict(list)
    d = collections.defaultdict(list)
    d2 = collections.defaultdict(list)
    for i in range(Ecount): 
        d[EtoP[i]].append(Elist[i]) 
        d2[EtoP[i]].append(Ptype[i])
        PtoE['P{}'.format(EtoP[i])].append('E{}'.format(Elist[i]))

    PtoE = dict(sorted(PtoE.items()))
    Pconnect = dict(sorted(d.items())) 
    Plist = sorted(Pconnect)
    PtypeCo = dict(sorted(d2.items()))     
    PtypeCo = PtypeCo

    d = collections.defaultdict(list)
    for p, Elist in PtoE.items():
        for e in Elist:
            d[p] += EtoC[e]
    PtoC = dict(d)

    d = collections.defaultdict(list)
    for i in range(Ecount): 
        d[EtoP[i]].append(Ptype[i])
    PtypeCo = dict(sorted(d.items()))         
    
    # Name of indiv based of edges nodes
    List_EtoC = [['E{}-C{}'.format(start, end) for end in List] for start , List in Econnect.items()]
    List_PtoE = [['P{}-E{}'.format(start, end) for end in List] for start , List in Pconnect.items()]
    edgesEtoC = [('E{}'.format(e),'C{}'.format(c))  for e,v in Econnect.items() for c in v]
    edgesPtoE = [('P{}'.format(p),'E{}'.format(e))  for p,v in Pconnect.items() for e in v]
    Enodes = ['E{}'.format(n) for n in Elist]
    Pnodes = ['P{}'.format(n) for n in Plist]        
    Name = list(itertools.chain.from_iterable(List_EtoC + List_PtoE))
    Name_txt = ','.join(Name)   
    
    indiv = dict(
        CtoE = CtoE,
        Econnect = Econnect,
        EtoC = EtoC,
        Ecount = Ecount,
        EtoP = EtoP,
        Pconnect = Pconnect,
        PtoE = PtoE,
        Plist = Plist,
        Ptype = Ptype,
        PtoC = PtoC,
        Enodes = Elist,
        Pnodes = Pnodes,
        PtypeCo = PtypeCo,
        edgesEtoC = edgesEtoC,
        edgesPtoE = edgesPtoE,
        ID = algo.Nindiv,
        parent = [],
        Name_txt = Name_txt,
        Epoch = algo.epoch,  
        ICount = {'Pa': 0, 'Pb' : 0, 'Pc' : 0, 'EV' : 0,'Y' : 0,'T':0},
    )    

    indiv = Gen_Objectif(algo, indiv)
    algo.indivs.append(indiv)
    algo.Nrepro += 1  
    algo.Nindiv += 1   

    return indiv

def Gen_Objectif(algo, indiv):
    # process the differents indiv parameters for graph generation et genetic algorithm fitness 

    # Graph 
    indiv = Indiv_Graph(algo, indiv, mode = algo.Tmode)
    G = indiv['G']
    EtoC = indiv['EtoC']

    # ptype attribution in edges attributes if Bus = random 1 pt 
    # masse cout for ev, P  nodes
    for p, Elist in indiv['PtoE'].items():
        ptList = indiv['PtypeCo'][int(p[1:])]
        p , ptList
        actif = algo.G0.nodes[p]['Bus']
        Masse , Cout = 0, 0
        if actif:
            pt = np.random.choice(ptList)
            indiv['ICount'][pt] += 1
            end = list(G.edges(p))[0]
            nx.set_edge_attributes(G, {end: {"pt": pt}})         
            Masse += algo.DataCategorie['Pompe']['Values'][pt]['Masse']
            Cout  += algo.DataCategorie['Pompe']['Values'][pt]['Cout']
        else :            
            # multiple pump / slot 
            pts = ptList[0]
            for i, e in enumerate(Elist):
                pt = ptList[i]
                indiv['ICount'][pt] += 1
                if (algo.PompeB) & (i > 0): 
                    if pts == 'Pa' :
                        pt, pts  = 'Pb',''
                        indiv['ICount']['Pb'] += 1
                        indiv['ICount']['Pa'] -= 2
                    else :
                        pts = 'Pa'
                nx.set_edge_attributes(G, {(p,e): {"pt": pt}}) 
                Masse+= algo.DataCategorie['Pompe']['Values'][pt]['Masse']
                Cout += algo.DataCategorie['Pompe']['Values'][pt]['Cout']

        G.nodes[p]['Masse'] = Masse
        G.nodes[p]['Cout']  = Cout

        # EV, take into account direct connexion PtoC
        for e, Clist in EtoC.items():
            if algo.Group & algo.Split:
                G.nodes[e]['Masse'] = 0
                G.nodes[e]['Cout'] = 0
            else : 
                Ncapteurs = len(Clist)
                if (len(G.adj[e].items()) == 1) : Ncapteurs=0
                indiv['ICount']['EV'] += Ncapteurs
                G.nodes[e]['Masse'] = algo.G0.nodes[e]['Masse'] * Ncapteurs
                G.nodes[e]['Cout']  = algo.G0.nodes[e]['Cout']  * Ncapteurs
    
    # Ptypes
    # permet de filtrer les lignes au depart d'une pompe et stocker pt dans Ptypes
    d = collections.defaultdict(list)    
    for tup in list(nx.subgraph_view(indiv['G'], filter_edge= lambda n1, n2 : n1[0] == 'P').edges.data("pt")):
        p = tup[0]
        d[p].append(tup[2])
    indiv['Ptypes'] = dict(d)

    # print(list(nx.subgraph_view(indiv['G'], filter_edge= lambda n1, n2 : n1[0] == 'P').edges.data("pt")))
    # list(nx.subgraph_view(indiv['G'], filter_edge= lambda n1, n2 : n1[0] == 'P').edges.data("pt"))
    # nx.get_node_attributes(G, 'Masse')
    # nx.get_node_attributes(G, 'Cout')

    # DEBIT calculation
    indiv['IndivLine2'] = IndivLine_(algo,indiv)
    indiv = debit_calculation(algo,indiv)

    # OBJECTIF ALIVE
    indiv['dist']  = round(G.size('dist'),1)

    indiv['Masse'] = G.size('Masse') + sum(nx.get_node_attributes(G, 'Masse').values())
    indiv['Masse'] += algo.DataCategorie['Reservoir']['Values'][algo.NameReservoir]['Masse']
    indiv['Masse'] = round(indiv['Masse'],1)

    indiv['Cout']  = G.size('Cout')  + sum(nx.get_node_attributes(G, 'Cout').values())
    indiv['Cout'] += algo.DataCategorie['Reservoir']['Values'][algo.NameReservoir]['Cout']
    indiv['Cout'] = round(indiv['Cout'],1)

    indiv['PressionList'] = np.array(list(nx.get_node_attributes(G, "Pi").values())).round(1)
    indiv['DebitList'] = np.array(list(nx.get_node_attributes(G, "Qi").values())).round(1)
    indiv['Debit'] = round(indiv['DebitList'].sum(),1)
  
    ListFitness = ['dist','Masse','Cout']
    fitness = 0
    for i in range(3): 
        fitness+= indiv[ListFitness[i]] * algo.fitnessCompo[i]  
    indiv['fitness'] = round(fitness,5)
    indiv['Alive'] = False if  (indiv['PressionList'] < algo.SlotsDict['C'].limit.values).any() else True 

    # Pompe limit 
    DictPNmax = algo.SlotsDict['P'].set_index('Slot').Nmax.to_dict()
    cond = True
    for p, v in indiv['Ptypes'].items():
        if len(v) > DictPNmax[p] : cond = cond & False 

    # check if indiv alive
    DictEVNmax = algo.SlotsDict['E'].set_index('Slot').Nmax.to_dict()
    cond = True
    for e,v in indiv['EtoC'].items():
        if len(v) > DictEVNmax[e] : cond = cond & False 
    indiv['Alive'] = indiv['Alive'] & cond 

    # format pression & debit
    indiv['PressionList']  = dict(zip(algo.CombNodes['C'], indiv['PressionList']))
    indiv['DebitList']  = dict(zip(algo.CombNodes['C'], indiv['DebitList']))

    return indiv

def Indiv_reverse(Name,algo):
    # return indiv with name input
    
    NameList = Name.split(',')
    Clist = algo.CombNodes['C']
    CtoE = {}
    EtoP = {}
    for n in NameList:
        slot1, slot2 = n.split('-')
        if slot1[0] == 'E':
            c = int(slot2[1:])
            e = int(slot1[1:])
            CtoE[c] = e
            
        if slot1[0] == 'P':
            e = int(slot2[1:])
            p = int(slot1[1:])
            EtoP[e] = p

    d = dict(sorted(CtoE.items()))
    Clist = list(d.keys())
    CtoE = list(d.values())
    d = dict(sorted(EtoP.items()))
    EtoP = list(d.values())
    print('Indiv_reverse',Clist,CtoE, EtoP)
    indiv = indiv_create(algo, row = None, NewCtoE = CtoE, IniEtoP = EtoP)
    return indiv

def indiv_init(algo, pop):
    # initialise pop indivs 
    df = []
    if pop >0 : 
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

def Top_indiv(algo): 
    # generate best indiv 
    Cnodes = algo.CombNodes['C']
    Enodes = algo.CombNodes['E']
    Pnodes = algo.CombNodes['P']
    EtoCconnect = Cluster_(algo, Enodes, Cnodes)
    PtoEconnect = Cluster_(algo, Pnodes, Enodes)
    print(EtoCconnect , PtoEconnect)
    Name = []
    Name += ['{}-{}'.format(start, end) for start , List in EtoCconnect.items() for end in List]    
    Name += ['{}-{}'.format(start, end) for start , List in PtoEconnect.items() for end in List]
    print(Name)
    
    return ','.join(Name)

def Indiv_Graph(algo, indiv,mode = None):
    # graph generation , mode is the optimisation option none , bus or T

    tic = time.perf_counter()
    edgesEtoC = indiv['edgesEtoC']
    edgesPtoE  = indiv['edgesPtoE']
    Elist = algo.CombNodes['E']
    
    Cnodes = algo.CombNodes['C']
    Pnodes = indiv['Pnodes']    
    mode = algo.Tmode

    #G creation
    if mode == 'T':
        Tlist = algo.CombNodes['T']
        ListEdges  = []
        ListEdges += edgesEtoC
        Tconnect = Cluster_(algo, Pnodes, algo.CombNodes['T'])
        for p in Pnodes:
            Elist = indiv['PtoE'][p]            
            if algo.G0.nodes()[p]['Bus']:
                if (p in Tconnect):
                    Tlist = Tconnect[p]
                    ListEdges += [Adj_Shortest_edge(algo,p,Tlist)]
                    nodes  = Elist + Tlist       
                    Gx = nx.subgraph(algo.G0,nodes).copy()    
                    edges = (nx.minimum_spanning_tree(Gx,'dist')).edges() 
                    # edges = nx.approximation.steiner_tree(Gx, nodes, weight='dist', method = 'kou').edges()
                    ListEdges += edges
                else : 
                    ListEdges += Bus_(algo, p, Elist)
            else : 
                ListEdges += [(p,e) for e in Elist]

        G = algo.G0.edge_subgraph(ListEdges).copy() 
    
    
    elif mode == 'Bus': 
        G = algo.G0.edge_subgraph(edgesPtoE + edgesEtoC).copy()
        for p in Pnodes: 
            if algo.G0.nodes()[p]['Bus']:
                Pedges = list(G.edges(p))
                G.remove_edges_from(Pedges)
                Elist =  indiv['PtoE'][p]
                EdgesBus = Bus_(algo, p, Elist)
                G.add_edges_from(EdgesBus)
    
    else : 
        G = algo.G0.edge_subgraph(edgesPtoE + edgesEtoC).copy() 

    #G to Digraph = filter T 
    Gd = nx.DiGraph()
    for p in Pnodes:
        path = nx.shortest_path(G, source = p)
        for c in Cnodes:
            if c in path:
                p = path[c]
                Gd.add_edges_from(nx.utils.pairwise(p))


    # graph optimisation delete T node with 1 successors
    if mode == 'T':
        for T in algo.CombNodes['T']: 
            if T in Gd :      
                # l = len(Gd.adj[T].items())
                successors = list(Gd.successors(T))
                # print(list(Gd.predecessors(T)))
                if len(successors) == 1:
                    n0 = successors[0]
                    n1 = list(Gd.predecessors(T))[0]
                    Gd.remove_node(T)
                    Gd.add_edge(n1, n0)
                else :
                    indiv['ICount']['T'] += 1 

    # nodes & edges attributes
    nx.set_node_attributes(Gd,algo.G0.nodes)

    attrs = {(n0, n1) : algo.G0[n0][n1] for  n0, n1 in Gd.edges}
    nx.set_edge_attributes(Gd,attrs)

    toc = time.perf_counter()
    indiv['G'] = Gd
    indiv['time'] = round(toc - tic , 6)
    return indiv

def IndivLine_(algo, indiv):
    # schema for indiv debit calculation, manage group

    gr = algo.gr
    G = indiv['G']

    indivline = {}
    for p in indiv['Pnodes']:
        Pline = []
        for g , Clist in gr:

            for n2 in G.successors(p):
                pt = G[p][n2]['pt']
                ns = [n2]
                d = collections.defaultdict(list)
                # print(p,g,Clist,ns)
                i = 0
                while ns:
                    n2 = ns[-1]
                    ns = ns[:-1]
                    # print(p,g,Clist,n2, list(G.successors(n2)),ns)
                    for n in G.successors(n2):             
                        if n in Clist: 
                            d[n2].append(n)
                        if n[0]!='C': 
                            ns.append(n)
                    i+=1
                    if i>=100:
                        break
                if dict(d):
                    Pline.append((g,pt, dict(d)))
                
        indivline[p] = Pline

    return indivline

def debit_calculation(algo,indiv):
    # debit calculation based on formula and excel example provided by VALEO

    G = indiv['G']
    Q0 = np.arange(0.1,80,0.1)
    res = {}
    PressionList, DebitList = {}, {}
    attrs = {}

    # parcours IndivLine
    for p, line in indiv['IndivLine2'].items():    
        for g ,pt , Edict in line:

            ax,bx,cx = [algo.DataCategorie['Pompe']['Values'][pt][i] for i in ['a','b','c']]           
            F = ax * Q0**2 + bx*Q0 +cx
            Qx = 0
            Qlist, Alist, CcList = [] , [], []
            Start = p
            CListTotal = []
            for e, Clist in Edict.items():
                coef_PtoE = nx.shortest_path_length(G,Start,e,'coeff')
                CoeffEtoC = np.array([G[e][c]['coeff'] for c in Clist])
                CoeffC    = np.array([G.nodes[c]['a'] for c in Clist])
                CcList.append(CoeffC)

                # direct connexion P to C 
                if (len(G.adj[e].items()) > 1) & (algo.Split == False): 
                    CoeffE = G.nodes[e]['a']
                else :
                    CoeffE = 0

                if algo.Split:
                    indiv['ICount']['Y'] += 1 
                    G.nodes[e]['Masse'] += 10
                    G.nodes[e]['Cout']  += 0.2

                
                F = F - coef_PtoE*(Q0-Qx)**2
                F[F<0] = 0              

                A = CoeffE + CoeffEtoC + CoeffC
                Alist.append(A)
                Qlist.append(np.sqrt(F / A[:,np.newaxis]))
                Qi = np.vstack(Qlist)
                Qx = Qi.sum(0)
                Start = e  # attention si T il est perdu 
                CListTotal += Clist
            idx = np.searchsorted(Q0 - Qx, -0.1)
            Qi  = np.vstack(Qlist)[:,idx].round(2)
            Pi  = (np.concatenate(CcList)* (Qi**2)).round(2)

            for i in range(len(CListTotal)):
                c = CListTotal[i]
                attrs[c] = dict(
                    Pi = Pi[i],
                    Qi = Qi[i]                
                )
    nx.set_node_attributes(G,attrs)
                
    indiv['G'] = G
    return indiv

# -------------Genetic algorithm-------------
def Gen_CrossOver(dfx, algo):  
    # compute Crossover
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
        parents = dfx.ID.tolist()
        for i in range(2):
            row = dfx.iloc[i]            
            indiv = indiv_create(algo, row,NewCtoE[i]) 
            indiv['parent'] =  parents
            L.append(indiv)
        return L 

def Gen_Mutation(row, algo): 
    # compute Mutation for single
    NewCtoE = copy.deepcopy(row.CtoE)
    idx = np.random.randint(len(NewCtoE))
    D = algo.Comb
    l = [e for e in D['E'] if e != NewCtoE[idx]]
    e = np.random.choice(l,1)[0]
    NewCtoE[idx] = e
    indiv = indiv_create(algo, row,NewCtoE)
    indiv['parent'] =  [row.ID]
    return indiv

# -------------utilities-------------
def PF_forward(A,start): 
    # forward on all the grid from start
    N = len(A)
    v0 = np.array([-1,1,-N,N])
    #     v0 = np.array([-1,1,-N,N, -N-1, -N+1,N+1,N-1])
    Dim = len(v0)
    e = 2
    A[start] = e
    a = A.reshape(-1)
    v = np.where(a == e)  

    while len(v) > 0 :
        v = np.tile(v, (Dim, 1)).T + v0
        v = v[np.where(a[v]==0)]
        v = np.unique(v)        
        e+=1
        a[v]=e
    return a.reshape((N,N))

def PF_backward(A,start,goal):
    # backward from goal to start
    N = len(A)
    v0 = np.array([-1,1,-N,N])
    #     v0 = np.array([-1,1,-N,N, -N-1, -N+1,N+1,N-1])
    Dim = len(v0)
    e1,e2  = A[start] , A[goal]
    a = A.reshape(-1)
    v = goal[1] + goal[0]*N
    L  = [goal]
    while e2 > 2:
        v = v + v0
        v[v > len(a)] = len(a)-1
        v = v[np.where((a[v] < e2) & (a[v] >= 2))]
        idx = a[v].argmin()
        v = v[idx]
        e2 = a[v]
        pos = (int(np.ceil(v/N)-1),  v%N)
        L.insert(0,pos)
    return L  

def Bus_(algo, start, nodes):
    # create bus connection 
    G0 = algo.G0.copy()
    G = G0.subgraph([start] + nodes).copy()
    n = start
    lines = []
    while G.size()>0 :
        NodesAdj  = [x[0] for x in G.adj[n].items()]
        NodesDist = [x[1]['dist'] for x in G.adj[n].items()]
        ns = n
        G.remove_node(n)
        n = NodesAdj[np.array(NodesDist).argmin()]
        lines.append((ns,n))
    return lines

def Cluster_(algo, starts, targets):
    # cluster of targets around start slot
    G0 = algo.G0
    G = G0.subgraph(starts + targets).copy()
    DictCluster = collections.defaultdict(list)
    for i, t in enumerate(targets):  
        NodesDist = [G[t][s]['dist'] for s in starts]
        n = starts[np.array(NodesDist).argmin()]
        DictCluster[n].append(t)
    return dict(DictCluster)

def Adj_Shortest_edge(algo,start,nodes):
    # snippet for find closest slot around start slot
    idx = np.argmin([algo.G0[start][n]['dist'] for n in nodes])
    node = nodes[idx]
    return  (start, node)

