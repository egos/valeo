import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import itertools 
import math
import io
from io import BytesIO
from math import factorial as f
from datetime import timedelta
from streamlit import session_state
import matplotlib.pyplot as plt
import json
import collections
import copy
from types import SimpleNamespace
import matplotlib.patches as mpatch
from collections import Counter
import xlsxwriter
import time

#ctrl k ctrl &  //// fold level1 


# DATA
def export_excel_test(algo, ListResultsExport):
    Col = 0
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()
    # plt.gcf().set_size_inches(4, 4)
    for i in range(len(ListResultsExport)) : 
        row = ListResultsExport[i]['row']
        fig = ListResultsExport[i]['fig']

        for i in range(len(row)):
            worksheet.write(i,0+Col*2, row.index[i])
            worksheet.write(i,1+Col*2, str(row.values[i]))
        imgdata = BytesIO()
        # fig.savefig(imgdata, format='png',bbox_inches='tight', dpi=50)
        fig.savefig(imgdata, format='png',bbox_inches='tight', dpi=100)
        worksheet.insert_image(i,0+Col*2, '', {'image_data': imgdata})
        Col+=1
    workbook.close()
    return output.getvalue() 
   
def export_excel(algo, Type):
    confs = algo.confs
    dfmap = algo.dfmap
    dfline = algo.dfline.drop(columns= 'path')
    # dfline['Select'] = 'o'
    dfcapteur = algo.dfc
    
    output = BytesIO()

    writer = pd.ExcelWriter(output, engine='xlsxwriter')   
    confs.to_excel(writer, sheet_name='confs')  
    dfmap.to_excel(writer, sheet_name='map') 
    if Type : 
        dfline.drop(columns = 'duriteVal').to_excel(writer, sheet_name='lines') 
        dfcapteur.to_excel(writer, sheet_name='slot') 

    writer.close()
    processed_data = output.getvalue()
    return processed_data

def load_data_brut(File , select = None):
    print('Init algo namespace')
    uploaded_file = File['uploaded_file']
    SheetMapName  = File ['SheetMapName']
    DistFactor = File['DistFactor']
    uploaded_file = uploaded_file if uploaded_file else 'data.xlsx'
    # print('algo', uploaded_file)
    # DistFactor = 1
    dfmap = pd.read_excel(uploaded_file, sheet_name= SheetMapName, index_col=0)
        
    DictLine, DictPos, A0,Comb, ListWall = new_import_T(dfmap, DistFactor)
    CombNodes = {slot :  ["{}{}".format(slot,n) for n in nList] for slot , nList  in  Comb.items()}
    # print(DictLine)
    sheet_names = pd.ExcelFile(uploaded_file).sheet_names
    print(sheet_names)
    SheetMapNameList = [n for n in sheet_names if "map" in n]
    dfline = pd.DataFrame(DictLine).T

    dfline.index.name = 'ID'

    dfline.reset_index(inplace = True)
    dfline['duriteType'] = 4
    if "lines" in sheet_names : 
        print("Load line from excel")
        lines_add = pd.read_excel(uploaded_file, sheet_name= 'lines', index_col=0)
        dfline['duriteType'] = lines_add['duriteType'].copy()
        dfline['dist'] = lines_add['dist'].copy()
        # dfline = dfline[lines_add.Select=='o']
        # dfline = dfline[]
    # print(DictLine.keys(),dfline.set_index('ID').to_dict().keys())
    # dfline['dist'] = dfline.dist.astype('f')
    dfline['s'] = dfline.ID.str.split('-').str[0]
    dfline['t'] = dfline.ID.str.split('-').str[1]
    dfline['Connect'] = dfline['s'].str[0] + dfline['t'].str[0]
    # DictLine = dfline.set_index('ID').to_dict(orient = 'index')
    CombAll = list(DictPos.keys())
    
    confs = pd.read_excel(uploaded_file, sheet_name= 'confs', index_col=0)
    # print(confs)
    DataCategorie = {}
    mask = confs['Actif'].notnull()
    df = confs[mask].copy()
    for Categorie in df.Categorie.unique():        
        dfx = df[df.Categorie == Categorie]
        DataCategorie[Categorie] = {
            'Unique' : dfx.Name.unique().tolist(),
            'Values' : dfx.set_index('Name').dropna(axis = 1).to_dict('index')
                    }         
    # print(DataCategorie['Tuyau'])
    Dict_durite_val = {
        4 : DataCategorie['Tuyau']['Values'][4]['a'],
        8 : DataCategorie['Tuyau']['Values'][8]['a']
    }
    dfline['duriteVal'] = dfline['duriteType'].map(Dict_durite_val)
    dfline['durite'] = 4
    # dfline['Cout'] = algo.DataCategorie['Tuyau']['Values'][4]['Cout']
    dfline['edge'] = dfline.ID.str.split('-').apply(lambda x : (x[0] , x[1]))
    DictLine = dfline.set_index('ID').to_dict(orient = 'index')   
    DictEdge = dfline.set_index('edge').to_dict(orient = 'index') 
    # print(Dict_durite_val)
    # print(dfline['durite_val'])
    
    Clist = Comb['C']
    # Nozzles = [DataCategorie['Nozzle']['Unique'][0]] * len(Comb['C'])
    # Nvals   = [DataCategorie['Nozzle']['Values'][n]['a'] for n in Nozzles]
    # Nvals  = dict(zip(Clist, Nvals))
    # GroupDict = dict(zip(Clist,[0] * len(Clist)))
    GroupDict = np.zeros(len(Clist), dtype= int)
    Group = ~(GroupDict == 0).all()    
    
    # Group0  = [0] * len(Clist)
    # Nature0 = [0] * len(Clist)
    # Limit0  = np.array([2] * len(Clist))

    # dfc dfe dfp
    if "slot" in sheet_names:
        dfc = pd.read_excel(uploaded_file, sheet_name= 'slot', index_col=0)
        dfc.limit = dfc.limit.astype(float)
        print(dfc)
    else : 
        dfc = pd.DataFrame()        
        size = len(CombNodes['C'])
        dfc['capteur'] = CombNodes['C']
        dfc['group']   = [0]*size
        dfc['nature']  = ['F']*size
        dfc['limit']   = [2.0]*size

        # EV
        
        size = len(CombNodes['E'])
        dfe = pd.DataFrame()
        dfe['EVname'] = CombNodes['E']
        dfe['Nmax'] = [len(CombNodes['C'])]*size

        # POMPE
        dfp = pd.DataFrame()
        size = len(CombNodes['P'])
        # ListPlimSlot = [len(Comb['E'])] * Len
        dfp = pd.DataFrame()
        dfp['pompe'] = CombNodes['P']
        dfp['Nmax'] = [len(CombNodes['E'])]*size
        dfp['Bus'] = [False]*size


    # print(dfc)
    # Group0 = dfc.group.tolist()        
    # Nature0 = dfc.nature.map({'F':0,'R':1}).tolist()
    # Limit0 = dfc.limit.values
    # PompeMaxPerSlot = ['P{}'.format(p) for p in Comb['P']]
    # Nozzlelimits = Limit0   
    
    SlotsDict0 = dict(
        dfc = dfc,
        dfe = dfe,
        dfp = dfp, 
                )

    #-----------GRAPH
    G0 = nx.from_pandas_edgelist(dfline, 's', 't', ['dist','duriteVal','durite','Connect','path'])
    print(G0)
    SlotList = CombAll
    DictNodeGraph = {k: {'pos' : DictPos[k]} for k in SlotList}
    nx.set_node_attributes(G0, DictNodeGraph)

    algo = dict(
        SheetMapName = SheetMapName,
        sheet_names = sheet_names,
        SheetMapNameList = SheetMapNameList,
        uploaded_file = uploaded_file,
        DistFactor = DistFactor,
        dfmap = dfmap,
        Group = True,
        GroupDict = GroupDict,
        gr = {},
        pop = 10,
        fitness = 'dist',
        fitnessCompo = np.array([1,0,0]),
        crossover = 20,
        mutation = 20,
        # Nlim = 2.0,   
        SlotsDict0 = SlotsDict0,
        SlotsDict = copy.deepcopy(SlotsDict0),
        # Nozzlelimits = Nozzlelimits,      
        Plot = False,
        dfc = dfc,
        dfp = dfp,
        dfe = dfe,
        dfline = dfline,
        # DictPos = DictPos,        
        DictLine = DictLine,
        DictEdge = DictEdge,
        epoch = 0,
        Nindiv = 0,
        Nrepro = 0,
        indivs = [],
        df = pd.DataFrame(),
        DataCategorie =  DataCategorie,
        # Tuyau = [4],     
        Npa = 10,
        Npc = 4,
        PompesSelect = ['Pa'] * 10 + ['Pc'] * 0,  
        Pmax = 10,
        PompeB = False,
        BusActif = True, 
        Split = 'Deactivate', 
        EV = ['Ea'],    
        # Nozzles  = Nozzles,  
        # Nvals = Nvals,               
        confs = confs,
        Clist = Clist,
        Comb = Comb,
        CombAll = CombAll,
        CombNodes = CombNodes,
        # dist       = dfline.set_index('ID').dist.astype(float).round(1).to_dict(),
        # duriteType = dfline.set_index('ID').duriteType.to_dict(),
        # duriteVal  = dfline.set_index('ID').duriteVal.to_dict(),
        A0 = A0,
        ListWall = ListWall,
        # Group0 = Group0,
        # Nature0 = Nature0, 
        # Limit0 = Limit0,
        # dfcapteur = dfc,
        # ListPlimSlot = ListPlimSlot,
        SaveRun = [],
        iterations = 1,
        PlotLineWidth = [1,3],
        ListBusPactif = [False] * len(Comb['P']),
        # DebitCalculationNew = True,
        G0 = G0,
        PtoTConnect = {},
        Tmode= False,
        indivMode = None,

        )
    algo = SimpleNamespace(**algo)

    return algo

def new_import_T(dfmap, DistFactor):
    # print('new_import')    
    SlotColor = {'C' : 10, 'E': 20, 'P' : 30,"T":40}
    slots = ['C','P','E',"T"]    
    
    A0 = dfmap.values
    Size = max(A0.shape)
    # DistFactor = Size / DistFactor
    
    Comb = collections.defaultdict(list)
    DictPos = {}    
    ListBegin = []
    ListEnd = []   
    ListWall = (np.argwhere(A0[1:-1,1:-1] == 1)+1).tolist()
        
    slots = ['C','P','E',"T"]  
    slotsN = dict(zip(slots,[0,0,0,0]))

    for iy, ix in np.ndindex(A0.shape):
        v = A0[iy, ix]
        if type(v) == str: 
            slot = v[0]
            n = slotsN[slot]
            slotsN[slot] = n+1
            v = slot + str(n)
            A0[iy,ix] = SlotColor[slot]*20
            #Comb[v[0]].append(int(v[1:]))
            Comb[v[0]].append(int(n))
            DictPos[v] = (iy,ix)
            
            if slot == "E" : ListBegin.append(v)
            else : ListEnd.append(v)    
               
    A0 = A0.astype(float)      
    Ax = np.ones((Size,Size))
    Ax[:A0.shape[0],:A0.shape[1]] = A0
    ListBegin, ListEnd = sorted(ListBegin), sorted(ListEnd)
        
    Path = {}
    DictLine = {}
    
    for begin in ListBegin:
        start = DictPos[begin]
        A = Ax.copy()
        A1 = Path1(A,start)
        for end in ListEnd: 
            goal = DictPos[end]        
            path = Path2(A1.copy() ,start,  goal)
            path = np.array(path)       
            dist = (np.abs(np.diff(path.T)).sum())#.astype(int)
            
            if end[0] == 'C' : ID = begin + '-' + end
            else : ID = end + '-' + begin

            DictLine[ID] = {'path' : path, 'dist' : dist}        

    ListEv =  ['E' + str(n) for n in Comb['E']]
    it = itertools.permutations(ListEv, 2) # pour avoir les 2 sens sur EtoE
    # it = itertools.combinations(ListEv, 2)
    ListEtoE = list(it)
    ListEtoE
    # print(ListEv, ListEtoE)

    DictEtoE = {}
    for begin, end in ListEtoE:
        start = DictPos[begin]
        A = Ax.copy()
        A1 = Path1(A,start)
        goal = DictPos[end]        
        path = Path2(A1.copy() ,start,  goal)
        path = np.array(path)       
        dist = (np.abs(np.diff(path.T)).sum())#.astype(int)

        # if end[0] == 'C' : ID = begin + '-' + end
        # elif begin[0] == 'P' : ID = begin + '-' + end    
        # else :
        ID = end + '-' + begin
        DictEtoE[ID] = {'path' : path, 'dist' : dist}
    DictLine.update(DictEtoE)

    DictTt = {}
    if "T" in Comb:
        List1 =  ['T' + str(n) for n in Comb['T']]
        List2 =  ['P' + str(n) for n in Comb['P']]
        ListSlots = List1 + List2
        it = itertools.permutations(ListSlots, 2) # pour avoir les 2 sens sur EtoE
        # it = itertools.combinations(ListSlots, 2)
        ListTt= list(it)
        # print(ListSlots, ListTt)

        DictTt = {}
        for begin, end in ListTt:
            start = DictPos[begin]
            A = Ax.copy()
            A1 = Path1(A,start)
            goal = DictPos[end]        
            path = Path2(A1.copy() ,start,  goal)
            path = np.array(path)       
            dist = (np.abs(np.diff(path.T)).sum() )#.astype(int)

            # if end[0] == 'C' : ID = begin + '-' + end
            # elif begin[0] == 'P' : ID = begin + '-' + end    
            # else :
            ID = end + '-' + begin
            DictTt[ID] = {'path' : path, 'dist' : dist}

    DictLine.update(DictTt)       
    
    return DictLine, DictPos, A0,dict(Comb), ListWall  

def Update_Algo(algo):
    G0 = algo.G0

    #EDGES
    for n0, n1  in algo.G0.edges():
        durite = 4
        # G0[n0][n1]['dist'] = G0[n0][n1]['dist']/10
        dist      = G0[n0][n1]['dist']/10
        duriteVal = algo.DataCategorie['Tuyau']['Values'][durite]['a']
        dmasse    = algo.DataCategorie['Tuyau']['Values'][durite]['Masse']
        dcout     = algo.DataCategorie['Tuyau']['Values'][durite]['Cout']
        
        attrs = dict(
            coeff = round(duriteVal * dist,5),
            Masse = round(dmasse * dist,2),
            Cout  = round(dcout * dist,2),
        )
        nx.set_edge_attributes(G0, {(n0, n1): attrs})

    #NODES
    DictNodeAttr = {}
    # print(algo.ListBusPactif)
    for i, n in enumerate(algo.dfc.nature.values):
        c = 'C{}'.format(i)
        DictNodeAttr[c] = algo.DataCategorie['Nozzle']['Values'][n]
    for i, e in enumerate(algo.CombNodes['E']):
        DictNodeAttr[e] = algo.DataCategorie['EV']['Values']['Ea']
        # DictNodeAttr[e].update({'Bus' : algo.dfp.Bus.values[i]})
    for i, p in enumerate(algo.CombNodes['P']):
        # DictNodeAttr[p] = algo.DataCategorie['Pompe']['Values']['Pa']
        DictNodeAttr[p] = {'Bus' : algo.dfp.Bus.values[i]}
    # print(DictNodeAttr)
    nx.set_node_attributes(G0, DictNodeAttr)

    #MODE INDIV T Tx T0 Bus & group ... 
    if np.any(algo.dfp.Bus.values) :
        if ('T' in algo.CombNodes) & (algo.Tmode != False):
            mode = algo.Tmode
            if algo.Tmode == 'T0':
                algo.PtoTConnect = PToT_path_2(algo)
            algo.Group = False
        else : 
            mode = 'Bus'
    else : 
        mode = None
        algo.BusActif = False
    algo.indivMode = mode

    # GROUP passage crappy tant que j'ai pas reform groupdict
    if algo.Group :
        GroupDict = algo.dfc.group.values
    else : 
        GroupDict = [0]*len(algo.Comb['C'])
    # print(GroupDict)
    
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
    # print(algo.GroupDict, algo.gr)

def new_plot_2(algo,indiv = None, hideEtoC = False, plotedges = True, rs = 0.4):

    DictLine = algo.DictEdge
    if indiv is not None: 
        G = indiv['G'].copy() 
    else :
        G = algo.G0.copy()       
    edges = G.edges(data = True) 
    nodes = G.nodes(data = True)
    LenPath = len(edges) # = nombre de ligne 
    ListWall = algo.ListWall
    A0 = algo.A0
    Ymax , Xmax = A0.shape
    # print(A0.shape)
    PlotColor = {'C' : "#93c9ee", 'E': '#a2ee93', 'P' : "#c593ee", 'T': '#C0C0C0'}
    fig, ax = plt.subplots(figsize = (8,8))
    ax.imshow(np.zeros(A0.shape), cmap='gray',vmin=0,vmax=1)  
    ax.add_patch(mpatch.Rectangle((0,0), Xmax-1, Ymax-1, color='#d8d8d8'))
    # masked = np.ma.masked_where(A0 <= 1, A0)
    # MinOffset = LenPath*0.03
    
    lc = len(algo.CombNodes['C'])
    lpe = LenPath - lc
    MinOffset = LenPath*rs
    # MinOffset = 0.02
    MinOffset = rs*0.8
    MinOffset = MinOffset if MinOffset < rs else rs
    offset  = np.linspace(-MinOffset,MinOffset,lpe)
    offsetC = np.linspace(-MinOffset,MinOffset,lc)
    # print(len(offset))
    i, j  = 0,0
    if plotedges : 
        for n0 , n1, data in edges:
            # print(n0 , n1)
            if  (data['Connect'] == 'EC'): 
                n = offsetC[j] 
                j+=1
            else : 
                n = offset[i]
                i+=1
            # n = offset[i]  
            # n=0
            p = data['path']        
            if  (data['Connect'] == 'EC')  & (not hideEtoC):
                ax.plot(p[:,1]+n,p[:,0]+n,"white", linewidth=1, zorder=1, linestyle ='-')
            elif  (n0[0]  == 'E') & (n1[0]!='C'):
                ax.plot(p[:,1]+n,p[:,0]+n,'#1FA063', linewidth=2, zorder=1, linestyle ='-')
            elif n0[0]  == 'P' : 
                ax.plot(p[:,1]+n,p[:,0]+n,PlotColor['P'], linewidth=2, zorder=1, linestyle ='-')
            elif data['Connect']  != 'EC' : 
                ax.plot(p[:,1]+n,p[:,0]+n,"#3286ff", linewidth=2, zorder=1, linestyle ='-')


    for x,y in ListWall: 
        ax.add_patch(mpatch.Rectangle((y-0.4,x-0.4), 1, 1, color='black')) 
    # style = dict(size= 15 * 9 / Ymax, color='black')
    style = dict(size = 8, color='black')
    # rs = 0.4 # taille slot
    nature = algo.dfc.nature.values
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
                N = len(indiv['EtoC'][n])
                for i in range(N):
                    size = rs*2/8
                    circle1 = plt.Circle((y-rs+size+i*size*2,x+rs -size), size, color='k',fill=False)
                    ax.add_patch(circle1)
            if (slot == 'P'):
                N = len(indiv['Ptypes'][n])
                for i in range(N):
                    size = rs*2/8
                    circle1 = plt.Circle((y-rs+size+i*size*2,x+rs -size), size, color='k',fill=False)
                    ax.add_patch(circle1)        
    return fig

# INDIV
def indiv_create(algo, row = None, NewCtoE = None, IniEtoP = None): 
    # print(algo.Pmax)
    dfline = algo.dfline
    D = algo.Comb    
    Clist = D['C']
    Ccount = len(D['C'])
    PompesSelect = algo.PompesSelect
    PompesSelect = ['Pa'] * algo.Npa + ['Pc'] * algo.Npc
        
    ElistMax = np.random.choice(D['E'],algo.Pmax) if len(D['E']) >  algo.Pmax  else D['E']
    #EcMax = algo.Pmax 
        
    if NewCtoE is not None : 
        # si repro on verifie que le nombre de EV slot : nombre P  < Pmax
        Pmax = algo.Pmax         
        # NewCtoE = np.array([0,1,2,0])
        Elist = np.unique(NewCtoE)
        Ecount = len(Elist)
        n = Ecount -  Pmax 
        # print(Pmax, Ecount, n , NewCtoE)
        if n > 0:
            Edrop = np.random.choice(Elist,n, replace=False)
            Edispo = Elist[~np.isin(Elist, Edrop)]
            mask = np.isin(NewCtoE,Edrop)            
            NewCtoE[mask] = np.random.choice(Edispo,mask.sum())        
        CtoE = NewCtoE        
    else : CtoE = np.random.choice(ElistMax,Ccount)
    # print(CtoE)
    EtoC = collections.defaultdict(list)
    d = collections.defaultdict(list)
    for i in range(Ccount): 
        d[CtoE[i]].append(D['C'][i])
        EtoC['E{}'.format(CtoE[i])].append('C{}'.format(D['C'][i]))
    EtoC = dict(sorted(EtoC.items()))
    Econnect = dict(sorted(d.items()))
    # Edist = dict(sorted(d.items()))
    Elist = sorted(Econnect)
    # print(Econnect)
    Ecount = len(Elist)      
    
    if row is not None :          
        if Ecount > row.Ecount:
            
            NewEtoP = np.random.choice(D['P'],Ecount - row.Ecount)
            EtoP = np.append(row.EtoP, NewEtoP)
            # print(row.ID, row.Name ,Ecount , row.Ecount, row.Ptype , PompesSelect)
            if not algo.BusActif : 
                for pt in row.Ptype:  PompesSelect.remove(pt)   
            NewPtype = np.random.choice(PompesSelect,Ecount - row.Ecount, replace=False)
            Ptype = np.append(row.Ptype, NewPtype)
            
        elif Ecount < row.Ecount : 
            EtoP = np.random.choice(row.EtoP,Ecount)
            Ptype = np.random.choice(row.Ptype,Ecount, replace=False)
            # print("Ecount",EtoP)
        else :
            EtoP  = row.EtoP
            Ptype = row.Ptype
        # print(NewCtoE , 'avant',row.Elist,row.EtoP,'apres',Elist,EtoP,row.Ecount, Ecount)
    else : 
        EtoP = np.random.choice(D['P'],Ecount)
        Ptype = np.random.choice(PompesSelect,Ecount, replace=False)
    if IniEtoP is not None : 
        EtoP = IniEtoP
        Ptype = np.random.choice(PompesSelect,Ecount, replace=False)
    # print(IniEtoP, EtoP, Elist)

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
    # Pcount  = len(Plist) 
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
    
    List_EtoC = [['E{}-C{}'.format(start, end) for end in List] for start , List in Econnect.items()]
    List_PtoE = [['P{}-E{}'.format(start, end) for end in List] for start , List in Pconnect.items()]
    edgesEtoC = [('E{}'.format(e),'C{}'.format(c))  for e,v in Econnect.items() for c in v]
    edgesPtoE = [('P{}'.format(p),'E{}'.format(e))  for p,v in Pconnect.items() for e in v]
    # edgesEtoC = [tuple(line.split('-')) for line in EtoC]
    Enodes = ['E{}'.format(n) for n in Elist]
    Pnodes = ['P{}'.format(n) for n in Plist]
    print(Enodes, Elist, Plist, Econnect, EtoC)
        
    Name = list(itertools.chain.from_iterable(List_EtoC + List_PtoE))
    Name_txt = ','.join(Name)   
    
    indiv = dict(
        Clist = Clist,
        CtoE = CtoE,
        Econnect = Econnect,
        EtoC = EtoC,
        # Elist = Elist,
        Ecount = Ecount,
        EtoP = EtoP,
        Pconnect = Pconnect,
        PtoE = PtoE,
        Plist = Plist,
        # Pcount = Pcount,
        # Ptype0 = Ptype,
        Ptype = Ptype,
        PtoC = PtoC,
        Enodes = Elist,
        Pnodes = Pnodes,
        PtypeCo = PtypeCo,
        # List_EtoC = List_EtoC,
        # List_PtoE = List_PtoE,
        edgesEtoC = edgesEtoC,
        edgesPtoE = edgesPtoE,
        # Name = Name,
        ID = algo.Nindiv,
        parent = [],
        Name_txt = Name_txt,
        Epoch = algo.epoch,  
        # ListBusActif = [True] * len(Plist) ,
        # ListBusActif = [algo.ListBusPactif[p] for p in Plist] ,   
    )    
    # print(PtoE)
    # print(EtoC)
    # print(Name_txt)
    indiv = Gen_Objectif_New(algo, indiv)
    # indiv['IndivLine'] = indiv_lines_conf(algo, indiv)
    algo.indivs.append(indiv)
    algo.Nrepro += 1  
    algo.Nindiv += 1   
    # print(indiv.keys())
    return indiv

def Gen_Objectif_New(algo, indiv):

    indiv = Indiv_Graph(algo, indiv, mode = algo.indivMode)
    G = indiv['G']
    EtoC = indiv['EtoC']

    # attribution des ptypes si Bus = random 1 pt parmi ptlist dans first edges from p
    # masse cout pour nodes EV et P 
    for p, Elist in indiv['PtoE'].items():
        ptList = indiv['PtypeCo'][int(p[1:])]
        p , ptList
        actif = algo.G0.nodes[p]['Bus']
        Masse , Cout = 0, 0
        if actif:
            pt = np.random.choice(ptList)
            end = list(G.edges(p))[0]
            # G[p][end]['pt'] = pt
            nx.set_edge_attributes(G, {end: {"pt": pt}})         
            Masse += algo.DataCategorie['Pompe']['Values'][pt]['Masse']
            Cout  += algo.DataCategorie['Pompe']['Values'][pt]['Cout']
        else : 
            for i, e in enumerate(Elist):
                pt = ptList[i]
                nx.set_edge_attributes(G, {(p,e): {"pt": pt}}) 
                Masse+= algo.DataCategorie['Pompe']['Values'][pt]['Masse']
                Cout += algo.DataCategorie['Pompe']['Values'][pt]['Cout']
            
                Ncapteurs = len(EtoC[e])
                G.nodes[e]['Masse'] = algo.G0.nodes['E0']['Masse'] * Ncapteurs
                G.nodes[e]['Cout']  = algo.G0.nodes['E0']['Cout']  * Ncapteurs
        G.nodes[p]['Masse'] = Masse
        G.nodes[p]['Cout']  = Cout
    
    # Ptypes
    d = collections.defaultdict(list)
    for tup in list(nx.subgraph_view(indiv['G'], filter_edge= lambda n1, n2 : n1[0] == 'P').edges.data("pt")):
        p = tup[0]
        d[p].append(tup[2])
    indiv['Ptypes'] = dict(d)

    # print(list(nx.subgraph_view(indiv['G'], filter_edge= lambda n1, n2 : n1[0] == 'P').edges.data("pt")))
    # list(nx.subgraph_view(indiv['G'], filter_edge= lambda n1, n2 : n1[0] == 'P').edges.data("pt"))
    # nx.get_node_attributes(G, 'Masse')
    # nx.get_node_attributes(G, 'Cout')

    # DEBIT
    indiv['IndivLine2'] = IndivLine_(algo,indiv)
    indiv = debit_v3(algo,indiv)

    # OBJECTIF ALIVE
    indiv['dist'] = round(G.size('dist')/10,1)
    indiv['Masse'] = G.size('Masse') + sum(nx.get_node_attributes(G, 'Masse').values())
    indiv['Cout'] = G.size('Cout') + sum(nx.get_node_attributes(G, 'Cout').values())

    indiv['PressionList'] = np.array(list(nx.get_node_attributes(G, "Pi").values()))
    indiv['DebitList'] = np.array(list(nx.get_node_attributes(G, "Qi").values()))
    indiv['Debit'] = round(indiv['DebitList'].sum(),1)

    ListFitness = ['dist','Masse','Cout']
    fitness = 0
    for i in range(3): 
        fitness+= indiv[ListFitness[i]] * algo.fitnessCompo[i]  
    indiv['fitness'] = round(fitness,5)
    indiv['Alive'] = False if  (indiv['PressionList'] < algo.dfc.limit.values).any() else True 

    # Pompe limit 
    DictPNmax = algo.dfp.set_index('pompe').Nmax.to_dict()
    cond = True
    for p, v in indiv['Ptypes'].items():
        if len(v) > DictPNmax[p] : cond = cond & False 

    DictEVNmax = algo.dfe.set_index('EVname').Nmax.to_dict()
    cond = True
    for e,v in indiv['EtoC'].items():
        if len(v) > DictEVNmax[e] : cond = cond & False 
    indiv['Alive'] = indiv['Alive'] & cond 
    # print(len(G.adj['P0']))
    return indiv

def Indiv_reverse(Name,algo, Ptype = None):
    NameList = Name.split(',')
    Clist = algo.Clist
    CtoE = {}
    EtoP = {}
    for n in NameList:
        # print(n)
        slot1, slot2 = n.split('-')
        if slot1[0] == 'E':
            c = int(slot2[1:])
            e = int(slot1[1:])
            CtoE[c] = e
            # print(e,c) 
            
        if slot1[0] == 'P':
            e = int(slot2[1:])
            p = int(slot1[1:])
            EtoP[e] = p
            # print(e,p)   
    d = dict(sorted(CtoE.items()))
    Clist = list(d.keys())
    CtoE = list(d.values())
    d = dict(sorted(EtoP.items()))
    EtoP = list(d.values())
    print('Indiv_reverse',Clist,CtoE, EtoP)
    indiv = indiv_create(algo, row = None, NewCtoE = CtoE, IniEtoP = EtoP)
    return indiv

def indiv_init(algo, pop):
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
    Cnodes = algo.CombNodes['C']
    Enodes = algo.CombNodes['E']
    Pnodes = algo.CombNodes['P']
    EtoCconnect = Cluster_(algo, Enodes, Cnodes)
    PtoEconnect = Cluster_(algo, Pnodes, Enodes)
    print(EtoCconnect)
    print(PtoEconnect)
    Name = []
    Name += ['{}-{}'.format(start, end) for start , List in EtoCconnect.items() for end in List]
    print(Name)
    
    Name += ['{}-{}'.format(start, end) for start , List in PtoEconnect.items() for end in List]
    print(Name)
    
    return ','.join(Name)

def Indiv_Graph(algo, indiv,mode = None):
    tic = time.perf_counter()
    edgesEtoC = indiv['edgesEtoC']
    edgesPtoE  = indiv['edgesPtoE']
    Elist = algo.CombNodes['E']
    
    Cnodes = algo.CombNodes['C']
    Pnodes = indiv['Pnodes']    

    #G creation
    if mode == 'Tx':
        Tlist = algo.CombNodes['T']
        ListEdges  = []
        ListEdges += edgesEtoC
        Tconnect = Cluster_(algo, Pnodes, algo.CombNodes['T'])
        # print(Tconnect)
        for p in Pnodes:
            Elist = indiv['PtoE'][p]
            # print(Elist)
            
            if algo.G0.nodes()[p]['Bus']:
                if (p in Tconnect):
                    Tlist = Tconnect[p]
                    ListEdges += [Adj_Shortest_edge(algo,p,Tlist)]
                    nodes  = Elist + Tlist       
                    Gx = nx.subgraph(algo.G0,nodes).copy()    
                    edges = (nx.minimum_spanning_tree(Gx,'dist')).edges() 
                    # edges = nx.approximation.steiner_tree(Gx, nodes, weight='dist', method = 'kou').edges()
                    ListEdges += edges
                    # print(p,Elist,Tlist,Adj_Shortest_edge(algo,p,Tlist), edges)
                else : 
                    ListEdges += Bus_(algo, p, Elist)

            else : 
                ListEdges += [(p,e) for e in Elist]

        G = algo.G0.edge_subgraph(ListEdges).copy() 

    elif mode == 'T0':
        Tlist = algo.CombNodes['T']
        G = algo.G0.edge_subgraph(edgesPtoE + edgesEtoC).copy() 
        PtoTConnect = algo.PtoTConnect
        for p, PtoTedges in PtoTConnect.items(): 
            if algo.G0.nodes()[p]['Bus']:
                Pedges = list(G.edges(p))
                G.remove_edges_from(Pedges)
                PtoTedges  = PtoTConnect[p]
                G.add_edges_from(PtoTedges)        

                Tlist = np.unique(np.array(PtoTedges).ravel())
                Tlist = Tlist[Tlist != p].tolist()
                Elist = indiv['PtoE'][p]

                TtoEconnect = Cluster_(algo, Tlist, Elist)   
                # print(TtoEconnect)
                for t, Elist in TtoEconnect.items():
                    G.add_edges_from(Bus_(algo, t, Elist))
    
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

    #G to Digraph
    Gd = nx.DiGraph()
    for p in Pnodes:
        path = nx.shortest_path(G, source = p)
        for c in Cnodes:
            if c in path:
                p = path[c]
                Gd.add_edges_from(nx.utils.pairwise(p))

    nx.set_node_attributes(Gd,algo.G0.nodes)
    # galere set_edge_attributes marche pas pour digraph obliger de reconstruire manuellement
    attrs = {(n0, n1) : algo.G0[n0][n1] for  n0, n1 in Gd.edges}
    nx.set_edge_attributes(Gd,attrs)
    toc = time.perf_counter()
    indiv['G'] = Gd
    indiv['time'] = round(toc - tic , 6)
    return indiv

def IndivLine_(algo, indiv):

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

def debit_v3(algo,indiv):

    G = indiv['G']
    Q0 = np.arange(0.1,80,0.1)
    res = {}
    PressionList, DebitList = {}, {}
    attrs = {}
    for p, line in indiv['IndivLine2'].items():    
        for g ,pt , Edict in line:
            # print(p, Edict)

            ax,bx,cx = [algo.DataCategorie['Pompe']['Values'][pt][i] for i in ['a','b','c']]           
            F = ax * Q0**2 + bx*Q0 +cx
            Qx = 0
            Qlist, Alist, CcList = [] , [], []
            Start = p
            CListTotal = []
            for e, Clist in Edict.items():
                # print(p,g,Start,e)
                coef_PtoE = nx.shortest_path_length(G,Start,e,'coeff')
                CoeffEtoC = np.array([G[e][c]['coeff'] for c in Clist])
                CoeffC    = np.array([G.nodes[c]['a'] for c in Clist])
                CcList.append(CoeffC)
                CoeffE    = G.nodes[e]['a']
                
                F = F - coef_PtoE*(Q0-Qx)**2
                F[F<0] = 0              

                A = CoeffE + CoeffEtoC + CoeffC
                Alist.append(A)
                Qlist.append(np.sqrt(F / A[:,np.newaxis]))
                Qi = np.vstack(Qlist)
                Qx = Qi.sum(0)
                Start = e  # attention si T il est perdu 
                CListTotal += Clist
                #print(Clist, coef_PtoE, distStoE, CoeffEtoC, CoeffC, CoeffE)  
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
    # print(nx.get_node_attributes(G, "Pi"))
                
    indiv['G'] = G
    return indiv

# GEN
def AG_CrossOver(dfx, algo):    
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

def Mutation(row, algo): 
    NewCtoE = copy.deepcopy(row.CtoE)
    idx = np.random.randint(len(NewCtoE))
    D = algo.Comb
    l = [e for e in D['E'] if e != NewCtoE[idx]]
    e = np.random.choice(l,1)[0]
    NewCtoE[idx] = e
    indiv = indiv_create(algo, row,NewCtoE)
    indiv['parent'] =  [row.ID]
    return indiv

# UTILS
def Path1(A,start): 
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

def Path2(A,start,goal):
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

def Multiconnect_(algo,start, nodes):
    G0 = algo.G0
    d = algo.dfline.set_index('ID').dist.to_dict()
    idx = np.argmin([G0[start][n]['dist'] for n in nodes])
    firstnode = nodes[idx]
    G = algo.G0.subgraph(nodes).copy()
    l = list(G.edges)
    print(l)
    l = pd.Series(l).str.join('-').tolist()
    l = tuple(itertools.combinations(l, len(nodes) - 1))
    l = np.array(l)
    idx = np.vectorize(d.__getitem__)(l).sum(1).argmin()
    lines = l[idx]
    return [(start,firstnode)] + [tuple(l.split('-')) for l in lines]

def Bus_(algo, start, nodes):
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
    # print(starts, targets)
    G0 = algo.G0
    G = G0.subgraph(starts + targets).copy()
    # print(G.edges())
    # print(G['T2'])
    DictCluster = collections.defaultdict(list)
    for i, t in enumerate(targets):  
        NodesDist = [G[t][s]['dist'] for s in starts]
        n = starts[np.array(NodesDist).argmin()]
        DictCluster[n].append(t)
    return dict(DictCluster)

def PToT_path_2(algo):
    # T cluster / P 

    G0 = algo.G0
    starts  = ['P{}'.format(t) for t in algo.Comb['P']]
    targets = ['T{}'.format(t) for t in algo.Comb['T']]
    DictCluster = Cluster_(algo, starts, targets)
    PtoTConnect = {}
    for p, Tlist in DictCluster.items():
        if len(Tlist) > 2:
            PtoTConnect[p] = Multiconnect_(algo,p, Tlist)  
        else : 
            PtoTConnect[p] = Bus_(algo, p, Tlist)
    return PtoTConnect

def Line_Name(nodes):    
    LineName = [] 
    ns = nodes[0]
    for n in nodes[1:]:
        LineName.append("{}-{}".format(ns,n))
        ns = n
    return LineName

def Adj_Shortest_edge(algo,start,nodes):
    idx = np.argmin([algo.G0[start][n]['dist'] for n in nodes])
    node = nodes[idx]
    return  (start, node)

