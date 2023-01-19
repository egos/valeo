import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import itertools 
import math
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

def export_excel(algo, Type):
    confs = algo.confs
    dfmap = algo.dfmap
    dfline = algo.dfline.drop(columns= 'path')
    dfcapteur = algo.dfcapteur
    
    output = BytesIO()

    writer = pd.ExcelWriter(output, engine='xlsxwriter')   
    confs.to_excel(writer, sheet_name='confs')  
    dfmap.to_excel(writer, sheet_name='map') 
    if Type : 
        dfline.drop(columns = 'duriteVal').to_excel(writer, sheet_name='lines') 
        dfcapteur.to_excel(writer, sheet_name='slot') 

    writer.save()
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
        
    DictLine, DictPos, A0,Comb, ListWall = new_import(dfmap, DistFactor)
    # print(DictLine)
    sheet_names = pd.ExcelFile(uploaded_file).sheet_names
    print(sheet_names)
    
    dfline = pd.DataFrame(DictLine).T

    dfline.index.name = 'ID'
    dfline.reset_index(inplace = True)
    dfline['duriteType'] = 4
    if "lines" in sheet_names : 
        print("Load line from excel")
        lines_add = pd.read_excel(uploaded_file, sheet_name= 'lines', index_col=0)
        dfline['duriteType'] = lines_add['duriteType'].copy()
        dfline['dist'] = lines_add['dist'].copy()
    # print(DictLine.keys(),dfline.set_index('ID').to_dict().keys())
    DictLine = dfline.set_index('ID').to_dict(orient = 'index')   
  
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
    # print(Dict_durite_val)
    # print(dfline['durite_val'])
    
    Clist = Comb['C']
    Nozzles = [DataCategorie['Nozzle']['Unique'][0]] * len(Comb['C'])
    Nvals   = [DataCategorie['Nozzle']['Values'][n]['a'] for n in Nozzles]
    Nvals  = dict(zip(Clist, Nvals))
    GroupDict = dict(zip(Clist,[0] * len(Clist)))
    GroupDict = np.zeros(len(Clist), dtype= int)
    Group = ~(GroupDict == 0).all()
    
    
    Group0  = [0] * len(Clist)
    Nature0 = [0] * len(Clist)
    Limit0  = np.array([2] * len(Clist))
    if "slot" in sheet_names:
        dfc = pd.read_excel(uploaded_file, sheet_name= 'slot', index_col=0)
        dfc.limit = dfc.limit.astype(float)
        # print(dfc)
    else : 
        dfc = pd.DataFrame()
        dfc['capteur'] = ['C' + str(c) for c in Clist]
        dfc['group'] = [0]*len(dfc)
        dfc['nature'] = ['F']*len(dfc)
        dfc['limit'] = [2.0]*len(dfc)
        # print(dfc)
    Group0 = dfc.group.tolist()        
    # Ctype = algo.DataCategorie['Nozzle']['Unique']  
    Nature0 = dfc.nature.map({'F':0,'R':1}).tolist()
    Limit0 = dfc.limit.values
    # PompeMaxPerSlot = ['P{}'.format(p) for p in Comb['P']]
    Len = len(Comb['P'])
    ListPlimSlot = [len(Comb['E'])] * Len

    Nozzlelimits = Limit0
    algo = dict(
        SheetMapName = SheetMapName,
        uploaded_file = uploaded_file,
        DistFactor = DistFactor,
        dfmap = dfmap,
        Group = Group,
        GroupDict = GroupDict,
        pop = 10,
        fitness = 'dist',
        fitnessCompo = np.array([1,0,0]),
        crossover = 20,
        mutation = 20,
        Nlim = 2.0,   
        Nozzlelimits = Nozzlelimits,      
        Plot = False,
        dfline = dfline,
        DictPos = DictPos,        
        DictLine = DictLine,
        epoch = 0,
        Nindiv = 0,
        Nrepro = 0,
        indivs = [],
        df = [],
        DataCategorie =  DataCategorie,
        Tuyau = [4],     
        Npa = 4,
        Npc = 0,
        PompesSelect = ['Pa'] * 4 + ['Pc'] * 0,  
        Pmax = 4,
        PompeB = False,
        BusActif = True, 
        Split = 'Deactivate', 
        EV = ['Ea'],    
        Nozzles  = Nozzles,  
        Nvals = Nvals,               
        confs = confs,
        Clist = Clist,
        Comb = Comb,
        CombAll = CombAll,
        dist       = dfline.set_index('ID').dist.astype(float).round(1).to_dict(),
        duriteType = dfline.set_index('ID').duriteType.to_dict(),
        duriteVal  = dfline.set_index('ID').duriteVal.to_dict(),
        A0 = A0,
        ListWall = ListWall,
        Group0 = Group0,
        Nature0 = Nature0, 
        Limit0 = Limit0,
        dfcapteur = dfc,
        ListPlimSlot = ListPlimSlot,
        )
    algo = SimpleNamespace(**algo)
    return algo

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
    
    d = collections.defaultdict(list)
    for i in range(Ccount): 
        d[CtoE[i]].append(D['C'][i])
    Econnect = dict(sorted(d.items()))
    Edist = dict(sorted(d.items()))
    Elist = sorted(Econnect)
    Ecount = len(Elist)      
    
    if row is not None :          
        if Ecount > row.Ecount:
            
            NewEtoP = np.random.choice(D['P'],Ecount - row.Ecount)
            EtoP = np.append(row.EtoP, NewEtoP)
            
            for pt in row.Ptype: PompesSelect.remove(pt)  
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
    
    d = collections.defaultdict(list)
    d2 = collections.defaultdict(list)
    for i in range(Ecount): 
        d[EtoP[i]].append(Elist[i]) 
        d2[EtoP[i]].append(Ptype[i])
    Pconnect = dict(sorted(d.items()))   
    Plist = sorted(Pconnect)
    PtypeCo = dict(sorted(d2.items()))     
    Pcount  = len(Plist) 
    PtypeCo = PtypeCo        
    
    List_EtoC = [['E{}-C{}'.format(start, end) for end in List] for start , List in Econnect.items()]
    List_PtoE = [['P{}-E{}'.format(start, end) for end in List] for start , List in Pconnect.items()]
        
    Name = list(itertools.chain.from_iterable(List_EtoC + List_PtoE))
    Name_txt = ','.join(Name)
    
          
    
    indiv = dict(
        Clist = Clist,
        CtoE = CtoE,
        Econnect = Econnect,
        Elist =Elist,
        Ecount = Ecount,
        EtoP = EtoP,
        Pconnect = Pconnect,
        Plist = Plist,
        Pcount = Pcount,
        Ptype0 = Ptype,
        List_EtoC = List_EtoC,
        List_PtoE = List_PtoE,
        Name = Name,
        ID = algo.Nindiv,
        parent = [],
        Name_txt = Name_txt,
        Epoch = algo.epoch,                 
    )    
    indiv = Gen_Objectif(algo, indiv)
    algo.indivs.append(indiv)
    algo.Nrepro += 1  
    algo.Nindiv += 1   

    return indiv

def Gen_Objectif(algo, indiv):
    # Pompe excepetion Pb & Bus 
    Option = 'Na'
    Name = indiv['Name']
    Ptype = indiv['Ptype0']    
    Ecount = indiv['Ecount']
    EtoP = indiv['EtoP']
    BusName = [],
    BusDist= [],
    d = collections.defaultdict(list)
    for i in range(Ecount): 
        d[EtoP[i]].append(Ptype[i])
    PtypeCo = dict(sorted(d.items()))   
    PompesCo = PtypeCo 
    duriteVal = dict(zip(Name, [algo.duriteVal[line] for line in Name]))  
    distList = [algo.dist[line] for line in Name]    
    dist_Connect = dict(zip(Name,distList)) 
    
    DictPompesCount = collections.defaultdict(list)
    DictPompesFinal = collections.defaultdict(list)
    if algo.PompeB & (not algo.Group) :
        # Pompe 2 on change 'Pa' en Pb si pas de group et PompeB True 
        # 1er passage pour identifier Pb
        for slot , ptList in PtypeCo.items():
            # print(slot , ptList)
            idx = []
            for i, pt in enumerate(ptList): 
                if pt =='Pa' :
                    idx.append(i)
                    if len(idx) == 2: 
                        ptList[idx[0]] = 'Pb'
                        ptList[idx[1]] = 'Pb'
                        idx = []
            PtypeCo[slot] = ptList  
        # 2ieme passage pour creer decompte pompe final    
        for slot , ptList in PtypeCo.items():
            idx = []
            for i, pt in enumerate(ptList): 
                if pt =='Pb' : 
                    idx.append(i)
                    if len(idx) == 2: 
                        DictPompesFinal[slot].append('Pb') 
                        idx = []
                else :
                   DictPompesFinal[slot].append(pt) 
        PompesCo = dict(DictPompesFinal)   
        Option = 'Pb'   
                  
    elif algo.BusActif & (not algo.Group):

        dist_Connect, BusName = Bus_Connection(algo, indiv, dist_Connect)
        duriteVal    = dict(zip(Name, [algo.duriteVal[line] for line in BusName]))
        BusDist      = dict(zip(Name, [algo.dist[line] for line in BusName]))
        
        # partie pompe P rand sur pt dipo et replique sur len(ptype) =  tromper calcul debit
        Ptype = []
        # PtypeCo = {}
        for p , ptList in PtypeCo.items():
            pt = np.random.choice(ptList)
            Ptype += [pt] * len(ptList)
            PtypeCo[p] = [pt] * len(ptList)
            DictPompesFinal[p].append(pt)
        PompesCo = dict(DictPompesFinal)  
        Option = 'Bus'
    elif algo.Group:
        Option = 'Gr'
        if algo.Split : Option= 'Split'
    
    DictPompesCount = {} 
    PompesTot = {}  
    L = []
    for i, ptlist in PompesCo.items():
        DictPompesCount['P{}'.format(i)] = dict(Counter(ptlist))
        L += ptlist
    PompeSum= dict(Counter(L))
    dist = sum([dist_Connect[line] for line in Name])
    # dist = round(dist,2)
    dist = int(100*dist)
        
    d = dict(        
        Option = Option,
        Ptype = Ptype,   
        dist_Connect = dist_Connect, 
        dist = dist,
        BusName = BusName,
        BusDist= BusDist,
        PtypeCo = PtypeCo,  
        PompesCo = PompesCo ,
        PompeCount = DictPompesCount,
        PompeSum = PompeSum,
        duriteVal  = duriteVal ,        
    )
    indiv.update(d)
           
    # calcul debit ['PressionList','DebitList','Esplit','Debit']
    d =  Calcul_Debit(algo ,indiv, Split = algo.Split)
    indiv.update(d)
    
    Esplit = indiv['Esplit']
    EvCount = {}
    EvTot = 0
    for Eslot, Elist in Esplit.items():
        EvCount['E{}'.format(Eslot)] = len(Elist)
        EvTot += len(Elist)
    indiv['EvCount'] = EvCount
    indiv['EvSum'] = EvTot
    
    
    info , d = calcul_Masse_cout(indiv, algo)
    indiv.update(d)
    
    # d =  Calcul_Debit(algo ,indiv, Split = 'Forced')
    # d = {k + '_S' : v for k,v in d.items()}
    
    ListFitness = ['dist','Masse','Cout']
    fitness = 0
    for i in range(3): fitness+= indiv[ListFitness[i]] * algo.fitnessCompo[i]        
    indiv['fitness'] = round(fitness,5)
    
    indiv['Alive'] = False if  (np.array(indiv['PressionList']) < algo.Nozzlelimits).any() else True 
    ListPlimSlot = algo.ListPlimSlot
    cond = True
    for p, v in indiv['PompesCo'].items():
        Cible =  ListPlimSlot[p] 
        if len(v) > Cible : cond = cond & False 
        # print(p, v , len(v), Cible, cond)
    indiv['Alive'] = indiv['Alive'] & cond   
    return indiv 

def Bus_Connection(algo, indiv, dist_Connect):
    New_dist_Connect = copy.deepcopy(dist_Connect)
    BusName = copy.deepcopy(indiv['Name'])
    Pconnect = indiv['Pconnect']
    
    dfx0 = algo.dfline.copy()
    dfx0['a'] = dfx0.ID.str.split('-').str[0]
    dfx0['b'] = dfx0.ID.str.split('-').str[1]
    
    NameListNew  = []
    DictMapName = {}
    
    for i, (p,Elist) in enumerate(Pconnect.items()):

        s = 'P{}'.format(p)
        ElistName = ['E{}'.format(e) for e in Elist]
        ListMask = [s] + ElistName
        mask0 = dfx0.a.isin(ListMask) & dfx0.b.isin(ListMask)
        dfx = dfx0[mask0].copy()
        path,dist,lines = [s] ,[], []
        
        # pour chaque slot de P commence avec s = Px et avance par iter sur le plus proche E & crop  dfx = dfline 
        # l'astuce c'est que qu'on conserve les lines P-E mais en changeant les valeurs de dist
        while len(dfx)>0:
            mask = dfx.ID.str.contains(s)
            x  = dfx[mask].dist.values.argmin()    
            cx = dfx[mask][['a','b']].iloc[x].values
            line = dfx[mask].iloc[x].ID
            lines.append(line)
            NameListNew.append(line)
            dist.append(dfx[mask].dist.values.min())    
            dfx = dfx[~mask]
            s = cx[cx!=s][0]
            path.append(s)
            
        distCumsum = np.array(dist).cumsum()
        PxConnect = ['{}-{}'.format(path[0],s) for s in path[1:]]
        DictMapName.update(dict(zip(PxConnect,lines)))
        d = dict(zip(PxConnect,distCumsum))
        New_dist_Connect.update(d)
        # print(path,lines,dist, distCumsum, PxConnect)

    # quelle galere obliger de passer par Series pour map les old  et new name 
    BusName = pd.Series(BusName).replace(DictMapName).tolist()
    
    return New_dist_Connect, BusName

def Indiv_reverse(Name,algo, Ptype = None):
    NameList = Name.split(',')
    Clist = algo.Clist
    CtoE = {}
    EtoP = {}
    for n in NameList:
        if n[0] == 'E':
            c = int(n[-1])
            e = int(n[1])
            CtoE[c] = e

        if n[0] == 'P':
            e = int(n[-1])
            p = int(n[1])
            EtoP[e] = p
            
    d = dict(sorted(CtoE.items()))
    Clist = list(d.keys())
    CtoE = list(d.values())
    d = dict(sorted(EtoP.items()))
    EtoP = list(d.values())
    print('Indiv_reverse',Clist,CtoE, EtoP)
    indiv = indiv_create(algo, row = None, NewCtoE = CtoE, IniEtoP = EtoP)
    return indiv

def calcul_Masse_cout(indiv, algo): 
    dmasse = {}
    dcout = {}
    
    for Categorie in ['Pompe', 'Tuyau','EV']:
        v = algo.DataCategorie[Categorie]['Values']
        if Categorie == 'Pompe' : 
            # a = indiv['PompesCo'].values()
            Ptype = list(itertools.chain.from_iterable(indiv['PompesCo'].values()))
            masse = 0
            cout  = 0
            for pt in Ptype: 
                factor = 1
                masse += factor * algo.DataCategorie[Categorie]['Values'][pt]['Masse']
                cout  += factor * algo.DataCategorie[Categorie]['Values'][pt]['Cout']             
            dmasse[Categorie] = int(masse)
            dcout[Categorie]  = int(cout)
        if Categorie == 'Tuyau' :
            # distPerLine  = np.array([algo.dist[line] for line in indiv['Name']])
            distPerLine  = np.array(list(indiv['dist_Connect'].values()))
            Name = indiv['Name']
            if indiv['Option'] == 'Bus' : 
                distPerLine  = np.array(list(indiv['BusDist'].values()))
                Name = indiv['BusName']
            MassePerLine = np.array([v[algo.duriteType[line]]['Masse'] for line in Name])
            CoutPerLine  = np.array([v[algo.duriteType[line]]['Cout']  for line in Name])
            dmasse[Categorie] = (distPerLine * MassePerLine).sum()
            dcout[Categorie]  = (distPerLine * CoutPerLine).sum()
        if Categorie == 'EV' :
            Ccount = len(algo.Comb['C'])
            Ccount = sum([len(l) for l in indiv['Esplit'].values()])
            Factor = Ccount
            Name = algo.EV  
            dmasse[Categorie] = int(sum([Factor * v[n]['Masse'] for n in Name]))
            dcout[Categorie]  = int(sum([Factor * v[n]['Cout']  for n in Name]))
            
    dmasse['Reservoir'] = 600
    dcout['Reservoir']  = 30  
    info = [dmasse, dcout]
    # print(dmasse.values())
    Masse = round(sum(dmasse.values()),2)
    Masse = int(Masse)
    Cout = round(sum(dcout.values()),2)
    Cout = int(Cout)
    # print(info)
    return  info, { 'Masse' : Masse, 'Cout' : Cout}

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

def debit(algo,indiv,debitinput, grouped = True, split = True):
    # print(d_EtoC_list,d_PtoE,Clist)
    pompe,ev,ClistG,pt = debitinput.values()
    d_EtoC_list = np.array([algo.dist['E{}-C{}'.format(ev,c)] for c in ClistG])
    d_PtoE      = indiv['dist_Connect']['P{}-E{}'.format(pompe,ev)]
    # print(d_EtoC_list,d_PtoE,ClistG)
    
    if not grouped : split = False
    PompeType = pt

    p = [-5.16e-04, -1.54e-02, 4.87]
    p = [algo.DataCategorie['Pompe']['Values'][pt][i] for i in ['a','b','c']]

    cE0 = 7.64e-04
    coef_E = 0 if split else cE0
    
    coef_C  = 0.036
    coef_C  = [algo.Nvals[i] for i in ClistG]
    coef_C  = np.array(coef_C)
    coef_d_EtoC  = 2.35e-04
    coef_d_EtoC = np.array([algo.duriteVal['E{}-C{}'.format(ev,c)] for c in ClistG])
    # coef_d_PtoE = algo.duriteVal['P{}-E{}'.format(pompe,ev)] 
    coef_d_PtoE = indiv['duriteVal']['P{}-E{}'.format(pompe,ev)] 
    
    
    A = coef_E + d_EtoC_list * coef_d_EtoC + coef_C 
    # print(A)
    Z = ( A**-0.5).sum() if grouped else A**-0.5
    coef_E = cE0 if split else 0
    As = p[0] - (coef_d_PtoE * d_PtoE) - 1/(Z**2) - coef_E
    Bs = p[1]
    Cs = p[2]
    delta = (Bs**2) - (4 * As * Cs)
    Qt  = np.array((- Bs - delta**0.5)/(2*As))
    
    Pt = np.array(Qt**2 / Z**2)
    # print(Pt)
    if (PompeType == 'Pc') &  (Pt >=  algo.Nozzlelimits[ClistG]).all():
        Pt = algo.Nozzlelimits[ClistG]*1.1 #* Pt/Pt

    a0 = p[0] * (Qt**2) + p[1] * Qt + p[2] - Pt
    Qi = (Pt / A)**0.5
    Pi = coef_C * (Qi**2)
    key = ['Qt','Pt','Qi','Pi']
    val = [Qt, Pt, Qi, Pi]
    val = [v.round(2) for v in val]
    return dict(zip(key,val))

def Calcul_Debit(algo ,indiv, Split):
    D = algo.Comb  
    Group = algo.Group
    gr = algo.GroupDict 
    Clist = D['C']
    Econnect = indiv['Econnect']
    Pconnect = indiv['Pconnect']
    dist_Connect = indiv['dist_Connect']
    EtoP = indiv['EtoP']
    Ptype = indiv['Ptype']
    Pression = []
    Debit = []
    # Data = {}
    # Pression_C = []
    # on loop sur chaque EV pour connect to C et faire calcul Pt Qt coté pompe et Pi Qi coté Capteur
    Cpression = {}
    Cdebit = {}
    grouped = False
    EsplitDict = collections.defaultdict(list)
    VerifPression = True
    for i, (e,EClist) in enumerate(Econnect.items()):
        p = EtoP[i]
        pt = Ptype[i]
        name = 'P{}-E{}'.format(p,e)
        d = collections.defaultdict(list)
        # differencier groupe et non groupé dans Eclist        
        for c in EClist:
            d[gr[c]].append(c)
            
        for g,ClistG in d.items():
            
            if g == 0 : grouped = False
            else : grouped = True
            
            d_EtoC_list = np.array([algo.dist['E{}-C{}'.format(e,c)] for c in ClistG])
            d_PtoE = dist_Connect['P{}-E{}'.format(p,e)]
            # d_PtoE = dist_Connect['P{}-E{}'.format(p,e)]
            # print(d_EtoC_list, d_PtoE)
            debitinput = dict(
                p = p,
                e = e,
                ClistG = ClistG,
                pt = pt,                
            )
            # print(ClistG)            
            if grouped : 
                if Split ==  'Deactivate' :
                    res = debit(algo,indiv,debitinput, grouped = True, split = False)
                    for c in ClistG : EsplitDict[e].append(c) 
                else   :
                    res = debit(algo,indiv,debitinput, grouped = True, split = True)              
                    Pi = list(res['Pi'])         
                    VerifPression = (np.array(Pi) < algo.Nozzlelimits[ClistG]).any()
                    if VerifPression & (Split == 'Auto'):
                        res = debit(algo,indiv,debitinput, grouped = True, split = False)     
                        for c in ClistG : EsplitDict[e].append(c)   
                    #Split == Forced
                    else : 
                        if len(ClistG) > 1 :EsplitDict[e].append(tuple(ClistG))   
                        else : EsplitDict[e].append(ClistG[0]) 
            else : 
                res = debit(algo,indiv,debitinput, False, split = False)          
                for c in ClistG : EsplitDict[e].append(c)
                                
            Pi = list(res['Pi']) 
            PressionConnect = dict(zip(ClistG, Pi))
            Cpression.update(PressionConnect)
            
            Qi = list(res['Qi'])            
            Cdebit.update(dict(zip(ClistG, Qi)))
            Debit = Debit + list(res['Qi'])                
            
            # Data[name] = res        
            # Pression = Pression + list(res['Pi'])          
            # print(dc,dp,Clist,list(res['Pi']))
            # Pression_C = Pression_C + [PressionConnect]
            # print(i,e,EClist, grouped, VerifPression, g,ClistG, PressionConnect, dict(EsplitDict))
    PressionList = [Cpression[i] for i in D['C']]
    DebitList    = [Cdebit[i] for i in D['C']]
    # print(Cpression)
    SumDebit = round(sum(Debit),1)
    # keys = ['info','Data','Pression','Debit','SumDebit']
    # vals = [info, Data,Pression, Debit, SumDebit]     
    keys = ['PressionList','DebitList','Esplit','Debit']
    vals = [PressionList, DebitList,dict(EsplitDict), SumDebit] 
    d = dict(zip(keys,vals))
    return d

def new_import(dfmap, DistFactor):
    # print('new_import')    
    SlotColor = {'C' : 10, 'E': 20, 'P' : 30}
    slots = ['C','P','E']    
    
    A0 = dfmap.values
    Size = max(A0.shape)
    # DistFactor = Size / DistFactor
    
    Comb = collections.defaultdict(list)
    DictPos = {}    
    ListBegin = []
    ListEnd = []   
    ListWall = (np.argwhere(A0[1:-1,1:-1] == 1)+1).tolist()
        
    slots = ['C','P','E']
    slotsN = dict(zip(slots,[0,0,0]))

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
            dist = (np.abs(np.diff(path.T)).sum() * DistFactor).round(2)
            
            if end[0] == 'C' : ID = begin + '-' + end
            else : ID = end + '-' + begin

            DictLine[ID] = {'path' : path, 'dist' : dist}        

    ListEv =  ['E' + str(n) for n in Comb['E']]
    it = itertools.combinations(ListEv, 2)
    ListEtoE = list(it)
    ListEtoE
    print(ListEtoE)

    DictEtoE = {}
    for begin, end in ListEtoE:
        start = DictPos[begin]
        A = Ax.copy()
        A1 = Path1(A,start)
        goal = DictPos[end]        
        path = Path2(A1.copy() ,start,  goal)
        path = np.array(path)       
        dist = (np.abs(np.diff(path.T)).sum() * DistFactor).round(2)

        # if end[0] == 'C' : ID = begin + '-' + end
        # elif begin[0] == 'P' : ID = begin + '-' + end    
        # else :
        ID = end + '-' + begin
        DictEtoE[ID] = {'path' : path, 'dist' : dist}

    DictLine.update(DictEtoE)   
    
    
    return DictLine, DictPos, A0,dict(Comb), ListWall
      
def new_plot(algo,SelectLine, SelectSlot):
    DictLine = {k:v for k,v in algo.DictLine.items() if k in SelectLine}
    DictPos  = {k:v for k,v in algo.DictPos.items()  if k in SelectSlot}
    LenPath = len(DictLine)
    ListWall = algo.ListWall
    A0 = algo.A0
    Ymax , Xmax = A0.shape
    PlotColor = {'C' : "#93c9ee", 'E': '#a2ee93', 'P' : "#c593ee"}
    fig, ax = plt.subplots(figsize = (8,8))
    f = ax.imshow(np.zeros(A0.shape), cmap='gray',vmin=0,vmax=1)  
    f = ax.add_patch(mpatch.Rectangle((0,0), Xmax-1, Ymax-1, color='#d8d8d8'))
    # masked = np.ma.masked_where(A0 <= 1, A0)
    MinOffset = LenPath*0.03
    MinOffset = MinOffset if MinOffset < 0.4 else 0.4
    offset = np.linspace(-MinOffset,MinOffset,LenPath)
    for i, (slot,data) in enumerate(DictLine.items()):
        n = offset[i]  
        p = data['path']
        if slot[0] == 'E' : 
            f= ax.plot(p[:,1]+n,p[:,0]+n,"#32cdff", linewidth=2, zorder=1, linestyle ='-')
        else : 
            f =ax.plot(p[:,1]+n,p[:,0]+n,"#3286ff", linewidth=3, zorder=1, linestyle ='-')

    style = dict(size= 15 * 9 / Ymax, color='black')
    for slot, pos in DictPos.items(): 
        x , y = pos
        Type = slot[0]
        color = PlotColor[Type]
        f = ax.add_patch(mpatch.Rectangle((y-0.45,x-0.45), 0.9, 0.9, color=color))
        f = ax.add_patch(mpatch.Rectangle((y-0.45,x-0.45), 0.9, 0.9, color='black', fill = None))
        f = ax.text(y, x+0.1,slot[1:] , **style,  ha='center', weight='bold') 
        
    for x,y in ListWall: 
        f = ax.add_patch(mpatch.Rectangle((y-0.5,x-0.5), 1, 1, color='black'))    
    
    
    return  fig

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


  