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
from sklearn.cluster import KMeans

#ctrl k ctrl &  //// fold level1 

def export_excel_test(algo, ListResultsExport):
    Col = 0
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()
    # plt.gcf().set_size_inches(4, 4)
    for i in range(len(ListResultsExport)) : 
        row = ListResultsExport[i]['row']
        fig = ListResultsExport[i]['fig']
    # for idx , row in df1.iterrows(): 
    # row = df1.loc[0]
        # ElemsList = ['Clist','Elist','Plist']
        # Elems = ['C','E','P']
        # SelectSlot = []
        # List_EtoC = row.List_EtoC
        # List_PtoE = row.List_PtoE
        # for n in range(3):
        #     SelectSlot+= ['{}{}'.format(Elems[n],i) for i in row[ElemsList[n]]]
        # SelectLine = row.Name
        # if row.Option == 'Bus' :   SelectLine = row.BusName
                    
        # fig = new_plot(algo, SelectLine, SelectSlot)
        # Width , Heigth  = fig.get_size_inches()
        # print(Width , Heigth , fig.dpi)
        # # print(fig.get_size_inches()*fig.dpi)
        # plt.gcf().set_size_inches(Width/2, Heigth/2)
        
        # worksheet.write(0,0, 'Hello')
        # print(row)
        # plt.gcf().set_size_inches(4, 4)
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
    dfcapteur = algo.dfcapteur
    
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
    dfline['s'] = dfline.ID.str.split('-').str[0]
    dfline['t'] = dfline.ID.str.split('-').str[1]
    dfline['Connect'] = dfline['s'].str[0] + dfline['t'].str[0]
  
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
    DictLine = dfline.set_index('ID').to_dict(orient = 'index')   
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
        sheet_names = sheet_names,
        SheetMapNameList = SheetMapNameList,
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
        df = pd.DataFrame(),
        DataCategorie =  DataCategorie,
        Tuyau = [4],     
        Npa = 10,
        Npc = 4,
        PompesSelect = ['Pa'] * 10 + ['Pc'] * 0,  
        Pmax = 10,
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
        SaveRun = [],
        iterations = 1,
        PlotLineWidth = [1,3],
        ListBusPactif = [True] * len(Comb['P']),
        DebitCalculationNew = True,
        G0 = nx.from_pandas_edgelist(dfline, 's', 't', ['dist']),
        PtoTConnect = {},
        Tactived = False
        )
    algo = SimpleNamespace(**algo)
    if 'T' in Comb:
        algo.PtoTConnect = PToT_path(algo)  
    
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
        # ListBusActif = [True] * len(Plist) ,
        ListBusActif = [algo.ListBusPactif[p] for p in Plist]          
    )    
    indiv = Gen_Objectif(algo, indiv)
    algo.indivs.append(indiv)
    algo.Nrepro += 1  
    algo.Nindiv += 1   
    # print(indiv.keys())
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
    BusConnectDict = []
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
                  
    elif algo.BusActif:

        dist_Connect, BusName, BusConnectDict = Bus_Connection(algo, indiv, dist_Connect)
        duriteVal    = dict(zip(Name, [algo.duriteVal[line] for line in BusName]))
        BusDist      = dict(zip(Name, [algo.dist[line] for line in BusName]))
        
        # partie pompe P rand sur pt dipo et replique sur len(ptype) =  tromper calcul debit
        Ptype = []
        # PtypeCo = {}
        for p , ptList in PtypeCo.items():
            pt = np.random.choice(ptList)
            # pt = p
            Ptype += [pt] * len(ptList)
            PtypeCo[p] = [pt] * len(ptList)
            DictPompesFinal[p].append(pt)
        PompesCo = dict(DictPompesFinal)  
        Option = 'Bus'
    elif algo.Group:
        Option = 'GroupDict'
        if algo.Split : Option= 'Split'
    
    DictPompesCount = {} 
    PompesTot = {}  
    L = []
    for i, ptlist in PompesCo.items():
        DictPompesCount['P{}'.format(i)] = dict(Counter(ptlist))
        L += ptlist
    PompeSum= dict(Counter(L))
    dist = sum([dist_Connect[line] for line in Name])
    if Option == 'Bus':
        dist = sum([BusDist[line] for line in Name])
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
        BusConnectDict = BusConnectDict       
    )
    indiv.update(d)
           
    # calcul debit ['PressionList','DebitList','Esplit','Debit']
    
    # print(d)
    # print(algo.GroupDict,algo.BusActif)
    if ('T' in algo.Comb) & algo.Tactived:
        BusTNames = []
        for p in indiv['Plist']:
            L = Tconnection_v4(algo, indiv,p)
            BusTNames = BusTNames + L
        indiv['BusName'] = BusTNames
        indiv['Option'] = 'Na'


    if algo.DebitCalculationNew:           
        IndivLine = indiv_lines_conf(algo, indiv)        
        indiv['IndivLine'] = IndivLine
        res,d = New_debit(algo,indiv)
        indiv.update(d)
    else : 
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
    # indiv['mc_details'] = info  
    indiv['DetailsMasse'] = info[0]
    indiv['DetailsCout'] = info[1]
    # d =  Calcul_Debit(algo ,indiv, Split = 'Forced')
    # d = {k + '_S' : v for k,v in d.items()}
    
    ListFitness = ['dist','Masse','Cout']
    fitness = 0
    for i in range(3): fitness+= indiv[ListFitness[i]] * algo.fitnessCompo[i]        
    indiv['fitness'] = round(fitness,5)
    # print(indiv['PressionList'], algo.Nozzlelimits) 
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
    ListBusActif = indiv['ListBusActif']
    
    dfx0 = algo.dfline.copy()
    dfx0['a'] = dfx0.ID.str.split('-').str[0]
    dfx0['b'] = dfx0.ID.str.split('-').str[1]
    
    NameListNew  = []
    DictMapName = {}
    BusConnectDict = []
    for i, (p,Elist) in enumerate(Pconnect.items()): 
        if ListBusActif[i]: 
            # print(i, (p,Elist))
            # # EvDrop = 4
            # if EvDrop in Elist:         Elist.remove(EvDrop)

            s = 'P{}'.format(p)
            ElistName = ['E{}'.format(e) for e in Elist]
            ListMask = [s] + ElistName
            mask0 = dfx0.a.isin(ListMask) & dfx0.b.isin(ListMask)
            dfx = dfx0[mask0].copy()
            path,dist,lines = [s] ,[], []
            Elist = []
            
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
                Elist.append(int(s[1:]))
                path.append(s)
            BusConnectDict.append((p,Elist))  
            # BusConnectDict.append({'p': p ,'e' :Elist}) 
            # print(BusConnectDict)
            distCumsum = np.array(dist).cumsum()
            PxConnect = ['{}-{}'.format(path[0],s) for s in path[1:]]
            DictMapName.update(dict(zip(PxConnect,lines)))
            # print(p,Elist, PxConnect)   
            d = dict(zip(PxConnect,distCumsum))
            # BusConnectDict[p].append()
            New_dist_Connect.update(d)
            # print('bus',path[0], p,Elist, PxConnect,path,lines,dist, distCumsum, PxConnect)
        else : 
            for e in Elist:
                line =  (p, [e])
                BusConnectDict.append(line)

    # quelle galere obliger de passer par Series pour map les old  et new name 
    # mais pratique si pas de changement cas non bus 

    BusName = pd.Series(BusName).replace(DictMapName).tolist()
    
    return New_dist_Connect, BusName, BusConnectDict

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
            dmasse[Categorie] = round(masse,1)
            dcout[Categorie]  = round(cout,1)
        if Categorie == 'Tuyau' :
            # distPerLine  = np.array([algo.dist[line] for line in indiv['Name']])
            distPerLine  = np.array(list(indiv['dist_Connect'].values()))
            Name = indiv['Name']
            if indiv['Option'] == 'Bus' : 
                distPerLine  = np.array(list(indiv['BusDist'].values()))
                Name = indiv['BusName']
            MassePerLine = np.array([v[algo.duriteType[line]]['Masse'] for line in Name])
            CoutPerLine  = np.array([v[algo.duriteType[line]]['Cout']  for line in Name])
            masse = (distPerLine * MassePerLine).sum()
            cout = (distPerLine * CoutPerLine).sum()
            dmasse[Categorie] = round(masse,1)
            dcout[Categorie]  = round(cout,1)
        if Categorie == 'EV' :
            Ccount = len(algo.Comb['C'])
            Ccount = sum([len(l) for l in indiv['Esplit'].values()])
            Factor = Ccount
            Name = algo.EV  
            masse = sum([Factor * v[n]['Masse'] for n in Name])
            cout  = sum([Factor * v[n]['Cout']  for n in Name])
            dmasse[Categorie] = round(masse,1)
            dcout[Categorie]  = round(cout,1)
            
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
    
    # print(debitinput, coef_d_PtoE, d_EtoC_list, coef_d_EtoC, coef_C, coef_d_PtoE * d_PtoE)
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
    # if (PompeType == 'Pc') &  (Pt >=  algo.Nozzlelimits[ClistG]).all():
    #     Pt = algo.Nozzlelimits[ClistG]*1.1 #* Pt/Pt

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
    GroupDict = algo.GroupDict 
    # print(GroupDict)
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
            d[GroupDict[c]].append(c) 
        for g,ClistG in d.items():
            # g = 0 c'est le code des list de non groupés 
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
            # print(debitinput)
            # print(ClistG) 
            # partie deguelasse a cause du split            
            if grouped : 
                if Split ==  'Deactivate' :
                    res = debit(algo,indiv,debitinput, grouped = True, split = False)
                    for c in ClistG : EsplitDict[e].append(c) 
                else   :
                    # je test si split possible sinon je desactive = galere de code
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
    # it = itertools.permutations(ListEv, 2) # pour avoir les 2 sens sur EtoE
    it = itertools.combinations(ListEv, 2)
    ListEtoE = list(it)
    ListEtoE
    print(ListEv, ListEtoE)

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
      
def new_plot(algo,SelectLine, SelectSlot, hideEtoC = False):
    DictLine = {k:v for k,v in algo.DictLine.items() if k in SelectLine}
    DictPos  = {k:v for k,v in algo.DictPos.items()  if k in SelectSlot}
    LenPath = len(DictLine) # = nombre de ligne 
    ListWall = algo.ListWall
    A0 = algo.A0
    Ymax , Xmax = A0.shape
    print(A0.shape)
    PlotColor = {'C' : "#93c9ee", 'E': '#a2ee93', 'P' : "#c593ee", 'T': '#C0C0C0'}
    fig, ax = plt.subplots(figsize = (8,8))
    f = ax.imshow(np.zeros(A0.shape), cmap='gray',vmin=0,vmax=1)  
    f = ax.add_patch(mpatch.Rectangle((0,0), Xmax-1, Ymax-1, color='#d8d8d8'))
    # masked = np.ma.masked_where(A0 <= 1, A0)
    # MinOffset = LenPath*0.03
    MinOffset = LenPath*0.4
    MinOffset = MinOffset if MinOffset < 0.4 else 0.4
    # MinOffset = 0.5
    offset = np.linspace(-MinOffset,MinOffset,LenPath)
    for i, (line,data) in enumerate(DictLine.items()):
        n = offset[i]  
        p = data['path']
        
        slots = line.split('-')
        # print(line, slots)
        if (slots[1][0] == 'C') :
            if (not hideEtoC): 
                f= ax.plot(p[:,1]+n,p[:,0]+n,"white", linewidth=2, zorder=1, linestyle ='-')
        elif slots[1][0] == 'E' : 
            f= ax.plot(p[:,1]+n,p[:,0]+n,"#2CB73D", linewidth=2, zorder=1, linestyle ='-')
        else : 
            f =ax.plot(p[:,1]+n,p[:,0]+n,"#3286ff", linewidth=2, zorder=1, linestyle ='-')
            
    for x,y in ListWall: 
        f = ax.add_patch(mpatch.Rectangle((y-0.4,x-0.4), 1, 1, color='black')) 
    # style = dict(size= 15 * 9 / Ymax, color='black')
    style = dict(size = 8, color='black')
    rs = 0.4 # taille slot
    for slot, pos in DictPos.items(): 
        x , y = pos
        Type = slot[0]
        color = PlotColor[Type]
        f = ax.add_patch(mpatch.Rectangle((y-rs,x-rs), rs*2, rs*2, color=color))
        f = ax.add_patch(mpatch.Rectangle((y-rs,x-rs), rs*2, rs*2, color='black', fill = None))
        f = ax.text(y, x + 0.2,slot[1:] , **style,  ha='center', weight='bold') 
        
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

def indiv_lines_conf(algo, indiv):
    GroupDict = algo.GroupDict
    Econnect = indiv['Econnect']
    Ptype    = indiv['Ptype']
    Pconnect = indiv['Pconnect']
    GroupDict = algo.GroupDict
    Ptype = indiv['Ptype']
    Cconnect =  { v: k for k, l in indiv['Econnect'].items() for v in l } # a bouger de place 

    Pline = indiv['BusConnectDict']
    # print(Pline)
    IndivLine = []
    Gconnect = {}
    # loop sur BusConnectDict generer via func bus_connection qui doit prendre en compte le remove des PtoE 
    # list les mutiple connection au meme slot de pompe = remove ou autre solution 
    
    for i, (p, Elist) in enumerate(Pline):
        pt = Ptype[i]    
        gr = collections.defaultdict(list)
        Gconnect = {}

        # gr dict pour une line
        for e in Elist: 
            # on fait 1 le dict des group par {gr : c}  
            Clist = indiv['Econnect'][e]
            for c in Clist:
                gr[GroupDict[c]].append(c) 
        gr = dict(gr)
        # print(p, Elist, gr, sorted(gr))

        # etape la plus bizarre on utile le nouveau Connect pour organiser les ev dans l'ordre et par groupe pour juste lister ensuite dans calcul debit
        for e in Elist:
            for n in sorted(gr): # pas sur quie le sorted serve a quelque chose 
                g = gr[n]
                d = collections.defaultdict(list)
                for c in g :
                    e = Cconnect[c]
                    d[e].append(c)
                Gconnect[n] = dict(d)       

        LineConf = dict(
            p = p,        
            e = Elist,
            gr = gr,
            pt = Ptype[i],
            Gconnect = Gconnect
        )
        
        IndivLine.append(LineConf)
    return IndivLine

def New_debit(algo,indiv):
    res = []
    DictLine = algo.DictLine
    IndivLine = indiv['IndivLine']

    for line in IndivLine:
        pt = line['pt']
        p = line['p']
        Q0 = np.arange(0.1,80,0.1)
        for g , ClistDict in line['Gconnect'].items():
            g , ClistDict
            Elist = ClistDict.keys()
            # ClistDict = line['Gconnect']
            PtoElist  = ['P{}-E{}'.format(p,e) for e in Elist]
             # tres important puisque ordre de calucl a respecter on utilise les valeurs modifié de busDist
            coef_PtoE = [DictLine[line]['duriteVal'] * indiv['BusDist'][line] for line in PtoElist]
            coef_PtoE = np.round(coef_PtoE,6)
            a,b,c = [algo.DataCategorie['Pompe']['Values'][pt][i] for i in ['a','b','c']]
            G = a * Q0**2 + b*Q0 +c
            Qx = 0

            Qlist, Alist = [] , []

            for i, (e,Clist) in enumerate(ClistDict.items()): # changer en list pour avoir l'ordre !!
    #             print(i, e,EClist)
                # calcul des coef comme avant 
                coef_C      = np.array([algo.Nvals[i] for i in Clist])
                d_EtoC      = np.array([algo.dist['E{}-C{}'.format(e,c)] for c in Clist])
                coef_d_EtoC = np.array([algo.duriteVal['E{}-C{}'.format(e,c)] for c in Clist])
                coef_E = 7.64e-04  

                # new si pas de capteur on a juste le coef de droite = 0 normalement 
                # le truc c'est quil faudra faire cette loop dans le cas de group ou le premier Ev dn'est pas activer 
                # donc prendre en compte pour le calcul de coef_PtoE
                G = G - coef_PtoE[i]*(Q0-Qx)**2
                G[G<0] = 0
                # print(coef_E, d_EtoC, coef_d_EtoC, coef_C)
                A = coef_E + d_EtoC * coef_d_EtoC + coef_C 
    #             print(coef_E ,d_EtoC , coef_d_EtoC ,coef_C,  A)
                Alist.append(A.round(3))
                # magie on tile la mtrice sur A step correspond au tableau excel etendu sur la droite
                Qlist.append(np.sqrt(G / A[:,np.newaxis]))
                Qi = np.vstack(Qlist)  # a essayer de passer en append coté numpy 
                Qx = Qi.sum(0)           

            # le offset de -1 pour retrouver la plus proche valeur de Q0 -Qx facile 
            idx = np.searchsorted(Q0 - Qx, -0.1)
            # idx
            Qi = np.vstack(Qlist)[:,idx].round(3)
            Pi = (np.concatenate(Alist)* (Qi**2)).round(3)
            r = dict(
                g = g,
                ClistDict = ClistDict,
                PtoElist = PtoElist,
                coef_PtoE = coef_PtoE,
                Qi = Qi, 
                Pi = Pi,
                Alist = Alist,
            )
            res.append(r)
    
    Debit = {}
    Pression = {}
    for v in res:
        Clist = list(itertools.chain.from_iterable(v['ClistDict'].values()))
        Clist ,v['Qi']
        for i , c in enumerate(Clist):
            Debit[c]    = v['Qi'][i]
            Pression[c] = v['Pi'][i] 
    DebitList    = list((dict(collections.OrderedDict(sorted(Debit.items()))).values()))
    PressionList = list((dict(collections.OrderedDict(sorted(Pression.items()))).values()))
    SumDebit = round(sum(DebitList),1)
    keys = ['PressionList','DebitList','Esplit','Debit']
    vals = [PressionList, DebitList,{}, SumDebit] 
    d = dict(zip(keys,vals))
    return res,d

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
            dist = (np.abs(np.diff(path.T)).sum() * DistFactor).round(2)
            
            if end[0] == 'C' : ID = begin + '-' + end
            else : ID = end + '-' + begin

            DictLine[ID] = {'path' : path, 'dist' : dist}        

    ListEv =  ['E' + str(n) for n in Comb['E']]
    it = itertools.permutations(ListEv, 2) # pour avoir les 2 sens sur EtoE
    # it = itertools.combinations(ListEv, 2)
    ListEtoE = list(it)
    ListEtoE
    print(ListEv, ListEtoE)

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

    DictTt = {}
    if "T" in Comb:
        List1 =  ['T' + str(n) for n in Comb['T']]
        List2 =  ['P' + str(n) for n in Comb['P']]
        ListSlots = List1 + List2
        it = itertools.permutations(ListSlots, 2) # pour avoir les 2 sens sur EtoE
        # it = itertools.combinations(ListSlots, 2)
        ListTt= list(it)
        print(ListSlots, ListTt)

        DictTt = {}
        for begin, end in ListTt:
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
            DictTt[ID] = {'path' : path, 'dist' : dist}

    DictLine.update(DictTt)       
    
    return DictLine, DictPos, A0,dict(Comb), ListWall  

def PToT_path(algo):
    G0 = algo.G0
    Start  = ['P{}'.format(t) for t in algo.Comb['P']]
    Target = ['T{}'.format(t) for t in algo.Comb['T']]
    SlotNames =Start + Target
    L = []
    # backward sur chaque E pour connaitre le bus le plus court vers le plus proche T
    DictPath = collections.defaultdict(list)
    G = G0.subgraph(SlotNames).copy()
    for i, t in enumerate(Target):  
        NodesDist = [G[t][p]['dist'] for p in Start]
        n = Start[np.array(NodesDist).argmin()]
        DictPath[n].append(t)

    DictPath2 = collections.defaultdict(list)
    for k, Slist in dict(DictPath).items():
        G = G0.subgraph([k] + Slist).copy()
        # print([p] + Tlist)
        n = k
        Sconnect = [] 
        SconnectPath = []
        while G.size()>0 :
            NodesAdj  = [x[0] for x in G.adj[n].items()]
            NodesDist = [x[1]['dist'] for x in G.adj[n].items()]
            ns = n
            G.remove_node(n)
            n = NodesAdj[np.array(NodesDist).argmin()]
    #         Sconnect.append(n)
    #         SconnectPath.append("{}-{}".format(ns,n))
            DictPath2[k].append(n)
    DictPath2 = dict(DictPath2)
    return DictPath2

def Tconnection_v4(algo, indiv, pn):
    p = 'P{}'.format(pn)
    PtoTConnect = algo.PtoTConnect[p]
    Tconnect = PtoTConnect
    TconnectPath = Line_Name([p] + Tconnect)
    Tlist = Tconnect
    # print(p , Tconnect,TconnectPath)

    G0 = algo.G0.copy()
    Elist = indiv['Pconnect'][pn]
    ElistName = ['E{}'.format(e) for e in Elist]
    Elist = ElistName.copy()

    SlotName = ElistName + Tlist
    L = []

    # on cherche le T le plus porche de chaque E 
    Start  = Tconnect
    Target = Elist
    SlotNames =Start + Target
    L = []
    DictPath = collections.defaultdict(list)
    G = G0.subgraph(SlotNames).copy()
    for i, t in enumerate(Target):  
        NodesDist = [G[t][p]['dist'] for p in Start]
        n = Start[np.array(NodesDist).argmin()]
        DictPath[n].append(t)
    DictTpath = dict(DictPath)

    Tconnect2 = Tconnect.copy()
    for T in Tconnect[::-1]:
        if  T not in DictTpath:
            Tconnect2.remove(T)
        else : 
            break
    Tconnect = Tconnect2
    # print(p , Tconnect,TconnectPath , DictTpath)
    BusTNames = []
    DictPath2 = collections.defaultdict(list)

    for i in range(len(Tconnect)):
            t = Tconnect[i]  
            BusTNames.append(TconnectPath[i])
            if t in DictTpath.keys():
                e_list = DictTpath[t]
                Glist = [t] + list(e_list)
                G = G0.subgraph(Glist).copy()
                n = t
                FinalPath  = [n]  
                i+=1  
                while G.size()>0 :
                    NodesAdj  = [x[0] for x in G.adj[n].items()]
                    NodesDist = [x[1]['dist'] for x in G.adj[n].items()]
                    ns = n
                    G.remove_node(n)
                    n = NodesAdj[np.array(NodesDist).argmin()]
                    FinalPath.append(n)
                    BusTNames.append("{}-{}".format(ns,n))
                for e in e_list:
                    Clist = indiv['Econnect'][int(e[1:])]
                    for c in Clist:
                        BusTNames.append("{}-C{}".format(e,c))
                FinalPath  
    return BusTNames           

def Line_Name(nodes):    
    LineName = [] 
    ns = nodes[0]
    for n in nodes[1:]:
        LineName.append("{}-{}".format(ns,n))
        ns = n
    return LineName
