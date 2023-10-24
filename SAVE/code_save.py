
# ancien pickdownload
with st.sidebar:
    pass
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



# PickleDonwload.download_button(
#     label="📥 download pickle Save_{}.pickle".format(today), key='pickle_Save_pickle',
#     data=pickle.dumps(vars(algo)),
#     file_name="Save_{}.pickle".format(today)) 


if mode == 'T0':
    Tlist = algo.CombNodes['T']
    G = algo.G0.edge_subgraph(edgesPtoE + edgesEtoC).copy() 
    PtoTConnect = algo.PtoTConnect
    for p in indiv['PtoE']:
        PtoTedges = PtoTConnect[p]
    # for p, PtoTedges in PtoTConnect.items(): 
        # print(indiv['PtoE'], p, PtoTedges)
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



def new_import_T_S(dfmap, DistFactor):
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
            # Comb[v[0]].append(int(n))

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



def T_connection(algo, indiv):
    
    DictLine = algo.DictLine
    Tlist = ['T{}'.format(t) for t in algo.Comb['T']]
    G0 = nx.from_pandas_edgelist(algo.dfline, 's', 't', ['dist'])
    for i, (p,Elist) in enumerate(indiv['Pconnect'].items()): 
        pass
    p = 'P{}'.format(p)
    ElistName = ['E{}'.format(e) for e in Elist]
    Elist = ElistName.copy()
    

    G = G0.subgraph([p] + Tlist).copy()
    n = p
    Tconnect = [n]    
    while G.size()>0 :
        NodesAdj  = [x[0] for x in G.adj[n].items()]
        NodesDist = [x[1]['dist'] for x in G.adj[n].items()]
        G.remove_node(n)
        n = NodesAdj[np.array(NodesDist).argmin()]
        Tconnect.append(n)

    # print('Tconnect',Tconnect,Tlist, G.size())

    SlotName = ElistName + ['T0']
    SlotName    
    L = []
    for i, e0 in enumerate(Elist):
        G = G0.subgraph(SlotName).copy()
        n = e0
        l = []
        while n!='T0':
            NodesAdj  = [x[0] for x in G.adj[n].items()]
            NodesDist = [x[1]['dist'] for x in G.adj[n].items()]
            ns = n
            l.append(ns)
            G.remove_node(n)
            n = NodesAdj[np.array(NodesDist).argmin()]
        L.append(l)
        # print(n,e0,l , path, np.argmax(A == e0, axis=1)) 
    # passage hotvectors pour clustering => regrouper les lignes avec des E communs
    X =pd.get_dummies(pd.Series(L).explode()).groupby(level=0).sum().values
    # print(L)
    # print(X)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    group = kmeans.labels_

    gr = collections.defaultdict(list)
    for i, g in enumerate(group):
        gr[g].append(Elist[i])

    distT = copy.deepcopy(indiv['dist_Connect'])
    BusTNames = ['P0-T0']
    for g , Egroup in gr.items():
        SlotName = Egroup + ['T0']
        G = G0.subgraph(SlotName).copy()
        n = "T0"
        L = [n]
        path = ["P0-T0"]
        dist = DictLine['P0-T0']['dist']
        while G.size()>0 :
            NodesAdj  = [x[0] for x in G.adj[n].items()]
            NodesDist = [x[1]['dist'] for x in G.adj[n].items()]
            ns = n
            G.remove_node(n)
            n = NodesAdj[np.array(NodesDist).argmin()]
            L.append(n)
            #ajouter calcul duritval ect .... 
            line = '{}-{}'.format(ns,n)
            path.append(line)
            BusTNames.append(line)
            dist += DictLine['{}-{}'.format(ns,n)]['dist']
            distT['P0-{}'.format(n)] = dist
        
        # print(g, Egroup , L    , path)
    for e in Elist:
        Clist = indiv['Econnect'][int(e[1:])]
        for c in Clist:
            BusTNames.append("{}-C{}".format(e,c))
    return BusTNames, distT


def T_connection_SAVE(algo, indiv):
    DictLine = algo.DictLine
    Tlist = algo.Comb['T']
    G0 = nx.from_pandas_edgelist(algo.dfline, 's', 't', ['dist'])
    for i, (p,Elist) in enumerate(indiv['Pconnect'].items()): 
        pass
    p = 'P{}'.format(p)
    ElistName = ['E{}'.format(e) for e in Elist]
    SlotName = ElistName + ['T0']
    SlotName
    G = G0.subgraph(SlotName).copy()

    Elist = ElistName.copy()
    L = []
    for i, e0 in enumerate(Elist):
        G = G0.subgraph(SlotName).copy()
        n = e0
        l = []
        while n!='T0':
            NodesAdj  = [x[0] for x in G.adj[n].items()]
            NodesDist = [x[1]['dist'] for x in G.adj[n].items()]
            ns = n
            l.append(ns)
            G.remove_node(n)
            n = NodesAdj[np.array(NodesDist).argmin()]
        L.append(l)
        # print(n,e0,l , path, np.argmax(A == e0, axis=1)) 
    X =pd.get_dummies(pd.Series(L).explode()).groupby(level=0).sum().values
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    group = kmeans.labels_

    gr = collections.defaultdict(list)
    for i, g in enumerate(group):
        gr[g].append(Elist[i])

    distT = copy.deepcopy(indiv['dist_Connect'])
    BusTNames = ['P0-T0']
    for g , Egroup in gr.items():
        SlotName = Egroup + ['T0']
        G = G0.subgraph(SlotName).copy()
        n = "T0"
        L = [n]
        path = ["P0-T0"]
        dist = DictLine['P0-T0']['dist']
        while G.size()>0 :
            NodesAdj  = [x[0] for x in G.adj[n].items()]
            NodesDist = [x[1]['dist'] for x in G.adj[n].items()]
            ns = n
            G.remove_node(n)
            n = NodesAdj[np.array(NodesDist).argmin()]
            L.append(n)
            #ajouter calcul duritval ect .... 
            line = '{}-{}'.format(ns,n)
            path.append(line)
            BusTNames.append(line)
            dist += DictLine['{}-{}'.format(ns,n)]['dist']
            distT['P0-{}'.format(n)] = dist
        
        # print(g, Egroup , L    , path)
    for e in Elist:
        Clist = indiv['Econnect'][int(e[1:])]
        for c in Clist:
            BusTNames.append("{}-C{}".format(e,c))
    return BusTNames, distT


def Tconnection_v3(algo, indiv, pn):
    p = 'P{}'.format(pn)
    PtoTConnect = algo.PtoTConnect[p]
    Tconnect = PtoTConnect
    TconnectPath = Line_Name([p] + Tconnect)
    Tlist = Tconnect
    print(p , Tconnect,TconnectPath)

    G0 = algo.G0.copy()

    p = 'P{}'.format(p)
    Elist = indiv['Pconnect'][pn]
    ElistName = ['E{}'.format(e) for e in Elist]
    Elist = ElistName.copy()

    SlotName = ElistName + Tlist
    L = []
    # backward sur chaque E pour connaitre le bus le plus court vers le plus proche T
    DictTpath = collections.defaultdict(list)
    Gx = G0.subgraph(SlotName).copy()
    for i, e0 in enumerate(Elist):
        G = Gx.copy()
        n = e0
        l = []
        path = []
        while n[0]!='T':
            NodesAdj  = [x[0] for x in G.adj[n].items()]
            NodesDist = [x[1]['dist'] for x in G.adj[n].items()]
            ns = n
            l.append(ns)
            G.remove_node(n)
            n = NodesAdj[np.array(NodesDist).argmin()]
            # print(e0,ns,n)
            path.append((ns,n))

        DictTpath[n] = DictTpath[n] + path
        L.append(l)
    DictTpath = dict(DictTpath)
    print(DictTpath, TconnectPath)
    # print(L)
    # ------ perf =  1000µs on prend 100µs par copy de graph !!! 
    Lgroup = []

    BusTNames = []
    i, j = 0,0
    for i in range(len(Tconnect)):
        t = Tconnect[i]         
        BusTNames.append(TconnectPath[i])
        if t in DictTpath.keys():
            path = dict(DictTpath)[t]
            Gg = nx.Graph(path)
            Gg.remove_node(t)
            List_sub_G = nx.connected_components(Gg)
            for e_list in List_sub_G: 
                print(t, e_list)       
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
                    j+=1
                for e in e_list:
                    Clist = indiv['Econnect'][int(e[1:])]
                    for c in Clist:
                        BusTNames.append("{}-C{}".format(e,c))
                FinalPath
    # print(i,j)
    # ------ perf =  1200µs -- solution mettre un weight actif on off 
    return BusTNames



def load_data_brut_S(file, select = None):
    
    with open(file, 'r') as f:
      data = json.load(f)

    dfslot = pd.DataFrame(data['layers'][1]['objects']).drop(columns= ['rotation','width','height','visible','gid']).rename(columns = {'class' : 'Class', 'name' : 'Name'})
    dfslot.x = (dfslot.x/16).astype(int)
    dfslot.y = (dfslot.y/16).astype(int) - 1
    dfslot.Name = pd.to_numeric(dfslot.Name)
    dfslot['ID'] = dfslot.Class + dfslot.Name.astype(str)
    dfslot['Color'] = dfslot['Class'].map({'C':10,'E':20,'P':30})

    dfline  = pd.DataFrame(data['layers'][2]['objects']).drop(columns=['rotation','width','name','height','visible','id']).rename(columns = {'class' : 'Class'})
    for idx, row in dfline.iterrows():
        properties = row.properties
        # properties = properties[:0] + properties[-2:]
        dfline.loc[idx, ['end','long','start']] = [d['value'] for d in properties]    
        polyline = row.polyline
        x,y = row.x,row.y
        polyline = [(p['x'] + x, p['y'] + y) for p in polyline]
        p = np.array(polyline)
        dfline.at[idx , 'polyline'] = p   
        dfline.loc[idx ,'dist'] = np.abs(np.diff(p.T)).sum()
    dfline.start = pd.to_numeric(dfline.start)
    dfline.end = pd.to_numeric(dfline.end)
    dfline.dist = (dfline.dist.astype(int)*2/100).round(2)
    dfline = dfline.drop(columns= ['properties','x','y'])
    t = dfline.Class.str.split('-')
    dfline['ID'] = t.str[1] + dfline.end.astype(str) + '-' + t.str[0] + dfline.start.astype(str)
    dfline = dfline.sort_values('ID').reset_index(drop = True)
    
    CombAll = dfslot.ID.tolist()

    # DropList = ['C0','E2']
    # DropList = []
    # if len(DropList) > 0 : 
    # if select is not None : 
    #     dfline = dfline[~dfline.ID.str.contains('|'.join(DropList))]
    #     dfslot = dfslot[~dfslot.ID.isin(DropList)]
    if select is not None : 
        dfline = dfline[~dfline.ID.str.contains('|'.join(select))]
        dfslot = dfslot[~dfslot.ID.isin(select)]  
    A0 = data['layers'][0]['data']
    height = data['height']
    A0 = np.array(A0).reshape(10,7)
    pas = 16
    unique = np.unique(A0)
    A0[A0 == unique[0]] = 0
    A0[A0 == unique[1]] = 1
    
    confs = pd.read_excel('test.xlsx')
    DataCategorie = {}
    mask = confs['Actif'].notnull()
    df = confs[mask].copy()
    for Categorie in df.Categorie.unique():        
        dfx = df[df.Categorie == Categorie]
        DataCategorie[Categorie] = {
            'Unique' : dfx.Name.unique().tolist(),
            'Values' : dfx.set_index('Name').dropna(axis = 1).to_dict('index')
                    }  
        
    Comb = dfslot.groupby('Class').Name.unique().apply(list).apply(sorted).to_dict()
    Clist = Comb['C']
    Pompes  = [DataCategorie['Pompe']['Unique'][0]]* len(Comb['P'])
    Pvals   = [DataCategorie['Pompe']['Values'][Pompes[0]][i] for i in ['a','b','c']]
    Nozzles = [DataCategorie['Nozzle']['Unique'][0]] * len(Comb['C'])
    Nvals   = [DataCategorie['Nozzle']['Values'][n]['a'] for n in Nozzles]
    Nvals  = dict(zip(Clist, Nvals))
        
    algo = dict(
        Group = [],
        pop = 50,
        fitness = 'dist',
        crossover = 0.4,
        mutation = 0.4,
        Nlim = 2.0,          
        Pmax = 3,
        Plot = False,
        dfslot = dfslot,
        dfline = dfline,
        epoch = 0,
        Nindiv = 0,
        Nrepro = 0,
        indivs = [],
        df = [],
        DataCategorie =  DataCategorie,
        Tuyau = ['Ta'],
        Pompes = Pompes, 
        Pvals = Pvals,     
        EV = ['Ea'],    
        Nozzles  = Nozzles,  
        Nvals = Nvals,
               
        confs = confs,
        Clist = Clist,
        Comb = Comb,
        CombAll = CombAll,
        dist = dfline.set_index('ID').dist.to_dict(),
        height = data['height'],
        A0 = A0,
        )
    algo = SimpleNamespace(**algo)
    return algo


def plot_(algo,dflineSelect, dfsSelect, name): 
    A0 = algo.A0.copy()
    
    for idx, row in dfsSelect.iterrows():
        A0[row.y, row.x] = row.Color + int(row.Name)   
        
    A = np.kron(A0, np.ones((16,16), dtype=int))
    fig, ax = plt.subplots(figsize = (8,8))
    ax.set_title(name)
    plt.xticks([])
    plt.yticks([])
    ax.imshow(A)
    for idx, row in dflineSelect.iterrows():
        p = row.polyline
        f = ax.plot(p[:,0],p[:,1],'k', linewidth=4)

    style = dict(size=15, color='black') 
    for idx, row in dfsSelect.iterrows():
        x = row.x*16
        y = row.y*16
        text = row.Class + str(row.Name)
        f = ax.text(row.x*16+8, row.y*16+8,text , **style,  ha='center', weight='bold') 
    return fig



    # with st.form("Select"): 
    #     select = st.multiselect('Pattern',algo.CombAll, algo.CombAll)
    #     select = [s for s in algo.CombAll if s not in select]
    #     if select == [] : select = None
    #     submitted = st.form_submit_button("Submit & Reset")      

    #     if submitted:
    #         # file = {'SheetMapName' : algo.SheetMapName, 'uploaded_file' : algo.uploaded_file} 
    #         print('submitted Select')
    #         algo = load_data_brut(file, select)
    #         algo.df = indiv_init(algo, pop)
    #         session_state['algo'] = algo
    
'''col select'''
   # ColdropTest = ColAlgo + ColSysteme + ColResults + ColBus
    # dfCol = df1.columns
    # print([c for c in dfCol if c not in ColdropTest])
    # ColBase ['Ptype0', 'Ptype', 'PtypeCo', 'PompeCountFinal', 'ID', 'Epoch', 'dist', 'Esplit', 'Debit', 'Masse', 'Cout', 'fitness', 'Alive']
    # print(df1)
    # ColSelect = []
    # ColSt = st.columns(5)
    # ColCatList = [ColAlgo, ColSysteme, ColResults,ColBus]
    # ColCatName= ['ColAlgo', 'ColSysteme', 'ColResults','ColBus']
    # for i in range(len(ColCatList)):
    #     ColSelect += ColSt[i].multiselect(label = ColCatName[i],options = ColCatList[i],default=ColCatList[i], help = str(ColCatName[i]))
    # ColSelect = st.multiselect(label = 'Columns',options = df1.columns,default=ColBase, help = 'Columns')    


    # if not c2.checkbox('Algo'   , value = False, help = str(ColAlgo)) : Col_drop += ColAlgo
    # if not c3.checkbox('System' , value = False, help = str(ColSysteme)) : Col_drop += ColSysteme
    # if not c4.checkbox('Results', value = False, help = str(ColResults)) : Col_drop += ColResults 
    # if not c5.checkbox('BUS'    , value = False, help = str(ColBus)) : Col_drop += ColBus       
    # if not c6.checkbox('Pompe'  , value = False, help = str(ColPompe)) : Col_drop += ColPompe  
    # print(len(algo.indivs))       
    
def Calcul_Debit(algo ,indiv, split):
    D = algo.Comb  
    Group = algo.Group  
    Clist = D['C']
    Econnect = indiv['Econnect']
    Pconnect = indiv['Pconnect']
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
    for i, (e,EClist) in enumerate(Econnect.items()):
        p = EtoP[i]
        pt = Ptype[i]
        name = 'P{}-E{}'.format(p,e)
        VerifGroup = np.isin(Group,  EClist)
        # EClistTotal = [EClist]
        # if VerifGroup.all() & (len(Group) > 0):
        EClistTotal = [[i for i in EClist if i in Group], [i for i in EClist if i not in Group]]          
        grouped = True
        for j,  EClist in enumerate(EClistTotal):
            if j >0 : grouped = False # bascule a No group apres le passage group 
            if len(EClist)>0: # bug avec calcul array
                d_EtoC_list = np.array([algo.dist['E{}-C{}'.format(e,c)] for c in EClist])
                d_PtoE = algo.dist['P{}-E{}'.format(p,e)]
                res = debit(algo, d_EtoC_list,d_PtoE, EClist,pt, grouped, split = split)

                Debit = Debit + list(res['Qi'])
                Pi = list(res['Pi'])
                PressionConnect = dict(zip(EClist, Pi))
                Cpression.update(PressionConnect)
                
                Qi = list(res['Qi'])
                Cdebit.update(dict(zip(EClist, Qi)))
                
                # Data[name] = res        
                # Pression = Pression + list(res['Pi'])          
                # print(dc,dp,Clist,list(res['Pi']))
                # Pression_C = Pression_C + [PressionConnect]
                # print(i, j ,Group,grouped, EClistTotal ,EClist, PressionConnect)
    PressionList = [Cpression[i] for i in D['C']]
    DebitList    = [Cdebit[i] for i in D['C']]
    # print(Cpression)
    SumDebit = round(sum(Debit),1)
    # keys = ['info','Data','Pression','Debit','SumDebit']
    # vals = [info, Data,Pression, Debit, SumDebit]     
    keys = ['PressionList','DebitList','Debit']
    vals = [PressionList, DebitList, SumDebit] 
    return dict(zip(keys,vals))