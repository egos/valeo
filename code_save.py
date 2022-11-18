
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



