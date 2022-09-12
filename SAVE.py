def indiv_verif_S(row, NewCtoE, dfs, dfline): 
    D = dfs.Class.value_counts().to_dict()
    Clist = list(range(D['C']))
    Clist = dfs[dfs.Class =='C'].Name.sort_values().unique().tolist()
    
    CtoE = NewCtoE
    Econnect = dict(collections.Counter(CtoE))
    Elist = sorted(Econnect)
    Ecount = len(Elist)   
    
    if Ecount > row.Ecount:
        NewEtoP = np.random.randint(0,D['P'],Ecount - row.Ecount)
        EtoP = np.append(row.EtoP, NewEtoP)
    elif Ecount < row.Ecount:
        # print(Elist,'avant',row.EtoP,row.Ecount, Ecount)
        EtoP = np.random.choice(row.EtoP,Ecount)
        # print("Ecount",EtoP)
    else : 
        EtoP = row.EtoP
    print('avant',Elist,'apres',EtoP,row.Ecount, Ecount)
    
    Pconnect = dict(collections.Counter(EtoP))
    Plist = sorted(Pconnect)
    Pcount = len(Plist)    

    # ID_CtoE = ['C-E{}{}'.format(i, CtoE[i]) for i in Clist]
    # ID_EtoP = ['E-P{}{}'.format(Elist[i]  , EtoP[i]) for i in range(Ecount)]
    ID_CtoE = ['C{}-E{}'.format(C, CtoE[i]) for i, C in enumerate(Clist)]
    ID_EtoP = ['E{}-P{}'.format(Elist[i]  , EtoP[i]) for i in range(Ecount)]

    dist = dfline.loc[dfline.ID.isin(ID_CtoE + ID_EtoP), 'dist'].sum()  
    l = [Clist, CtoE,Econnect,Elist,Ecount, EtoP,Pconnect,Plist,Pcount, ID_CtoE,ID_EtoP,dist]
    return l 


def indiv_create(algo):
    dfline = algo.dfline
    D = algo.Comb
    Clist = D['C']
    Ccount = len(D['C'])
    CtoE = np.random.choice(D['E'],Ccount)
    d = collections.defaultdict(list)
    for i in range(Ccount):      d[CtoE[i]].append(D['C'][i])
    Econnect = dict(sorted(d.items()))
    Elist = sorted(Econnect)
    Ecount = len(Elist)

    EtoP = np.random.choice(D['P'],Ecount)
    d = collections.defaultdict(list)
    for i in range(Ecount):      d[EtoP[i]].append(Elist[i])
    Pconnect = dict(sorted(d.items()))
    Plist = sorted(Pconnect)
    Pcount = len(Plist)

    List_EtoC = [['E{}-C{}'.format(start, end) for end in List] for start , List in Econnect.items()]
    List_PtoE = [['P{}-E{}'.format(start, end) for end in List] for start , List in Pconnect.items()]
    
    Name = list(itertools.chain.from_iterable(List_EtoC + List_PtoE))
    dist_Connect = (dfline.loc[dfline.ID.isin(Name), ['ID','dist']].set_index('ID').dist).to_dict()
    dist = dfline.loc[dfline.ID.isin(Name), 'dist'].sum()
    
    col = ['D', 'Clist','CtoE','Econnect','Elist','Ecount', 'EtoP','Pconnect','Plist','Pcount', 'List_EtoC','List_PtoE','dist_Connect', 'dist', 'Name','ID']
    l = [D,Clist, CtoE,Econnect,Elist,Ecount, EtoP,Pconnect,Plist,Pcount, List_EtoC,List_PtoE,dist_Connect, dist, Name,algo.epoch]
    indiv = SimpleNamespace(**dict(zip(col,l)))
    indiv = dict(zip(col,l))

    return indiv


# for idx,row  in df1.iterrows():
#     rowCopy = copy.deepcopy(row)
#     d = Calcul_All(algo ,rowCopy, False)
#     col  = ['Pression', 'Debit','SumDebit']
#     col2 = ['Pression_s', 'Debit_s','SumDebit_s']
#     df1.loc[idx, col2] = [str(d[c]) for c in col]
#     d = Calcul_All(algo , rowCopy, True)
#     col = ['Pression', 'Debit','SumDebit']
#     df1.loc[idx, col] = [str(d[c]) for c in col]