    if len(df1)>0 :     
        pass   
        # ColCatList = [ColBase, ColAlgo  , ColSysteme, ColResults,ColBus,ColPompe]
        # ColCatName = ['Base','Algo','Systeme','Results','Bus&EV','Pompe']
        # ColCat = dict(zip(ColCatName,ColCatList))
        # # ColCat = pd.DataFrame.from_dict(ColCat, orient='index')
        # # print(df1.columns)
        
        # with st.expander("ColSelect", False):
        #     c1, c2 = st.columns(2)
        #     ColSelect = c1.multiselect(label = 'Columns',options = df1.columns,default=ColBase, help = 'Columns') 
        #     # c2.write(ColCat.style.format(na_rep = ' '))
        #     for k,v in ColCat.items():
        #         c2.write('{} : {}'.format(k,v))
        # ColSelect = ColBase   

        # Col_drop = df1.columns[~df1.columns.isin(ColSelect)].tolist()
        # col1 = df1.columns[~df1.columns.isin(ColBase)]
        # df1 = df1[ColBase + df1.columns[~df1.columns.isin(ColBase)].tolist()]
        # dfx = df1.drop(columns= Col_drop)

# with st.expander("Figures", True): 
#     # hideEtoC = st.checkbox('hideEtoC',False)
#     c1 , c2 = st.columns(2)
#     Empty = c2.empty()
#     if algo.Plot: 
#         ListResultsExport = []
        
#         MinCol = 3 if  len(df1) >= 3 else len(df1)
#         Ncol = c1.number_input(label  = 'indiv number',value = MinCol, min_value = 1,
#                                  max_value  = len(df1),step = 1, label_visibility='collapsed')
#         # Ncol = 3 if len(df1) >=3 else len(df1)
#         Ncolmin  = 4 if Ncol < 4 else Ncol
#         col = st.columns(Ncolmin)               
                
#         for i in range(Ncol):   
#             c1, c2 = st.columns([0.3,0.7])   
#             ListSelectbox = df1.index
#             index = col[i].selectbox('indiv detail ' + str(i),options = ListSelectbox, index = i, label_visibility='collapsed')
#             row = df1.loc[index]
#             indiv = row.to_dict()
#             fig = new_plot_2(algo,indiv, hideEtoC)
#             col[i].pyplot(fig)
#             ListResultsExport.append({'row':row.drop(labels= Col_drop), 'fig': fig})
