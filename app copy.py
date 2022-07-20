import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import itertools 
import math
from math import factorial as f
from datetime import timedelta
from streamlit import session_state
import matplotlib.pyplot as plt
# $C^{k-1}_{n+k-1} = \frac{(n+k-1)!}{n! (k-1)!}$
st.set_page_config(page_title = "_IHM", layout="wide")
print('BEGIN')
st.title('combinatoire et temps de calcul')
with st.expander("hypotheses de calcul ", expanded=True):
    with open("assets/Combi_notes.md", 'r') as file:    
        TEXT = file.read()    
    st.markdown(TEXT)

d0 = {'Slots' : 3,'Capteurs' : 2, 'Elements' : 1}
d = d0.copy()

for i, (k,v) in enumerate(d.items()):
    d[k] = int(st.sidebar.number_input(k,min_value = v))

k = d['Slots'] * d['Elements']
n = d['Capteurs']
# d['Combinaison'] = int(f(n)/f(n-k))
a = f(n + k-1)
b = f(n) * f(k-1)
COMBINAISON = int(a/b)
d['Combinaison'] = COMBINAISON
d['Comb_C'] = d['Slots']**d['Capteurs']
Nslot = d['Slots']
r = 2
CombMaxSlot2 = [f(Nslot)/(f(l) * f(Nslot-l) ) for l in range(Nslot)]
CombMaxSlot2 = sum(CombMaxSlot2)

d['Comb_S'] = CombMaxSlot2 #f(Nslot)/(f(r) * f(Nslot-r))
d['Comb'] = d['Comb_C'] * d['Comb_S']
d['Time'] = str(timedelta(seconds=int(d['Comb']*0.001)))
# print(Values)
# print(d, session_state['Val'])
if st.button('Reset Values'):
    st.balloons()
    del session_state['Val']
if 'Val' not in session_state:
    session_state['Val'] = [d]

else : 
    session_state['Val'].append(d)

# st.write(session_state['Val'])

df = pd.DataFrame(session_state['Val']).sort_index(ascending= False)
c1, c2 = st.columns([0.7,0.3])
c2.image("assets/img_patter.png", use_column_width=True)
c1._legacy_dataframe(df, height= 1000)
# st.write('COMBINAISON = {:,.0f}   ;    TIME = {}'.format(COMBINAISON, TIME))

# st.title('Hello Networkx')
# st.markdown('Zachary´s Karate Club Graph')

# G = nx.karate_club_graph()
# L=["hello", "world", "how", "are", "you"]
# G=nx.complete_graph(len(L))
# nx.relabel_nodes(G,dict(enumerate(L)), copy = False)

# fig, ax = plt.subplots()
# pos = nx.kamada_kawai_layout(G)
# nx.draw(G,pos, with_labels=True)
# st.pyplot(fig)

A = pd.read_excel('pattern.xlsx',header = None).values

A = np.flip(A.T, axis=0)

S = list(zip(*np.where(A==1)))
C = tuple(zip(*np.where(A==2)))
PosSlots = ['S'+ str(i) for i in range(len(S))]
PosSlots = dict(zip(PosSlots,S))

PosCapt = ['C'+ str(i) for i in range(len(C))]
PosCapt = dict(zip(PosCapt,C))

PosTotal = PosSlots.copy()
PosTotal.update(PosCapt)
comb = list(itertools.combinations(list(PosTotal.keys()),2))
dist = {}
for edge in comb:
    x1,x2 = edge
    dist[edge] = round(math.dist(PosTotal[x1],PosTotal[x2]),2)
G = nx.Graph()
G.add_nodes_from([(node, {'pos': v}) for (node, v) in PosTotal.items()])
# G.nodes(data=True)
for (u,v),d in dist.items():
    G.add_edge(u, v, dist=d)
c1, c2 = st.columns([0.7,0.3])
fig, ax = plt.subplots(figsize = (5,7))
# nx.draw(G,  with_labels=True)
ax.imshow(A.T)
nx.draw(G, nx.get_node_attributes(G, 'pos'), with_labels=True, node_size=2000,  ax=ax)
c2.pyplot(fig)