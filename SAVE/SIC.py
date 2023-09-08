import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
import time
from types import SimpleNamespace

st.set_page_config(
    page_title="homm_",
    page_icon="🎯",
    layout="wide",
)

def Draw(algo, point = None): 
    pas = algo.pas
    arr = algo.arr
    if point is not None : 
        y,x = np.array(point)*pas
        arr[x:x+pas,y:y+pas,:] = [255,50,50]
    arr[::pas,:,:] = 255
    arr[-1,:,:] = 255
    arr[:,::pas,:] = 255
    arr[:,-1,:] = 255

def reset_algo():
    pas = 50
    size = 10
    points = []
    algo = dict(
        pas = pas,
        size = size, 
        points = points,
        arr = np.full((size*pas, size*pas,3), 0),
        A = np.full((size, size), 0)
        )
    algo = SimpleNamespace(**algo)
    return algo

if "algo" not in st.session_state:
    st.session_state["algo"] = reset_algo()
    st.session_state["value"]  = None

if st.button('reset'): 
    # st.session_state["points"] = []
    st.session_state["algo"] = reset_algo()
    st.experimental_rerun()

algo   = st.session_state["algo"]
# points = st.session_state["points"]
pas = algo.pas
size = algo.size
X = np.where(algo.A == 1)[0]
Y = np.where(algo.A == 1)[1]
# pos = tuple(zip(*np.where(algo.A == 1)))

# print(algo.pas)
pos = tuple(zip(*np.where(algo.A == 1)))
# print(pos)
algo.arr = np.full((size*pas, size*pas,3), 0)
Draw(algo)
# if len(pos) >0 : 
for point in pos:
    Draw(algo, point)
img = Image.fromarray(algo.arr.astype('uint8'), 'RGB')

value = None
run = st.checkbox('run')
if run:
    slot = st.empty()
    algo.arr = np.full((size*pas, size*pas,3), 0)
    pos = tuple(zip(*np.where(algo.A == 1)))
    Draw(algo)
    for point in pos:         
        Draw(algo, point)
        img = Image.fromarray(algo.arr.astype('uint8'), 'RGB')
        slot.image(img)
        time.sleep(0.1)
else :   
    value = streamlit_image_coordinates(img, key="pil")
print(value)

if (value is not None):
    if value!= st.session_state["value"] :
        point = int(value["x"]), int(value["y"])
        y,x = ((np.array(point))/pas).astype(int)
        # print(y,x)
        if algo.A[y,x] == 1:
            algo.A[y,x] = 0
        else : 
            algo.A[y,x] = 1
        # print(point)
        # points.append(point)
        # st.session_state["points"] = points
        st.session_state["value"] = value
        st.session_state["algo"] = algo
        st.experimental_rerun()

# print(algo.A.T)

