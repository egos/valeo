import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# size = 10
# if 'df' not in st.session_state:
#     df = pd.DataFrame(np.ones((size,size)))
#     df.columns= df.columns.astype(str)
#     # edited_df = st.data_editor(df)
#     st.session_state['df'] = df
# else :
#     df = st.session_state['df']

# edited_df = st.data_editor(df) 
# # edited_df
# # print(df)
# # print(edited_df)
# if st.button('run'):
#     edited_df.loc[0,'0'] = True
#     st.session_state['df'] = edited_df
#     st.experimental_rerun()



import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
from streamlit import session_state
import scipy.ndimage
# import skimage

st.set_page_config(
    page_title="Streamlit Image Coordinates: Image Update",
    page_icon="🎯",
    layout="wide",
)
print('run')
'''
question evoyé a 2 endroit y a un gars qui cherche la meme chose je pense 
https://discuss.streamlit.io/t/get-coordinate-of-an-image-by-a-click-with-streamlit/27403
https://discuss.streamlit.io/t/new-tiny-component-streamlit-image-coordinates/35150
'''
def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
    center = point
    radius = 10
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )
    
PAS = 20

with st.echo("below"):
    if 'img' not in session_state: 
        img0 = np.full((400, 400), 0)
        # print(img0)
        # img0[::PAS,::PAS] = 256
        img = Image.fromarray(img0, 'RGB')
    else : 
        img = session_state['img']

    # img0 = np.zeros((300,300), dtype='uint8')
    # img0[::PAS,:] = 250
    # img0[:,::PAS] = 250

    # print(img0)
    # print(img0.shape)
    # img = Image.fromarray(img0,'L')

    draw = ImageDraw.Draw(img)
    # value = None
    # c1, c2 = st.columns(2)
    # with c1:  
    value = streamlit_image_coordinates(img, key="pil")  
    # ShowImage = c2.image(img, width = 300)    
    # st.write(value)

    # print(ShowImage)
    if value is not None: 
        # time.sleep(1)
        point = value["x"], value["y"]
        coords = get_ellipse_coords(point)
        draw.ellipse(coords, fill="red")
        point = (value["x"], value["y"])
        xy = [point, (value["x"]+ PAS, value["y"] + PAS)]
        # draw.rectangle(xy, fill='white', outline=None, width=1)
        # ShowImage.image(img, width = 256, output_format = 'PNG')
        session_state['img'] = img
        st.experimental_rerun()
    
    
    
    
# placeholder = st.empty()

# # Replace the placeholder with some text:
# placeholder.text("Hello")

# # Replace the text with a chart:
# placeholder.line_chart({"data": [1, 5, 2, 6]})

# # Replace the chart with several elements:
# with placeholder.container():
#     st.write("This is one element")
#     st.write("This is another")

# Clear all those elements:
# placeholder.empty()    
    
# with c1 :
#     if value is not None:
#         x, y = value.values()
#         # print(x,y)
#         img0[x:(x+100), y:(y+100)] = 1
#         img = Image.fromarray(img0, 'RGB')
#         streamlit_image_coordinates(img)



# img0 = np.transpose(img0)
# img0[::PAS,::PAS,:] = 1
# img0 = skimage.transform.resize(img0, (300, 300), order =0)
# img0 = scipy.ndimage.zoom(img0, 20, order=1)

# img0 = np.full((400, 400), 0)
# # print(img0)
# img0[::PAS,::PAS] = 256
# img0 =np.random.randint(0,256,(28,28,3), dtype=np.uint8)
# img0 = np.full((20, 20,3), 0)



