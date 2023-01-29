import streamlit as st
import xlsxwriter
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

print('test')

def Export(): 
    output = BytesIO()

    # Write files to in-memory strings using BytesIO
    # See: https://xlsxwriter.readthedocs.io/workbook.html?highlight=BytesIO#constructor
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'Hello')
    
    x=np.linspace(-10,10,100)
    y=x**2
    fig,ax=plt.subplots()
    ax.plot(x,y)
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    worksheet.insert_image(2,2, '', {'image_data': imgdata})
    workbook.close()
    return output.getvalue()  

st.download_button(
    label="Download Excel workbook",
    data=Export(),
    file_name="workbook.xlsx",
    mime="application/vnd.ms-excel"
)