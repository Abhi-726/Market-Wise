import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image
from scipy import spatial

pickle_in=open("Ps4.pkl","rb")
similarity_score,pivot_table=pickle.load(pickle_in)


def welcome():
    return "WELCOME ALL"

def recommend(exhibitor):
    index=np.where(pivot_table.index==exhibitor)[0][0]
    similar_items=sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[2:6]
    lisst=[]
    for i in similar_items:
        lisst.append(pivot_table.index[i[0]])
    return lisst
 
def main():
    html_temp="""<h1 style="text-align: center; color: white; font-size: 20px;">Welcome All</h1>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    html_temp="""<h1 style="text-align: center; color: blue; font-size: 20px;">EVENT RECOMMENDATION SYSTEM</h1>"""
    st.markdown(html_temp,unsafe_allow_html=True)
    pro_category=['Technology serviceprovider', 'Ancillaries',
       'Boiler component manufacturer', 'Boiler manufacturer',
       'Dealers Traders Distributors', 'Turbine manufacturer',
       'Professionals in NDE, energy audit, RLA and R&M',
       'WTP ETP other pollution control equipment manufacturer']
    company_repname=st.text_input("Company Representative Name","Type Here")
    company_name=st.text_input("Company Name","Type Here")
    mobile_no=st.text_input("Mobile No","Type Here")
    designation=st.text_input("Designation","Type Here")
    email=st.text_input("Email","Type Here")
    profession=st.text_input("Profession","Type Here")
    result=[]
    if st.button("Recommend"):
        result=recommend(company_name)
        st.success("Recommended companies are : ")
    for i in result:
        st.success(i)
    if st.button("About"):
        st.text("lets Learn")
        st.text("Built with Streamlit")
 

if __name__=='__main__':
    main()
    