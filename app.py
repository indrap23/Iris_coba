import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict
from model import modelling

@st.cache(suppress_st_warning=True)
def get_fvalue(val):
    feature_dict = {"No":1,"Yes":2}    
    for key,value in feature_dict.items():
        if val == key:
            return value
def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value
app_mode = st.sidebar.selectbox('Select Page',['Home','Prediction','Relearning']) #three pages

if app_mode=='Home':
    st.title('Classifying Iris Flowers')
    st.markdown('we try to predict the type of Iris Flower')
    st.image('iris.png')
elif app_mode == 'Prediction':
    st.title('Classifying Iris Flowers')
    st.markdown('model for classify iris flowers into \setosa, versicolor, virginica')
    st.header('Plant Features')
    col1, col2 = st.columns(2)
    with col1:
        st.text('Sepal characteristics')
        sepal_l = st.number_input('Sepal lenght (cm)', 0.5, 8.0)
        sepal_w = st.number_input('Sepal width (cm)', 0.5, 4.4)
    with col2:
        st.text('Pepal characteristics')
        petal_l = st.number_input('Petal lenght (cm)', 0.5, 7.0)
        petal_w = st.number_input('Petal width (cm)', 0.5, 2.5)
    if st.button('Predict type of Iris'):
        result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
        st.text(result[0])
        if result == 'Setosa':
            st.image('setosa.jpg')
        elif result == 'Virginica':
            st.image('virginica.jpg')
        elif result == 'Versicolor':
            st.image('versicolor.jpg')
elif app_mode == 'Relearning':
    st.title('Relearning Classifying Iris Flowers')
    uploaded_file = st.file_uploader("Choose a CSV file")
    if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_csv(uploaded_file)
    if st.button('Relearning'):
        result = modelling(dataframe)
        st.text('Done Relearning')
        st.text(result)

