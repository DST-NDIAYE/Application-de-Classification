import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score, recall_score

st.title("Application Web de Classification Binaire")
st.sidebar.title("Application Web de Classification Binaire")
st.markdown("Vos champignons sont-ils comestibles ou toxiques ? 🍄")
st.sidebar.markdown("Vos champignons sont-ils comestibles ou toxiques ? 🍄")

@st.cache(persist=True) #pour ne pas recharger les données à chaque fois
def load_data():
    data = pd.read_csv("data/mushrooms.csv")
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data

df = load_data()
