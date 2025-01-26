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

@st.cache_data(persist=True) #pour ne pas recharger les données à chaque fois
def load_data():
    data = pd.read_csv("mushrooms.csv")
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


@st.cache_data(persist=True)
def split(df):
    y = df["class"]
    x = df.drop(columns=["class"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

df = load_data()
x_train, x_test, y_train, y_test = split(df)


def plot_metrics(metrics_list):
    if "Matrice de Confusion" in metrics_list:
        st.subheader("Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

    if "Courbe ROC" in metrics_list:
        st.subheader("Courbe ROC")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        st.write(fpr, tpr)

    if "Courbe PR" in metrics_list:
        st.subheader("Courbe PR")
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        st.write(precision, recall)

        





















if st.sidebar.checkbox("Afficher le Dataset", help="Cliquez ici pour afficher le dataset"):
    st.subheader("Dataset")
    st.write(df)
