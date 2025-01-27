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
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


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




def metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        disp = ConfusionMatrixDisplay.from_estimator(
            estimator=model,
            X=x_test,
            y=y_test,
            display_labels=class_names
        )
        st.pyplot(disp.figure_)  

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        roc_disp = RocCurveDisplay.from_estimator(model, x_test, y_test)
        st.pyplot(roc_disp.figure_)  

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        pr_disp = PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        st.pyplot(pr_disp.figure_)  
























if st.sidebar.checkbox("Afficher le Dataset", help="Cliquez ici pour afficher le dataset"):
    st.subheader("Dataset")
    st.write(df)

st.sidebar.subheader("Choisir un Classifieur")

classifier  = st.sidebar._selectbox("Classifieur", ("SVM", "Logistic Regression", "Random Forest"))

if classifier == "SVM":
    st.sidebar.subheader("Paramètres du modèle SVM")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar._selectbox("Kernel", ("rbf", "linear"), key="kernel")
    gamma = st.sidebar._selectbox("Gamma (Kernel coefficient)", ("scale", "auto"), key="gamma")

    metrics_list = st.sidebar.multiselect("Choisir les métriques à afficher", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    if st.sidebar.button("Classer", help="Cliquez ici pour classer"):
        st.subheader("Résultats du modèle SVM")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred).round(2))
        metrics(metrics_list, model, x_test, y_test, ["edible", "poisonous"])