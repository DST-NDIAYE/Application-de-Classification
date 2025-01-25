import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

st.title("Application Web de Classification Binaire")
st.sidebar.title("Application Web de Classification Binaire")
st.markdown("Vos champignons sont-ils comestibles ou toxiques ? 🍄")
st.sidebar.markdown("Vos champignons sont-ils comestibles ou toxiques ? 🍄")

