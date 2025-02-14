import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Chargement des ensembles de données
iris = load_iris(as_frame=True)
wine = load_wine(as_frame=True)

# Préparation des données
X_wine = wine.data
y_wine = wine.target
X_iris = iris.data
y_iris = iris.target

# Entraînement du modèle de régression logistique pour Wine
X_train_wine, X_test_wine, y_train_wine, y_test_wine = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)
logreg_wine = LogisticRegression(max_iter=200)
logreg_wine.fit(X_train_wine, y_train_wine)

# Interface utilisateur
st.title("Application de Prédiction et de Clustering")

# Choix de la base de données
database_option = st.sidebar.selectbox('Sélectionnez la base de données:', ['Iris', 'Wine'])

# Choix du modèle
model_option = st.sidebar.selectbox('Sélectionnez le modèle:', ['Régression Logistique', 'KMeans'])

if database_option == 'Iris':
    st.subheader("Base de données Iris")
    st.write(iris.data)

    if model_option == 'Régression Logistique':
        # Prédiction avec régression logistique
        features = {}
        for feature in iris.feature_names:
            features[feature] = st.number_input(f"{feature}", min_value=0.0)

        if st.button("Prédire la classe"):
            input_data = [[features[feature] for feature in iris.feature_names]]
            prediction = logreg_wine.predict(input_data)  # Utiliser le modèle de Wine pour simplifier
            st.success(f"La classe prédite est : {iris.target_names[prediction][0]}")

    elif model_option == 'KMeans':
        k = st.number_input("Nombre de clusters (k)", min_value=1, max_value=10, value=3)
        if st.button("Clustering"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_iris)
            st.success("Clustering effectué.")
            st.write("Centres des clusters :")
            st.write(kmeans.cluster_centers_)

elif database_option == 'Wine':
    st.subheader("Base de données Wine")
    st.write(wine.data)

    if model_option == 'Régression Logistique':
        features = {}
        for feature in wine.feature_names:
            features[feature] = st.number_input(f"{feature}", min_value=0.0)

        if st.button("Prédire la classe"):
            input_data = [[features[feature] for feature in wine.feature_names]]
            prediction = logreg_wine.predict(input_data)
            st.success(f"La classe prédite est : {wine.target_names[prediction][0]}")

    elif model_option == 'KMeans':
        k = st.number_input("Nombre de clusters (k)", min_value=1, max_value=10, value=3)
        if st.button("Clustering"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_wine)
            st.success("Clustering effectué.")
            st.write("Centres des clusters :")
            st.write(kmeans.cluster_centers_)

# Uploader de fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
    st.success("Fichier chargé avec succès")
else:
    st.warning("Avertissement : Veuillez télécharger un fichier CSV")
