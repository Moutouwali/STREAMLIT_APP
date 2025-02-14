import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Titre de l'application
st.title("Application de Classification et Détection d'Images")

# Chemin de l'image d'arrière-plan
background_image_url = "/home/fadoul/MDSMS2/Datacsience sous Python et R/STREAMLIT_APP/image.jpg"

# Ajouter une image d'arrière-plan
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url("{background_image_url}") no-repeat center center fixed; 
        background-size: cover;
        height: 100vh;
    }}
    .image-container {{
        position: absolute;
        top: 0;
        right: 0;
        width: 80%;
        height: 100%;
        background: url("image.jpg") no-repeat center center;
        background-size: cover;
        z-index: -1;
    }}
    </style>
    <div class="image-container"></div>
    """,
    unsafe_allow_html=True
)

# Fonction pour tracer la frontière de décision
def plot_model(X, y, model):
    X = np.array(X)
    y = np.array(y)

    plot_step = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contour(xx, yy, Z, colors='k', linewidths=3)
    plt.contourf(xx, yy, Z, colors=['red', 'blue'], alpha=0.2)

# Choix de l'activité
activity_option = st.sidebar.selectbox("Choisissez l'activité :", ["Charger un jeu de données", "Visualiser les données"])

# Charger les données
if activity_option == "Charger un jeu de données":
    st.header("Chargement du jeu de données")
    uploaded_data = st.file_uploader("Téléchargez un fichier CSV", type="csv")

    if uploaded_data is not None:
        # Charger les données
        data = pd.read_csv(uploaded_data)
        st.write("Aperçu des données :", data.head())

        # Analyse exploratoire
        st.subheader("Analyse exploratoire des données")
        st.write(data.describe())
        st.write(data.info())

        # Sélectionner les colonnes pour X et y
        features = st.multiselect("Sélectionnez les caractéristiques", data.columns.tolist())
        target = st.selectbox("Sélectionnez la cible", data.columns.tolist())

        if len(features) > 0 and target:
            X = data[features]
            y = data[target]

            # Diviser les données
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Visualiser les données avant l'exécution du modèle
            st.header("Visualisation des Données")
            fig, ax = plt.subplots()
            scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis')
            ax.set_xlabel(features[0])
            ax.set_ylabel(features[1])
            ax.set_title("Nuage de Points des Données (Avant Modèle)")
            plt.colorbar(scatter)
            st.pyplot(fig)

            # Visualisations supplémentaires
            st.header("Visualisations Statistiques")

            # Barplot
            if st.checkbox("Afficher un Barplot"):
                st.subheader("Barplot")
                plt.figure(figsize=(10, 5))
                sns.countplot(data=data, x=target)
                plt.title(f"Distribution de {target}")
                st.pyplot(plt)

            # Boxplot
            if st.checkbox("Afficher un Boxplot"):
                st.subheader("Boxplot")
                plt.figure(figsize=(10, 5))
                sns.boxplot(data=data, x=target, y=features[0])  # Utiliser la première caractéristique
                plt.title(f"Boxplot de {features[0]} par {target}")
                st.pyplot(plt)

            # Pie chart
            if st.checkbox("Afficher un Pie Chart"):
                st.subheader("Pie Chart")
                pie_data = data[target].value_counts()
                plt.figure(figsize=(8, 8))
                plt.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=140)
                plt.title(f"Répartition de {target}")
                st.pyplot(plt)

            # Choisir le modèle
            model_choice = st.selectbox("Choisissez le modèle", ["SVM", "K-Means", "Régression Logistique"])

            # Exécution du modèle sélectionné
            if model_choice == "SVM":
                if len(features) >= 2:  # Au moins deux caractéristiques pour SVM
                    model_svm = svm.SVC(probability=True)
                    model_svm.fit(X_train, y_train)

                    if st.button("Prédire avec SVM"):
                        prediction = model_svm.predict(X_test)
                        st.write("Prédictions : ", prediction)

                        unique_classes = np.unique(y)
                        predicted_classes = [unique_classes[int(pred)] for pred in prediction]
                        st.write("Classes prédites :", predicted_classes)

                        accuracy = accuracy_score(y_test, prediction)
                        precision = precision_score(y_test, prediction, average='weighted', zero_division=0)
                        recall = recall_score(y_test, prediction, average='weighted', zero_division=0)
                        f1 = f1_score(y_test, prediction, average='weighted', zero_division=0)

                        st.write(f"Précision du modèle : {accuracy:.2f}")
                        st.write(f"Précision : {precision:.2f}")
                        st.write(f"Rappel : {recall:.2f}")
                        st.write(f"F1-Score : {f1:.2f}")

                        cm = confusion_matrix(y_test, prediction)
                        st.write("Matrice de Confusion :")
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                        disp.plot(cmap='Blues')
                        st.pyplot(plt)

                        if len(features) == 2:  # Visualisation seulement si 2 caractéristiques
                            fig, ax = plt.subplots()
                            plot_model(X_test, y_test, model_svm)
                            scatter = ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=prediction, cmap='viridis')
                            ax.set_xlabel(features[0])
                            ax.set_ylabel(features[1])
                            ax.set_title("Nuage de Points et Droite de Séparation après SVM")
                            plt.colorbar(scatter)
                            st.pyplot(fig)
                        else:
                            st.warning("Pour visualiser la frontière de décision SVM, sélectionnez exactement deux caractéristiques.")

            elif model_choice == "K-Means":
                k = st.slider("Choisissez le nombre de clusters K", min_value=2, max_value=10, value=3)
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(X)

                fig, ax = plt.subplots()
                scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans.labels_, cmap='viridis')
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_title("Clusters K-Means (Avant Prédiction)")
                plt.colorbar(scatter)
                st.pyplot(fig)

                st.write("Classes K-Means prédites :", kmeans.labels_)

            elif model_choice == "Régression Logistique":
                if len(features) == 2:  # Vérifiez si deux caractéristiques sont sélectionnées
                    model_logistic = LogisticRegression(max_iter=200)
                    model_logistic.fit(X_train, y_train)

                    if st.button("Prédire avec Régression Logistique"):
                        prediction_logistic = model_logistic.predict(X_test)
                        st.write("Prédictions : ", prediction_logistic)

                        unique_classes = np.unique(y)
                        predicted_classes_logistic = [unique_classes[int(pred)] for pred in prediction_logistic]
                        st.write("Classes prédites :", predicted_classes_logistic)

                        accuracy_logistic = accuracy_score(y_test, prediction_logistic)
                        precision_logistic = precision_score(y_test, prediction_logistic, average='weighted', zero_division=0)
                        recall_logistic = recall_score(y_test, prediction_logistic, average='weighted', zero_division=0)
                        f1_logistic = f1_score(y_test, prediction_logistic, average='weighted', zero_division=0)

                        st.write(f"Précision du modèle : {accuracy_logistic:.2f}")
                        st.write(f"Précision : {precision_logistic:.2f}")
                        st.write(f"Rappel : {recall_logistic:.2f}")
                        st.write(f"F1-Score : {f1_logistic:.2f}")

                        cm_logistic = confusion_matrix(y_test, prediction_logistic)
                        st.write("Matrice de Confusion :")
                        disp_logistic = ConfusionMatrixDisplay(confusion_matrix=cm_logistic)
                        disp_logistic.plot(cmap='Blues')
                        st.pyplot(plt)

                        fig, ax = plt.subplots()
                        plot_model(X_test, y_test, model_logistic)
                        scatter = ax.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=prediction_logistic, cmap='viridis')
                        ax.set_xlabel(features[0])
                        ax.set_ylabel(features[1])
                        ax.set_title("Nuage de Points et Droite de Séparation après Régression Logistique")
                        plt.colorbar(scatter)
                        st.pyplot(fig)
                else:
                    st.warning("Pour visualiser la frontière de décision de la régression logistique, sélectionnez exactement deux caractéristiques.")

elif activity_option == "Visualiser les données":
    st.header("Visualisation des Données")
    uploaded_data = st.file_uploader("Téléchargez un fichier CSV pour visualiser les données", type="csv")
    
    if uploaded_data is not None:
        data = pd.read_csv(uploaded_data)
        st.write("Aperçu des données :", data)
        st.write("Statistiques descriptives :")
        st.write(data.describe())
        st.write("Données complètes :")
        st.dataframe(data)

# Lancer l'application
if __name__ == "__main__":
    st.write("Application prête à être utilisée !")