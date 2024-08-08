import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le modèle et le scaler
knn_model = joblib.load('KNN.pkl')
scaler = joblib.load('scaler.pkl')

# Fonction pour déterminer le cluster
def determine_cluster(score):
    if score > 1:
        return 1
    elif 0 <= score <= 1:
        return 2
    elif -0.035 <= score < 0:
        return 2
    elif -0.07 <= score < -0.035:
        return 3
    else:
        return 3
    
    # Fonction pour formater les montants
def format_currency(x):
    return f"{x:,.2f}"

# Fonction pour formater les montants
def format_currency(x):
    return f"{x:,.2f}".replace(',', ' ').replace('.', ',')

# Fonction pour prédire le cluster à partir des features
def predict_cluster(features):
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    prediction = knn_model.predict(features_scaled)
    return determine_cluster(prediction[0])

# Fonction pour charger les données des clients
@st.cache_data
def load_clients_data():
    df = pd.read_csv('df.csv')
    return df

# Charger les données des clients
df = load_clients_data()

# Ajout du sélecteur de thème
theme = st.sidebar.radio("Choisissez un thème", ("Clair", "Sombre"))

if theme == "Sombre":
    st.markdown(
        """
        <style>
        body {
            background-color: #2e2e2e;
            color: #ffffff;
        }
        .stApp {
            background-color: #2e2e2e;
            color: #ffffff;
        }
        .stButton button {
            background-color: #444444;
            color: #ffffff;
        }
        .stSidebar {
            background-color: #2e2e2e;
        }
        .stSelectbox {
            color: #ffffff;
        }
        .stTextInput label,
        .stNumberInput label {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    title_color = "#ffffff"
else:
    st.markdown(
        """
        <style>
        body {
            background-color: #ffffff;
            color: #000000;
        }
        .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        .stButton button {
            background-color: #dddddd;
            color: #000000;
        }
        .stSidebar {
            background-color: #ffffff;
        }
        .stSelectbox {
            color: #000000;
        }
        .stTextInput label,
        .stNumberInput label {
            color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    title_color = "#000000"

# Utilisation de markdown avec couleur dynamique
st.markdown(f"<h2 style='color: {title_color};'>Prédiction des classements des Clients</h2>", unsafe_allow_html=True)

# Menu de navigation
st.sidebar.header('Navigation')
page = st.sidebar.selectbox('Choisissez une page', ['Téléverser un fichier', 'Sélectionner un client', 'Entrer les données manuellement', 'Afficher les métriques', 'Visualiser les top clients'])

if page == 'Téléverser un fichier':
    st.markdown("""
    Veuillez téléverser un fichier Excel contenant les colonnes suivantes :
    - Nombre d'impayée
    - Total d'impayées
    - Chiffre d'affaire
    - Encaissement
    """)

    # Téléversement de fichier
    uploaded_file = st.file_uploader("Choisissez un fichier Excel", type="xlsx")

    if uploaded_file:
        # Charger les données du fichier téléversé
        input_df = pd.read_excel(uploaded_file)
        st.write("Aperçu du fichier téléversé :")
        st.write(input_df.head())

         # Vérifier si les colonnes requises sont présentes
        required_columns = ["Nombre d'impayée", "Total d'impayées", "Chiffre d'affaire", 'Encaissement']
        if all(column in input_df.columns for column in required_columns):
           # Remplacer les valeurs NaN par 0 dans les colonnes requises
           input_df[required_columns] = input_df[required_columns].fillna(0)

           # Prédire les clusters pour chaque client du fichier téléversé
           input_features = input_df[required_columns]
           input_features_scaled = scaler.transform(input_features)
           input_predictions = knn_model.predict(input_features_scaled)
           input_clusters = [determine_cluster(score) for score in input_predictions]

            # Ajouter les prédictions au dataframe
           input_df['Classement'] = input_clusters

            # Afficher les résultats
           st.write("Résultats de la prédiction :")
           st.write(input_df)

            # Visualisation
           st.write("Visualisation des clusters prédits :")
           plt.figure(figsize=(10, 6))
           sns.countplot(data=input_df, x='Classement')
           plt.title('Répartition des clusters')
           st.pyplot(plt)
            
            # Option pour ajouter les nouvelles données à la base de données principale
           if st.button("Ajouter à la base de données"):
            # Supprimer les doublons avant d'ajouter
             df = pd.concat([df, input_df]).drop_duplicates(subset=['Code'])  # Remplacez 'Identifiant_Unique' par la colonne unique de votre dataframe
             st.write("Les nouvelles données ont été ajoutées à la base de données.")
             st.write("Base de données mise à jour :")
             st.write(df)
        else:
            st.write("Le fichier téléversé ne contient pas les colonnes requises.")


elif page == 'Sélectionner un client':
    # Sélectionner un client
    st.sidebar.header('Sélectionner un client')
    client_id = st.sidebar.selectbox('Client', df['Client'].unique())

    if client_id:
     client_data = df[df['Client'] == client_id].iloc[0]
    title_color = '#FF5733'  # Définir la couleur du titre
    st.markdown(f"<h3 style='color: {title_color}; text-align: center;'>Détails du client : {client_id}</h3>", unsafe_allow_html=True)
    
    selected_columns = ['Total d\'impayées', "Chiffre d'affaire", 'Encaissement', 'Nombre d\'impayée']
    client_data_display = client_data[selected_columns].copy()

    # Formater les montants
    client_data_display['Total d\'impayées'] = format_currency(client_data_display['Total d\'impayées'])
    client_data_display["Chiffre d'affaire"] = format_currency(client_data_display["Chiffre d'affaire"])
    client_data_display['Encaissement'] = format_currency(client_data_display['Encaissement'])

    # Convertir en HTML
    client_data_display_html = client_data_display.to_frame().T.to_html(classes='dataframe', index=False, header=True)

    # CSS pour styliser le tableau
    st.markdown("""
    <style>
    .dataframe {
        margin: auto;
        width: 80%;
        font-size: 18px;
        border-collapse: collapse;
        border: 1px solid #dddddd;
    }
    .dataframe th, .dataframe td {
        text-align: center;
        padding: 8px;
    }
    .dataframe th {
        background-color: #f2f2f2;
    }
    </style>
    """, unsafe_allow_html=True)

    # Afficher le tableau
    st.markdown(client_data_display_html, unsafe_allow_html=True)

    if st.button('Voir le classement pour ce client'):
        features = [client_data['Nombre d\'impayée'], client_data['Total d\'impayées'], client_data["Chiffre d'affaire"], client_data['Encaissement']]
        predicted_cluster = predict_cluster(features)

        st.markdown(f"""
        <style>
        .predicted-cluster {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            width: 50%;
            margin: auto;
        }}
        </style>
        <div class="predicted-cluster">
            La classe du client <span style="color: #2196F3;">{client_id}</span> est : {predicted_cluster}
        </div>
        """, unsafe_allow_html=True)
            

elif page == 'Entrer les données manuellement':
    st.markdown(f"<h3 style='color: {title_color};'>Entrez les données manuellement</h3>", unsafe_allow_html=True)

    # Utilisation de CSS pour structurer et styliser le formulaire
    st.markdown(
        """
        <style>
        .form-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .form-row {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        .form-row label {
            margin-right: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Formulaire pour entrer manuellement les données
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    # Champ "Nom du client" seul sur une ligne
    client_name = st.text_input('Nom du client', key="client_name")

    # Champs "Nombre de impaye" et "Montant" sur la même ligne
    st.markdown('<div class="form-row">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        nombre_de_redundances = st.number_input('Nombre d\'impayée', min_value=0, value=0, key="nombre_de_redundances")
    with col2:
        montant = st.number_input('Total d\'impayées', min_value=0.0, value=0.0, key="montant")
    st.markdown('</div>', unsafe_allow_html=True)

    # Champs "Chiffre d'affaire" et "Encaissement" sur la même ligne
    st.markdown('<div class="form-row">', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        ca_23 = st.number_input("Chiffre d'affaire", min_value=0.0, value=0.0, key="ca_23")
    with col4:
        mtant_23 = st.number_input('Encaissement', min_value=-1.797e+308, value=0.0, key="mtant_23")
    st.markdown('</div>', unsafe_allow_html=True)


    if st.button('Prédire'):
        features = [nombre_de_redundances, montant, ca_23, mtant_23]
        predicted_cluster = predict_cluster(features)
        

        st.markdown(f"""
        <style>
        .predicted-cluster {{
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
            background-color: #f0f0f0;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }}
        </style>
        <div class="predicted-cluster">
            Cluster prédit pour le client <span style="color: #2196F3;">{client_name}</span>: {predicted_cluster}
        </div>
        """, unsafe_allow_html=True)

elif page == 'Afficher les métriques':
    # Calcul des métriques sur un échantillon de test
    def load_test_data():
        test_data = pd.read_csv('df.csv')
        X_test = test_data[['Nombre d\'impayée', 'Total d\'impayées', "Chiffre d'affaire", 'Encaissement']]
        y_true_clusters_test = test_data['Classement']
        return X_test, y_true_clusters_test 

    def compute_metrics():
        X_test, y_true_clusters_test = load_test_data()
        X_test_scaled = scaler.transform(X_test)
        y_pred_knn = knn_model.predict(X_test_scaled)
        y_pred_clusters_knn = [determine_cluster(score) for score in y_pred_knn]
        accuracy_knn = accuracy_score(y_true_clusters_test, y_pred_clusters_knn)
        report = classification_report(y_true_clusters_test, y_pred_clusters_knn)
        return accuracy_knn, report

    accuracy_knn, report = compute_metrics()
    st.write(f'Accuracy k-NN: {accuracy_knn:.2f}')
    st.write('Classification Report:')
    st.text(report)   

elif page == 'Visualiser les top clients':
    col1, col2 = st.columns(2)

    with col1:

     st.write("Répartition des classement des clients :")
    
    # Fonction pour afficher le nombre et le pourcentage de chaque catégorie dans le graphique circulaire
    def func(pct, allvalues):
        absolute = int(round(pct / 100. * sum(allvalues)))
        return f'{absolute} ({pct:.1f}%)'

    # Répartition des clients par cluster
    cluster_counts = df['Classement'].value_counts().sort_index()
    plt.figure(figsize=(4, 2))
    plt.pie(cluster_counts, labels=cluster_counts.index, autopct=lambda pct: func(pct, cluster_counts),
            colors=sns.color_palette('viridis', len(cluster_counts)), textprops={'fontsize': 6})
    plt.title('Répartition des Clients par Cluster')
    st.pyplot(plt)
    
