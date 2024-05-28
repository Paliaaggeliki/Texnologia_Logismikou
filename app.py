import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClusteringfrom sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# ΒΗΜΑ 1ο: Φόρτωση Δεδομένων: Η εφαρμογή θα πρέπει να είναι σε θέση να φορτώνει tabular data (csv)
def load_data():
    uploaded_file = st.file_uploader("heart_failure_clinical_records.csv")
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.success("File loaded successfully!")    
        return data

# ΒΗΜΑ 2ο: Προδιαγραφές Πίνακα
def validate_data(data):
    if data is not None:
        rows, cols = data.shape
        if cols < 2:
            st.error("The data must have at least one feature column and one label column")
            return False
        
        # Assuming the last column is the label
        features = data.iloc[:, :-1]
        label = data.iloc[:, -1]
        
        if label.nunique() < 2:
            st.error("The label column must contain at least two unique values")
            return False
        
        st.success(f"Data has {rows} samples and {cols-1} features with 1 label column.")
        return True
    return False

def preprocess_data(data):
    # Remove rows with any non-numeric data in features
    features = data.iloc[:, :-1]
    features = features.apply(pd.to_numeric, errors='coerce')
    label = data.iloc[:, -1]
    
    # Drop rows with NaN values
    data_clean = features.dropna()
    data_clean[label.name] = label.loc[data_clean.index]
    
    return data_clean

# ΒΗΜΑ 3ο: 2D Visualization Tab

def plot_tsne(data):
    if data.empty:
        st.error("No valid data available for t-SNE.")
        return
    
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette='viridis')
    plt.title('t-SNE 2D Visualization')
    st.pyplot(plt)
 
def plot_eda(data):
    if data.empty:
        st.error("No valid data available for EDA.")
        return
    
    st.write("Exploratory Data Analysis:")
    
    # Pairplot
    st.write("Pairplot:")
    sns.pairplot(data, hue=data.columns[-1])
    st.pyplot(plt)
    
    # Heatmap
    st.write("Heatmap:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)
    
    # Boxplot
    st.write("Boxplot:")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    st.pyplot(plt)

# ΒΗΜΑ 4
def run_classification(data):
    st.write("Classification Tab")
    
    # Split the data
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Check if labels are continuous and convert them to categorical if necessary
    if y.dtype.kind in 'fc':
        y = pd.cut(y, bins=5, labels=False)  # Discretize continuous labels into 5 categories

    # Encode labels to ensure they are numeric
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Get user input for k-NN
    k = st.sidebar.slider("Select k for k-NN", 1, 15, 3)
    
    # k-NN Classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    
    # Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Calculate metrics
    metrics = {
        "k-NN": {
            "accuracy": accuracy_score(y_test, y_pred_knn),
            "f1": f1_score(y_test, y_pred_knn, average='weighted'),
            "precision": precision_score(y_test, y_pred_knn, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred_knn, average='weighted')
        },
        "Random Forest": {
            "accuracy": accuracy_score(y_test, y_pred_rf),
            "f1": f1_score(y_test, y_pred_rf, average='weighted'),
            "precision": precision_score(y_test, y_pred_rf, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred_rf, average='weighted')
        }
    }
    
    # Display metrics
    st.write("## Performance Metrics")
    for model, metric in metrics.items():
        st.write(f"### {model}")
        st.write(f"Accuracy: {metric['accuracy']:.4f}")
        st.write(f"F1 Score: {metric['f1']:.4f}")
        st.write(f"Precision: {metric['precision']:.4f}")
        st.write(f"Recall: {metric['recall']:.4f}")
    
    # Confusion Matrices
    st.write("## Confusion Matrices")
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', ax=ax[0])
    ax[0].set_title('k-NN Confusion Matrix')
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    
    sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', ax=ax[1])
    ax[1].set_title('Random Forest Confusion Matrix')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Actual')
    
    st.pyplot(fig)
    
    # Classification Reports
    st.write("## Classification Reports")
    st.write("### k-NN")
    st.text(classification_report(y_test, y_pred_knn, zero_division=0))
    
    st.write("### Random Forest")
    st.text(classification_report(y_test, y_pred_rf, zero_division=0))

    #ΒΗΜΑ 5 Determine the best model
    best_model = max(metrics, key=lambda x: metrics[x]['f1'])
    st.write(f"## Best Model: {best_model}")
    

def run_clustering(data):
    st.write("Clustering Tab")
    
    # Split the data
    X = data.iloc[:, :-1]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get user input for k-means
    k = st.sidebar.slider("Select k for k-Means", 2, 10, 3)
    
    # k-Means Clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels_kmeans = kmeans.fit_predict(X_scaled)
    
    # Hierarchical Clustering
    agg = AgglomerativeClustering(n_clusters=k)
    labels_agg = agg.fit_predict(X_scaled)

    # Calculate metrics
    metrics = {
        "k-Means": {
            "silhouette": silhouette_score(X_scaled, labels_kmeans),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels_kmeans)
        },
        "Agglomerative Clustering": {
            "silhouette": silhouette_score(X_scaled, labels_agg),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels_agg)
        }
    }

    # Display metrics
    st.write("## Performance Metrics")
    for model, metric in metrics.items():
        st.write(f"### {model}")
        st.write(f"Silhouette Score: {metric['silhouette']:.4f}")
        st.write(f"Davies-Bouldin Score: {metric['davies_bouldin']:.4f}")
    
    # Plot clustering results
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels_kmeans, palette='viridis', ax=ax[0])
    ax[0].set_title('k-Means Clustering')
    
    sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels_agg, palette='viridis', ax=ax[1])
    ax[1].set_title('Agglomerative Clustering')
    
    st.pyplot(fig)

    #ΒΗΜΑ 5 Determine the best model
    best_model = max(metrics, key=lambda x: metrics[x]['silhouette'])
    st.write(f"## Best Model: {best_model}")

#ΒΗΜΑ 6 
def display_info():
    st.header("Σχετικά με την Εφαρμογή")
    st.write("""
        Αυτή η εφαρμογή αναπτύχθηκε για την ανάλυση δεδομένων με διάφορους αλγόριθμους μηχανικής μάθησης.
        Παρέχει δυνατότητες όπως:
        - Φόρτωση και προεπεξεργασία δεδομένων
        - Οπτικοποίηση δεδομένων σε 2D με χρήση t-SNE
        - Αναλυτική Εξερεύνηση Δεδομένων (EDA)
        - Ταξινόμηση δεδομένων με χρήση αλγορίθμων k-NN και Random Forest
        - Συσταδοποίηση δεδομένων με χρήση αλγορίθμων k-Means και Agglomerative Clustering
        - Παρουσίαση των αποτελεσμάτων και καθορισμός του καλύτερου μοντέλου βάσει των μετρικών απόδοσης
    """)

    
def main():
    st.title("Application Development for Data Analysis")

    # Load the data
    data = load_data()

    # Display the data
    if data is not None:
        if validate_data(data): 
            st.write("Data Preview:")
            st.dataframe(data.head())

            # Preprocess the data
            data_clean = preprocess_data(data)
            
            # Check if cleaned data is empty
            if data_clean.empty:
                st.error("No valid data available after cleaning. Please upload a dataset with valid numeric values.")
                return
                
            # Tabs for different visualizations
            tab1, tab2,tab3, tab4, tab5 = st.tabs(["T-SNE", "EDA","CLASSIFICATION","CLUSTERING","INFO"])

            with tab1:
                st.header("t-SNE Visualization")
                plot_tsne(data_clean)
                
            with tab2:
                st.header("Exploratory Data Analysis")
                plot_eda(data_clean)

            with tab3:
                st.header("Classification")
                run_classification(data_clean)
                
            with tab4:
                st.header("Clustering")
                run_clustering(data_clean)

            with tab5:
                st.header("INFO")
                display_info()
       else:
            st.error("Invalid data structure. Please upload a dataset with the correct format.")

if __name__ == "__main__":
    main()
