# Import the required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import seaborn as sns
# for SVM choice
from sklearn.svm import SVC
# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
# instantiate classifier with default hyperparameters

from sklearn.neural_network import MLPClassifier



st.header('Using Neural Network')
dataset_name = st.selectbox('Select Data base: ',
                            ('Breast Cancer', 'Iris', 'Digits'))
stat = st.selectbox('Info of Dataframe: ', ('Head', 'Tail', 'Correlation', 'Describe'))
def get_dataset(dataset_name):
    if dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    elif dataset_name == 'Iris':
        data = datasets.load_iris()
    else:
        data = datasets.load_digits()
    X = data.data
    y = data.target
    return data, X, y
df_data = st.radio("Which Dataset you choose", ('Your Data', 'Prepared'))
if df_data == 'Your Data':
    spectra = st.file_uploader("Upload file", type={"csv", "txt"})
    if spectra is not None:
        df = pd.read_csv(spectra)
        for col_name in df.columns:
            if (df[col_name].dtype == 'object'):
                df[col_name] = df[col_name].astype('category')
                df[col_name] = df[col_name].cat.codes
        y = df.iloc[:, -1]
        X = df.iloc[:, 1:-1]
        st.write('You selected your data')
        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            st.write(df.corr())
        else:
            st.write(df.describe())
        st.write('Number of data', X.shape)
        st.write('Number of Class= ', len(np.unique(y)))
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        C0 = st.slider('max_iter', 1, 1000)
        CA = st.slider('hidden_layer_sizes', 1, 1000)
        clf = MLPClassifier(hidden_layer_sizes=CA, learning_rate_init=0.1, max_iter=C0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy score = {acc:.5f}')
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        fig,ax=plt.subplots(figsize=(2,2))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues',ax=ax)
        ax.set_title('Confusion Matrix with labels\n\n');
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ');
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig)
        pca = PCA(2)
        X_projected = pca.fit_transform(X)
        x1 = X_projected[:, 0]
        x2 = X_projected[:, 1]
        fig = plt.figure()
        plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
        plt.xlabel('Principle Component 1')
        plt.ylabel('Principle Component 2')
        plt.colorbar()
        st.pyplot(fig)

else:
    data, X, y = get_dataset(dataset_name)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    st.write("You select prepared data.")
    if stat == 'Head':
        st.write(df.head())
    elif stat == 'Tail':
        st.write(df.tail())
    elif stat == 'Correlation':
       st.write(df.corr())
    else:
        st.write(df.describe())

    st.write('Number of data', X.shape)
    st.write('Number of Class= ', len(np.unique(y)))
    C0 = st.slider('max_iter', 1, 1000)
    CA = st.slider('hidden_layer_sizes', 1, 1000)
    from sklearn.neural_network import MLPClassifier
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = MLPClassifier(hidden_layer_sizes=CA, learning_rate_init=0.1, max_iter=C0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy score = {acc:.5f}')
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(2, 2))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
    ax.set_title('Seaborn Confusion Matrix with labels\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ## Display the visualization of the Confusion Matrix.
    st.pyplot(fig)

    pca = PCA(2)
    X_projected = pca.fit_transform(X)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
    plt.xlabel('Principle Component 1')
    plt.ylabel('Principle Component 2')
    plt.colorbar()
    st.pyplot(fig)


