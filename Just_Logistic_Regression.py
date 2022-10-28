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
from sklearn.neural_network import MLPClassifier



st.header('Using Logistic Regression')
df_data = st.radio("Which Dataset you choose", ('Your Data', 'Prepared'))
dataset_name = st.selectbox('Select Data base: ',
                                ('Titanic', 'Diabetes', 'Iris'))
stat = st.selectbox('Info of Dataframe: ', ('Head', 'Tail', 'Correlation', 'Describe'))
if df_data == 'Your Data':
    spectra = st.file_uploader("Upload file", type={"csv", "txt"})
    if spectra is not None:
        df = pd.read_csv(spectra)
        df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 2})
        y = df.iloc[:, 1]
        X = df.iloc[:, 2:-1]
        st.write('You selected your data')
        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        from sklearn.linear_model import LogisticRegression
        C0 = st.slider('Maximum iteration', 1, 1000)
        logmodel = LogisticRegression(max_iter=C0)
        logmodel.fit(X_train, y_train)
        y_pred = logmodel.predict(X_test)
        from sklearn.metrics import confusion_matrix
        confusion_matrix(y_test, y_pred)
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
        ax.set_title('Seaborn Confusion Matrix with labels');
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values ');
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig)
        from sklearn.metrics import accuracy_score
        mm = accuracy_score(y_test, y_pred)
        st.subheader(f'Accuracy score = {mm:.5f}')
        from sklearn.metrics import classification_report
        st.subheader('Model Report:\n ' + classification_report(y_test, y_pred))



else:
    if dataset_name == 'Titanic':
        data =pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
        data = data.dropna()
        X = data[['Pclass','Sex','Parch','Embarked','Age']]
        for col_name in X.columns:
            if (X[col_name].dtype == 'object'):
                X[col_name] = X[col_name].astype('category')
                X[col_name] = X[col_name].cat.codes

        y = data['Survived']
        df = data
        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        from sklearn.linear_model import LogisticRegression
        C0 = st.slider('Maximum iteration', 1, 1000)
        logmodel = LogisticRegression(max_iter=C0)
        logmodel.fit(X_train, y_train)
        y_pred = logmodel.predict(X_test)
        from sklearn.metrics import confusion_matrix
        confusion_matrix(y_test, y_pred)
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
        ax.set_title('Seaborn Confusion Matrix with labels');
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values ');
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig)
        from sklearn.metrics import accuracy_score
        mm = accuracy_score(y_test, y_pred)
        st.subheader(f'Accuracy score = {mm:.5f}')
        from sklearn.metrics import classification_report
        st.subheader('Model Report:\n ' + classification_report(y_test, y_pred))





    elif dataset_name == 'Diabetes':
        data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv')
        df = data
        feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                        'DiabetesPedigreeFunction', 'Age']
        X = data[feature_cols]  # Features
        y= data.Outcome  # Target variable

        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        from sklearn.linear_model import LogisticRegression
        C0 = st.slider('Maximum iteration', 1, 1000)
        logmodel = LogisticRegression(max_iter=C0)
        logmodel.fit(X_train, y_train)
        y_pred = logmodel.predict(X_test)
        from sklearn.metrics import confusion_matrix
        confusion_matrix(y_test, y_pred)
        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        fig, ax = plt.subplots(figsize=(2, 2))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
        ax.set_title('Seaborn Confusion Matrix with labels');
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values ');
        ## Display the visualization of the Confusion Matrix.
        st.pyplot(fig)
        from sklearn.metrics import accuracy_score
        mm = accuracy_score(y_test, y_pred)
        st.subheader(f'Accuracy score = {mm:.5f}')
        from sklearn.metrics import classification_report
        st.subheader('Model Report:\n ' + classification_report(y_test, y_pred))

    else:
        data = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
        data['species'] = data['species'].replace({'setosa': 1, 'versicolor': 2, 'virginica': 3})
        df = data
        X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        Y = df['species'].values
        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            # st.write(fig)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
            from sklearn.linear_model import LogisticRegression
            C0 = st.slider('Maximum iteration', 1, 1000)
            logmodel = LogisticRegression(max_iter=C0)
            logmodel.fit(X_train, y_train)
            y_pred = logmodel.predict(X_test)
            from sklearn.metrics import confusion_matrix
            confusion_matrix(y_test, y_pred)
            conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
            fig, ax = plt.subplots(figsize=(2, 2))
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
            ax.set_title('Seaborn Confusion Matrix with labels');
            ax.set_xlabel('Predicted Values')
            ax.set_ylabel('Actual Values ');
            ## Display the visualization of the Confusion Matrix.
            st.pyplot(fig)
            from sklearn.metrics import accuracy_score
            mm = accuracy_score(y_test, y_pred)
            st.subheader(f'Accuracy score = {mm:.5f}')
            from sklearn.metrics import classification_report
            st.subheader('Model Report:\n ' + classification_report(y_test, y_pred))



