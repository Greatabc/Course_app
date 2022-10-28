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
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
# instantiate classifier with default hyperparameters



st.set_page_config(layout="wide")


# Functions for each of the pages
def data_NN():
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
            X = df.iloc[:, :-1]
            st.write('You selected your data')
            if stat == 'Head':
                st.write(df.head())
            elif stat == 'Tail':
                st.write(df.tail())
            elif stat == 'Correlation':
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
                st.pyplot(fig)
                st.write(df.corr())
            else:
                st.write(df.describe())
            st.write('Number of data', X.shape)
            st.write('Number of Class= ', len(np.unique(y)))

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            C0 = st.slider('max_iter', 1, 1000)
            CA = st.slider('hidden_layer_sizes', 1, 1000)
            clf = MLPClassifier(hidden_layer_sizes=CA, learning_rate_init=0.1, max_iter=C0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f'Accuracy score = {acc:.5f}')
            conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
            fig, ax = plt.subplots(figsize=(2, 2))
            sns.heatmap(conf_matrix, annot=True, cmap='Blues', ax=ax)
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
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())

        st.write('Number of data', X.shape)
        st.write('Number of Class= ', len(np.unique(y)))
        C0 = st.slider('max_iter', 1, 1000)
        CA = st.slider('hidden_layer_sizes', 1, 1000)

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


def data_Linear_Regression():
    st.header('Linear Regression')
    #st.header('Using Linear Regression')
    df_data = st.radio("Which Dataset you choose", ('Your Data', 'Prepared'))

    dataset_name = st.selectbox('Select Data base: ',
                                ('california_housing_test', 'Headbrain', 'Cruise ship info'))
    stat = st.selectbox('Info of Dataframe: ', ('Head', 'Tail', 'Correlation', 'Describe'))

    if df_data == 'Your Data':
        spectra = st.file_uploader("Upload file", type={"csv", "txt"})
        if spectra is not None:
            df = pd.read_csv(spectra)
            Y = df.iloc[:, -1]
            X = df.iloc[:, 0]
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
                # Calculating coefficient
            # Mean X and Y
            mean_x = np.mean(X)
            mean_y = np.mean(Y)

            # Total number of values
            n = len(X)

            # Using the formula to calculate theta1 and theta2
            numer = 0
            denom = 0
            for i in range(n):
                numer += (X[i] - mean_x) * (Y[i] - mean_y)
                denom += (X[i] - mean_x) ** 2
            b1 = numer / denom
            b0 = mean_y - (b1 * mean_x)
            y = b0 + b1 * X
            rmse = 0
            for i in range(n):
                y_pred = b0 + b1 * X[i]
                rmse += (Y[i] - y_pred) ** 2

            rmse = np.sqrt(rmse / n)
            # Calculating R2 Score
            ss_tot = 0
            ss_res = 0
            for i in range(n):
                y_pred = b0 + b1 * X[i]
                ss_tot += (Y[i] - mean_y) ** 2
                ss_res += (Y[i] - y_pred) ** 2
            r2 = 1 - (ss_res / ss_tot)
            # st.write(f'Root Mean Squares Error = {rmse:.5f}')
            st.write(f'R2 Score = {r2:.5f}')
            # Ploting Line

            fig = plt.figure()
            plt.plot(X, y, color='blue', label='Regression Line')
            plt.scatter(X, Y, c='green', label='Data points')
            plt.xlabel('Hours')
            plt.ylabel('Score')
            plt.colorbar()
            plt.grid()
            st.pyplot(fig)


    else:
        if dataset_name == 'california_housing_tes':
            data = pd.read_csv(
                'https://raw.githubusercontent.com/timothypesi/Data-Sets-For-Machine-Learning-/main/california_housing_test.csv')
            X = data[['total_rooms', 'population', 'households']]
            y = data['total_bedrooms']
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

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
            from sklearn.linear_model import LinearRegression

            ml = LinearRegression()
            ml.fit(X_train, y_train)
            y_prediction = ml.predict(X_test)
            from sklearn.metrics import r2_score

            aa = r2_score(y_test, y_prediction)
            st.write(f'Prediction = {aa:.5f}')
            st.subheader('Percent difference ')
            Delta_Y = pd.DataFrame({'Actual Data': y_test, 'Predicted': y_prediction,
                                    '% Difference': (np.abs(y_test - y_prediction) / y_test) * 100})
            st.write(Delta_Y.head())
            fig = plt.figure()
            plt.scatter(y_test, y_prediction, alpha=0.8, cmap='viridis')
            plt.xlabel('Actual Data')
            plt.ylabel('Predicted Data')
            plt.colorbar()
            plt.grid()
            st.pyplot(fig)

        elif dataset_name == 'Headbrain':
            data = pd.read_csv('https://raw.githubusercontent.com/RupeshMohan/Linear_Regression/master/headbrain.csv')
            df = data
            X = df['Head Size(cm^3)'].values
            Y = df['Brain Weight(grams)'].values

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

            # Calculating coefficient

            # Mean X and Y
            mean_x = np.mean(X)
            mean_y = np.mean(Y)

            # Total number of values
            n = len(X)

            # Using the formula to calculate theta1 and theta2
            numer = 0
            denom = 0
            for i in range(n):
                numer += (X[i] - mean_x) * (Y[i] - mean_y)
                denom += (X[i] - mean_x) ** 2
            b1 = numer / denom
            b0 = mean_y - (b1 * mean_x)
            y = b0 + b1 * X
            rmse = 0
            for i in range(n):
                y_pred = b0 + b1 * X[i]
                rmse += (Y[i] - y_pred) ** 2

            rmse = np.sqrt(rmse / n)
            # Calculating R2 Score
            ss_tot = 0
            ss_res = 0
            for i in range(n):
                y_pred = b0 + b1 * X[i]
                ss_tot += (Y[i] - mean_y) ** 2
                ss_res += (Y[i] - y_pred) ** 2
            r2 = 1 - (ss_res / ss_tot)
            # st.write(f'Root Mean Squares Error = {rmse:.5f}')
            st.write(f'R2 Score = {r2:.5f}')
            # Ploting Line

            fig = plt.figure()
            plt.plot(X, y, color='blue', label='Regression Line')
            plt.scatter(X, Y, c='green', label='Data points')
            plt.xlabel('Head Size in cm3')
            plt.ylabel('Brain Weight in grams')
            plt.colorbar()
            plt.grid()
            st.pyplot(fig)
        else:
            data = pd.read_csv(
                'https://raw.githubusercontent.com/LeondraJames/Hyundai-Cruise-Ship-Crew-Prediction/master/cruise_ship_info.csv')
            df = data
            X = df['passengers'].values
            Y = df['crew'].values
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

            # Calculating coefficient

            # Mean X and Y
            mean_x = np.mean(X)
            mean_y = np.mean(Y)

            # Total number of values
            n = len(X)

            # Using the formula to calculate theta1 and theta2
            numer = 0
            denom = 0
            for i in range(n):
                numer += (X[i] - mean_x) * (Y[i] - mean_y)
                denom += (X[i] - mean_x) ** 2
            b1 = numer / denom
            b0 = mean_y - (b1 * mean_x)
            y = b0 + b1 * X
            rmse = 0
            for i in range(n):
                y_pred = b0 + b1 * X[i]
                rmse += (Y[i] - y_pred) ** 2

            rmse = np.sqrt(rmse / n)
            # Calculating R2 Score
            ss_tot = 0
            ss_res = 0
            for i in range(n):
                y_pred = b0 + b1 * X[i]
                ss_tot += (Y[i] - mean_y) ** 2
                ss_res += (Y[i] - y_pred) ** 2
            r2 = 1 - (ss_res / ss_tot)
            # st.write(f'Root Mean Squares Error = {rmse:.5f}')
            st.write(f'R2 Score = {r2:.5f}')

            fig = plt.figure()
            plt.plot(X, y, color='blue', label='Regression Line')
            plt.scatter(X, Y, c='steelblue', edgecolor='white', s=70)
            plt.xlabel('passengers')
            plt.ylabel('crew')
            plt.title('scatter plot of passengers vs crew')
            plt.colorbar()
            plt.grid()
            st.pyplot(fig)

        # data,X,y= get_dataset(dataset_name)
        # def get_dataset(dataset_name):
        # return data, X, y

    # end of Linear Regression

def data_Logistic_Regression():
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
            data = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')
            data = data.dropna()
            X = data[['Pclass', 'Sex', 'Parch', 'Embarked', 'Age']]
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
            y = data.Outcome  # Target variable

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

    # end of Logistic Regression
def data_Cover():
    st.title('CE 548 Machine Learning and Artificial Intellegent')
def data_Random_Forest():
    st.header('Random Forest')


def data_SVM():
    #beginning svm

    #st.set_page_config(layout="wide")
    st.header('Using SVM')
    dataset_name = st.selectbox('Select Data base: ',
                                ('Breast Cancer', 'Iris', 'Digits'))
    classifier_name = st.selectbox('Select kernel: ', ('rbf', 'poly', 'sigmoid', 'linear'))
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
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
            st.write('You selected your data')
            if stat == 'Head':
                st.write(df.head())
            elif stat == 'Tail':
                st.write(df.tail())
            elif stat == 'Correlation':
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
                st.pyplot(fig)
                st.write(df.corr())
            else:
                st.write(df.describe())

            st.write('Number of data', X.shape)
            st.write('Number of Class= ', len(np.unique(y)))
            C0 = st.slider('C', 1.00, 1000.00)
            clf = SVC(kernel=classifier_name, C=C0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f'Accuracy score = {acc:.5f}')
            conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

            #
            # Print the confusion matrix using Matplotlib
            #
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

    else:
        data, X, y = get_dataset(dataset_name)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        st.write("You select prepared data.")
        if stat == 'Head':
            st.write(df.head())
        elif stat == 'Tail':
            st.write(df.tail())
        elif stat == 'Correlation':
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, ax=ax)
            st.pyplot(fig)
            st.write(df.corr())
        else:
            st.write(df.describe())

        st.write('Number of data', X.shape)
        st.write('Number of Class= ', len(np.unique(y)))
        C0 = st.slider('C', 1.00, 1000.00)
        clf = SVC(kernel=classifier_name, C=C0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy score = {acc:.5f}')

        conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
        #
        # Print the confusion matrix using Matplotlib
        #
        fig, ax = plt.subplots()
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

    #end svm
# Add a title and intro text
#st.title('Machine Learning')
#st.text('This is a web app to allow exploration of Machine Learning')

# Sidebar setup
st.sidebar.title('Sidebar')
# Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', ['Cover','NN', 'Linear Regression','Logistic Regression', 'SVM', 'Random Forest'])

# Navigation options
if options == 'Cover':
    data_Cover()
elif options == 'NN':
    data_NN()
elif options == 'Linear Regression':
    data_Linear_Regression()
elif options == 'Logistic Regression':
    data_Logistic_Regression()
elif options == 'SVM':
    data_SVM()
elif options == 'Random Forest':
    data_Random_Forest()

