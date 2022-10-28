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



st.header('Using Linear Regression')
df_data = st.radio("Which Dataset you choose", ('Your Data', 'Prepared'))

dataset_name = st.selectbox('Select Data base: ',
                                ('california_housing_tes', 'Headbrain', 'Cruise ship info'))
stat = st.selectbox('Info of Dataframe: ', ('Head', 'Tail', 'Correlation', 'Describe'))

if df_data == 'Your Data':
    spectra = st.file_uploader("Upload file", type={"csv", "txt"})
    if spectra is not None:
        df = pd.read_csv(spectra)
        Y = df.iloc[:, -1]
        X = df.iloc[:, 0::-1]
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
            #st.write(f'Root Mean Squares Error = {rmse:.5f}')
            st.write(f'R2 Score = {r2:.5f}')
            # Ploting Line

            fig = plt.figure()
            plt.plot(X, Y, color='blue', label='Regression Line')
            plt.scatter(X, Y, c='green', label='Data points')
            plt.xlabel('Head Size in cm3')
            plt.ylabel('Brain Weight in grams')
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
        #st.write(f'Root Mean Squares Error = {rmse:.5f}')
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
        #st.write(f'Root Mean Squares Error = {rmse:.5f}')
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


    #data,X,y= get_dataset(dataset_name)
    #def get_dataset(dataset_name):
        #return data, X, y




