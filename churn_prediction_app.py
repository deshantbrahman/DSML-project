import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Churn Prediction App")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Analysis", "Prediction"])

df =pd.read_csv("C:\\Users\\brahm\\OneDrive\\Data sets\\Churn_Modelling.csv")
x = df.drop(columns=['Exited', 'Surname', 'RowNumber'])
y = df['Exited']

x =pd.get_dummies(x)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)


if page == "Introduction":
    st.title("Welcome to the Churn Prediction Model")
    st.write("Top five rows of the dataset")
    st.write(df.head())
    st.write("### Dataset Statistics")
    st.write(df.describe())

elif page == "Analysis":
    st.title("Dataset Analysis")
    st.write("Use the options below to explore the dataset and model results.")

    if st.button("Show Confusion Matrix"):
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(2,2))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Not Churned", "Churned"], yticklabels=["Not Churned", "Churned"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)


    if st.button("Show Scatter Plot (Age vs. Balance)"):
        st.subheader("Scatter Plot: Age vs. Balance")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Age", y="Balance", hue="Exited", palette="viridis", alpha=0.7)
        plt.title("Age vs. Balance with Churn Status")
        plt.xlabel("Age")
        plt.ylabel("Balance")
        st.pyplot(fig)

        # Geography-wise Customer Count (Bar Plot)
    if st.button("Show Geography-wise Customer Count"):
        st.subheader("Geography-wise Customer Count")
        geography_count = df['Geography'].value_counts()
        fig, ax = plt.subplots()
        geography_count.plot(kind="bar", color=["skyblue", "orange", "green"], alpha=0.8)
        plt.title("Customer Count by Geography")
        plt.xlabel("Geography")
        plt.ylabel("Customer Count")
        st.pyplot(fig)
        
    if st.button("Show Correlation Matrix"):
        st.subheader("Correlation Matrix")
        corr = x.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        st.pyplot(fig)

    if st.button("Show Dataset Statistics"):
        st.subheader("Dataset Statistics")
        st.write(df.describe())

elif page == "Prediction":
    st.title("Customer Churn Prediction")
    st.write("Enter the input data of the customer to check wehther the customer will churn or not")

    st.write(f"The accuracy of the model on testing data is  : {accuracy}")


    input_data = {}
    for col in x.columns:
        input_data[col] = st.number_input(f"Enter value for {col}", value=0.0, format="%.2f")

    
    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.success(f"Predicted Churn: {'Yes' if prediction[0] == 1 else 'No'}")