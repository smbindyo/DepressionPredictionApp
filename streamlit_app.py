#import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

st.title("DEPRESSION PREDICTION WEB APPLICATION")
st.write("This Web application is used to predict whether one has depression or not based on the data provided. A user is supposed to fill in a form provided where the application will predict if that particular user has depression or not."
)
st.subheader('Data information')

#ML modelling
url = "https://raw.githubusercontent.com/smbindyo/Datasets/refs/heads/main/student_depression_dataset.csv"
df = pd.read_csv(url)
st.subheader("Sample entry Data")
st.dataframe(df.head(3))

#Cleaning and preprocessing the data
# Drop the ? values
df = df.replace('?', np.nan)
def preprocess_data(df):
    df.dropna(inplace=True)
    # encode our dataset using OrdinalEncoder
    encoder = OrdinalEncoder()
    categorical_cols = ['Gender', 'City', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

    # Save the encoder for use with user input
    global fitted_encoder
    fitted_encoder = encoder
    global cat_columns
    cat_columns = categorical_cols
   
    return df

# process the data
preprocessed_df = preprocess_data(df)
st.subheader("Preprocessed Data")
st.dataframe(preprocessed_df.head())

## Train our ML Model
X = preprocessed_df.drop(['Depression', 'id'], axis=1)
y = preprocessed_df['Depression'].map({1:"Yes", 0:"No"})


# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# instantiate model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# prediction
y_pred = model.predict(X_test)
# evaluate model
from sklearn.metrics import classification_report


# display accuracy results if user wants to see them
if st.checkbox("Display accuarcy score?"):
    st.text(classification_report(y_test, y_pred))


## User interface
# create a form for user input

st.subheader("Fill in the form below to predict if you have depression or not")

col1, col2, col3 = st.columns(3)


# work on column 1
with col1:
    city = st.selectbox("City", ["Mumbai", "Delhi", "Agra", "Delhi", "Faridabad", "Hyderabad", "Kalyan", "Kanpur", "Ludhiana", "Meerut", "Nagpar", "Srinagar", "Vasai-Virar", "Visakhapatnam"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    profession = st.selectbox("Profession", ["Student", "Architect", "Civil Engineer", "Content Writer", "Digital Marketer", "Doctor", "Lawyer", "Manager", "Teacher", "UX/UI Designer", "Others"])
    self_employed = st.selectbox("Self employed", ["Yes", "No"])
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01, format="%.2f")
    age = st.slider("Age", min_value=1, max_value=100, value=25, step=1)
    

# work on column 2
with col2:
   
    sleepDuration = st.selectbox("Sleep Duration", ["'5-6 hours'", "'7-8 hours'", "'Less than 5 hours'", "'More than 8 hours'", "'Others'" ])
    dietaryHabits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy", "Moderate"])
    studySatisfaction = st.selectbox("Study Satisfaction", ["0", "1", "2", "3", "4", "5"])
    wsours = st.selectbox("Work/Study Hours", ["1", "2", "3", "4", "6", "7", "8", "9", "10", "12"])
    fstress = st.selectbox("Financial Stress", ["1", "2", "3", "4", "5"])
    degree = st.selectbox("Degree", ["B.Ed", "BA", "MTECH", "MCA", "B.Arc", "BE", "BSc", "B.Pharm", "MBA", "MSc", "MTech", "PhD", "B.Com"])
    

    

# work on column 3
with col3:
    acPressure = st.selectbox("Academic Pressure", ["0", "1", "2", "3", "4", "5"])
    workPressure = st.selectbox("Work Pressure", ["0", "2", "5"])
    jobSatisfaction = st.selectbox("Job Satisfaction", ["0", "1", "2", "3", "4", "5"])
    thoughts = st.selectbox("Have you ever had suicidal thoughts ?", ["Yes", "No"])
    fhistory = st.selectbox("Family History of Mental Illness", ["Yes", "No"])
    

# preprocess user input data
def preprocess_input(data):
    # Create a DataFrame from user input
    input_df = pd.DataFrame(data, index=[0])

    # Convert string values to appropriate numeric types
    
    if 'CGPA' in input_df:
        input_df['CGPA'] = input_df['CGPA'].astype(float)
    if 'Study_Satisfaction' in input_df:
        input_df['Study_Satisfaction'] = input_df['Study_Satisfaction'].astype(float)
    if 'Work/Study Hours' in input_df:
        input_df['Work/Study Hours'] = input_df['Work/Study Hours'].astype(float)
    if 'Financial Stress' in input_df:
        input_df['Financial Stress'] = input_df['Financial Stress'].astype(float)
    if 'Academic Pressure' in input_df:
        input_df['Academic Pressure'] = input_df['Academic Pressure'].astype(float)
    if 'Work Pressure' in input_df:
        input_df['Work Pressure'] = input_df['Work Pressure'].astype(float)
    if 'Job Satisfaction' in input_df:
        input_df['Job Satisfaction'] = input_df['Job Satisfaction'].astype(float) 
    if 'Age' in input_df:
        input_df['Age'] = input_df['Age'].astype(float)
   
    # Use the fitted encoder to transform categorical features


    categorical_data = input_df[cat_columns]
    input_df[cat_columns] = fitted_encoder.transform(categorical_data)
   
    return input_df

# Create a button to predict
if st.button("Predict Depression Status"):
    # Collect all user input into a dictionary
    user_input = {
        'Gender': gender,
        'Age': age,
        'City': city,
        'Profession': profession,
        'Academic Pressure': acPressure,
        'Work Pressure': workPressure,
        'CGPA': cgpa,
        'Study Satisfaction': studySatisfaction,
        'Job Satisfaction': jobSatisfaction,
        'Sleep Duration': sleepDuration,
        'Dietary Habits': dietaryHabits,
        'Degree': degree,
        'Have you ever had suicidal thoughts ?': thoughts,
        'Work/Study Hours': wsours,
        'Financial Stress': fstress,
        'Family History of Mental Illness': fhistory
       }
        
 
   
    # Preprocess the user input
    processed_input = preprocess_input(user_input)
   
    # Display the processed input data
    st.subheader("Processed User Input")
    st.dataframe(processed_input)
    

# Make prediction using the model
    prediction = model.predict(processed_input)
   
    # Display the prediction
    st.subheader("Depression Status Prediction")
    if prediction[0] == "Yes":
        st.success("Hey! You are likely to be depressed.")
    else:
        st.error("Hey! You are likely not to be depressed.")
   


