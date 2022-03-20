import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'Model/ld_tuned_final.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

options_cause_acc = ['Moving Backward', 'Overtaking', 'Changing lane to the left',
       'Changing lane to the right', 'Overloading', 'Other',
       'No priority to vehicle', 'No priority to pedestrian',
       'No distancing', 'Getting off the vehicle improperly',
       'Improper parking', 'Overspeed', 'Driving carelessly',
       'Driving at high speed', 'Driving to the left', 'Unknown',
       'Overturning', 'Turnover', 'Driving under the influence of drugs',
       'Drunk driving']

options_collision_typ = ['Collision with roadside-parked vehicles',
       'Vehicle with vehicle collision',
       'Collision with roadside objects', 'Collision with animals',
       'Other', 'Rollover', 'Fall from vehicles',
       'Collision with pedestrians', 'With Train']

options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']

options_junction_typ = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other',
       'Unknown', 'T Shape', 'X Shape']

options_age = ['na', '31-50', '18-30', 'Under 18', 'Over 51', '5']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
       
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']

options_minute = [5, 10, 15, 30, 20, 40, 45, 35, 25,  0, 50, 55]

options_edu = ['Above high school', 'Junior high school', 'Elementary school',
       'High school', 'Unknown', 'Illiterate', 'Writing & reading']

features = ['hour','Cause_of_accident','Type_of_collision','Minute','Type_of_vehicle','Types_of_junction','Area_accident_occured','Age_band_of_casualty','Day_of_week','Educational_level']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        hour = st.slider("Hour of Accident: ", 0, 23, value=0, format="%d")
        Cause_of_accident = st.selectbox("Select Cause of Accident : ", options=options_cause_acc)
        Type_of_collision = st.selectbox("Select Type Of Collision : ", options=options_collision_typ)
        Minute = st.selectbox("Select Minute of Accident : ", options=options_minute)
        Type_of_vehicle = st.selectbox("Select type of vehicle : ", options=options_vehicle_type)
        Types_of_junction = st.selectbox("Select type of vehicle : ", options=options_junction_typ)
        Area_accident_occured = st.selectbox("Select Accident Area: ", options=options_acc_area)
        Age_band_of_casualty = st.selectbox("Select Age of casualty: ", options=options_age)
        Day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        Educational_level = st.selectbox("Select Cause of Accident : ", options=options_edu)

        submit = st.form_submit_button("Predict")


    if submit:
        Cause_of_accident = ordinal_encoder(Cause_of_accident, options_cause_acc)
        Type_of_collision = ordinal_encoder(Type_of_collision, options_collision_typ)
        Type_of_vehicle = ordinal_encoder(Type_of_vehicle, options_vehicle_type)
        Types_of_junction = ordinal_encoder(Types_of_junction, options_junction_typ)
        Area_accident_occured = ordinal_encoder(Area_accident_occured, options_acc_area)
        Age_band_of_casualty = ordinal_encoder(Age_band_of_casualty, options_age)
        Day_of_week = ordinal_encoder(Day_of_week, options_day)
        Educational_level = ordinal_encoder(Educational_level, options_edu)

        data = np.array([hour,Cause_of_accident,Type_of_collision,Minute,Type_of_vehicle,Types_of_junction,
                            Area_accident_occured,Age_band_of_casualty,Day_of_week,Educational_level]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()