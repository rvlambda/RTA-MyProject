import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'Model/rf_deploy.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

options_age = ['5','Under 18', '18-30', '31-50', 'Over 51', 'Unknown' ]

options_light = ['Daylight', 'Darkness - lights lit','Darkness - no lighting','Darkness - lights unlit']

options_drv_exp = ['No Licence','Below 1yr','1-2yr','2-5yr','5-10yr','unknown']

options_junction_typ = ['No junction', 'Y Shape', 'Crossing', 'O Shape', 'Other',
       'Unknown', 'T Shape', 'X Shape']

options_casualty_sex = ['Male','Female','na']

options_casualty_class = ['Driver or rider','na','Pedestrian','Passenger']

options_cause_acc = ['Moving Backward', 'Overtaking', 'Changing lane to the left',
       'Changing lane to the right', 'Overloading', 'Other',
       'No priority to vehicle', 'No priority to pedestrian',
       'No distancing', 'Getting off the vehicle improperly',
       'Improper parking', 'Overspeed', 'Driving carelessly',
       'Driving at high speed', 'Driving to the left', 'Unknown',
       'Overturning', 'Turnover', 'Driving under the influence of drugs',
       'Drunk driving']


options_lane = ['Two-way (divided with broken lines road marking)','Undivided Two way',
       'other','Double carriageway (median)','One way','Two-way (divided with solid lines road marking)',
       'Unknown']


options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
       

features = ['Number_of_vehicles_involved','Number_of_casualties','Age_band_of_driver','Day_of_week',
       'Light_conditions','Driving_experience','Types_of_Junction','Sex_of_casualty','Casualty_class',
       'Area_accident_occured','Lanes_or_Medians']


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        Number_of_vehicles_involved = st.slider("Number of Vehicles Involved: ", 1, 7, value=1, format="%d")
        Number_of_casualties = st.slider("Number of casualties: ", 1, 8, value=1, format="%d")
        Age_band_of_driver = st.selectbox("Select Age of driver: ", options=options_age)
        Day_of_week = st.selectbox("Select Day of the Week: ", options=options_day)
        Light_conditions = st.selectbox("Select Light Condition: ", options=options_light)
        Driving_experience = st.selectbox("Select Driver Experience: ", options=options_drv_exp)
        Types_of_junction = st.selectbox("Select type of vehicle : ", options=options_junction_typ)
        Sex_of_casualty = st.selectbox("Select type of vehicle : ", options=options_casualty_sex)
        Casualty_class = st.selectbox("Select class of casualty : ", options=options_casualty_class)
        Area_accident_occured = st.selectbox("Select Accident Area: ", options=options_acc_area)
        Lanes_or_Medians = st.selectbox("Select Lane or Median type: ", options=options_lane)

        submit = st.form_submit_button("Predict")


    if submit:
        Age_band_of_driver = ordinal_encoder(Age_band_of_driver, options_age)
        Day_of_week = ordinal_encoder(Day_of_week, options_day)
        Light_conditions = ordinal_encoder(Light_conditions, options_light)
        Driving_experience = ordinal_encoder(Driving_experience, options_drv_exp)
        Types_of_junction = ordinal_encoder(Types_of_junction, options_junction_typ)
        Sex_of_casualty = ordinal_encoder(Sex_of_casualty, options_casualty_sex)
        Casualty_class = ordinal_encoder(Casualty_class, options_casualty_class)
        Area_accident_occured = ordinal_encoder(Area_accident_occured, options_acc_area)
        Lanes_or_Medians = ordinal_encoder(Lanes_or_Medians, options_lane)


        data = np.array([Number_of_vehicles_involved,Number_of_casualties,Age_band_of_driver,
              Day_of_week,Light_conditions,Driving_experience,Types_of_junction,Sex_of_casualty,
              Casualty_class,Area_accident_occured,Lanes_or_Medians]).reshape(1,-1)

        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred}")

if __name__ == '__main__':
    main()