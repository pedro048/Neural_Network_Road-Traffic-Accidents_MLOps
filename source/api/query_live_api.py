"""
Creators: Pedro Victor and Beatriz Soares
Date: 23 July 2022
Script that POSTS to the API using the requests 
module and returns both the result of 
model inference and the status code
"""
import requests
import json
# import pprint

driver = {
    "Age_band_of_driver": '18-30',
    "Sex_of_driver": 'Female',
    "Educational_level": 'High school',
    "Vehicle_driver_relation": 'Employee',
    "Driving_experience":  '2-5yr',
    "Lanes_or_Medians": 'One way',
    "Types_of_Junction": 'Y Shape',
    "Road_surface_type": 'Asphalt roads',
    "Light_conditions": 'Daylight',
    "Weather_conditions": 'Normal',
    "Type_of_collision": 'Vehicle with vehicle collision',
    "Vehicle_movement": 'Going straight',
    "Pedestrian_movement": 'Not a Pedestrian',
    "Cause_of_accident": 'No priority to vehicle'
    }

#url = "http://127.0.0.1:8000"
url = "https://nn-road-traffic-accidents-app.herokuapp.com"
response = requests.post(f"{url}/predict",
                         json=driver)

print(f"Request: {url}/predict")
print(f"Person: \n Age_band_of_driver: {driver['Age_band_of_driver']}\n Sex_of_driver: {driver['Sex_of_driver']}\n"\
      f" Educational_level: {driver['Educational_level']}\n"\
      f" Vehicle_driver_relation: {driver['Vehicle_driver_relation']}\n"\
      f" Driving_experience: {driver['Driving_experience']}\n"\
      f" Lanes_or_Medians: {driver['Lanes_or_Medians']}\n"\
      f" Types_of_Junction: {driver['Types_of_Junction']}\n"\
      f" Road_surface_type: {driver['Road_surface_type']}\n"\
      f" Light_conditions: {driver['Light_conditions']}\n"\
      f" Weather_conditions: {driver['Weather_conditions']}\n"\
      f" Type_of_collision: {driver['Type_of_collision']}\n"\
      f" Vehicle_movement: {driver['Vehicle_movement']}\n"\
      f" Pedestrian_movement: {driver['Pedestrian_movement']}\n"
      f" Cause_of_accident: {driver['Cause_of_accident']}\n"
     )
print(f"Result of model inference: {response.json()}")
print(f"Status code: {response.status_code}")
