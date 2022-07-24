"""
Creators: Pedro Victor and Beatriz Soares
Date: 23 July 2022
Create API
"""
# from typing import Union
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi import FastAPI
import pandas as pd
import joblib
import os
import wandb
import tensorflow
import sys
from source.api.pipeline import FeatureSelector, CategoricalTransformer

# global variables
setattr(sys.modules["__main__"], "FeatureSelector", FeatureSelector)
setattr(sys.modules["__main__"], "CategoricalTransformer", CategoricalTransformer)

# name of the model artifact
artifact_model_name = "neural_network1/model_export:latest"

# initiate the wandb project
run = wandb.init(project="neural_network1",job_type="api")

# create the api
app = FastAPI()

# declare request example data using pydantic
# a person in our dataset has the following attributes
class Driver(BaseModel):
    Age_band_of_driver: str
    Sex_of_driver: str
    Educational_level: str
    Vehicle_driver_relation: str
    Driving_experience: str
    Lanes_or_Medians: str
    Types_of_Junction: str
    Road_surface_type: str
    Light_conditions: str
    Weather_conditions: str
    Type_of_collision: str
    Vehicle_movement: str
    Pedestrian_movement: str
    Cause_of_accident: str

    class Config:
        schema_extra = {
            "example": {
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
        }

# give a greeting using GET
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <p><span style="font-size:28px"><strong>Hello World</strong></span></p>"""\
    """<p><span style="font-size:20px">In this project, we will apply the skills """\
        """acquired in the Deploying a Scalable ML Pipeline in Production course to develop """\
        """a classification model on publicly available"""\
        """<a href="http://archive.ics.uci.edu/ml/datasets/Adult"> Census Bureau data</a>.</span></p>"""

# run the model inference and use a Person data structure via POST to the API.
@app.post("/predict")
async def get_inference(driver: Driver):
    
    # Download inference artifact
    model_export_path = run.use_artifact(artifact_model_name).file()
    pipe = joblib.load(model_export_path)
    
    #load best model
    best_model = wandb.restore('model-best.h5', run_path="pedro_victor046/neural_network1/zmketyas")
    model = tensorflow.keras.models.load_model(best_model.name)
    
    # Create a dataframe from the input feature
    # note that we could use pd.DataFrame.from_dict
    # but due be only one instance, it would be necessary to
    # pass the Index.
    df = pd.DataFrame([driver.dict()])
    
    #pipeline to transform
    data = pipe.transform(df)

    # Predict test data
    predict = model.predict(data)

    if predict[0] == 2:
        return "Accident_severity: 2" 
    elif predict[0] == 1:
        return "Accident_severity: 1" 
    elif predict[0] == 0:
        return "Accident_severity: 0"
    else:
        return "Incorrect result"
    
