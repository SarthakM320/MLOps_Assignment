import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

import pycaret.classification as pycr

from dotenv import load_dotenv
import pandas as pd
import os

# Load the environment variables from the .env file into the application
load_dotenv()

# Initialize the FastAPI application
app = FastAPI()

# Create a class to store the deployed model & use it for prediction
class Model:
    def __init__(self, modelname, bucketname):
        """
        Function to initalize the model
        modelname: Name of the model stored in the S3 bucket
        bucketname: Name of the S3 bucket
        """
        # Load the deployed model from Amazon S3
        self.model = pycr.load_model(modelname, platform = 'aws', authentication = { 'bucket' : bucketname })

    def predict(self, data):
        """
        Function to use the loaded model to make predictions on the data
        data: Pandas DataFrame to perform predictions
        """
        # After predicting, we return only the column containing the predictions (i.e. 'Label') after converting it to a list

        predictions = pycr.predict_model(self.model, data = data).Label.to_list()

        return predictions
    

model1 = Model('LDA_deployed', 'mlopsassignment2')
model2 = Model('KNC_deployed', 'mlopsassignment2')


# def model_deploy(model_name):
#     async def create_upload_file(file: UploadFile = File(...)):
#     # Handle the file only if it is a CSV
#         if file.filename.endswith('.csv'):
#             with open(file.filename, 'wb') as f:
#                 f.write(file.file.read())
#             data = pd.read_csv(file.filename)

#             # Return a JSON object containing the model predictions on the data
#             return {
#                 "Labels": model_name.predict(data)
#             }

#         else:
#             # Raise a HTTP 400 Exception, indicating Bad Request (you can learn more about HTTP response status codes here)
#             raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")
#     return create_upload_file()


# Create the POST endpoint with path '/predict'
@app.post("LDA/predict")
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
        if file.filename.endswith('.csv'):
            with open(file.filename, 'wb') as f:
                f.write(file.file.read())
            data = pd.read_csv(file.filename)

            # Return a JSON object containing the model predictions on the data
            return {
                "Labels": model1.predict(data)
            }

        else:
            # Raise a HTTP 400 Exception, indicating Bad Request (you can learn more about HTTP response status codes here)
            raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")
# To understand how to handle file uploads in FastAPI, visit the documentation here

@app.post("/KNC/predict")
async def create_upload_file(file: UploadFile = File(...)):
    # Handle the file only if it is a CSV
        if file.filename.endswith('.csv'):
            with open(file.filename, 'wb') as f:
                f.write(file.file.read())
            data = pd.read_csv(file.filename)

            # Return a JSON object containing the model predictions on the data
            return {
                "Labels": model2.predict(data)
            }

        else:
            # Raise a HTTP 400 Exception, indicating Bad Request (you can learn more about HTTP response status codes here)
            raise HTTPException(status_code=400, detail="Invalid file format. Only CSV Files accepted.")
# Check if the necessary environment variables for AWS access are available. If not, exit the program
if os.getenv("AWS_ACCESS_KEY_ID") == None or os.getenv("AWS_SECRET_ACCESS_KEY") == None:
    print("AWS Credentials missing. Please set required environment variables.")
    exit(1)
