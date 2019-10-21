# [START setup]
import datetime
import pandas as pd

from google.cloud import storage

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer

# TODO: REPLACE 'YOUR_BUCKET_NAME' with your GCS Bucket name.
BUCKET_NAME = 'ml-tuto-fred'
# [END setup]

# ---------------------------------------
# 1. Add code to download the data from GCS (in this case, using the publicly hosted data).
# AI Platform will then be able to use the data when training your model.
# ---------------------------------------
# [START download-data]
# Public bucket holding the data
bucket = storage.Client().bucket('ml-tuto-fred')

# Path to the data inside the bucket
blob = bucket.blob('data/loan_approvals.csv')
# Download the data
blob.download_to_filename('loan_approvals.csv')
# [END download-data]

BUCKET_NAME="ml-tuto-fred"
PROJECT_ID="fmolina-is"
REGION="europe-west1"

#TODO DL data here from blob to pandas DF
with open('./loan_approvals.csv', 'r') as train_data:
    data = pd.read_csv(train_data)

# ---------------------------------------
# This is where your model code would go. Below is an example model using the dataset.
# ---------------------------------------
# [START define-and-load-data]


# prepare the data by Droping empty lines, encoding the textual values as numerical, build training dataset and the validation one
data.dropna(inplace=True)

#this Gradient Boosting Model only accepts numerical values as input
# here is how we replace the textual values by numerical values (also see one hot encoding method :: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html )
data['Gender'] = data['Gender'].replace({'Male': 0, 'Female': 1})
data['Married'] = data['Married'].replace({'No' : 0,'Yes' : 1})
data['Dependents'] = data['Dependents'].replace({'0':0,'1':1,'2':2,'3+':3})
data['Education'] = data['Education'].replace({'Not Graduate' : 0, 'Graduate' : 1})
data['Self_Employed'] = data['Self_Employed'].replace({'No' : 0,'Yes' : 1})
data['Property_Area'] = data['Property_Area'].replace({'Semiurban' : 0, 'Urban' : 1,'Rural' : 2})

#the prepared data compatible with ML model training is in data variable

#the Training dataset is the prepared data without the Loan_ID, because we dont want our model to learn a pattern based on LoanID value as its not the case in real life loan approval only depends on the data relatives to loan application
# the training dataset dont contain the Loan_Status column as it's the column we want to predict
trainning_data = data.drop(columns=['Loan_ID','Loan_Status'])

#The target variable to try to predict in the case of supervised learning: here, if the loan has been approved or not
target_prediction_data = data['Loan_Status']


# Model creation, checking and training
#Import the scikit learn libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

#model creation : prepare the model handler: model_gbc
model_gbc = GradientBoostingClassifier()

#model training: execute the model training using the GradientBoostingClassifier.fit method
model_gbc.fit(trainning_data, target_prediction_data)
model = 'model.joblib'
joblib.dump(model_gbc, model)

# [START export-to-gcs]
# Upload the model to GCS
bucket = storage.Client().bucket(BUCKET_NAME)
blob = bucket.blob('{}/{}'.format(
    datetime.datetime.now().strftime('loanApproval_%Y%m%d_%H%M%S'),
    model))
blob.upload_from_filename(model)
# [END export-to-gcs]
