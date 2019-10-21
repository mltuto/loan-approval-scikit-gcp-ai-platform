PROJECT_ID=[YOUR-PROJECT-ID]
BUCKET_ID=[YOUR-BUCKET-NAME]
JOB_NAME=loanapp_training_$(date +"%Y%m%d_%H%M%S")
JOB_DIR=gs://$BUCKET_ID/scikit_learn_job_dir
TRAINING_PACKAGE_PATH="[YOUR_LOCAL_PATH]"
MAIN_TRAINER_MODULE=loanapp.train
REGION=europe-west1
RUNTIME_VERSION=1.14
PYTHON_VERSION=2.7
SCALE_TIER=BASIC

puis : 
gcloud ai-platform jobs submit training $JOB_NAME \
  --job-dir $JOB_DIR \
  --package-path $TRAINING_PACKAGE_PATH \
  --module-name $MAIN_TRAINER_MODULE \
  --region $REGION \
  --runtime-version=$RUNTIME_VERSION \
  --python-version=$PYTHON_VERSION \
  --scale-tier $SCALE_TIER
