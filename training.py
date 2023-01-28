import boto3
import sagemaker 

session = sagemaker.Session()
region = boto3.Session().region_name
role = sagemaker.get_execution_role()

#Retrieving BlazingText image container
image_uri = sagemaker.image_uris.retrieve(region=region,framework='blazingtext')

#estimator for the training job and hyperparameter setting
estimator = sagemaker.estimator.Estimator(
image_uri=image_uri, 
    role=role, 
    instance_count=1, 
    instance_type='ml.m5.large',
    volume_size=5,
    max_run=1200,
    output_path=session.default_bucket()
    sagemaker_session=session)

estimator.set_hyperparameters(mode='supervised',   
                              epochs=20,min_epochs=5
                              learning_rate=0.05  
                              min_count=5,                          
                              vector_dim=200, sampling_threshold=0.0001
                              word_ngrams=2,batch_size=11)

#creating the training and validation channels
train_data = sagemaker.inputs.TrainingInput(
    train_s3_uri, 
    distribution='FullyReplicated', 
   s3_data_type='S3Prefix')

validation_data = sagemaker.inputs.TrainingInput(
    validation_s3_uri, 
    distribution='FullyReplicated', 
   s3_data_type='S3Prefix')

# Organize the data channels defined above as a dictionary
data_channels = {
    'train': train_data, # Replace None
    'validation': validation_data # Replace None
}

#fitting the model to the dataset
estimator.fit(input=data_channels, job_name='BlazingText_SentimentReview')

