import boto3
import sklearn
from sagemaker import get_execution_role
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

#role
role = get_execution_role()

#creating sklearn processor
sklearn_processor = SKLearnProcessor(framework_version='0.20.0',
                                     instance_type='ml.m5.large',
                                     instance_count=1,
                                     role=role)


if name == __main__:
sklearn_processor.run(code='processing.py', # preprocessing code
                      inputs=[ProcessingInput(
                          source = 's3://ml-workflow-training-job/ToysandGames/Toys_and_Games_5.json.zip', # the S3 path of the unprocessed data
                          destination='/opt/ml/processing/input', # a 'local' directory path for the input to be downloaded into
                      )],
                      outputs=[ProcessingOutput(source='/opt/ml/processing/output/train/' ),# a 'local' directory path for where you expect the output for train data to be
                               ProcessingOutput(source='/opt/ml/processing/output/test/' )]) # a 'local' directory path for where you expect the output for test data to be 

sklearn_processor.jobs[-1].describe()
