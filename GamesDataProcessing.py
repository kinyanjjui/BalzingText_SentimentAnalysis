import os
import boto3
import sagemaker
import json
import zipfile

import pandas as pd
import numpy as np

def label_data(input_data):
    labeled_data = []
    HELPFUL_LABEL = "__label__1"
    UNHELPFUL_LABEL = "__label__2"
     
    for l in open(input_data, 'r'):
        l_object = json.loads(l)
        helpful_votes = float(l_object['helpful'][0])
        total_votes = l_object['helpful'][1]
        reviewText = l_object['reviewText']
        if total_votes != 0:
            if helpful_votes / total_votes > .5:
                labeled_data.append(" ".join([HELPFUL_LABEL, reviewText]))
            elif helpful_votes / total_votes < .5:
                labeled_data.append(" ".join([UNHELPFUL_LABEL, reviewText]))
          
    return labeled_data


# Labeled data is a list of sentences, starting with the label defined in label_data. 

def split_sentences(labeled_data):
    split_sentences = []
    for d in labeled_data:
        label = d.split()[0]        
        sentences = " ".join(d.split()[1:]).split(".") # Initially split to separate label, then separate sentences
        for s in sentences:
            if s: # Make sure sentences isn't empty. Common w/ "..."
                split_sentences.append(" ".join([label, s]))
    return split_sentences


input_data  = unzip_data('Toys_and_Games_5.json.zip')
labeled_data = label_data('Toys_and_Games_5.json')
split_sentence_data = split_sentences(labeled_data)

#print(split_sentence_data[0:9])

import boto3
from botocore.exceptions import ClientError
# Note: This section implies that the bucket below has already been made and that you have access
# to that bucket. You would need to change the bucket below to a bucket that you have write
# premissions to. This will take time depending on your internet connection, the training file is ~ 40 mb

BUCKET = "ml-workflow-training-job"
s3_prefix = "ToysAndGames"


def cycle_data(fp, data):
    for d in data:
        fp.write(d + "\n")

def write_trainfile(split_sentence_data):
    train_path = "hello_blaze_train"
    with open(train_path, 'w') as f:
        cycle_data(f, split_sentence_data)
    return train_path

def write_validationfile(split_sentence_data):
    validation_path = "hello_blaze_validation"
    with open(validation_path, 'w') as f:
        cycle_data(f, split_sentence_data)
    return validation_path 

def upload_file_to_s3(file_name, s3_prefix):
    object_name = os.path.join(s3_prefix, file_name)
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, BUCKET, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    
# Split the data
split_data_trainlen = int(len(split_sentence_data) * .9)
split_data_validationlen = int(len(split_sentence_data) * .1)

# Todo: write the training file
train_path = write_trainfile(split_sentence_data[:split_data_trainlen])
print("Training file written!")

# Todo: write the validation file
validation_path = write_validationfile(split_sentence_data[split_data_trainlen:])
print("Validation file written!")

upload_file_to_s3(train_path, s3_prefix)
print("Train file uploaded!")
upload_file_to_s3(validation_path, s3_prefix)
print("Validation file uploaded!")

print(" ".join([train_path, validation_path]))
