import os
import boto3
import sagemaker
import json
import zipfile

#Bucket path containing data
BUCKET = "ml-workflow-training-job"
s3_prefix = "ToysAndGames"
file_name = "Toys_and_Games_5.json.zip"

source_path = 's3://' + "/".join([s3_bucket, s3_prefix, file_name])


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


def write_data(data, train_path, test_path, proportion):
    border_index = int(proportion * len(data))
    train_f = open(train_path, 'w')
    test_f = open(test_path, 'w')
    index = 0
    for d in data:
        if index < border_index:
            train_f.write(d + '\n')
        else:
            test_f.write(d + '\n')
        index += 1


def upload_file_to_s3(file_name, s3_prefix):
    object_name = os.path.join(s3_prefix, file_name)
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, BUCKET, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    
    
if __name__ == "__main__":
    unzipped_path = unzip_data('/opt/ml/processing/input/Toys_and_Games_5.json.zip')
    labeled_data = label_data(unzipped_path)
    new_split_sentence_data = split_sentences(labeled_data)
    write_data(new_split_sentence_data, '/opt/ml/processing/output/train/hello_blaze_train_scikit', '/opt/ml/processing/output/test/hello_blaze_test_scikit', .9)
    upload_file_to_s3(file_name)
    print(source_path)

