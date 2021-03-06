from preprocess import clean_text
from custom_loss import weighted_sparse_categorical_cross_entropy
from model import create_model

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
vocabulary_size = 100 # Choosing size of vocabulary
tokenizer = Tokenizer(num_words=vocabulary_size)
# Pads sequences to the same length: MAXLEN
MAXLEN = 10

from datetime import datetime
from packaging import version


from sklearn.utils import class_weight
import numpy as np
import pandas as pd
import click
import yaml

@click.command()
@click.argument('config_file', type=str, default="config.yml")
def cli(config_file):
    print(config_file)
    config_path = config_file
    print("config_path"+ config_path)
    config = parse_config(config_path)
    print("Path to Dataset"+config["dataset"]["data_path"])
    data_path = config["dataset"]["data_path"]
    model_path = config["model"]["model_path"]
    print(data_path)
    print(model_path)
    
    train_df = pd.read_csv(data_path+"train_df.csv")
    valid_df = pd.read_csv(data_path+"valid_df.csv")

    total = pd.concat([train_df,valid_df])
    total['transcription'] = total['transcription'].map(lambda x: clean_text(x))

    total['action'] = pd.Categorical(total['action'])
    total['object'] = pd.Categorical(total['object'])
    total['location'] = pd.Categorical(total['location'])

    total['action_codes'] = total['action'].cat.codes
    total['object_codes'] = total['object'].cat.codes
    total['location_codes'] = total['location'].cat.codes

    df_train, df_valid = total.iloc[:11566], total.iloc[11566:]
    
    print("Got Training and Validation Data...")

    #Computing class weights to penalize more frequent classes
    object_class_weight = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(df_train['object_codes']), y = df_train['object_codes'])
    object_class_weight = dict(zip(np.unique(df_train['object_codes']), object_class_weight))
    #object_class_weight

    action_class_weight = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(df_train['action_codes']), y = df_train['action_codes'])
    action_class_weight = dict(zip(np.unique(df_train['action_codes']), action_class_weight))
    #action_class_weight

    location_class_weight = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(df_train['location_codes']), y = df_train['location_codes'])
    location_class_weight = dict(zip(np.unique(df_train['location_codes']), location_class_weight))
    #location_class_weight

    print("Computed Class weights...")
    tokenizer.fit_on_texts(total['transcription'])

    train_sequences = tokenizer.texts_to_sequences(df_train['transcription'])
    X_train = pad_sequences(train_sequences, maxlen=MAXLEN)

    valid_sequences = tokenizer.texts_to_sequences(df_valid['transcription'])
    X_valid = pad_sequences(valid_sequences, maxlen=MAXLEN)

    y_train = df_train[['action_codes','object_codes','location_codes']]
    y_valid = df_valid[['action_codes','object_codes','location_codes']]

    y_train_output = []
    y_train_output.append(y_train['action_codes'])
    y_train_output.append(y_train['object_codes'])
    y_train_output.append(y_train['location_codes'])

    y_valid_output = []
    y_valid_output.append(y_valid['action_codes'])
    y_valid_output.append(y_valid['object_codes'])
    y_valid_output.append(y_valid['location_codes'])
    print("Ready to start training")

    #Compiling the loss functions for each of the computed weights
    loss1 = weighted_sparse_categorical_cross_entropy(list(action_class_weight.values()))
    loss2 = weighted_sparse_categorical_cross_entropy(list(object_class_weight.values()))
    loss3 = weighted_sparse_categorical_cross_entropy(list(location_class_weight.values()))
    print("Compiled custom losses")
    print(loss1, loss2, loss3)
    
    print("Starting to train model...")
    train_model(X_train,y_train_output,X_valid,y_valid_output,loss1, loss2, loss3, config_path, model_path)

def parse_config(config_file):
    print("Inside parse_config")
    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config

def train_model(X_train,y_train_output,X_valid,y_valid_output, loss1, loss2, loss3,config_path, model_path):
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    with strategy.scope():
      model = create_model()
      model.compile(optimizer='adam',
                  loss={'action_output':loss1, 'object_output':loss2, 'location_output':loss3}, loss_weights=[2,0.75,0.75],
                  metrics={'action_output':'accuracy', 'object_output':'accuracy', 'location_output':'accuracy'})


    logdir = config_path + "logs\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(X_train, y_train_output,epochs=20, batch_size=128, callbacks=[tensorboard_callback], validation_data = (X_valid, y_valid_output), verbose=1);
    model.save(model_path)
    

if __name__ == "__main__":
    cli()   