import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
vocabulary_size = 100 # Choosing size of vocabulary
tokenizer = Tokenizer(num_words=vocabulary_size)
# Pads sequences to the same length: MAXLEN
MAXLEN = 10

def create_model():
  model = keras.Model()
  #creating inputs
  main_input = layers.Input(shape=(MAXLEN,), dtype='int32', name='main_input')
  x = layers.Embedding(input_dim=vocabulary_size, output_dim=50, input_length=MAXLEN)(main_input)
  x = layers.Dropout(0.3)(x)
  x = layers.Conv1D(128, 5, activation='relu')(x)
  x = layers.Conv1D(128, 5, activation='relu')(x)
  x = layers.MaxPooling1D(2, padding="same")(x)
  x = layers.LSTM(128, activation='tanh')(x)
  x = layers.Dropout(0.3)(x)

  #creating outputs
  output_array = [] 
  metrics_array = {}
  loss_array = {}

  action_output = layers.Dense(6, activation='softmax', name='action_output')(x)
  output_array.append(action_output)

  object_output = layers.Dense(14, activation='softmax', name='object_output')(x)
  output_array.append(object_output)

  location_output = layers.Dense(4, activation='softmax', name='location_output')(x)
  output_array.append(location_output)

  model = keras.Model(inputs=main_input, outputs=output_array)
  return(model)