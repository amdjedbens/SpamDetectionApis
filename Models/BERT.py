from json import load
import pickle
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM
import numpy as np
import sys, os
from sklearn import preprocessing
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow as tf
# from keras import model_from_json 
from keras.models import model_from_json 
# import keras.models.model_fro

from sklearn.metrics import classification_report, accuracy_score

# import Preprocessing

class BERT_Model:
    def __init__(self) -> None:
        train_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'DataSets','spam_ham_dataset.csv'))
        train_dataset.drop({'Unnamed: 0'},axis=1,inplace=True)
        train_dataset.rename(columns={"text":"body"}, inplace=True)
        train_dataset.drop({'label'}, axis = 1, inplace=True)
        label_encoder = preprocessing.LabelEncoder()

        # Encode labels in column 'species'.
        train_dataset['label_num']= label_encoder.fit_transform(train_dataset['label_num'])

        train_dataset['label_num'].unique()
        Y = train_dataset['label_num']
        X = train_dataset['body']   

        

        try:
            #Trying to load the model from the disk

            json_file = open(os.path.join(os.path.dirname(__file__), '..', 'Stored_Models','bert_model.json'),'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'Stored_Models','bert_model.h5'))
            METRICS = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
            

            self.model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=METRICS)
            
            self.model = loaded_model
            print("Loaded Model from disk")
            
        except:
            print("Loading failed, Training the model over")
            bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",trainable=True)
            bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
            # Bert Layer
            text_input = tf.keras.layers.Input(shape = (),dtype = tf.string, name = "text")

            preprocessed_text = bert_preprocess(text_input)

            outputs= bert_encoder(preprocessed_text)
            # Neural Network layer
            l = tf.keras.layers.Dropout(0.1,name = 'dropout')(outputs['pooled_output'])
            l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

            # Use inputs and outputs to construct a final model
            self.model = tf.keras.Model(inputs=[text_input], outputs = [l])
            METRICS = [
                tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
            

            self.model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=METRICS)
            self.model.fit(X, Y, epochs = 2)
           
            model_json = self.model.to_json()
            with open(os.path.join(os.path.dirname(__file__), '..', 'Stored_Models','model.json'), "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights(os.path.join(os.path.dirname(__file__), '..', 'Stored_Models','model.h5'))
            # pickle.dump(self.model1,open(r"C:\Users\assas\OneDrive\Desktop\Apis\APis\Models\LstmModel.pkl","wb"))
            pass
        

        
    def Predict(self,X):
        X = pd.Series([X])

        try:
            y_lstm_predict = self.model.predict(X)
            y_predicted = list(np.where(y_lstm_predict > 0.5, True, False))
            y_lstm = y_predicted.count(True)/len(y_predicted)
        except:
            raise Exception("Error while predicting the email")
            
        return y_lstm
            
        

