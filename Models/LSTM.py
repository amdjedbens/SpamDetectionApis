from json import load
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM
import numpy as np
import sys, os

# from keras import model_from_json 
from keras.models import model_from_json 
# import keras.models.model_fro
try:
    
    sys.path.append(os.path.join(os.path.dirname(__file__), '..',"Helpers"))
    from Preprocessing import PreProcess
except:
    raise Exception("Server Error")
  
from sklearn.metrics import classification_report, accuracy_score

# import Preprocessing

class LSTM_Model:
    def __init__(self) -> None:
        train_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'DataSets','spam_ham_dataset.csv'))
        Y = train_dataset['label_num']
        X = train_dataset['text']   

        try:
            #Trying to load the model from the disk

            json_file = open(os.path.join(os.path.dirname(__file__), '..', 'Stored_Models','model.json'),'r')
            loaded_model_json = json_file.read()
            json_file.close()


            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'Stored_Models','model.h5'))


            loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
            self.model1 = loaded_model

            print("Loaded Model from disk")
            
        except:
            print("Loading failed, Training the model over")
            early_stop = EarlyStopping(monitor='val_loss', patience=3)
            #LSTM Spam detection architecture
            self.x_train,self.y_train,vocab_size,word_index = PreProcess(X,Y)
            self.MAX_SEQUENCE_LENGTH = 50
            n_lstm = 200
            drop_lstm = 0.2
            embeding_dim = 16
            drop_value = 0.2
            n_dense = 24
            num_epochs = 7
        
            self.model1 = Sequential()
            self.model1.add(Embedding(vocab_size, embeding_dim, input_length=self.MAX_SEQUENCE_LENGTH))
            self.model1.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
            self.model1.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
            self.model1.add(Dropout(0.1))
            self.model1.add(Dense(1, activation='sigmoid'))
            self.model1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

            
            self.model1.fit(self.x_train, self.y_train, epochs=5, verbose=2)
            early_stop = EarlyStopping(monitor='accuracy', patience=2)
            model_json = self.model1.to_json()
            with open(os.path.join(os.path.dirname(__file__), '..', 'Stored_Models','model.json'), "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.model1.save_weights(os.path.join(os.path.dirname(__file__), '..', 'Stored_Models','model.h5'))
            # pickle.dump(self.model1,open(r"C:\Users\assas\OneDrive\Desktop\Apis\APis\Models\LstmModel.pkl","wb"))
            pass
        

        
    def Predict(self,X):
        X = PreProcess(pd.Series([X]))[0]
        print(X.shape)
        try:
            y_lstm_predict = self.model1.predict(X)
            print(y_lstm_predict)
            y_lstm = []
            for i in y_lstm_predict:
                y_lstm.append(list(i).count([True])/len(i))
            y_lstm = sum(y_lstm)/len(y_lstm)
        except:
            raise Exception("Error while predicting the email")
            
        return y_lstm
            
        

