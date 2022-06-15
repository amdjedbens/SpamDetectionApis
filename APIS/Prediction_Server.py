import email
from flask import Flask, request, jsonify
from flask_restful import Api, Resource,reqparse
from sympy import arg

import sys, os
try:
    
    sys.path.append(os.path.join(os.path.dirname(__file__), '..',"Models"))
    from LSTM import LSTM_Model
    # from BERT import BERT_Model
except:
    raise Exception("Server Error")
  

app = Flask(__name__)
api = Api(app)

# bert = BERT_Model()

lstm = LSTM_Model()

email_put_args = reqparse.RequestParser()
email_put_args.add_argument("header",type=str,help="Error no Header found")
email_put_args.add_argument("content",type=str,help="Error no Content found")


@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    try:
        #print(data["content"],data)
        prediction = lstm.Predict(data["content"])
        

        output = prediction

    except:
        output = {"error":"Server Error"}
    # Take the first value of prediction
    return jsonify(output)
@app.route('/api',methods=['GET'])
def hello():
    return({"res":"hello"})
if __name__ == "__main__":
    app.run(host="localhost",port=8085,debug=True)
    # pass
