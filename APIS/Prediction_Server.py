import email
from flask import Flask, request, jsonify
from flask_restful import Api, Resource,reqparse
from sympy import arg

import sys, os
try:
    
    sys.path.append(os.path.join(os.path.dirname(__file__), '..',"Models"))
    from LSTM import LSTM_Model
except:
    raise Exception("Server Error")
  

app = Flask(__name__)
api = Api(app)
lstm = LSTM_Model()



email_put_args = reqparse.RequestParser()
email_put_args.add_argument("header",type=str,help="Error no Header found")
email_put_args.add_argument("content",type=str,help="Error no Content found")

# class Predict(Resource):
#     def get(self):
#         return {"data":"hello there"}
#     def post(self):
#         try:
#             args = email_put_args.parse_args()
#             content = (args["content"])
#             resp = lstm.Predict(content)
#         except:
#             return {"Error","Server Error"}
#         return {"type":str(resp)}


# api.add_resource(Predict,"/predict")


@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    try:
        print(data["content"],data)
        prediction = lstm.Predict(data["content"])
        output = prediction

    except:
        output = {"error":"Server Error"}
    # Take the first value of prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(host="localhost",port=8085,debug=True)
    # pass