from flask import Flask, request, jsonify
import pickle
import numpy as np
# from flask_cors import CORS
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

# import model
# model = pickle.load(open('/home/patty/THESIS/projects/IA/final/model.pkl', 'rb'))
model = pickle.load(open('/home/patty/THESIS/projects/IA/ml-ultracasas/4-v2-ULTRACASAS/model.pkl', 'rb'))
modeld = pickle.load(open('/home/patty/THESIS/projects/IA/ml-ultracasas/4-v2-ULTRACASAS/modeld.pkl', 'rb'))



def risk(r):
    risk = {'alto':[1,0,0,0], 'bajo':[0,1,0,0], 'moderado':[0,0,1,0], 'muy_bajo':[0,0,0,1]}
    if r in risk:
        return risk[r]
    return [0,0,0,0]

@app.route('/api/v1/price',methods=['POST'])
def predict():
    # get the data FrOm the POST REQUEST
    data = request.get_json()
    # making prediction using the model form disk as per data
    # prediction = model.predict([[np.array(data["exp"])]])


    # take the first value of prediction
    # result = prediction[-1]
    # features = np.array([data["amoblado"],data["bathroom"],0,data["bedroom"],data["dimension"],data["garage"],-16.53,-68.04,data["pool"],data["status"],0])
    # prediction = model.predict([np.array([0,2,0,4,297,2,-16.53,-68.04,0,2,2000])])
    # prediction = model.predict([np.array([int(data["amoblado"]),int(data["bathroom"]),0,int(data["bedroom"]),int(data["dimension"]),int(data["garage"]),-16.53,-68.04,int(data["pool"]),int(data["status"]),2000])])

    if data["type_offer"] == 'house':
        # amoblado	bathroom	baulera	bedroom	dimension_built	dimension_ground	garage	latitud	longitud	status	year_built	riesgo__alto	riesgo__bajo	riesgo__moderado	riesgo__muy bajo	neighborhood_encoded
        pp = np.array([
            int(data["amoblado"]),
            int(data["bathroom"]),
            int(data["baulera"]),
            int(data["bedroom"]),
            int(data["dimension_built"]),
            int(data["dimension_ground"]),
            int(data["garage"]),
            -16.53,
            -68.04,
            int(data["status"]),
            int(data["year_built"]),
        ])
        pp = np.concatenate([pp, risk(data['riesgo']), np.array([data["neighborhood_encoded"]])])

        prediction = model.predict([pp])
    else:
        # amoblado	bathroom	baulera	bedroom	dimension_built	dimension_ground	elevator	garage	latitud	longitud	status	year_built	riesgo__alto	riesgo__bajo	riesgo__moderado	riesgo__muy bajo	neighborhood_encoded
        pp = np.array([
            int(data["amoblado"]),
            int(data["bathroom"]),
            int(data["baulera"]),
            int(data["bedroom"]),
            int(data["dimension_built"]),
            int(data["dimension_ground"]),
            int(data["elevator"]),
            int(data["garage"]),
            -16.53,
            -68.04,
            int(data["status"]),
            int(data["year_built"]),
        ])
        pp = np.concatenate([pp, risk(data['riesgo']), np.array([data["neighborhood_encoded"]])])

        prediction = modeld.predict([pp])

    # amoblado	bathroom	baulera	bedroom	dimension	garage	latitud	longitud	status	year_built	riesgo__alto	riesgo__bajo	riesgo__moderado	riesgo__muy bajo


    return jsonify(prediction[0])

    # return pp

if __name__ == '__main__':
    app.run(port=5000, debug=True)



# !!!!!! TODO PARSEAR LOS NROS DE LOS FEATURES DEL FORM