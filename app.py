from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
import pyrebase


config = {
  "apiKey": "AIzaSyB1407Jf6sosGEFnO-1h6rCKwWpzbW1REQ",
  "authDomain": "farmgrid-67b05.firebaseapp.com",
  "databaseURL": "https://farmgrid-67b05-default-rtdb.firebaseio.com",
  "projectId": "farmgrid-67b05",
  "storageBucket": "farmgrid-67b05.appspot.com",
  "messagingSenderId": "602553499578",
  "appId": "1:602553499578:web:644ee05fa1be2fddb5e783",
  "measurementId": "G-FF6T2X548B"
}


"""

config = {
    "apiKey": "AIzaSyB_k9SPujVHdm-8EHwuy9OQU1AEAhBA4Ro",
    "authDomain": "irisscan-94a4e.firebaseapp.com",
    "databaseURL": "https://irisscan-94a4e-default-rtdb.firebaseio.com",
    "projectId": "irisscan-94a4e",
    "storageBucket": "irisscan-94a4e.appspot.com",
    "messagingSenderId": "157605262331",
    "appId": "1:157605262331:web:9f18539654fca271a7515e",
    "measurementId": "G-BW0H8V7E2D"
}
"""
firebase = pyrebase.initialize_app(config)

storage = firebase.storage()

db = firebase.database()

auth=firebase.auth()

try:
    login = auth.sign_in_with_email_and_password("farmgrid2021@gmail.com", "shambo1234%")
except:
    print("hello")


# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

#---------------------------------------------------------------------------------------------------------------------------------------------------------

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(img)
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'FarmGrid - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page


@ app.route('/crop-recommend')
def crop_recommend():
    title = 'FarmGrid - Crop Recommendation'
    return render_template('crop.html', title=title)

# render fertilizer recommendation form page


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'FarmGrid - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)

# render disease prediction input page

@ app.route('/disease')
def disease_upload():
    title = 'FarmGrid - Diesease Prediction'

    return render_template('disease.html', title=title)




# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'FarmGrid - Crop Recommendation'

    #if request.method == 'POST':
    DN = str(request.form['nitrogen'])

    rainfall = float(request.form['rainfall'])

    X1 = db.child("LandParameters/"+ DN + "/N").get().val()
    N = int(X1)
    X2 = db.child("LandParameters/"+ DN + "/P").get().val()
    P = int(X2)
    X3 = db.child("LandParameters/"+ DN + "/K").get().val()
    K = int(X3)
    X4 = db.child("LandParameters/"+ DN + "/ph").get().val()
    ph = float(X4)
    X5 = db.child("LandParameters/"+ DN + "/Temperature").get().val()
    temperature = int(X5)
    X6 = db.child("LandParameters/"+ DN + "/Humidity").get().val()
    humidity = int(X6)


    #N = db.child("LandParameters/"+ DN + "/N").get().val()

    # state = request.form.get("stt")



    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    my_prediction = crop_recommendation_model.predict(data)
    final_prediction = my_prediction[0]

    return render_template('crop-result.html', prediction=final_prediction, title=title)
        #return final_prediction
      

# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'FarmGrid - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    DN = str(request.form['nitrogen'])
    #P = int(request.form['phosphorous'])
    #K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    X1 = db.child("LandParameters/"+ DN + "/N").get().val()
    N = int(X1)
    X2 = db.child("LandParameters/"+ DN + "/P").get().val()
    P = int(X2)
    X3 = db.child("LandParameters/"+ DN + "/K").get().val()
    K = int(X3)

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

# render disease prediction result page


@app.route('/disease-predict', methods=['POST'])
def disease_prediction():
    title = 'FarmGrid - Disease Detection'   
    
    
    DN = str(request.form['nitrogen'])
    Image_name = db.child("LeafImages/"+ DN ).get().val()
        
    links = storage.child("Images/" + DN + "/" + Image_name).get_url(None)
            
    response = requests.get(links)
    image_data = io.BytesIO(response.content)

    prediction = predict_image(image_data)

    prediction = Markup(str(disease_dic[prediction]))
    
    return render_template('disease-result.html', prediction=prediction, title=title)
        
        #try:                
        #except:
        #    pass
    #return render_template('disease.html', title=title)


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
