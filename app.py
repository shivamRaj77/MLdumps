# 1. Library imports
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pandas as pd
import pickle

# 2. Create the app object
app = FastAPI()
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1.8000
@app.get('/')
def index():
    return {'message': 'HELLO'}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello {name}'}

# 4. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with confidence(probability)
@app.post('/predict')
def predict_banknote(data: BankNote):
    data = data.dict()
    print(data)
    print("Hello")
    variance = data['variance']
    print(variance)
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print('Hello')
    if prediction[0] > 0.5:
        prediction = "Fake Note"
    else:
        prediction = "It's a Bank Note"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http:127.0.0.1:8000
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
# uvicorn app:app --reload