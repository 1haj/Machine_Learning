from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    df=pd.read_csv('notebook\data\data_main.csv')
    neighbourhoods = list(df['neighbourhood_cleansed'].unique())
    return render_template('home.html', neighbourhoods=neighbourhoods)


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    df=pd.read_csv('notebook\data\data_main.csv')
    neighbourhoods = list(df['neighbourhood_cleansed'].unique())

    if request.method=='GET':
        return render_template('home.html', neighbourhoods=neighbourhoods)
    else:
        data = CustomData(
    # Numeric features
        bathrooms=float(request.form.get('bathrooms')),
        bedrooms=float(request.form.get('bedrooms')),
        beds=float(request.form.get('beds')),
        accommodates=float(request.form.get('accommodates')),
        guests_included=float(request.form.get('guests_included')),
        minimum_nights_avg_ntm=float(request.form.get('minimum_nights_avg_ntm')),
        maximum_nights_avg_ntm=float(request.form.get('maximum_nights_avg_ntm')),
        cleaning_fee=float(request.form.get('cleaning_fee')),
        extra_people=float(request.form.get('extra_people')),
        security_deposit=float(request.form.get('security_deposit')),
        host_response_rate=float(request.form.get('host_response_rate')),
        

        # Categorical features (keep them as strings)
        room_type=request.form.get('room_type'),
        property_type=request.form.get('property_type'),
        neighbourhood_cleansed=request.form.get('neighbourhood_cleansed'),
        host_is_superhost=request.form.get('host_is_superhost'),
        bed_type=request.form.get('bed_type'),
        cancellation_policy=request.form.get('cancellation_policy')
    )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    port = 5000 
    print(f"Running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0",debug=True)      
    


