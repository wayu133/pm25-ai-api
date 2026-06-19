import os
import uvicorn
import joblib
import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI
from tensorflow.keras.models import load_model

app = FastAPI()

# โหลดโมเดลและ Scaler
model = load_model('pm25_hourly_trend_bilstm_model.h5', compile=False)
scaler_x = joblib.load('scaler_x.pkl')
scaler_y = joblib.load('scaler_y.pkl')

API_KEY = "ac368bfbb4f021f19ad47b9df364a803"
LAT = 14.0208
LON = 100.5250

@app.get("/")
def home():
    return {"status": "AI System is Online!"}

@app.get("/predict")
def predict():
    # 1. โหลดข้อมูลประวัติจาก CSV
    df = pd.read_csv('history_data.csv')
    
    # 2. ดึงสภาพอากาศจาก API
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
    weather = requests.get(url).json()
    
    # 3. สร้างข้อมูลใหม่ (ต้องจัดคอลัมน์ให้ตรงกับตอนเทรน)
    # สมมติว่าตอนเทรนคุณใช้ [pm25, temp, humidity, pressure, wind]
    last_pm25 = df['pm25'].iloc[-1]
    new_data = np.array([[
        last_pm25, 
        weather['main']['temp'], 
        weather['main']['humidity'], 
        weather['main']['pressure'], 
        weather['wind']['speed']
    ]])
    
    # 4. ปรับสเกลข้อมูล
    scaled_data = scaler_x.transform(new_data)
    
    # 5. รันโมเดล (ปรับเป็น 3D Array ตามที่โมเดลต้องการ)
    prediction = model.predict(scaled_data.reshape(1, 1, 5)) 
    final_result = scaler_y.inverse_transform(prediction)
    
    return {"predicted_pm25": float(final_result[0][0])}

# ส่วนนี้สำคัญมากสำหรับการรันบนคลาวด์
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)