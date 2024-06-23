import os
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


# Получаем правильные пути
SCRIPTS_PATH = os.path.dirname(os.path.abspath(__file__))      # Каталог со скриптами
PROJECT_PATH = os.path.dirname(SCRIPTS_PATH)                   # Каталог проекта

# API приложение
app = FastAPI()

class IrisPredictionInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Загрузка сохраненной модели
model_path = os.path.join(PROJECT_PATH, 'model', 'model.joblib')

try:
    model = joblib.load(model_path)
    print("Модель успешно загружена.")
except Exception as e:
    print('Произошла ошибка при открытии модели:', e)

# Обработчик API для предсказания видов ирисов
@app.post("/predict")
async def predict(payload: IrisPredictionInput):
     features = np.array([[payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width]])
     prediction = model.predict(features)
     prediction = int(prediction[0])            # Преобразуем предсказание в целое число
     return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
