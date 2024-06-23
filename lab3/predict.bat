curl -X POST -H "Content-Type: application/json" -d "{\"sepal_length\": 7.7, \"sepal_width\": 2.6, \"petal_length\": 6.9, \"petal_width\": 2.3}" http://127.0.0.1:8000/predict

curl -X POST -H "Content-Type: application/json" -d "{\"sepal_length\": 5.4, \"sepal_width\": 3.4, \"petal_length\": 1.5, \"petal_width\": 0.4}" http://127.0.0.1:8000/predict

