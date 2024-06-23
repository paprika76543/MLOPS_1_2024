#создаем образ
docker build -t jenkins_python .

#создаем и запускаем контейнер
docker run -d --restart always --name jenkins -p 8080:8080 -p 50000:50000 jenkins_python:latest