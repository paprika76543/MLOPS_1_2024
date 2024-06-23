# Используем официальный образ Jenkins
FROM jenkins/jenkins:lts-jdk17

# Переключаемся на пользователя root для установки пакетов
user root

# Обновление списка пакетов
RUN apt-get update

# Установка Python, pip, venv
RUN apt-get install -y python3 python3-pip python3-venv

# Создание и активация виртуального окружения
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Установка пакетов для Python
RUN pip install --upgrade pip && \
    pip install numpy pandas scikit-learn joblib

# Переключаемся обратно на пользователя Jenkins
user jenkins