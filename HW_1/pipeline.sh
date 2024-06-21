#!/bin/bash

# Функция для создания виртуального окружения
create_venv() {
    local env_name=${1:-".venv"}
    python3 -m venv "$env_name"
    echo "The virtual environment '$env_name' has been created."
}

# Функция активации виртуального окружения
activate_venv() {
    local env_name=${1:-".venv"}
    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' to create."
        return 1
    fi
    if [ -z "$VIRTUAL_ENV" ]; then
        source "./$env_name/bin/activate"
        echo "Virtual environment '$env_name' is activated."
    else
        echo "The virtual environment has already been activated."
    fi
}

# Функция для установки зависимостей из файла requirements.txt
install_deps() {
    if [ ! -f "requirements.txt" ]; then
        echo "File requirements.txt not found."
        return 1
    fi

    # Проверка установки зависимостей из requirements.txt
    for package in $(cat requirements.txt | cut -d '=' -f 1); do
        if ! pip freeze | grep -q "^$package=="; then
            echo "Dependency installation..."
            pip install -r requirements.txt
            echo "Dependencies installed."
            return 0
        fi
    done

    echo "All dependencies are already installed."
}

# Создаем виртуальное окружение
if [ ! -d ".venv" ]; then
    create_venv
fi

# Активируем виртуальное окружение
activate_venv

# Устанавливаем зависимости
install_deps

# Получаем количество дата сетов
n_datasets=$1

# Запускаем файл генерирования данных
python python_scripts/data_creation.py $n_datasets

# Запускаем препроцессинг
python python_scripts/model_preprocessing.py $n_datasets

# Запускаем компиляцию и обучение модели
python python_scripts/model_preparation.py $n_datasets

# Запускаем тестирование модели
python python_scripts/model_testing.py $n_datasets
