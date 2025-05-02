@echo off
echo Установка зависимостей для лабораторной работы...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo Зависимости успешно установлены!
pause