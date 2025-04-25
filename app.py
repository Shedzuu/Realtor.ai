from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import sys

# Добавляем директорию src в путь для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Импортируем функции из модуля model.py
from src.model import get_real_estate_details, extract_json_from_string, calculate_similarity_score

# Инициализация Flask приложения
app = Flask(__name__, static_url_path='')
CORS(app)  # Включаем CORS для всех маршрутов

# Определяем абсолютный путь к файлам данных
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "data", "apartments.csv")

# Загружаем CSV файл с данными о квартирах
try:
    df = pd.read_csv(data_path, index_col=0)
    print(f"Загружено {len(df)} объявлений о квартирах")
except FileNotFoundError:
    print(f"Ошибка: Файл {data_path} не найден")
    df = pd.DataFrame()

@app.route('/')
def index():
    """Отображение главной страницы"""
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Обрабатывает запрос на поиск квартир и возвращает топ-10 результатов
    """
    if df.empty:
        return jsonify([]), 500

    # Получаем пользовательский запрос из JSON
    data = request.json
    user_input = data.get('user_input', '')
    
    if not user_input:
        return jsonify([]), 400
    
    try:
        # Получаем признаки из LLM
        print(f"Анализ запроса пользователя: {user_input[:50]}...")
        formatted_response = get_real_estate_details(user_input)
        real_estate_dict = extract_json_from_string(formatted_response)
        
        if "error" in real_estate_dict:
            print(f"Ошибка при извлечении данных: {real_estate_dict['error']}")
            return jsonify([]), 500
        
        # Вычисление схожести и сортировка
        print("Поиск подходящих вариантов...")
        df_copy = df.copy()
        df_copy["similarity_score"] = calculate_similarity_score(df_copy, real_estate_dict, user_input)
        df_sorted = df_copy.sort_values("similarity_score", ascending=False)
        
        # Преобразуем топ-10 результатов в список словарей для JSON
        top_results = []
        for _, row in df_sorted.head(10).iterrows():
            apartment = {col: row[col] for col in df_sorted.columns if pd.notna(row[col])}
            # Добавляем процент схожести в формате для отображения
            similarity = row.get('similarity_score', 0)
            apartment['match_percent'] = int(similarity * 100) if similarity <= 1 else 100
            top_results.append(apartment)
        
        print(f"Найдено {len(top_results)} подходящих вариантов")
        
        # Сохраняем результаты поиска для анализа
        results_path = os.path.join(BASE_DIR, "data", "search_results.csv")
        df_sorted.to_csv(results_path)
        
        return jsonify(top_results)
    
    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Запуск API-сервера на порту 5000...")
    app.run(debug=True, port=5000)