import pandas as pd
import re
import json
import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import colorama
from colorama import Fore, Style
import textwrap

# Инициализируем colorama для цветного вывода
colorama.init()

# Загружаем переменные окружения из .env файла если он есть
load_dotenv()

# Определяем абсолютный путь к файлам данных
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "apartments.csv")

# Загружаем CSV файл с данными о квартирах
try:
    df = pd.read_csv(data_path, index_col=0)
except FileNotFoundError:
    print(f"Ошибка: Файл {data_path} не найден")
    df = pd.DataFrame()

# Получаем API ключ из переменных окружения или используем значение по умолчанию
api_key = os.environ.get("DEEPSEEK_API_KEY", "sk-74b87290351b47acacfc94680907ed09")

# Настройка клиента DeepSeek
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

# Функция запроса к LLM для извлечения признаков недвижимости
def get_real_estate_details(client_prompt):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": (
                    "You are a professional realtor. "
                    "Extract the following real estate features from the user's input and make sure values are in Russian and return them as a Python dictionary:\n\n"
                    "{ 'rooms': value, 'adress': value, 'price': value, 'city_region': value, 'floor': value, 'area': value, 'kitchen_studio': value,"
                    " 'apartment_condition': value, 'is_apartment_has_furniture(full or no)': value, 'is_previously_dormitory': value, 'security': value,"
                    " 'furniture_detailed': value, 'facilities': value, 'bathroom': value, 'zhiloi_complex': value, 'house_year': value,"
                    " 'parking': value, 'is_separated_toilet': value, 'toilet_count': value, 'description': value }\n\n"
                    "If any information is missing, replace it with 'No Information'. Ensure the response is a properly formatted JSON string without any markdown formatting."
                )},
                {"role": "user", "content": client_prompt},
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Ошибка при запросе к API: {e}")
        return "{}"

# Функция для извлечения JSON из строки
def extract_json_from_string(response_str):
    # Сначала проверяем наличие JSON в блоке кода
    json_match = re.search(r'```(?:json)?\n(.*?)\n```', response_str, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        # Если блока кода нет, пробуем найти JSON напрямую
        # Ищем первую открывающую и последнюю закрывающую фигурные скобки
        start = response_str.find('{')
        end = response_str.rfind('}')
        
        if start != -1 and end != -1 and end > start:
            json_str = response_str[start:end+1]
        else:
            return {"error": "No JSON found in response"}
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"Ошибка декодирования JSON: {json_str}")
        return {"error": "Invalid JSON format"}

# Подсчёт схожести по признакам
def calculate_similarity_score(df, feature_dict, user_prompt):
    similarity_scores = []
    if df.empty:
        return []
        
    # Если DataFrame содержит URL в первом столбце, пропускаем его
    data_df = df.iloc[:, 1:] if 'url' in df.columns else df

    # Определяем важные признаки и их веса
    key_features = {
        'price': 3.0,      # Цена - очень важный параметр
        'rooms': 5.0,      # Количество комнат - критически важный параметр
        'city_region': 3.0, # Район - критически важный параметр
        'description': 0.5 # Описание - важный параметр
    }
    
    # Стандартный вес для остальных признаков
    default_weight = 1.0
    
    # Дополнительный фактор для улицы (будем проверять как часть адреса)
    street_bonus = 5.0

    # Отбираем релевантные признаки (исключаем с 'No Information')
    relevant_features = [col for col in data_df.columns if 
                         col in feature_dict and 
                         feature_dict.get(col, "No Information") != "No Information" and 
                         col != "description"]
    
    # Проверяем, есть ли информация об улице в запросе
    street_info = None
    if 'adress' in feature_dict and feature_dict['adress'] != "No Information":
        street_info = feature_dict['adress'].lower()
    
    for index, row in data_df.iterrows():
        # Базовая оценка - начинаем с 0.2 (20% совпадения по умолчанию)
        base_score = 0.2
        feature_similarity = 0
        feature_weights_sum = 0
        total_score = 0
        
        # Проверяем точные совпадения по ключевым параметрам
        exact_matches = 0
        key_features_count = 0
        
        # 1. Проверка совпадения по комнатам
        if 'rooms' in relevant_features and 'rooms' in row:
            key_features_count += 1
            row_rooms = str(row.get('rooms', "")).strip()
            dict_rooms = str(feature_dict.get('rooms', "")).strip()
            
            if row_rooms and dict_rooms and row_rooms == dict_rooms:
                exact_matches += 1
                # Бонус за точное совпадение комнат
                base_score += 0.2
        
        # 2. Проверка совпадения по району
        if 'city_region' in relevant_features and 'city_region' in row:
            key_features_count += 1
            row_region = str(row.get('city_region', "")).lower().strip()
            dict_region = str(feature_dict.get('city_region', "")).lower().strip()
            
            # Проверяем вхождение района
            if row_region and dict_region and (dict_region in row_region or row_region in dict_region):
                exact_matches += 1
                # Бонус за совпадение района
                base_score += 0.2
        
        # 3. Проверка совпадения по улице (в адресе)
        if street_info and 'adress' in row:
            row_address = str(row.get('adress', "")).lower().strip()
            if row_address and street_info in row_address:
                # Значительный бонус за совпадение улицы
                base_score += 0.3
        
        # Если есть хотя бы одно точное совпадение по ключевым параметрам, 
        # увеличиваем базовую оценку
        if exact_matches > 0:
            base_score = max(base_score, 0.3 + (exact_matches / key_features_count) * 0.4)
        
        # TF-IDF для признаков с учетом весов
        if relevant_features:
            try:
                weighted_similarities = []
                
                for feature in relevant_features:
                    # Получаем вес для текущего признака
                    weight = key_features.get(feature, default_weight)
                    feature_weights_sum += weight
                    
                    # Создаем отдельные векторы для текущего признака
                    feature_vectorizer = TfidfVectorizer()
                    row_value = str(row.get(feature, ""))
                    dict_value = str(feature_dict.get(feature, ""))
                    
                    if row_value.strip() and dict_value.strip():
                        tfidf_matrix = feature_vectorizer.fit_transform([dict_value, row_value])
                        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                        weighted_similarities.append(similarity * weight)
                
                # Считаем средневзвешенное значение
                if feature_weights_sum > 0 and weighted_similarities:
                    feature_similarity = sum(weighted_similarities) / feature_weights_sum
                    feature_similarity = feature_similarity * 0.5  # Вклад TF-IDF в общую оценку - 50%
                    
            except Exception as e:
                print(f"Ошибка при расчете схожести признаков: {str(e)}")

        # TF-IDF для описания (с меньшим весом)
        description_similarity = 0
        if "description" in data_df.columns:
            try:
                description_text = str(row.get("description", ""))
                if description_text.strip() and user_prompt.strip():
                    desc_vectorizer = TfidfVectorizer()
                    tfidf_desc_matrix = desc_vectorizer.fit_transform([user_prompt, description_text])
                    description_similarity = cosine_similarity(tfidf_desc_matrix[0:1], tfidf_desc_matrix[1:2])[0][0] * 0.3 # Меньший вес для описания
            except Exception as e:
                print(f"Ошибка при расчете схожести описания: {str(e)}")

        # Общий результат - сумма базовой оценки, TF-IDF оценки признаков и оценки описания
        total_score = base_score + feature_similarity + description_similarity
        
        # Нормализация оценки до диапазона 0-1
        total_score = min(total_score, 1.0)
        
        similarity_scores.append(total_score)

    return similarity_scores

# Функция для красивого вывода результатов
def print_detailed_results(df_sorted, client_input, extracted_features, top_n=3):
    terminal_width = os.get_terminal_size().columns
    separator = "=" * terminal_width
    
    # Выводим информацию о запросе пользователя
    print(f"\n{Fore.CYAN}{separator}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🔍 ЗАПРОС ПОЛЬЗОВАТЕЛЯ:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{textwrap.fill(client_input, width=terminal_width)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{separator}{Style.RESET_ALL}\n")
    
    # Выводим извлеченные характеристики
    print(f"{Fore.CYAN}📋 ИЗВЛЕЧЕННЫЕ ХАРАКТЕРИСТИКИ:{Style.RESET_ALL}")
    for key, value in extracted_features.items():
        if key != "error" and value != "No Information" and value:
            print(f"{Fore.GREEN}{key}{Style.RESET_ALL}: {value}")
    print(f"\n{Fore.CYAN}{separator}{Style.RESET_ALL}\n")
    
    # Выводим топ результаты
    print(f"{Fore.CYAN}🏆 ТОП-{top_n} ПОДХОДЯЩИХ ОБЪЕКТОВ:{Style.RESET_ALL}\n")
    
    # Проверяем, есть ли результаты
    if len(df_sorted) == 0:
        print(f"{Fore.RED}Не найдено подходящих объектов.{Style.RESET_ALL}")
        return
    
    # Выводим детальную информацию по каждому объекту
    for i, (idx, row) in enumerate(df_sorted.head(top_n).iterrows()):
        similarity = row.get('similarity_score', 0)
        similarity_percent = int(similarity * 100) if similarity <= 1 else 100
        
        print(f"{Fore.CYAN}#{i+1} - Совпадение: {Fore.YELLOW}{similarity_percent}%{Style.RESET_ALL}")
        
        # URL объявления
        url = row.get('url', '')
        if url:
            print(f"{Fore.BLUE}🔗 Ссылка:{Style.RESET_ALL} {url}")
        
        # Основная информация
        print(f"{Fore.BLUE}🏠 Адрес:{Style.RESET_ALL} {row.get('adress', 'Нет данных')}")
        print(f"{Fore.BLUE}💰 Цена:{Style.RESET_ALL} {row.get('price', 'Нет данных')} тг")
        print(f"{Fore.BLUE}🚪 Комнат:{Style.RESET_ALL} {row.get('rooms', 'Нет данных')}")
        print(f"{Fore.BLUE}📍 Район:{Style.RESET_ALL} {row.get('city_region', 'Нет данных')}")
        print(f"{Fore.BLUE}📏 Площадь:{Style.RESET_ALL} {row.get('area', 'Нет данных')}")
        print(f"{Fore.BLUE}🏢 Этаж:{Style.RESET_ALL} {row.get('floor', 'Нет данных')}")
        
        # Дополнительная информация при наличии
        if 'apartment_condition' in row and row['apartment_condition']:
            print(f"{Fore.BLUE}🔧 Состояние:{Style.RESET_ALL} {row['apartment_condition']}")
        
        if 'furniture_detailed' in row and row['furniture_detailed']:
            print(f"{Fore.BLUE}🛋️ Мебель:{Style.RESET_ALL} {row['furniture_detailed']}")
            
        if 'bathroom' in row and row['bathroom']:
            print(f"{Fore.BLUE}🚿 Санузел:{Style.RESET_ALL} {row['bathroom']}")
            
        if 'house_year' in row and row['house_year']:
            print(f"{Fore.BLUE}📅 Год постройки:{Style.RESET_ALL} {row['house_year']}")
        
        # Описание
        if 'description' in row and row['description']:
            desc = row['description']
            if len(desc) > 200:
                desc = desc[:200] + "..."
            print(f"{Fore.BLUE}📝 Описание:{Style.RESET_ALL} {textwrap.fill(desc, width=terminal_width-10, initial_indent='  ', subsequent_indent='  ')}")
        
        print(f"{Fore.CYAN}{'-' * terminal_width}{Style.RESET_ALL}\n")

# Пример пользовательского запроса
client_input = (
    "Ищу 2-комнатную квартиру в Алматы, Алмалинский район, желательно по улице Айтеки би")

if __name__ == "__main__":
    if not df.empty:
        # Получение признаков из LLM
        print(f"{Fore.CYAN}Анализ запроса пользователя...{Style.RESET_ALL}")
        formatted_response = get_real_estate_details(client_input)
        real_estate_dict = extract_json_from_string(formatted_response)
        
        if "error" not in real_estate_dict:
            # Вычисление схожести и сортировка
            print(f"{Fore.CYAN}Поиск подходящих вариантов...{Style.RESET_ALL}")
            df["similarity_score"] = calculate_similarity_score(df, real_estate_dict, client_input)
            df_sorted = df.sort_values("similarity_score", ascending=False)
            
            # Вывод детальной информации о топ результатах
            print_detailed_results(df_sorted, client_input, real_estate_dict)
            
            # Сохраняем результаты в CSV для детального анализа
            results_path = os.path.join(BASE_DIR, "data", "search_results.csv")
            df_sorted.to_csv(results_path)
            print(f"{Fore.CYAN}Полные результаты сохранены в {results_path}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Ошибка при извлечении данных из ответа LLM: {real_estate_dict['error']}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}Данные о квартирах не загружены, проверьте путь к файлу.{Style.RESET_ALL}")
