from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import os
import random
import logging
from requests.exceptions import RequestException

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Определяем абсолютный путь к директории с данными
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Функции для извлечения информации
def get_info_from_header(text):
    try:
        info = text.split("·")
        if len(info) >= 2:
            rooms = info[0].strip()[0] if info[0].strip() and info[0].strip()[0].isdigit() else "1"
            address = ""
            for part in info:
                if "," in part:
                    address = part.split(",")[1].strip() if len(part.split(",")) > 1 else ""
                    break
            return [rooms, address]
        else:
            return ["1", ""]
    except Exception as e:
        logger.error(f"Ошибка при обработке заголовка: {e}")
        return ["", ""]

def get_info_from_price(price):
    try:
        price_text = price.strip()
        price_text = price_text.replace("&nbsp;", "").replace("\xa0", "")
        # Находим цифры в тексте
        digits = ''.join(filter(str.isdigit, price_text))
        return digits if digits else "0"
    except Exception as e:
        logger.error(f"Ошибка при обработке цены: {e}")
        return "0"

def get_first_int(s):
    try:
        f = 0
        for i in s:
            if i.isnumeric():
                f = f * 10 + int(i)
            else:
                break
        return f
    except Exception as e:
        logger.error(f"Ошибка при извлечении числа: {e}")
        return 0

def make_request(url, max_retries=3, delay=1):
    """Делает запрос с повторными попытками и задержкой"""
    for attempt in range(max_retries):
        try:
            # Добавляем случайную задержку между запросами
            if attempt > 0:
                time.sleep(delay + random.uniform(1, 3))
            
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
                timeout=10
            )
            response.raise_for_status()  # Вызовет исключение для ошибок HTTP
            return response
        except RequestException as e:
            logger.warning(f"Попытка {attempt+1} запроса к {url} не удалась: {e}")
    
    logger.error(f"Не удалось получить данные с {url} после {max_retries} попыток")
    return None

def main():
    # Логика скрапинга
    base_link = "https://krisha.kz/"
    list_of_apartments = []
    start_time = time.time()
    
    # Увеличиваем количество страниц для скрапинга
    pages_to_scrape = 5  # Увеличьте это значение для большего количества данных
    
    logger.info(f"Начинаем собирать данные с {pages_to_scrape} страниц")
    
    for p in range(1, pages_to_scrape + 1):
        logger.info(f"Обработка страницы {p}")
        page_url = f"{base_link}arenda/kvartiry/almaty/?rent-period-switch=%2Farenda%2Fkvartiry&page={p}"
        
        page_response = make_request(page_url)
        if not page_response:
            continue
        
        try:
            soup = BeautifulSoup(page_response.text, "html.parser")
            links = soup.findAll("a", attrs={"class": "a-card__title"})
            
            logger.info(f"Найдено {len(links)} объявлений на странице {p}")
            
            for l in links:
                try:
                    cell_url = f"{base_link}{l.get('href')}"
                    logger.info(f"Обработка объявления: {cell_url}")
                    
                    cell_response = make_request(cell_url)
                    if not cell_response:
                        continue
                    
                    soup_cell = BeautifulSoup(cell_response.text, "html.parser")
                    
                    # Извлечение данных с защитой от ошибок
                    header = soup_cell.find("h1")
                    cell_main_info = get_info_from_header(header.text if header else "")
                    
                    price_html = soup_cell.find("div", attrs={"class": "offer__price"})
                    price_info = get_info_from_price(price_html.text if price_html else "")
                    
                    city_html = soup_cell.find("div", attrs={"class": "offer__location offer__advert-short-info"})
                    city_info = city_html.find('span').text.strip() if city_html and city_html.find('span') else ""
                    
                    # Дополнительные данные
                    description_html = soup_cell.find("div", attrs={"class": "offer__description"})
                    description = description_html.text.strip() if description_html else ""
                    
                    # Собираем параметры квартиры из списка характеристик
                    params = {}
                    params_list = soup_cell.findAll("dl", attrs={"class": "offer__advert-short-info"})
                    for param in params_list:
                        try:
                            dt = param.find("dt")
                            dd = param.find("dd")
                            if dt and dd:
                                key = dt.text.strip().lower()
                                value = dd.text.strip()
                                params[key] = value
                        except Exception as e:
                            logger.warning(f"Ошибка при обработке параметра: {e}")
                    
                    # Формируем словарь с данными о квартире
                    apartment = {
                        "url": cell_url,
                        "rooms": cell_main_info[0],
                        "adress": cell_main_info[1],
                        "price": price_info,
                        "city_region": city_info,
                        "description": description
                    }
                    
                    # Добавляем параметры из списка характеристик
                    apartment.update({
                        "floor": params.get("этаж", ""),
                        "area": params.get("площадь, м²", ""),
                        "apartment_condition": params.get("состояние", ""),
                        "house_year": params.get("год постройки", ""),
                        "bathroom": params.get("санузел", ""),
                        "furniture_detailed": params.get("мебель", "")
                    })
                    
                    list_of_apartments.append(apartment)
                    
                    # Задержка между запросами для избежания блокировки
                    time.sleep(random.uniform(0.5, 2.0))
                    
                except Exception as e:
                    logger.error(f"Ошибка при обработке объявления {l.get('href') if l else 'неизвестно'}: {e}")
        
        except Exception as e:
            logger.error(f"Ошибка при обработке страницы {p}: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Скрапинг завершен за {elapsed_time:.2f} секунд")
    logger.info(f"Собрано {len(list_of_apartments)} объявлений")
    
    if list_of_apartments:
        try:
            # Сохраняем данные в CSV
            df = pd.DataFrame(list_of_apartments)
            output_path = os.path.join(DATA_DIR, "apartments.csv")
            df.to_csv(output_path)
            logger.info(f"Данные сохранены в {output_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных: {e}")
    else:
        logger.warning("Нет данных для сохранения")

if __name__ == "__main__":
    main()