import os
import re
import unicodedata
from urllib.parse import urlparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Конфігураційні параметри ---
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
# Шлях до директорії з моделлю відносно поточного файлу detector.py
MODEL_DIR = os.path.join(MODULE_DIR, "spam40k_v7")
TOKENIZER_MAX_LENGTH = 128  # Має відповідати тому, що використовувалося при тренуванні

# Глобальні змінні для моделі та токенізатора (ліниве завантаження)
_model = None
_tokenizer = None
_device = None


# --- Функція очищення тексту (адаптована з вашого коду) ---
def _clean_text(text: str, max_url_domain_length: int = 50) -> str | None:
    """
    Внутрішня функція для очищення тексту перед подачею в модель.
    """
    if not isinstance(text, str):
        return None

    # 1. Обробка URL
    def replace_url_with_domain_token(match):
        url = match.group(0)
        try:
            if not re.match(r'^[a-zA-Z]+://', url):
                url_to_parse = 'http://' + url
            else:
                url_to_parse = url

            parsed_url = urlparse(url_to_parse)
            netloc = parsed_url.netloc

            if not netloc:  # Спроба знайти домен без схеми
                simple_domain_match = re.match(
                    r'^([\w.-]+\.(?:com|org|net|gov|edu|info|biz|ua|рф|укр|io|co|ai|app|dev|me|xyz|online|shop|store|click|link|top|club|site|space|website|tech|agency|solutions|services|blog|news|finance|life|live|world|today|digital|expert|guru|company|community|center|zone|systems|capital|foundation|global|international|investments|management|marketing|media|network|partners|productions|studio|support|team|technology|ventures|vision|wiki|works|best|chat|codes|deals|download|express|plus|pro|promo|gold|guru|jet|one|rocks|star|vip|world|cc))\b',
                    url, re.IGNORECASE)
                if simple_domain_match:
                    netloc = simple_domain_match.group(1)
                else:
                    return ' [URL_UNKNOWN] '  # Якщо домен не вдалося витягти

            if netloc.lower().startswith('www.'):
                netloc = netloc[4:]

            netloc = netloc.lower()
            netloc = re.sub(r':\d+$', '', netloc)  # Видалення порту

            if len(netloc) > max_url_domain_length:
                netloc = netloc[:max_url_domain_length] + '...'

            # Перевірка на мінімальну валідність домену
            if '.' not in netloc or len(netloc) < 3:  # Дуже проста перевірка
                if not re.search(r'[a-zA-Z]', netloc):  # Якщо немає букв, то це не домен
                    return ' [URL_UNKNOWN] '
            return f' [URL:{netloc}] '
        except Exception:
            return ' [URL_INVALID] '  # У випадку будь-якої помилки парсингу

    # Оновлений патерн для URL, що включає домени без http/https та www
    common_tlds = r'(?:com|org|net|gov|edu|info|biz|ua|рф|укр|io|co|ai|app|dev|me|xyz|online|shop|store|click|link|top|club|site|space|website|tech|agency|solutions|services|blog|news|finance|life|live|world|today|digital|expert|guru|company|community|center|zone|systems|capital|foundation|global|international|investments|management|marketing|media|network|partners|productions|studio|support|team|technology|ventures|vision|wiki|works|best|chat|codes|deals|download|express|plus|pro|promo|gold|guru|jet|one|rocks|star|vip|world)'
    url_pattern = re.compile(
        r'https?://[^\s<>"\']+|'  # Стандартні URL з http/https
        r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s<>"\']*|'  # URL, що починаються з www.
        r'\b([a-zA-Z0-9.-]+\.' + common_tlds + r')(?:/[^\s<>"\']*)?\b',  # Домени типу example.com/path
        flags=re.IGNORECASE
    )
    text = url_pattern.sub(replace_url_with_domain_token, text)

    # НЕ ПЕРЕВОДИМО ВЕСЬ ТЕКСТ В НИЖНІЙ РЕГІСТР (якщо ваша модель чутлива до регістру)

    # Заміна email
    text = re.sub(r'\b[\w.-]+?@[\w.-]+\.\w{2,}\b', ' [EMAIL] ', text, flags=re.IGNORECASE)
    # Заміна телефонних номерів
    text = re.sub(r'\+?\d[\d\s\-\(\)]{6,}\d', ' [PHONE] ', text)
    # Заміна юзернеймів (наприклад, @username)
    text = re.sub(r'@\w+', ' [USERNAME] ', text)
    # Заміна хештегів
    text = re.sub(r'#\w+', ' [HASHTAG] ', text)

    # Заміна символів валют
    currency_symbols = r'[\$\u20AC\u00A3\u00A5\u20B4\u20B1\u20BD\u20A8\u20AA\u20A9\u0E3F\u20AB\u20AD\u20AF\u20B9\u20B8\u20B2\u20A1\u20A2\u20A3\u20A4\u20A5\u20A6\u20A7\u20BA\u20BB\u20BC\u20BE\u20BF\u20C0-\u20C5]'
    text = re.sub(currency_symbols, ' [CURRENCY] ', text)

    # Емодзі залишаються в тексті, оскільки BERT може їх обробляти.

    # Нормалізація Unicode для консистентності символів
    text = unicodedata.normalize('NFKC', text)
    # Видалення нерелевантних символів, залишаючи літери, цифри, пунктуацію, символи та пробіли
    # Також залишаємо квадратні дужки, бо вони використовуються в наших спеціальних токенах
    cleaned_text_chars = []
    for char_text in text:
        char_text_cat = unicodedata.category(char_text)
        if char_text_cat[0] in ['L', 'N', 'P', 'S', 'Z'] or char_text in ['[',
                                                                          ']']:  # Літери, Цифри, Пунктуація, Символи, Розділювачі (пробіли)
            cleaned_text_chars.append(char_text)
        elif char_text_cat[0] == 'M':  # Mark characters (e.g., accents) - зазвичай обробляються нормалізацією NFKC
            pass  # Якщо NFKC їх не прибрало, можна їх ігнорувати або замінити на пробіл
        else:
            cleaned_text_chars.append(' ')  # Інші символи (наприклад, контрольні) замінюємо на пробіл
    text = "".join(cleaned_text_chars)

    # Видалення зайвих пробілів
    text = re.sub(r'\s+', ' ', text).strip()
    # Видалення послідовностей однакових спеціальних токенів (наприклад, "[EMAIL] [EMAIL]" -> "[EMAIL]")
    text = re.sub(
        r'(\s*(\[URL:[^\]]+\]|\[EMAIL\]|\[PHONE\]|\[USERNAME\]|\[HASHTAG\]|\[CURRENCY\]|\[URL_UNKNOWN\]|\[URL_INVALID\])\s*)\1+',
        r'\1', text)

    # Перевірка на мінімальну довжину тексту *після* видалення спец. токенів
    content_text = text
    for token_pattern in [r'\[URL:[^\]]+\]', r'\[EMAIL\]', r'\[PHONE\]', r'\[USERNAME\]', r'\[HASHTAG\]',
                          r'\[CURRENCY\]', r'\[URL_UNKNOWN\]', r'\[URL_INVALID\]']:
        content_text = re.sub(token_pattern, '', content_text, flags=re.IGNORECASE)
    content_text = re.sub(r'\s+', '', content_text)  # Видаляємо всі пробіли для перевірки довжини

    return text if len(content_text) >= 3 else None  # Повертаємо None, якщо текст занадто короткий


def _load_model_and_tokenizer():
    """
    Завантажує модель та токенізатор, якщо вони ще не завантажені.
    Використовує глобальні змінні _model, _tokenizer, _device.
    """
    global _model, _tokenizer, _device
    if _model is None or _tokenizer is None:
        # print("Завантаження моделі та токенізатора...") # Можна розкоментувати для дебагу
        try:
            _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if not os.path.isdir(MODEL_DIR):
                # Ця помилка важлива, оскільки без моделі модуль не працюватиме.
                raise FileNotFoundError(
                    f"Директорію моделі не знайдено: {MODEL_DIR}. "
                    f"Переконайтеся, що модель ({os.path.basename(MODEL_DIR)}) "
                    f"знаходиться в {os.path.dirname(MODEL_DIR)} всередині пакета spam_detector_module."
                )

            _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
            _model.eval()  # Переводимо модель в режим оцінки (важливо для стабільних результатів)
            # print(f"Модель та токенізатор завантажено на пристрій: {_device}") # Можна розкоментувати для дебагу
        except Exception as e:
            # Обробка помилок завантаження моделі
            # Можна додати логування тут, якщо у вас є система логування в модулі
            raise RuntimeError(f"Не вдалося завантажити модель або токенізатор з {MODEL_DIR}: {e}")


def predict(raw_text: str, threshold: float = 0.5) -> int:
    """
    Прогнозує, чи є наданий текст спамом.

    Args:
        raw_text (str): Сирий текст для аналізу.
        threshold (float): Поріг для класифікації спаму.
                           Якщо ймовірність спаму вища за цей поріг, текст вважається спамом.
                           Значення за замовчуванням 0.5.

    Returns:
        int: 1 якщо текст є спамом, 0 якщо не спам.
             Повертає 0, якщо текст занадто короткий після очищення, порожній, або не є рядком.

    Raises:
        RuntimeError: Якщо виникає помилка під час завантаження моделі або класифікації.
                      Наприклад, якщо директорія моделі не знайдена.
    """
    _load_model_and_tokenizer()  # Переконуємося, що модель та токенізатор завантажені

    if not isinstance(raw_text, str) or not raw_text.strip():
        # Якщо вхідний текст не рядок або порожній рядок, вважаємо не спамом
        return 0, 0.0

    cleaned_text = _clean_text(raw_text)

    if not cleaned_text:
        # Якщо текст після очищення порожній або занадто короткий, вважаємо не спамом
        return 0, 0.0

    try:
        # Токенізація
        inputs = _tokenizer(
            cleaned_text,
            truncation=True,  # Обрізати текст, якщо він довший за max_length
            max_length=TOKENIZER_MAX_LENGTH,
            padding="max_length",  # Доповнити до max_length, якщо коротший
            return_tensors="pt"  # Повернути PyTorch тензори
        ).to(_device)  # Переміщуємо тензори на відповідний пристрій (CPU/GPU)

        # Передбачення
        with torch.no_grad():  # Вимикаємо обчислення градієнтів для інференсу
            logits = _model(**inputs).logits
            # Застосовуємо softmax для отримання ймовірностей
            # probs[0] - ймовірність класу "не спам" (мітка 0)
            # probs[1] - ймовірність класу "спам" (мітка 1)
            probs = F.softmax(logits, dim=-1)[0]
            spam_prob = probs[1].item()  # Ймовірність того, що це спам
            prediction_label = 1 if spam_prob > threshold else 0

        # Повертаємо 1, якщо ймовірність спаму перевищує поріг, інакше 0
        return prediction_label, spam_prob

    except Exception as e:
        # Обробка будь-яких інших помилок під час класифікації
        # Можна додати логування
        raise RuntimeError(f"Помилка під час класифікації тексту: \"{str(raw_text)[:50]}...\". Деталі: {e}")

