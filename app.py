
import streamlit as st
import google.generativeai as genai
from google.colab import userdata

st.title("Анализ характеристик товаров с Gemini")

# Получаем API ключ из секретов Colab
try:
    GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Ошибка получения API ключа: {e}")
    st.warning("Пожалуйста, убедитесь, что у вас установлен секрет 'GOOGLE_API_KEY' в настройках Colab.")
    st.stop() # Останавливаем выполнение, если нет ключа

# Инициализируем модель Gemini
try:
    model = genai.GenerativeModel('gemini-pro') # Используйте модель, подходящую для ваших задач
except Exception as e:
    st.error(f"Ошибка инициализации модели Gemini: {e}")
    st.stop()


st.write("Введите описание товара для анализа его характеристик.")

product_description = st.text_area("Описание товара", "")

if st.button("Анализировать"):
    if product_description:
        prompt = f'''
        Извлеки ключевые характеристики из следующего описания товара.
        Представь результат в структурированном формате JSON, где ключами являются названия характеристик, а значениями - их значения.
        Если характеристика не указана, пропустите ее.

        Примеры товаров и характеристик:

        Холодильник:
        - Тип: двухкамерный, однокамерный, side-by-side
        - Общий объем: в литрах
        - Класс энергопотребления: A++, A+, A, B и т.д.
        - Система размораживания: No Frost, капельная
        - Размеры (ВхШхГ): в см

        Телевизор:
        - Диагональ экрана: в дюймах
        - Разрешение: Full HD, 4K UHD, 8K UHD
        - Тип экрана: LED, OLED, QLED
        - Смарт ТВ: да/нет
        - Поддержка HDR: да/нет

        Компьютер:
        - Тип: настольный, ноутбук, моноблок
        - Процессор: модель и поколение (например, Intel Core i5 12th Gen, AMD Ryzen 7 5800H)
        - Оперативная память (RAM): объем в ГБ
        - Накопитель: тип и объем (например, SSD 512 ГБ, HDD 1 ТБ)
        - Видеокарта: модель

        Миксер:
        - Тип: ручной, стационарный
        - Мощность: в Вт
        - Количество скоростей: число
        - Объем чаши (для стационарных): в литрах
        - Насадки: перечисление (венчики, крюки для теста и т.д.)

        Описание товара для анализа:
        {product_description}

        JSON формат:
        '''
        try:
            response = model.generate_content(prompt)
            st.subheader("Результат анализа:")
            st.json(response.text) # Отображаем результат в формате JSON
        except Exception as e:
            st.error(f"Ошибка при вызове Gemini API: {e}")
            st.write("Пожалуйста, проверьте ваше описание и попробуйте снова.")
    else:
        st.warning("Пожалуйста, введите описание товара.")

