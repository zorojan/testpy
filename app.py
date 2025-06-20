import streamlit as st

st.set_page_config(page_title="Моё первое приложение Zorojan")

st.title("Привет от Streamlit, Zorojan!")
st.write("Это простое приложение, развернутое из GitHub.")
st.write("---")
st.write("Я обновил это приложение!")
st.write("---")
st.success("Успешно обновлено через Colab и GitHub!") # Новая строка

user_name = st.text_input("Как вас зовут?")
if user_name:
    st.write(f"Приятно познакомиться, {user_name}!")

favorite_number = st.slider("Выберите любимое число", 0, 100, 50)
st.write(f"Ваше любимое число: {favorite_number}")