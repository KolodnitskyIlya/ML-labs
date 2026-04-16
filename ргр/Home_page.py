import streamlit as st

pages = {
    "Моя информация": [
        st.Page("info.py", title="Главный экран"),
    ],
    "Непосредственно ML": [
        st.Page("data.py", title="Конструктивные данные"),
        st.Page("graphics.py", title="Визуализация"),
        st.Page("predict.py", title="Предсказание дианоза"),
    ],
}

pg = st.navigation(pages)
pg.run()