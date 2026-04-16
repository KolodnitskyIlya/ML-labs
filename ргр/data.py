import streamlit as st
import pandas as pd

st.title('Информация о наборе данных')

st.header("Описание предметной области датасета:")

st.markdown("**Тематика датасета:**")
st.markdown("Предсказание сердечно-сосудистых заболеваний на основе клинико-лабораторных показателей.")
st.markdown("**Описание:**")
st.markdown("Здоровье сердца — одна из важнейших медицинских проблем современности. Своевременная диагностика ишемической болезни сердца и других сердечных патологий " \
"может спасти множество жизней. В рамках данного проекта используется медицинский датасет, содержащий информацию о пациентах, таких как возраст, пол, уровень холестерина, " \
"артериальное давление и другие важные параметры. Цель анализа — построить модель машинного обучения, способную предсказывать наличие сердечных заболеваний.")
st.markdown("**Целевая переменная:**")
st.markdown("num — наличие диагноза сердечного заболевания (целочисленное значение, где, например, 0 означает отсутствие заболевания, а значения от 1 " \
"и выше — различные степени выраженности заболевания).")

st.header("Описание признакового пространства 🫀:")

df = pd.DataFrame({'Variable Name': ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'],
                  'Role': ['Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Feature', 'Target'],
                  'Type': ['Integer', 'Categorical', 'Categorical', 'Integer', 'Integer', 'Categorical', 'Categorical', 'Integer', 'Categorical', 'Integer', 'Categorical', 'Integer', 'Categorical', 'Integer'],
                  'Demographic': ['Age', 'Sex', '', '', '', '', '', '', '', '', '', '', '', ''],
                  'Description': ['', '', '', 'resting blood pressure \n (on admission to the hospital)', 'serum cholestoral', 'fasting blood sugar > 120 mg/dl', '',
                                  'maximum heart rate achieved', 'exercise induced angina', 'ST depression induced by exercise relative to rest', '', 
                                  'number of major vessels (0-3) colored by flourosopy', '', 'diagnosis of heart disease'],
                   'Units': ['years', '', 'mm Hg', 'mg/dl', '', '', '', '', '', '', '', '', '', ''],
                   'Missing Values': ['no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no']})

st.dataframe(df, width=1150, height=550)

heart_disease = pd.read_csv('heart_main.csv')

st.header("Особенности предобработки данных:")

st.markdown("Мой исходный датафрейм:")

st.dataframe(heart_disease.head())

st.markdown('''Получили следующую характеристику исходных данных:

- в столбцах 'ca', 'thal' отсутствуют значения;
- тип данных столбцов 'ca', 'thal' нужно перевести в целочисленный
''')

st.markdown('''#### Заполнение пропущенных значений''')

missing_values = heart_disease.isna().sum()
missing_values = missing_values[missing_values > 0]
missing_values.rename()

st.dataframe(missing_values.head())
