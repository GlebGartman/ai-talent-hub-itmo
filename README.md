

#  AI Talent Hub ITMO

## 📌 Описание проекта

Практические задания по курсу "Основы программирования на Python". В репозитории размещены решения домашних заданий

Проект включает решение практических задач по Python в рамках курса AI Talent Hub — проектной магистратуры и профессионального сообщества в области искусственного интеллекта на базе ИТМО.

В рамках проекта проведён исследовательский анализ (EDA) датасета пассажиров Титаника.  
Цель — изучить структуру данных, выявить закономерности выживаемости и продемонстрировать навыки анализа данных с использованием Python.

В ходе выполнения заданий были применены ключевые подходы к программированию, анализу данных и алгоритмическому мышлению, используемые в задачах Data Science и машинного обучения.

---

## 🎯 Цели проекта

- Применение синтаксиса и основных конструкций Python  
- Развитие алгоритмического мышления  
- Работа с коллекциями и структурами данных  
- Реализация функций и модульного подхода  
- Подготовка к решению задач анализа данных и ML  

---

## 📂 Описание данных

Датасет содержит информацию о пассажирах Титаника:

- **PassengerId** — уникальный идентификатор пассажира  
- **Survived** — выжил ли пассажир (1 — да, 0 — нет)  
- **Pclass** — класс билета (1 — высокий, 2 — средний, 3 — низкий)  
- **Name** — имя пассажира  
- **Sex** — пол  
- **Age** — возраст  
- **SibSp** — количество братьев, сестёр и супругов на борту  
- **Parch** — количество родителей и детей на борту  
- **Ticket** — номер билета  
- **Fare** — стоимость билета  
- **Cabin** — номер каюты  
- **Embarked** — порт посадки  

---

## 🛠️ Используемые технологии

- Python  
- Jupyter Notebook  
- Pandas  
- Matplotlib  
- Seaborn  
- Plotly  

---

## 🔍 Первичный анализ данных

```python
df.info()
df.isna().sum()
```

## 📊 Описательная статистика
```python
df.describe()
```
Получены основные статистические показатели:

- средние значения  
- медианы  
- стандартные отклонения  
- минимальные и максимальные значения

## 📈 Процент выживаемости по классам
```python
survival_by_class = df.groupby('Pclass')['Survived'].mean() * 100
survival_by_class
```
Показывает зависимость вероятности выживания от класса пассажира.


## 👤 Самые популярные имена на корабле

**Извлечение имени из столбца `Name`:**
```python
name_in_brackets = df['Name'].str.extract(r'\(([^ ]+)')
name_after_title = df['Name'].str.extract(r',\s*[^\.]+\.\s*([A-Za-z]+)')
df['FirstName'] = name_in_brackets[0].fillna(name_after_title[0])
```

**Определение самых популярных имён:**

```python
male_name = df[df['Sex'] == 'male']['FirstName'].value_counts().idxmax()
female_name = df[df['Sex'] == 'female']['FirstName'].value_counts().idxmax()

print('Самое популярное мужское имя:', male_name)
print('Самое популярное женское имя:', female_name)
```

## 👥 Популярные имена по классам

```python
for pclass in sorted(df['Pclass'].unique()):

    subset = df[df['Pclass'] == pclass]

    male_name = subset[subset['Sex'] == 'male']['FirstName'].value_counts().idxmax()
    female_name = subset[subset['Sex'] == 'female']['FirstName'].value_counts().idxmax()

    print(f'Класс {pclass}')
    print(f'Мужское имя: {male_name}')
    print(f'Женское имя: {female_name}')
    print()
```
## 🚢 Анализ кают

```python
cabins = df['Cabin'].dropna()
cabins_count = cabins.str.split().apply(len).value_counts()
cabins_count
```

## 👨‍👩‍👧 Пассажиры без родственников

```python
no_relatives = df[(df['SibSp'] == 0) & (df['Parch'] == 0)]
len(no_relatives)
```

## 📊 Визуализация распределений признаков

Визуализируем распределение значений признаков: `Survived`, `Pclass`, `Age`, `Sex`, `Parch`.

```python
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

df['Survived'].value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title('Survived')

df['Pclass'].value_counts().plot(kind='bar', ax=axes[1])
axes[1].set_title('Pclass')

axes[2].hist(df['Age'].dropna(), bins=30)
axes[2].set_title('Age')

df['Sex'].value_counts().plot(kind='bar', ax=axes[3])
axes[3].set_title('Sex')

df['Parch'].value_counts().plot(kind='bar', ax=axes[4])
axes[4].set_title('Parch')

plt.tight_layout()
plt.show()
```

## 📦 Boxplot возраста (Age)

Визуализируем распределение возраста пассажиров и выбросы:

```python
plt.figure(figsize=(6, 4))

plt.boxplot(df['Age'].dropna())

plt.title('Boxplot возраста')
plt.ylabel('Возраст')

plt.show()
```

## 📌 Интерпретация boxplot возраста
```python
q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)
median_age = df['Age'].median()

print(f'Медианный возраст пассажиров: {median_age:.1f} лет.')
print(f'Основной диапазон (межквартильный): {q1:.1f}–{q3:.1f} лет.')
print('На графике также наблюдаются выбросы среди пассажиров старшего возраста.')
```
- **Медианный возраст пассажиров:** 28.0 лет  
- **Основной диапазон (межквартильный):** 20.1–38.0 лет  
- **Наблюдаются выбросы:** присутствуют пассажиры более старшего возраста, выходящие за пределы основного распределения  

## 🥧 Pie chart распределений (Survived и Pclass)

Построение круговых диаграмм для признаков `Survived` и `Pclass`:

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

df['Survived'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    ax=axes[0]
)
axes[0].set_title('Доля выживших')

df['Pclass'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    ax=axes[1]
)
axes[1].set_title('Распределение по классам')

plt.show()
```

## 🔗 Pairplot числовых признаков

Визуализация взаимосвязей между числовыми признаками:

```python
pd.plotting.scatter_matrix(
    df.select_dtypes(include='number'),
    figsize=(10, 10)
)

plt.suptitle('Pairplot числовых признаков')
plt.show()
```

## 🎻 Violin plot (возраст по полу)

Сравнение распределения возраста мужчин и женщин:

```python
data = [
    df[df['Sex']=='male']['Age'].dropna(),
    df[df['Sex']=='female']['Age'].dropna()
]

plt.figure(figsize=(6, 4))

plt.violinplot(data)

plt.xticks([1, 2], ['Male', 'Female'])
plt.title('Возраст по полу')
plt.ylabel('Age')

plt.show()
```

## 🔥 Heatmap корреляций

Построение корреляционной матрицы:

```python
corr = df.corr(numeric_only=True)

plt.figure(figsize=(8, 6))

plt.imshow(corr)

plt.colorbar()

plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)

plt.title('Корреляционная матрица')

plt.show()
```

## 🌞 Sunburst plot (структура пассажиров)

Интерактивная визуализация структуры пассажиров по классу и полу:

```python
sunburst_data = df.groupby(['Pclass', 'Sex']).size().reset_index(name='count')

fig = px.sunburst(
    sunburst_data,
    path=['Pclass', 'Sex'],
    values='count',
    title='Структура пассажиров по классу и полу'
)

fig.show()
```
