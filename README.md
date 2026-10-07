# Система прогнозирования оттока клиентов и автоматизации удержания (Classical ML &amp; A/B).

Описание: Разработал сквозной ML-пайплайн для предсказания вероятности оттока пользователей (на примере телеком/банковских данных) и спроектировал архитектуру для интеграции с системами таргетированных предложений.

Результат:
•	Работа с данными: развернул локально PostgreSQL, написал сложные SQL-запросы (JOIN, оконные функции, агрегации) для формирования витрин данных и признаков.
•	EDA и Feature Engineering: Провёл разведочный анализ, обработал дисбаланс классов (применил SMOTE / class weights), сгенерировал бизнес-фичи (LTV, частота транзакций, time-since-last-login), отобрал признаки с помощью feature importance.
•	Моделирование: Обучил и сравнил градиентный бустинг (CatBoost, LightGBM, XGBoost). Оптимизировал PR-AUC (так как класс оттока несбалансирован) с учётом бизнес-метрик (стоимость ложного срабатывания vs стоимость удержания).
•	MLOps: Настроил трекинг экспериментов и артефактов в MLflow. Автоматизировал пайплайн ежедневного инференса и еженедельного переобучения модели с помощью Apache Airflow.
•	Спроектировал план A/B-теста для оценки эффекта от ML-модели: провел Power Analysis, рассчитал необходимый размер выборки и minimum detectable effect (MDE), определил первичные и вторичные метрики успеха.
Стек: Python, SQL, PostgreSQL, Pandas, Scikit-learn, CatBoost, LightGBM, MLflow, Apache Airflow, Docker, A/B-тестирование (Power Analysis).

dataset project-1

1. Brazilian E-Commerce Public Dataset by Olist
[https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)

---

