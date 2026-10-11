# Промпт-инженерия для EDA с помощью LLM (Задание 1)

Разведочный анализ датасета **Breast Cancer Wisconsin (Diagnostic)** из [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) с помощью LLM. Использовано 4 промпта: структура данных, закономерности, аномалии и бизнес-выводы.

- [`report.md`](report.md) — отчёт по шаблону задания: промпт → ответ LLM → комментарий;
- [`data/breast_cancer_wisconsin.csv`](data/breast_cancer_wisconsin.csv) — датасет, 569 строк × 31 столбец;
- [`verify_numbers.py`](verify_numbers.py) — скрипт, который пересчитывает все числа из ответов модели (`pip install pandas scipy`, затем `python verify_numbers.py`).
