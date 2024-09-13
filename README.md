# AI-Product-Hack. Команда "Фиксики"
Наш продукт решает задачу прогнозирования спроса товаров для ритейл-компаний. Мы предлагаем аналитикам и менеджерам удобный инструмент для оптимизации поставок и доставок, что в конечном итоге помогает ритейлерам снижать затраты и улучшать обслуживание клиентов.

#### Цель проекта: создание высокоэффективного инструмента, который поможет ритейл-компаниям прогнозировать спрос на товары, улучшая планирование поставок и избегая излишних запасов и нехватки товаров.

### Описание данных
В датасете представлены 3 таблицы: 
1) В shop_sales хранится информация по уникальным Id товара, Id магазина и даты, а также кол-во проданных конкретных товаров в конкретный день.
Количество проданных конкретных товаров в конкретный день - это целевая переменная (target), по которой мы можем построить временной ряд на всем промежутке времени(1819 дней)
2) В shop_sales_prices хранится агригированная по неделям цена продажи конкретного товара конкретного магазина.
Цена продажи является одним из признаков, которые участвуют в настройке и работы модели
3) В shop_sales_dates содержится информация об акциях и событиях, которые были актуальны в конкретный день.
Признаки наличия акции и наличия события также являются важными признаками, участвующими в работе модели для более корректного прогноза

### Build, test and deploy
`docker build . -t cr.yandex/crplondb28dlfjo3vi3t/product_app:<tag> --platform linux/amd64 `

Test run (optional) `docker run -p 8501:8501`

`docker push cr.yandex/crplondb28dlfjo3vi3t/product_app:trof1mov_09.09.1`

In yandex.cloud VM -> edit vm, select your tag, wait 

### Веб-интерфейс
http://fixiki.duckdns.org/

### Roadmap проекта
- [x] Формирование видения продукта
- [x] Исследование реализованных технических решений 
- [x] EDA
- [x] Разработка архитектуры системы, выбор моделей и методов обработки данныхпрогнозирование для новых товаров.
- [x] Создание ML-пайплайн решения 
- [x] Проектирование и разработка веб-интерфейс для визуализации данных и аналитики.
- [ ] Проведение тестирования и отладки сервиса.
- [ ] Проверка корректность работы моделей на тестовых данных
- [ ] Оптимизировать работу системы и устранить обнаруженные ошибки.

## Зависимости проекта

Для работы нашего проекта требуются следующие библиотеки:

- [Matplotlib](https://matplotlib.org/stable/contents.html)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [NumPy](https://numpy.org/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Streamlit](https://streamlit.io/)

## Используемые источники
* https://www.kaggle.com/code/nadezhda2019/regression
* https://www.kaggle.com/code/davidmsu/eda-for-russian-beginners-from-russian-beginner
* https://www.kaggle.com/c/rossmann-store-sales/overview
* https://www.kaggle.com/c/instacart-market-basket-analysis
* https://www.kaggle.com/datasets/aslanahmedov/walmart-sales-forecast
* https://vc.ru/marketplace/347350-kak-riteilery-predskazyvayut-spros
* https://vndit.io/en/5-common-inventory-management-challenges-facing-retailers/
