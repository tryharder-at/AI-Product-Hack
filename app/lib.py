import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from Tools import *
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from lightgbm import LGBMRegressor

def infer_granularity(dates):
    dates = dates.sort_values()
    delta = dates.diff().dropna().unique()
    if len(delta) == 1:
        if delta[0] == pd.Timedelta(days=1):
            return 'Day'
        elif delta[0] == pd.Timedelta(weeks=1):
            return 'Week'
        elif pd.Timedelta(days=28) <= delta[0] <= pd.Timedelta(days=31):
            return 'Month'
    raise ValueError("Cannot infer granularity from dates.")

def add_future_dates(pred_df, X_oot, date_column, horizont, granularity):
    """
    Добавляет будущие даты в pred_df в соответствии с гранулярностью.

    Параметры:
    - pred_df: DataFrame, куда нужно добавить даты.
    - X_oot: DataFrame с исходными данными, из которого берем последнюю дату.
    - date_column: название колонки с датами.
    - horizont: горизонт прогнозирования (количество будущих периодов).
    - granularity: гранулярность ('Day', 'Week', 'Month').

    Возвращает:
    - pred_df с добавленной колонкой дат.
    """
    last_date = X_oot[date_column].max()
    
    if granularity == 'Day':
        date_list = [last_date + pd.Timedelta(days=i) for i in range(1, horizont+1)]
    elif granularity == 'Week':
        date_list = [last_date + pd.Timedelta(weeks=i) for i in range(1, horizont+1)]
    elif granularity == 'Month':
        date_list = [last_date + pd.DateOffset(months=i) for i in range(1, horizont+1)]
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")
    
    pred_df[date_column] = date_list
    return pred_df



def simple_moving_average(df, column, window):
    df[f'{column}_SMA_{window}'] = df[column].rolling(window=window).mean()
    return df

# Пример использования:
# df = simple_moving_average(df, 'sales', window=3)


def weighted_moving_average(df, column, window):
    weights = np.arange(1, window + 1)  # Веса от 1 до размера окна
    wma = df[column].rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
    df[f'{column}_WMA_{window}'] = wma
    return df

# Пример использования:
# df = weighted_moving_average(df, 'sales', window=3)

def exponential_moving_average(df, column, span):
    df[f'{column}_EMA_{span}'] = df[column].ewm(span=span, adjust=False).mean()
    return df

# Пример использования:
# df = exponential_moving_average(df, 'sales', span=3)

# def line_plot_with_legend(df, variables):
#     # Построение линий для каждой переменной
#     plt.figure(figsize=(10, 6))
#
#     for var in variables:
#         sns.lineplot(data=df, x=df.index, y=var, label=var)
#
#     # Подписи осей
#     plt.xlabel('Index')
#     plt.ylabel('Values')
#
#     # Добавляем легенду
#     plt.legend(title="Variables")
#
#     # Отображаем график
#     plt.tight_layout()
#     plt.show()


def calculate_mape(df, target_col, predictions_list, add=1e-10):
    """
    Рассчитывает MAPE для нескольких прогнозов по отношению к таргету.
    
    Параметры:
    - df: DataFrame с данными.
    - target_col: колонка с истинными значениями (таргет).
    - predictions_list: список названий столбцов с прогнозами.
    - add: значение, добавляемое к таргету для избежания деления на ноль.
    
    Возвращает:
    - DataFrame с названиями предсказаний и их значениями MAPE.
    """
    
    # Проверка, что таргет и прогнозы есть в DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Колонка с таргетом '{target_col}' не найдена в DataFrame.")
    
    missing_predictions = [pred for pred in predictions_list if pred not in df.columns]
    if missing_predictions:
        raise ValueError(f"Прогнозы {missing_predictions} не найдены в DataFrame.")
    
    # Список для хранения MAPE для каждого прогноза
    mape_results = []
    
    # Рассчет MAPE для каждого прогноза
    for pred_col in predictions_list:
        # MAPE = (1/n) * sum(|(y_true - y_pred)| / (y_true + add)) * 100
        mape = (abs(df[target_col] - df[pred_col]) / (df[target_col] + add)).mean() * 100
        mape_results.append({'Prediction': pred_col, 'MAPE': mape})
    
    # Возвращаем результаты в виде DataFrame
    return pd.DataFrame(mape_results)



def create_lag_features_with_prediction(df, feature_list, min_lag, max_lag):
    """
    Создает лаговые признаки для обучения и для предсказания на следующие min_lag шагов.

    Параметры:
    - df: DataFrame с исходными данными.
    - feature_list: список колонок, для которых создаются лаги.
    - min_lag: минимальный лаг.
    - max_lag: максимальный лаг.

    Возвращает:
    - lagged_df: DataFrame для обучения, содержащий лаговые признаки и целевую переменную.
    - prediction_df: DataFrame с min_lag строками, содержащий лаговые признаки для предсказания на следующие min_lag шагов.
    """
    # Создаем лаговые признаки для обучения
    lagged_data = df.copy()
    for lag in range(min_lag, max_lag + 1):
        for feature in feature_list:
            lagged_data[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
    
    # Удаляем строки с NaN
    #lagged_df = lagged_data.dropna().reset_index(drop=True)
    
    # Создаем лаговые признаки для предсказания
    # Используем последние min_lag строк из исходного DataFrame
    prediction_data = df[feature_list].iloc[-min_lag:].copy().reset_index(drop=True)
    for lag in range(min_lag, max_lag + 1):
        for feature in feature_list:
            lag_feature_name = f'{feature}_lag_{lag}'
            # Сдвигаем данные, чтобы получить лаговые признаки
            prediction_data[lag_feature_name] = prediction_data[feature].shift(lag - min_lag)
    
    # Удаляем ненужные колонки исходных признаков
    prediction_df = prediction_data.drop(columns=feature_list)
    # Удаляем строки с NaN, если они есть
    #prediction_df = prediction_df.dropna().reset_index(drop=True)
    
    return lagged_data, prediction_df


def merge_files_to_dataset(
        shop_sales: pd.DataFrame,
        shop_sales_prices: pd.DataFrame,
        shop_sales_dates: pd.DataFrame
) -> pd.DataFrame:
    shop_sales_dates['date_id'] = shop_sales_dates.index + 1
    merged = shop_sales.merge(shop_sales_dates, on='date_id', how='left')
    #merged.fillna('Nothing', inplace=True)
    main_df = merged.merge(shop_sales_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    main_df['sell_price'] = main_df['sell_price'].ffill().bfill()
    return main_df


def summ_sales_data(df, date_column, granularity):

    df[date_column] = pd.to_datetime(df[date_column])
    if granularity == 'Day':
        return df
    elif granularity == 'Week':
        resample_rule = 'W' 
    elif granularity == 'Month':
        resample_rule = 'M'
    
    # Устанавливаем индекс на 'date_column' и 'item_id', чтобы выполнить группировку по обоим уровням
    df.set_index([date_column, 'item_id'], inplace=True)
    
    # Группируем по дате с учетом resample_rule и сохраняем группировку по 'item_id'
    grouped = df.groupby(['item_id']).resample(resample_rule, level=0).agg({
        'cnt': 'sum',
        'sell_price': 'mean',
        'CASHBACK_STORE_1': lambda x: x.sum() / len(x),
        'CASHBACK_STORE_2': lambda x: x.sum() / len(x),
        'CASHBACK_STORE_3': lambda x: x.sum() / len(x),
        'store_id': 'first',
        'date_id': 'first',  
        'wm_yr_wk': 'first',  
        'weekday': 'first',  
        'wday': 'first',  
        'month': 'first',  
        'year': 'first',  
        'event_name_1': lambda x: x.dropna().iloc[0] if not x.dropna().empty else None,
        'event_type_1': lambda x: x.dropna().iloc[0] if not x.dropna().empty else None,  
        'event_name_2': lambda x: x.dropna().iloc[0] if not x.dropna().empty else None,
        'event_type_2': lambda x: x.dropna().iloc[0] if not x.dropna().empty else None,
    })
    
    grouped_df = grouped.reset_index()

    return grouped_df


def preprocess_data_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df['store_number'] = df['store_id'].str.extract(r'(\d+)').astype(int)
    df['group_id'] = df['item_id'].str.extract(r'STORE_\d_(\d)').astype(int)
        
    return df


def get_preds(df, list_sku, horizont):
    df_m = df[df['item_id'].isin(list_sku)].copy()
    df_m = simple_moving_average(df_m, 'cnt', 3)
    df_m = weighted_moving_average(df_m, 'cnt', 3)
    df_m = exponential_moving_average(df_m, 'cnt', 3)

    technical_list = ['index_time', 'item_id', 'store_id', 'date_id', 'cnt', 'date', 'wm_yr_wk', 'wday']
    list_for_lags = [i for i in df_m.columns if i not in technical_list]
    lags, pred_df = create_lag_features_with_prediction(df_m, list_for_lags, horizont, horizont)
    lags_cols = [i for i in lags.columns if i not in df_m.columns]
    df_m = pd.concat([df_m, lags[lags_cols]], axis=1)

    seed_value = 23
    np.random.seed(seed_value)

    window_m = len(df_m.date.unique()) // 5
    test_m = window_m // 5
    cv_datetime = DateTimeSeriesSplit(window=window_m, n_splits=4, test_size=test_m, margin=0)
    group_dt = df_m['date']

    model = LGBMRegressor(max_depth=3, verbosity=-1, boosting_type= 'goss')
    selector1 = Kraken(model, cv_datetime, MAPE, 'exp1')
    selector1.get_rank_dict(df_m, df_m['cnt'], lags_cols, group_dt)
    vars_final = []
    vars_final = selector1.get_vars(df_m, df_m['cnt'], early_stopping_rounds=100, group_dt=group_dt)

    if len(vars_final) == 0:
        vars_final = [i for i in lags_cols if i.startswith('cnt_')]

    test_dates = pd.Series(df_m['date'].unique()).sort_values().tail(max(3, test_m // 2)).values

    X_train = df_m[~df_m['date'].isin(test_dates)]
    y_train = df_m[~df_m['date'].isin(test_dates)]['cnt']
    # Вычисляем медиану по y_train, исключая нули
    median_value = y_train[y_train != 0].median()

    # Заменяем нули на медианное значение
    y_train = y_train.replace(0, median_value)
    X_oot = df_m[df_m['date'].isin(test_dates)]

    model.fit(X_train[vars_final], y_train)
    granularity = infer_granularity(X_oot['date'])
    pred_df = add_future_dates(pred_df, X_oot, 'date', horizont, granularity)
    #pred_df['date'] = [X_oot.date.max() + pd.Timedelta(days=i) for i in range(1,horizont+1)]
    pred_df['cnt'] = np.nan
    pred_df['item_id'] = X_oot.item_id.max()
    pred_df['mean'] = X_oot.cnt.mean()
    X_oot['mean'] = X_oot.cnt.mean()
    list_cont = vars_final + ['mean', f'cnt_SMA_3_lag_{horizont}', 'item_id', 'date', 'cnt']
    data_prediction = pd.concat([X_oot[list_cont], pred_df[list_cont]], axis=0)
    # Шаг 1: Проверка на дубликаты столбцов
    all_columns = data_prediction[list_cont].columns
    duplicated_columns = all_columns[all_columns.duplicated()]
    data_prediction = data_prediction[list_cont].loc[:, ~all_columns.duplicated()]
    data_prediction['model_prediction'] = model.predict(data_prediction[vars_final])

    # Import ARIMA from statsmodels
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    warnings.filterwarnings('ignore')

    # Prepare the time series data for ARIMA
    df_m_ts = df_m[['date', 'cnt']].copy()
    df_m_ts['date'] = pd.to_datetime(df_m_ts['date'])
    df_m_ts.set_index('date', inplace=True)
    df_m_ts = df_m_ts.sort_index()
    df_m_ts = df_m_ts.asfreq('D')  # Ensure daily frequency
    df_m_ts['cnt'].fillna(method='ffill', inplace=True)  # Handle missing values

    # Fit the ARIMA model
    arima_order = (1, 1, 1)  # You can adjust the order based on your data
    model_arima = ARIMA(df_m_ts['cnt'], order=arima_order)
    arima_result = model_arima.fit()

    # Predict over the required period
    # Get the full date range for predictions
    last_date = df_m_ts.index[-1]
    pred_dates = data_prediction['date'].unique()
    full_dates = pd.date_range(start=df_m_ts.index[0], end=pred_dates.max(), freq='D')

    # Generate predictions for all dates in full_dates
    arima_predictions = arima_result.predict(start=0, end=len(full_dates) - 1)
    arima_pred_df = pd.DataFrame({'date': full_dates, 'arima_prediction': arima_predictions})
    arima_pred_df['date'] = pd.to_datetime(arima_pred_df['date'])

    # Merge ARIMA predictions with data_prediction
    data_prediction['date'] = pd.to_datetime(data_prediction['date'])
    data_prediction = pd.merge(data_prediction, arima_pred_df, on='date', how='left')

    return data_prediction, model
    
def calculate_and_plot_shap(model, X_data):
    """
    model - обученная модель
    X_data - данные для расчета SHAP values (например, X_oot[vars_final])
    
    Возвращает:
    fig - объект matplotlib.figure.Figure с графиком SHAP values.
    """
    # Создаем объект для расчета SHAP values
    explainer = shap.Explainer(model, X_data)
    
    # Рассчитываем SHAP values
    shap_values = explainer(X_data)
    
    # Создаем fig
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Строим график для визуализации SHAP values
    shap.plots.beeswarm(shap_values, show=False, ax=ax)
    
    return fig

def calculate_group_means(df: pd.DataFrame) -> pd.DataFrame:
    group_means = df.groupby(['store_number', 'group_id']).agg(
        cnt=('cnt', 'mean'),
        sell_price=('sell_price', 'mean'),
        cashback_1=('CASHBACK_STORE_1', 'sum'),
        cashback_2=('CASHBACK_STORE_2', 'sum'),
        cashback_3=('CASHBACK_STORE_3', 'sum')
    ).reset_index()
    
    group_means['cashback_2'] = group_means.apply(
        lambda row: 0 if row['cashback_1'] > 0 else row['cashback_2'], axis=1)
    group_means['cashback_3'] = group_means.apply(
        lambda row: 0 if row['cashback_1'] > 0 else row['cashback_3'], axis=1)
        
    return group_means


def line_plot_with_legend(df: pd.DataFrame, variables: list[str]) -> plt.Figure:
    # для теста))
    # df = df[df['item_id'] == 'STORE_2_085']

    fig, ax = plt.subplots(figsize=(10, 6))
    for var in variables:
        sns.lineplot(data=df, x=df.date, y=var, ax=ax, label=var)
    ax.set_xlabel('Index')
    ax.set_ylabel('Values')
    ax.legend(title="Variables")

    # правим отступы
    plt.tight_layout()
    # для стримлита
    return fig

def forecast_sales(trained_models, method, current_data, date_until, granularity='D'):
    """
    Строит численный прогноз продаж до заданной даты с помощью уже обученной модели.

    :param trained_models: Словарь обученных моделей.
    :param method: Метод, используемый для прогнозирования ('Constant', 'Exponential Smoothing', 'Gradient Boosting').
    :param current_data: Исходные данные о продажах.
    :param date_until: Дата (в виде строки 'YYYY-MM-DD') до которой нужен прогноз.
    :param granularity: Гранулярность прогноза (например, 'D' - ежедневно).
    :return: Прогноз продаж до заданной даты.
    """
    index = pd.date_range(start=current_data.index[-1] + pd.Timedelta(days=1), end=date_until, freq=granularity)
    forecast_length = len(index)

    if method == 'Constant':
        predictions = [current_data.iloc[-1]] * forecast_length

    elif method in trained_models:
        future_X = np.arange(len(current_data), len(current_data) + forecast_length).reshape(-1, 1)
        predictions = trained_models[method].predict(future_X)

    forecast_series = pd.Series(predictions, index=index)
    return forecast_series


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Рассчитывает среднее абсолютное процентное отклонение (MAPE).
    """
    y_true, y_pred = pd.Series(y_true), pd.Series(y_pred)
    return ((y_true - y_pred).abs() / y_true.abs()).mean() * 100


def calculate_mae(df, target_col, predictions_list):
    """ Рассчитывает MAE для нескольких прогнозов по отношению к таргету. """
    # Проверка, что таргет и прогнозы есть в DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Колонка с таргетом '{target_col}' не найдена в DataFrame.")

    missing_predictions = [pred for pred in predictions_list if pred not in df.columns]
    if missing_predictions:
        raise ValueError(f"Прогнозы {missing_predictions} не найдены в DataFrame.")

    mae_results = []
    # Рассчет MAE для каждого прогноза
    for pred_col in predictions_list:
        mae = np.mean(np.abs(df[target_col] - df[pred_col]))
        mae_results.append({'Prediction': pred_col, 'MAE': mae})

    return pd.DataFrame(mae_results)

def calculate_rmse(df, target_col, predictions_list):
    """ Рассчитывает RMSE для нескольких прогнозов по отношению к таргету. """
    # Проверка, что таргет и прогнозы есть в DataFrame
    if target_col not in df.columns:
        raise ValueError(f"Колонка с таргетом '{target_col}' не найдена в DataFrame.")

    missing_predictions = [pred for pred in predictions_list if pred not in df.columns]
    if missing_predictions:
        raise ValueError(f"Прогнозы {missing_predictions} не найдены в DataFrame.")

    rmse_results = []
    # Рассчет RMSE для каждого прогноза
    for pred_col in predictions_list:
        rmse = np.sqrt(np.mean((df[target_col] - df[pred_col]) ** 2))
        rmse_results.append({'Prediction': pred_col, 'RMSE': rmse})

    return pd.DataFrame(rmse_results)

def evaluate_predictions(df, horizont):
    """Функция для оценки различных методов предсказания."""
    methods = ['model_prediction', f'cnt_SMA_3_lag_{horizont}', 'mean', 'arima_prediction']

    # Вычисляем метрики
    df = df[df['cnt'] > 0]
    mape_df = calculate_mape(df, 'cnt', methods)
    mae_df = calculate_mae(df, 'cnt', methods)
    rmse_df = calculate_rmse(df, 'cnt', methods)

    # Объединение результатов в один DataFrame
    results_df = pd.merge(mape_df, mae_df, on='Prediction')
    results_df = pd.merge(results_df, rmse_df, on='Prediction')

    return results_df



def df_encoding(sku: pd.DataFrame) -> pd.DataFrame:
    df_one_hot = sku.copy()
    columns = ['event_name_1', 'event_type_1',	'event_name_2',	'event_type_2', 'weekday']
    # Применить one-hot encoding для указанных столбцов
    df_one_hot = pd.get_dummies(df_one_hot, columns=columns, dtype=int)
    return df_one_hot


def forecast_plot_from_df(df, date_column, value_column, forecast_columns):
    """
    Отрисовка реального временного ряда и нескольких прогнозов, которые хранятся в одном датафрейме.
    
    Параметры:
    - df: DataFrame, содержащий реальный временной ряд и несколько столбцов с прогнозами.
    - date_column: Столбец с датами.
    - value_column: Столбец с реальными значениями временного ряда.
    - forecast_columns: Список столбцов с прогнозами.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Определяем последнюю дату с реальными данными
    last_actual_date = df[date_column][df[value_column].notna()].max()
    
    # Получаем последнее реальное значение
    last_actual_value = df.loc[df[date_column] == last_actual_date, value_column].values[0]
    
    # Отрисовка реального временного ряда до последней даты с реальными данными
    actual_data = df[df[date_column] <= last_actual_date]
    ax.plot(actual_data[date_column], actual_data[value_column], label='Actual Series', color='blue')
    
    # Добавляем вертикальную прерывистую линию на последнюю дату с реальными данными
    ax.axvline(x=last_actual_date, color='black', linestyle='--', label='Forecast Start')
    
    # Цвета и стили для прогнозов (повторяются при большем числе столбцов)
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'magenta', 'cyan', 'black']
    linestyles = ['--', ':', '-.', '-', '--']
    
    # Отрисовка прогнозов, начиная с последнего реального значения
    for i, forecast_column in enumerate(forecast_columns):
        # Получаем данные прогноза начиная с даты после последней реальной даты
        forecast_data = df[df[date_column] > last_actual_date][[date_column, forecast_column]]
        
        # Создаем серии дат и значений прогноза, включая последнюю реальную точку
        forecast_dates = pd.concat([
            pd.Series([last_actual_date]),        # Добавляем последнюю реальную дату
            forecast_data[date_column]
        ], ignore_index=True)
        
        forecast_values = pd.concat([
            pd.Series([last_actual_value]),       # Добавляем последнее реальное значение
            forecast_data[forecast_column]
        ], ignore_index=True)
        
        # Отрисовываем прогноз как продолжение реального ряда
        ax.plot(
            forecast_dates, 
            forecast_values, 
            label=forecast_column, 
            color=colors[i % len(colors)], 
            linestyle=linestyles[i % len(linestyles)]
        )
    
    # Настройка меток и легенды
    ax.set_xlabel('Дата')
    ax.set_ylabel('Значение')
    ax.grid(True)
    ax.legend()
    
    return fig

# def decompose_series(df, date_col='date', column='cnt', model='additive', period=7):
#     # пример использования: decompose_series(df, date_col='date', column='cnt', model='additive', period=365)
#
#     # Создаем копию датафрейма, чтобы не изменять оригинал
#     df = df.copy()
#
#     # Проверяем, есть ли дата в колонках или в индексе
#     if date_col in df.columns:
#         # Убедимся, что столбец даты в формате datetime
#         df[date_col] = pd.to_datetime(df[date_col])
#         # Устанавливаем дату как индекс
#         df.set_index(date_col, inplace=True)
#     elif date_col in df.index.names:
#         # Если дата уже индекс, ничего не делаем
#         pass
#     else:
#         raise KeyError(f"Колонка или индекс '{date_col}' не найдена в датафрейме.")
#
#     # Сортируем индекс
#     df.sort_index(inplace=True)
#
#     # Выбираем временной ряд для анализа
#     ts = df[column]
#
#     # Проверяем на наличие пропущенных значений
#     if ts.isnull().any():
#         print("В данных обнаружены пропущенные значения. Они будут заполнены методом forward fill.")
#         ts = ts.fillna(method='ffill')
#
#     # Выполняем декомпозицию
#     decomposition = seasonal_decompose(ts, model=model, period=period)
#
#     # Получаем компоненты
#     observed = decomposition.observed
#     trend = decomposition.trend
#     residual = decomposition.resid
#
#     # Визуализируем результаты
#     plt.style.use('seaborn-whitegrid')
#     fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=False)
#
#     # Настройка формата отображения дат
#     date_format = '%Y-%m-%d'
#
#     # График исходных данных
#     axes[0].plot(observed.index, observed, label='Исходные данные', color='blue')
#     axes[0].set_ylabel('Исходные данные')
#     axes[0].set_xlabel('Дата')
#     axes[0].xaxis.set_tick_params(rotation=45)
#
#     # График тренда
#     axes[1].plot(trend.index, trend, label='Тренд', color='red')
#     axes[1].set_ylabel('Тренд')
#     axes[1].set_xlabel('Дата')
#     axes[1].xaxis.set_tick_params(rotation=45)
#
#     # График остатков
#     axes[2].plot(residual.index, residual, label='Остатки', color='purple')
#     axes[2].set_ylabel('Остатки')
#     axes[2].set_xlabel('Дата')
#     axes[2].xaxis.set_tick_params(rotation=45)
#
#     # Улучшаем компоновку и отображение
#     plt.tight_layout()
#     plt.show()
