import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from Tools import *
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from lightgbm import LGBMRegressor


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
    Создает лаговые признаки для обучения и для предсказания на следующий день.

    Параметры:
    - df: DataFrame с исходными данными.
    - feature_list: список колонок, для которых создаются лаги.
    - min_lag: минимальный лаг.
    - max_lag: максимальный лаг.

    Возвращает:
    - lagged_df: DataFrame для обучения, содержащий лаговые признаки и целевую переменную.
    - prediction_df: DataFrame с одной строкой, содержащей лаговые признаки для предсказания на следующий день.
    """
    # Создаем лаговые признаки для обучения
    lagged_features = []
    for lag in range(min_lag, max_lag + 1):
        lagged = df[feature_list].shift(lag).add_suffix(f'_lag_{lag}')
        lagged_features.append(lagged)

    # Объединяем исходные данные с лаговыми признаками
    lagged_df = pd.concat([df] + lagged_features, axis=1)
    lagged_df.dropna(inplace=True)

    # Создаем лаговые признаки для предсказания на следующий день
    prediction_dict = {}
    for feature in feature_list:
        for lag in range(min_lag, max_lag + 1):
            lag_feature_name = f'{feature}_lag_{lag}'
            # Получаем значение для соответствующего лага
            prediction_dict[lag_feature_name] = df[feature].iloc[-lag]

    prediction_df = pd.DataFrame([prediction_dict])

    return lagged_df, prediction_df


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

    model = LGBMRegressor(max_depth=3, verbosity=-1)
    selector1 = Kraken(model, cv_datetime, MAPE, 'exp1')
    selector1.get_rank_dict(df_m, df_m['cnt'], lags_cols, group_dt)
    vars_final = selector1.get_vars(df_m, df_m['cnt'], early_stopping_rounds=100, group_dt=group_dt)

    if len(vars_final) == 0:
        vars_final = [i for i in lags_cols if i.startswith('cnt')]

    test_dates = pd.Series(df_m['date'].unique()).sort_values().tail(max(3, test_m // 2)).values

    X_train = df_m[~df_m['date'].isin(test_dates)]
    y_train = df_m[~df_m['date'].isin(test_dates)]['cnt']
    X_oot = df_m[df_m['date'].isin(test_dates)]

    model.fit(X_train[vars_final], y_train)
    pred_df['date'] = X_oot.date.max() + pd.Timedelta(days=horizont)
    pred_df['cnt'] = np.nan
    pred_df['item_id'] = X_oot.item_id.max()
    pred_df['mean'] = X_oot.cnt.mean()
    X_oot['mean'] = X_oot.cnt.mean()
    list_cont = vars_final + ['mean', 'cnt_SMA_3_lag_1', 'item_id', 'date', 'cnt']
    data_prediction = pd.concat([X_oot[list_cont], pred_df[list_cont]], axis=0)
    data_prediction['model_prediction'] = model.predict(data_prediction[vars_final])

    # ARIMA predictions
    for sku in list_sku:
        sku_series = df_m[df_m['item_id'] == sku]
        sku_series.set_index('date', inplace=True, drop=False)
        if adfuller(sku_series['cnt'])[1] < 0.05:  # ADF test to check if series is stationary
            arima_model = ARIMA(sku_series['cnt'], order=(1, 0, 1))
            arima_results = arima_model.fit()
            data_prediction.loc[data_prediction['item_id'] == sku, 'arima_prediction'] = \
            arima_results.forecast(steps=horizont)[-1]

    return data_prediction, model
    
def calculate_and_plot_shap(model, X_data):
    """
    Построение SHAP values
    model - обученная модель 
    X_data - X_oot[vars_final]
    """
    # Создаем объект для расчета SHAP values в зависимости от типа модели
    explainer = shap.Explainer(model, X_data)
    
    # Рассчитываем SHAP values
    shap_values = explainer(X_data)
    
    # Строим beeswarm график для визуализации SHAP values
    shap.plots.beeswarm(shap_values)
    
    plt.show()

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

def evaluate_predictions(df):
    """Функция для оценки различных методов предсказания."""
    methods = ['model_prediction', 'cnt_SMA_3_lag_1', 'cnt_WMA_3_lag_1', 'mean', 'arima_prediction']

    # Вычисляем метрики
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


def forecast_plot_from_dfs(real_df, forecast1_df, forecast2_df, forecast3_df, date_column, value_column):
    """
    real_df - датафрейм с реальными данными
    forecast1_df - датафрейм с прогнозом 1
    forecast2_df - датафрейм с прогнозом 2
    forecast3_df - датафрейм с прогнозом 3
    date_column - столбец с датами в каждом из датафреймов
    value_column - столбец со значениями временного ряда
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Отрисовка реального временного ряда
    ax.plot(real_df[date_column], real_df[value_column], label='Actual Series', color='blue')

    # Отрисовка прогнозов
    ax.plot(forecast1_df[date_column], forecast1_df[value_column], label='Forecast 1', color='red', linestyle='--')
    ax.plot(forecast2_df[date_column], forecast2_df[value_column], label='Forecast 2', color='green', linestyle=':')
    ax.plot(forecast3_df[date_column], forecast3_df[value_column], label='Forecast 3', color='orange', linestyle='-.')

    # Добавляем пунктирную линию для обозначения начала прогноза
    ax.axvline(x=real_df[date_column].max(), color='black', linestyle='--', label='Forecast Start')

    # Настройка меток и легенды
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.legend()

    return fig
