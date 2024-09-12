import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import seaborn as sns


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
    df = df[df['item_id'] == 'STORE_2_085']

    fig, ax = plt.subplots(figsize=(10, 6))
    for var in variables:
        sns.lineplot(data=df, x=df.index, y=var, ax=ax, label=var)
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


def calculate_metrics(true_values, predictions_dict):
    """
    Рассчитывает метрики прогнозов для нескольких методов.

    :param true_values: Список (или массив) истинных значений.
    :param predictions_dict: Словарь, где ключи - названия методов, значения - списки (или массивы) прогнозов.
    :return: DataFrame с метриками для каждого метода.
    """
    metrics_data = []

    for method, predictions in predictions_dict.items():
        rmse = mean_squared_error(true_values, predictions, squared=False)
        mae = mean_absolute_error(true_values, predictions)
        mape = mean_absolute_percentage_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)

        metrics_data.append({
            'Method': method,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        })

    return pd.DataFrame(metrics_data)

