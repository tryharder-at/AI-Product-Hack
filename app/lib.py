import matplotlib.pyplot as plt
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
    columns_to_sum = ['cnt', 'CASHBACK_STORE_1', 'CASHBACK_STORE_2', 'CASHBACK_STORE_3', 'sell_price']
    if granularity == 'Day':
        return df
    elif granularity == 'Week':
        grouped_df = df.groupby(pd.Grouper(key=date_column, freq='W'))[columns_to_sum].sum().reset_index()
    elif granularity == 'Month':
        grouped_df = df.groupby(pd.Grouper(key=date_column, freq='M'))[columns_to_sum].sum().reset_index()
    else:
        raise ValueError("Неверно задана гранулярность. Используйте 'Day', 'Week' или 'Month'.")
    return grouped_df

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
