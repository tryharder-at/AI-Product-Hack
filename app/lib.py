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
    if granularity == 'Day':
        return df
    elif granularity == 'Week':
        resample_rule = 'W' 
    elif granularity == 'Month':
        resample_rule = 'M'
    
    grouped = df.set_index(date_column).resample(resample_rule).agg({
        'cnt': 'sum',
        'sell_price': 'mean',
        'CASHBACK_STORE_1': lambda x: x.sum() / len(x),
        'CASHBACK_STORE_2': lambda x: x.sum() / len(x),
        'CASHBACK_STORE_3': lambda x: x.sum() / len(x),
        'item_id': 'first', 
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
