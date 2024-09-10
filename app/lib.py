import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt


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


def line_plot_with_legend(df, variables):
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
    st.pyplot(fig)