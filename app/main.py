import lib
import pandas as pd
import streamlit as st


hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
hide_streamlit_footer = """<style>#MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)


def main_interface() -> None:
    st.sidebar.title("Load options")
    load_options: dict[str, str] = {
        "sep": st.sidebar.selectbox(
            "What is the separator?", [",", ";", "|"],
        ),
        "date_format": st.sidebar.text_input(
            "What is the date format?",
        )
    }

    main_file = st.sidebar.file_uploader(
        "Upload a shop sales file", type="csv", accept_multiple_files=False,
    )
    sales_prices_file = st.sidebar.file_uploader(
        "Upload a shop sales prices file", type="csv", accept_multiple_files=False
    )
    sales_date_file = st.sidebar.file_uploader(
        "Upload a shop sales date file", type="csv", accept_multiple_files=False
    )

    test_data: bool = st.sidebar.checkbox("Test data*")

    if test_data:
        main_file = "test_data/shop_sales.csv"
        sales_prices_file = "test_data/shop_sales_prices.csv"
        sales_date_file = "test_data/shop_sales_dates.csv"

    if all(files := [main_file, sales_prices_file, sales_date_file]):
        files = [pd.read_csv(file, **load_options) for file in files]
        dataset: pd.DataFrame = lib.merge_files_to_dataset(*files)
        # if all files are uploaded, start pipeline
        pipeline(dataset)


def pipeline(dataset: pd.DataFrame) -> None:
    with st.expander("Dataset"):
        st.write(dataset)
    feature = st.selectbox("Select feature", dataset.columns)
    # fig = lib.line_plot_with_legend(dataset, [feature])
    # st.pyplot(fig)

    item_ids: list[str] = st.multiselect(
        "Select item_id",
        sorted(set(dataset.item_id)),
        max_selections=1,
        default=['STORE_2_714']
    )
    if not item_ids:
        return
    granularity = st.selectbox("Select granularity", ['Day', 'Week', 'Month'])
    dataset = lib.summ_sales_data(dataset, 'date', granularity=granularity)
    dataset = lib.df_encoding(dataset)

    horizont = st.slider("Select horizont", 1, 10, 1)
    with st.spinner("Please wait"):
        data_prediction, model = lib.get_preds(dataset, item_ids, horizont)
        st.write(data_prediction)
        fig = lib.line_plot_with_legend(data_prediction, [f'cnt_SMA_3_lag_{horizont}', 'cnt', 'model_prediction'])
        st.pyplot(fig)

        st.write(lib.evaluate_predictions(data_prediction, horizont))
        fig = lib.forecast_plot_from_df(data_prediction, 'date', 'cnt', ['cnt_SMA_3_lag_1', 'arima_prediction', 'model_prediction'])
        st.pyplot(fig)


if __name__ == '__main__':
    main_interface()
