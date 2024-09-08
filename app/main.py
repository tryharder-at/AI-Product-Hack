import streamlit as st

hide_decoration_bar_style = '''<style>header {visibility: hidden;}</style>'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
hide_streamlit_footer = """<style>#MainMenu {visibility: hidden;}
                        footer {visibility: hidden;}</style>"""
st.markdown(hide_streamlit_footer, unsafe_allow_html=True)


def main_interface() -> None:
    load_options, datasets = dict(), dict()
    file = st.sidebar.file_uploader(
        "Upload a csv file", type="csv",
    )
    load_options["separator"] = st.sidebar.selectbox(
        "What is the separator?", [",", ";", "|"],
    )
    load_options["date_format"] = st.sidebar.text_input(
        "What is the date format?",
    )


if __name__ == '__main__':
    main_interface()
