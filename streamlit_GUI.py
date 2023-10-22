import streamlit as st
import os
import requests
import time

def fetch_data_from_api():
    response = requests.get("http://127.0.0.1:8000/prompts-get/get_prompt_metadata")
    response.raise_for_status()
    return response.json()

def show_sidebar_buttons(data):
    for item in data:
        button_label = f"💬**{item['title']}**\n\n{item['description']}"
        if st.sidebar.button(button_label, key=item['title']):
            st.session_state.selected_item = item

def display_selected_item_details():
    title = st.session_state.selected_item['title']
    dscription = st.session_state.selected_item['description']
    setting_dict = st.session_state.selected_item['setting']
    test_dict = st.session_state.selected_item['text']
    variables_dict = st.session_state.selected_item['variables']
    st.markdown(f"""
    ## <u>{title}</u>
    **{dscription}**
    """, unsafe_allow_html=True)
    st.markdown("* * *")

    settings_string = " <BR> ".join(f"**{key}**: {value}" for key, value in setting_dict.items())
    st.markdown(settings_string, unsafe_allow_html=True)
    st.markdown("* * *")

    prompt_data = display_prompt_text(test_dict)

    st.markdown("* * *")
    variables_dict = st.text_area(label='Variables',value=variables_dict)

    if st.button('Submit'):
        submit_data = {"title":title , "description":dscription} | {'text':prompt_data} | {'setting':setting_dict} | {'variables':variables_dict}
        print(submit_data)

def display_prompt_text(test_dict):
    text_box_value_dict = {}
    for key, value in test_dict.items():
        text_box_value = st.text_area(label=key, value=value,height=value.count('\n')*35)
        text_box_value_dict[key] = text_box_value
    return text_box_value_dict

def prompt_change(key, chage_value):
    print(f"Change {key} : {chage_value}")
    success = st.success(f"{key} data was saved.")
    time.sleep(1)
    success.empty()

if __name__ == "__main__":
    # 初期化
    if 'selected_item' not in st.session_state:
        st.session_state.selected_item = None
    
    prompts = fetch_data_from_api()
    show_sidebar_buttons(prompts)

    # 選択されたデータの詳細を表示
    if st.session_state.selected_item:
        display_selected_item_details()

    if not os.environ.get("STREAMLIT_RUN"):
        os.environ["STREAMLIT_RUN"] = "true"
        os.system(f"streamlit run {os.path.abspath(__file__)}")
