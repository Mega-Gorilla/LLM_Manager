import streamlit as st
import os
import requests
import time
import ast
import json

LLM_model_name=["gpt-3.5-turbo", "gpt-4"]

def fetch_data_from_api():
    response = requests.get("http://127.0.0.1:8000/prompts-get/get_prompt_metadata")
    response.raise_for_status()
    return response.json()

def edit_prompt_from_api(send_data):
    json_data= json.dumps(send_data, indent=4)
    response = requests.post("http://127.0.0.1:8000/prompts-post/add_edit_prompt", json_data)
    response=response.json()
    if response!=None:
        error = st.warning(f"Save Error.\nReason: {response}")
    else:
        success = st.success(f"Data was saved.")
        time.sleep(1)
        success.empty()
    return response

def show_sidebar_buttons(data):
    if st.sidebar.button("New Prompt",type="primary"):
        st.session_state.selected_item = {
            "title": "",
            "description": "",
            "text": {
                "system": ""
                },
                "setting": {
                    "model": "gpt-3.5-turbo",
                    "temperature": 1.0,
                    "top_p": 1,
                    "max_tokens": 256,
                    "presence_penalty": 0,
                    "frequency_penalty": 0
                    },
                "variables":{}
                    }
    for item in data:
        button_label = f"**{item['title']}**\n\n{item['description']}"
        if st.sidebar.button(button_label, key=item['title']):
            st.session_state.selected_item = item

def display_selected_item_details():
    title = st.session_state.selected_item['title']
    dscription = st.session_state.selected_item['description']
    setting_dict = st.session_state.selected_item['setting']
    test_dict = st.session_state.selected_item['text']
    variables_dict = st.session_state.selected_item['variables']

    if title == "":
        title = st.text_input('Title', 'Prompt Name')
        dscription = st.text_input('Dscription', 'Prompt dscription')
        st.markdown("* * *")
    else:
        st.markdown(f"""
        ## <u>{title}</u>
        **{dscription}**
        """, unsafe_allow_html=True)
    
    #Settings
    setting_model = st.selectbox(f"Model: {setting_dict['model']}",LLM_model_name,LLM_model_name.index(setting_dict['model']))
    setting_temp = st.slider(f"Temperature: {setting_dict['temperature']}",0.0,2.0,float(setting_dict['temperature']),0.01)
    setting_length = st.slider(f"Maximum length: {setting_dict['max_tokens']}",1,8191,int(setting_dict['max_tokens']),1)
    setting_topP = st.slider(f"Top P: {setting_dict['top_p']}",0.0,1.0,float(setting_dict['top_p']),0.01)
    setting_frequency = st.slider(f"Frequency penalty: {setting_dict['frequency_penalty']}",0.0,2.0,float(setting_dict['top_p']),0.01)
    setting_penalty = st.slider(f"Presence Penalty: {setting_dict['presence_penalty']}",0.0,2.0,float(setting_dict['presence_penalty']),0.01)
    st.markdown("* * *")

    #Display Prompts
    prompt_data = display_prompt_text(test_dict)
    st.markdown("* * *")

    variables_dict = st.text_area(label='Variables(未実装)',value=variables_dict)

    if st.button('Submit'):
        setting_dict = {'model': setting_model, 'temperature': setting_temp, 'top_p': setting_topP, 'max_tokens': setting_length, 'presence_penalty': setting_penalty, 'frequency_penalty': setting_frequency}
        submit_data = {"title":title , "description":dscription} | {'texts':prompt_data} | {'setting':setting_dict}
        
        response = edit_prompt_from_api(submit_data)

def display_prompt_text(test_dict):
    text_box_value_dict = {}
    for key, value in test_dict.items():
        text_box_value = st.text_area(label=key, value=value,height=value.count('\n')*35)
        text_box_value_dict[key] = text_box_value
    return text_box_value_dict

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
