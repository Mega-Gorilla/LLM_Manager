import streamlit as st
import os
import sys
import subprocess
import requests
import time
import ast
import json

LLM_model_name=["gpt-3.5-turbo", "gpt-4","gpt-4-1106-preview","gemini-pro"]
BASE_URL = "http://127.0.0.1:8000"

Deepl_API_key = os.getenv("DEEPL_API_KEY")

def translate_with_deepl(text,api_key,target_language='JA',source_language='EN'):
    base_url = "https://api-free.deepl.com/v2/translate"
    
    data = {
        'auth_key' : api_key,
        'text' : text,
        'source_lang' : source_language, # 翻訳対象の言語
        "target_lang": target_language  # 翻訳後の言語
    }
    
    response = requests.post(base_url, data=data)
    response_data = response.json()
    print(response_data)
    if "translations" in response_data:
        return response_data["translations"][0]["text"]
    else:
        return False

def fetch_data_from_api():
    response = requests.get(f"{BASE_URL}/prompts-get/get_prompt_metadata")
    response.raise_for_status()
    return response.json()

def edit_prompt_from_api(send_data,object):
    json_data= json.dumps(send_data, indent=4)
    response = requests.post(f"{BASE_URL}/prompts-post/add_edit_prompt", json_data)
    response=response.json()
    if response!=None:
        error = object.warning(f"Error: {response}")
    else:
        success = object.success(f"Data was saved.")
        time.sleep(1)
        success.empty()
    return response

def submit_openai(prompt_name,prompt,variables,object):
    print(f"data:  {variables}")
    json_data= {
            "variables": variables
            }
    json_data= json.dumps(json_data, indent=4)
    response = requests.post(f"{BASE_URL}/LLM/request/?prompt_name={prompt_name}&stream_mode=false", json_data)
    response = response.json()
    if response['ok']== True:
        success = object.info(f"GPT Request Done.")
        time.sleep(1)
        success.empty()
    else:
        error = object.warning(f"Error: {response}")
        response = None

def show_sidebar_buttons(data):
    def apply_css():
        st.markdown("""
        <style>
            .stButton>button {
                text-align: left;
            }
        </style>
        """, unsafe_allow_html=True)
    st.sidebar.title("GPT Manager")
    st.sidebar.markdown("[/API Doc/](http://127.0.0.1:8000/docs)\n* * *")
    if st.sidebar.button("Create New Prompt",type="primary",use_container_width=True):
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
                "variables":{},
                "history":[]
                    }
    
    st.sidebar.markdown("Select Prompt:")
    apply_css()
    for item in data:
        # キー 'title' が存在しない場合はデフォルト値を設定する
        title = item.get("title", "No Title")
        button_label = f"**{title}**"
        if st.sidebar.button(button_label, key=title,use_container_width=True):
            st.session_state.selected_item = item

def display_selected_item_details():
    title = st.session_state.selected_item['title']
    dscription = st.session_state.selected_item['description']
    setting_dict = st.session_state.selected_item['setting']
    text_dict = st.session_state.selected_item['text']
    variables_dict = st.session_state.selected_item['variables']
    history_dict = st.session_state.selected_item['history']
    if 'other' in st.session_state.get('selected_item', {}) and 'translate_text' in st.session_state.selected_item['other']:
        translate_text = st.session_state.selected_item['other']['translate_text']
    else:
        translate_text = None

    #タイトル・説明表示表示
    if title == "":
        title = st.text_input('Title', 'Prompt Name')
        dscription = st.text_input('Dscription', 'Prompt dscription')
        
    else:
        st.markdown(f"""
        ## <u>{title}</u>
        **{dscription}**
        """, unsafe_allow_html=True)
    
    #Display Prompt
    st.markdown("* * *\n**Prompt:**")
    prompt_data = display_prompt_text(text_dict)

    #Translate Prompt
    translate_button_clicked = st.button('Translate JP',use_container_width=True)
    if translate_text !=None:
        st.markdown("**Translate Prompt:**")
        translate_text = display_prompt_text(translate_text)

    #variables
    if variables_dict != {}:
        st.markdown("* * *\n**Variable:**")
        variables_data = display_prompt_text(variables_dict)
    else:
        variables_data = {}

    #Settings
    st.markdown("* * *\n**Settings:**")
    setting_model = st.selectbox(f"Model: {setting_dict['model']}",LLM_model_name,LLM_model_name.index(setting_dict['model']))
    setting_temp = st.slider(f"Temperature: {setting_dict['temperature']}",0.0,2.0,float(setting_dict['temperature']),0.01)
    setting_length = st.slider(f"Maximum length: {setting_dict['max_tokens']}",1,8191,int(setting_dict['max_tokens']),1)
    setting_topP = st.slider(f"Top P: {setting_dict['top_p']}",0.0,1.0,float(setting_dict['top_p']),0.01)
    setting_frequency = st.slider(f"Frequency penalty: {setting_dict['frequency_penalty']}",0.0,2.0,float(setting_dict['top_p']),0.01)
    setting_penalty = st.slider(f"Presence Penalty: {setting_dict['presence_penalty']}",0.0,2.0,float(setting_dict['presence_penalty']),0.01)

    #保存・問合せボタン
    st.markdown("* * *")
    save_button_clicked = st.button('Save',use_container_width=True)
    submit_buttion_clicked = st.button('Submit',use_container_width=True,type="primary")
    submit_info = st.empty()

    #submitデータ
    st.markdown("* * *")
    submit_result = st.empty()

    #history
    if history_dict != []:
        history_dict = history_dict[::-1]
        st.markdown("* * *\n**History**")
        for history in history_dict:
            expander_title = [f"{key} : {value}" for key, value in history['variables'].items()]
            with st.expander(" / ".join(expander_title)[:100]):
                st.markdown(f"### Responce\n```\n{history['content']}\n```")
                st.markdown(f"### Prompt")
                for key,value in history['prompt'][0].items():
                    st.markdown(f"**{key}:**")
                    st.markdown(f"```\n{value}\n```")
                st.markdown("* * *")
                formatted = [f"- **{item.split(' : ')[0]}:** {item.split(' : ')[1]}" for item in expander_title]
                formatted = '\n'.join(formatted)
                st.markdown("### Variables\n"+formatted)
                st.markdown(f"""
                            ### Info Data
                            - **Model:** `{history['model']}`
                            - **Finish Reason:** `{history['finish_reason']}`
                            - **process_time:** `{history['process_time']}`
                            - **Usage:**
                                - **prompt_tokens:** `{history['prompt_tokens']}`
                                - **completion_tokens:** `{history['completion_tokens']}`
                                - **total_tokens:** `{history['total_tokens']}`
                            """)
                st.markdown("### RAW Data")
                st.json(history,expanded=False)

    # ボタンリスト
    if translate_button_clicked:
        translate_text={}
        for key, value in text_dict.items():
            text=translate_with_deepl(value,Deepl_API_key)
            if text==None:
                text = "Transrate Error:"
            translate_text[key] = text
        save_button_clicked = True

    if save_button_clicked:
        setting_dict = {'model': setting_model, 'temperature': setting_temp, 'top_p': setting_topP, 'max_tokens': setting_length, 'presence_penalty': setting_penalty, 'frequency_penalty': setting_frequency}
        submit_data = {"title":title , "description":dscription}
        submit_data |= {'texts':prompt_data}
        submit_data |= {'setting':setting_dict}
        submit_data |= {'variables':variables_data}
        submit_data |= {'other':{'translate_text':translate_text}}
        edit_prompt_from_api(submit_data,submit_info)

    if submit_buttion_clicked:
        submit_info.info('Connecting to the server. Please wait...')
        submit_openai(title,[],variables_data,submit_info)

def display_prompt_text(text_dict):
    text_box_value_dict = {}

    for key, value in text_dict.items():
        height_value = max(value.count('\n') * 35, 68)  # 最低68ピクセルを保証
        text_box_value = st.text_area(label=key, value=value, height=height_value)
        text_box_value_dict[key] = text_box_value
    return text_box_value_dict

if __name__ == "__main__":
    # すでに自動再起動済みか（フラグがあるか）と、
    # Streamlit実行時にセットされる環境変数 STREAMLIT_SERVER_PORT の有無で判定
    if os.getenv("STREAMLIT_AUTO_RUN") is None and os.getenv("STREAMLIT_SERVER_PORT") is None:
        print("このアプリは 'streamlit run' で実行する必要があります。自動的に再起動します...")
        # 環境変数にフラグをセットして再実行する
        new_env = os.environ.copy()
        new_env["STREAMLIT_AUTO_RUN"] = "true"
        subprocess.run(["streamlit", "run", sys.argv[0]] + sys.argv[1:], env=new_env)
        sys.exit()
    
    # 初期化
    if "selected_item" not in st.session_state:
        st.session_state.selected_item = None
    
    prompts = fetch_data_from_api()
    show_sidebar_buttons(prompts)

    # 選択されたデータの詳細を表示
    if st.session_state.selected_item:
        display_selected_item_details()