import google.generativeai as genai
import os
import requests
import json

def tokenizer(text):
    api_key = os.getenv("GEMINI_API_KEY")
    url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:countTokens?key=' + api_key
    # POSTするデータ
    data = {
        "contents": [{
            "parts": [{
                "text": text
            }]
        }]
    }
    # リクエストヘッダ
    headers = {
        'Content-Type': 'application/json'
    }
    # POSTリクエストの送信
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_data = json.loads(response.text)
    return response_data['totalTokens']

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel(model_name='gemini-pro')
chat = model.start_chat()
prompt_text = '私は、あなたがどれぐらい長い文章を生成できるかテストしたいです。可能な限り長い物語を生成してください。'
prompt_token = tokenizer(prompt_text)
response = chat.send_message('私は、あなたがどれぐらい長い文章を生成できるかテストしたいです。可能な限り長い物語を生成してください。',generation_config={ "max_output_tokens": 500,"temperature": 1,"top_p": 1})
print("-" * 50)
print(response.text)
print("-" * 50)
output_token = tokenizer(response.text)
print(f"prompt Token:{prompt_token}\noutput Token: {output_token}\nTotal_token: {prompt_token+output_token}")
print("-" * 50)
print(response.__dict__)