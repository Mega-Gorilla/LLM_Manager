import requests

# 対象のURL
url = "http://127.0.0.1:8000/requst/openai-post"


def request_GPT(prompt_name,user_prompt):
    # 送信するJSONデータ
    data = {
        "user_assistant_prompt": user_prompt,
        "variables": {}
    }

    # POSTリクエストを送信
    request_URL=(f"{url}/{prompt_name}")
    response = requests.post(request_URL, json=data)
    return response

if __name__ == "__main__":
    request_data = [{"user":"こんにちわ！みらい。"}]
    
    #OpenAI問い合わせ実施
    result = request_GPT("mirai_old",request_data)

    #結果を表示
    print(result.text)