import requests
import keyboard

# 対象のURL
url = "http://127.0.0.1:8000/requst/openai-post"


def request_mirai_example(user_prompt):
    # 送信するJSONデータ
    data = {
        "user_assistant_prompt": user_prompt,
        "variables": {}
    }

    # POSTリクエストを送信
    response = requests.post(url+"/mirai_old", json=data)
    return response.text

if __name__ == "__main__":
    request_data = []
    while True:
        # ESCキーが押されたかどうかを確認
        if keyboard.is_pressed('esc'):
            break
        # 通常のinputの処理
        data = input("Enter Your text (Press ESC to exit): ")
        print(f"User: {data}")
        request_data.append({"user": data})

        # OpenAI問い合わせ実行
        result = request_mirai_example(request_data)
        print(f"Mirai: {result}")

        # メモリ保存
        request_data.append({"assistant": result})