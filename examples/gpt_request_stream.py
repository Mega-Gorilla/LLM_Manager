import httpx
import asyncio

# 対象のURL
openai_post = "http://127.0.0.1:8000/requst/openai-post"
openai_get = "http://127.0.0.1:8000/requst/openai-get/queue"

async def request_GPT(prompt_name, user_prompt, stream):
    data = {
        "user_assistant_prompt": user_prompt,
        "variables": {}
    }
    # オプションのクエリパラメータ
    params = {
        "stream": stream  # または True
    }
    request_URL = (f"{openai_post}/{prompt_name}")
    
    print(data)
    async with httpx.AsyncClient() as client:
        response = await client.post(request_URL, json=data, params=params)
        return response

async def get_request(url):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    except httpx.HTTPError as http_err:
        if http_err.response.status_code == 404:
            print("404 Error occurred. Stopping the loop.")
            return None
        else:
            print(f"An HTTP error occurred: {http_err}")
            return None
    except Exception as err:
        print(f"An error occurred: {err}")
        return None

async def main():
    request_data = [{"user":"こんにちわ！みらい。"}]
    
    task = asyncio.create_task(request_GPT("mirai_old", request_data, True))
    
    while not task.done():
        result = await get_request(openai_get)
        if result is None:
            continue
        if result.get('finish_reason') == 'stop':
            break
        
        message = result.get('message')
        print(message, end='')
    
    result = await task
    
    print()
    print(result.text)

if __name__ == "__main__":
    asyncio.run(main())
