#GPT_request.py
import openai
import asyncio
import time,sys,json,os
from rich import print
from rich.console import Console
from module.rich_desgin import error

class GPT_request:
    
    async def add_to_queue(self,queue, producer_id, content):
        await queue.put({"ID": producer_id, "message": content})

    async def GPT_Stream(self,queue, producer_id, OpenAI_key,Prompt=[{"system":"You are a helpful assistant."},{"user":"Hello!"}],temp=0,tokens_max=2000,model="gpt-4",max_retries=3,debug=False):
        openai.api_key=OpenAI_key
        if debug:
            print(f'Start {model}')
        #OpenAIのエラーリスト
        gpt_error_mapping = {
            openai.error.APIError: ("OpenAI API Error", "しばらく時間をおいてからリクエストを再試行し、問題が解決しない場合は弊社までご連絡ください。", 'sleep'),
            openai.error.Timeout: ("OpenAI Timeout Error", "リクエストがタイムアウトしました。しばらく時間をおいてからリクエストを再試行し、問題が解決しない場合は弊社までご連絡ください。", 'sleep'),
            openai.error.RateLimitError: ("OpenAI Rate Limit Error", "リクエストのペースを上げてください。詳しくはレート制限ガイドをご覧ください。", 'exit'),
            openai.error.APIConnectionError: ("OpenAI API Connection Error", "ネットワーク設定、プロキシ設定、SSL証明書、またはファイアウォールルールを確認してください。", 'exit'),
            openai.error.InvalidRequestError: ("OpenAI API Invalid Request Error", "エラーメッセージは、具体的なエラーについてアドバイスするはずです。呼び出している特定のAPIメソッドのドキュメントを確認し、有効で完全なパラメータを送信していることを確認してください。また、リクエストデータのエンコーディング、フォーマット、サイズを確認する必要があるかもしれません。", 'exit'),
            openai.error.AuthenticationError: ("OpenAI Authentication Error", "APIキーまたはトークンを確認し、それが正しく、アクティブであることを確認してください。アカウントダッシュボードから新しいものを生成する必要があるかもしれません。", 'exit'),
            openai.error.ServiceUnavailableError: ("OpenAI Service Unavailable Error", "しばらく時間をおいてからリクエストを再試行し、問題が解決しない場合はお問い合わせください。ステータスページをご確認ください。", 'sleep')
        }

        Prompts=[]
        for original_dict in Prompt:
            transformed_dict = {}
            for key, value in original_dict.items():
                transformed_dict["role"] = key
                transformed_dict["content"] = value
            Prompts.append(transformed_dict)

        retry_count = 0

        while retry_count < max_retries:
            try:
                gpt_result = openai.ChatCompletion.create(
                    model=model,
                    messages=Prompts,
                    stream=True,
                    temperature=temp,
                    max_tokens=tokens_max,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                for chunk in gpt_result:
                    content = chunk["choices"][0].get("delta", {}).get("content")
                    if content is not None:
                        #print(content,end=None)
                        await queue.put({"ID": producer_id, "message": content})
                        await asyncio.sleep(0.01)
                break
            except Exception as e:
                title, message, action = gpt_error_mapping.get(type(e), ("OpenAI Unknown Error", "不明なエラーです。", 'exit'))
                print(e)
                e=str(e)+(f"\n\nRaw Prompt: {Prompt}\nProcessed Prompt: {Prompts}\nTemp: {temp}\nMax Tokens: {tokens_max}")
                error(title, message, e if action == 'exit' else None)
                
                if action == 'exit':
                    sys.exit(1)
                elif action == 'sleep':
                    await asyncio.sleep(1)

async def handle_results(queue):
    while True:
        result = await queue.get()
        print(f"Received result: {result}")
        queue.task_done()

async def main():
    starttime=time.time()
    queue = asyncio.Queue()
    #GPT_stream = GPT_request.GPT_Stream(queue,"GPT_stream",os.getenv("OPENAI_API_KEY"),model='gpt-3.5-turbo')
    gpt_instance = GPT_request()
    GPT_stream = gpt_instance.GPT_Stream(queue, "GPT_stream", os.getenv("OPENAI_API_KEY"), Prompt=[{'system': gpt_instance.old_mirai_prompt()},{'user':"こんばんわ"}])

    consumer_task = asyncio.create_task(handle_results(queue))
    producer_task = GPT_stream
    
    await producer_task

    await queue.join()

    consumer_task.cancel()

    print(f"Result Time:{time.time()-starttime}")

if __name__ == "__main__":
    asyncio.run(main())