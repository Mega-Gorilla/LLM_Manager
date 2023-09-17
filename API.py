#API.py
from module.GPT_request import GPT_request
from module.rich_desgin import error
from rich import print
from fastapi import FastAPI
from pydantic import BaseModel
import os,json,shutil,re
import openai
import asyncio
import time

openai_key = os.getenv("OPENAI_API_KEY")
prompt_folder_path = "data"
app = FastAPI()
chank_list = []
result_list = []

#Get Job Status
@app.get("/jobs/{job_name}")
async def read_job_status(job_name: str):
    return {job_name: job_manager.get_status(job_name)}
#Get job list
@app.get("/job_list/")
async def read_job_list():
    return job_manager.get_job_list()

# 
def get_file_list():
    all_files = os.listdir("data")
    json_files = [f for f in all_files if f.endswith('.json')]
    return json_files

# Jsonデータを作成する
async def Create_or_add_json_data(title,prompt_text=None,settings=None,history=None):
    json_file_list = get_file_list()
    json_file_name = title + ".json"
    json_file_path = os.path.join(prompt_folder_path,json_file_name)
    
    if json_file_list not in json_file_name:
        #jsonファイルが存在しない場合新規作成する。
        tempfilepath=os.path.join(prompt_folder_path,"template.json")
        if os.path.exists(tempfilepath):
            error("template.json is Not Found.","[template.json] file not found in the [data] folder.")
            exit(1)
        shutil.copy(tempfilepath,json_file_path)

    #jsonファイルを読み込む
    with open(json_file_path, "r", encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    #データの書き込み
    if prompt_text is not None:
        json_data['text']=prompt_text
        #variables 設定
        placeholder_dict = {}
        placeholders = re.findall(r'{(.*?)}', prompt_text)
        for placeholder in placeholders:
            placeholder_dict[placeholder] = ""
        json_data['variables'] = placeholder_dict

    if settings is not None:
        for key, value in settings.items():
            json_data['setting'][key]= value
    
    if history is not None:
        json_data['history'].append(history)
    
    with open(json_file_path, "w", encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)

#新しいasyncioタスクを作成するプロセス
async def create_tasks(task_queue):
    #新しいタスクを作成
    if task_queue.qsize()!=0:
        tasks= await task_queue.get()
        if asyncio.iscoroutine(tasks[1]):
            asyncio.create_task(tasks[1])
        else:
            for task in tasks:
                asyncio.create_task(task[1])

async def Create_StreamGPT_task(task_queue,result_queue,producer_id,openai_key,prompt,temp=1,tokens_max=2000,model_name='gpt-4',max_retries=3,debug=False):
    gpt_instance = GPT_request()
    GPT_stream = gpt_instance.GPT_Stream(result_queue, 
                                        producer_id, 
                                        openai_key, 
                                        prompt,
                                        temp,
                                        tokens_max,
                                        model_name,
                                        max_retries,
                                        debug)
    await task_queue.put(["GPT_stream",GPT_stream])

# result_queueを監視し、新しい関数が入力されたら処理するプロセス
async def handle_results(result_queue,task_queue):

    while True:
        result = await result_queue.get()
        #print(f"Received result: {result}")
        print(result["message"],end=None)

        await asyncio.sleep(0.1)


async def main():
    task_queue = asyncio.Queue(maxsize=10)
    result_queue = asyncio.Queue(maxsize=100)

    speech_task = None

    # Start a task to handle results
    asyncio.create_task(handle_results(result_queue,task_queue))

    while True:
        
        await create_tasks(task_queue)
            
        await asyncio.sleep(0.5)

if __name__ == "__main__":
    speech_key = os.getenv("AZURE_API_KEY")
    speech_region = "japaneast"

    asyncio.run(main())