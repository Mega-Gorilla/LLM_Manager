#API.py
from module.GPT_request import GPT_request
from module.rich_desgin import error
from rich import print
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Union,Any
import os,json,shutil,re
import openai
import asyncio
import time

openai_key = os.getenv("OPENAI_API_KEY")
prompt_folder_path = "data"
app = FastAPI()
chank_list = []
result_list = []

class Settings(BaseModel):
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    presence_penalty: float
    frequency_penalty: float
    #logit_bias: Optional[Dict[int, Any]] = None  # Noneかfloatの辞書

#item class
class Prompts(BaseModel):
    title: str
    prompt_text: Optional[str] = Field(None, alias="promptText")
    setting: Settings

class Test(BaseModel):
    title: str
    prompt_text: Optional[str] = None

#Get Job Status
@app.post("/prompts/")
async def add_new_prompt(prompt: Prompts):
    await Create_or_add_json_data(prompt.title,prompt.prompt_text,prompt.setting)

#Get Job Status
@app.post("/tests/")
async def Test_item(test: Test):
    print(test.title)
    print(test.prompt_text)

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
    if json_file_name not in json_file_list:
        #jsonファイルが存在しない場合新規作成する。
        tempfilepath=os.path.join(prompt_folder_path,"template.json")
        if not os.path.exists(tempfilepath):
            error("template.json is Not Found.","[template.json] file not found in the [data] folder.")
            exit(1)
        shutil.copy(tempfilepath,json_file_path)

    #jsonファイルを読み込む
    with open(json_file_path, "r", encoding='utf-8') as json_file:
        json_data = json.load(json_file)

    #データの書き込み
    json_data['title']=title
    if prompt_text is not None:
        json_data['text']=prompt_text
        #variables 設定
        placeholder_dict = {}
        placeholders = re.findall(r'{(.*?)}', prompt_text)
        for placeholder in placeholders:
            placeholder_dict[placeholder] = ""
        json_data['variables'] = placeholder_dict

    if settings is not None:
        settings_dict = settings.dict()
        for key, value in settings_dict.items():
            json_data['setting'][key]= value
    
    if history is not None:
        json_data['history'].append(history)
    
    with open(json_file_path, "w", encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)

# result_queueを監視し、新しい関数が入力されたら処理するプロセス
async def handle_results(result_queue,task_queue):

    while True:
        result = await result_queue.get()
        #print(f"Received result: {result}")
        print(result["message"],end=None)

        await asyncio.sleep(0.1)


async def main():
    setting= {
        "model": "testmodel",
        "temperature": 0,
        "top_p": 0,
        "max_tokens": 0,
        "presence_penalty": 0,
        "frequency_penalty": 0
    }
    await Create_or_add_json_data("test_title","tsetPrompt",setting)

if __name__ == "__main__":
    speech_key = os.getenv("AZURE_API_KEY")
    speech_region = "japaneast"

    asyncio.run(main())