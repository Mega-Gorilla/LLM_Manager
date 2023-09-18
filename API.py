#API.py
from module.GPT_request import GPT_request
from module.rich_desgin import error
from rich import print
from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import Optional, Dict, List,Any
import os,json,shutil,re
import asyncio
import time

openai_key = os.getenv("OPENAI_API_KEY")
prompt_folder_path = "data"
app = FastAPI()

class Settings(BaseModel):
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    presence_penalty: float
    frequency_penalty: float
    #logit_bias: Optional[Dict[int, Any]] = None  # Noneかfloatの辞書

class TextItem(BaseModel):
    system: Optional[str] = None
    user: Optional[str] = None

#item class
class Prompts(BaseModel):
    title: str
    description: str
    text: List[TextItem]
    setting: Settings

class Test(BaseModel):
    title: str
    prompt_text: Optional[str] = None

class variables_dict(BaseModel):
    variables: dict = Field(default={})

class global_values:
    prompt_list = []
    stream_queue= asyncio.Queue()

@app.post("/prompts-post/new_prompt", tags=["Prompts"])
async def add_new_prompt(prompt: Prompts):
    await Create_or_add_json_data(prompt.title,prompt.description,prompt.text,prompt.setting)

@app.get("/prompts-get/list", tags=["Prompts"])
async def get_prompt_list():
    result = await get_prompts_list()
    global_values.prompt_list = result
    return result

@app.get("/prompts-get/names", tags=["Prompts"])
async def get_prompt_name():
    result = await get_prompts_list()
    global_values.prompt_list = result
    title_list = [d["title"] for d in result]
    return title_list

@app.get("/prompts-get/history/{prompt_name}", tags=["Prompts"])
async def get_history(prompt_name: str):
    result = await get_history(prompt_name)
    return result

@app.post("/requst/openai/{prompt_name}", tags=["OpenAI"])
async def OpenAI_request(prompt_name: str, item: variables_dict):
    responce = await GPT_request_API(prompt_name, item.variables)
    return responce

#Test
@app.post("/tests/")
async def Test_item(test: Test):
    print(test.title)
    print(test.prompt_text)

# jsonデータをリストで取得する
def get_file_list():
    all_files = os.listdir("data")
    json_files = [f for f in all_files if f.endswith('.json')]
    return json_files

#プロンプトリストの取得
async def get_prompts_list():
    json_file_list = get_file_list()
    result = []

    for json_file in json_file_list:
        with open(f"data/{json_file}", 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'history' in data:
                del data['history']
            result.append(data)
    return result

#プロンプト履歴の取得
async def get_history(name):
    result = None
    with open(f"data/{name}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'history' in data:
                result = data['history']
    return result

# Jsonデータを作成or編集する
async def Create_or_add_json_data(title,description=None,prompt_text=None,settings=None,history=None):
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
    if description is not None:
        json_data['description']=description

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

#GPTに問い合わせ実施
async def GPT_request_API(name,values):
    prompt_list = global_values.prompt_list
    filtered_list = [item for item in prompt_list if name.lower() == item['title'].lower()]
    if len(filtered_list) == 0:
        prompt_list=await get_prompts_list()
        global_values.prompt_list = prompt_list
        filtered_list = [item for item in prompt_list if name.lower() == item['title'].lower()]
    if len(filtered_list) == 0:
        return None
    filtered_list = filtered_list[0]

    text = filtered_list['text']
    for item in text:
        for key, value in item.items():
            item[key] = value.format(**values)
    response = await GPT_request().GPT_request(filtered_list['title'],
                              openai_key,
                              text,
                              filtered_list['setting']['temperature'],
                              filtered_list['setting']['max_tokens'],
                              filtered_list['setting']['model'])
    response['variables']= values
    response['prompt']= text
    await Create_or_add_json_data(name,history=response)
    return response["choices"][0]["message"]["content"]

async def main():
    #global_values.prompt_list = await get_prompts_list()
    process_time=time.time()
    await GPT_request_API("template",{"things":"weather"})
    print({(time.time())-process_time})

if __name__ == "__main__":
    speech_key = os.getenv("AZURE_API_KEY")
    speech_region = "japaneast"

    asyncio.run(main())