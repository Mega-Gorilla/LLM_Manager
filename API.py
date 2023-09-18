#API.py
from module.GPT_request import GPT_request
from module.rich_desgin import error
from rich import print
from fastapi import FastAPI, Body,HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List,Any,Union
import os,json,shutil,re
import asyncio
import time

openai_key = os.getenv("OPENAI_API_KEY")
prompt_folder_path = "data"
app = FastAPI()

class Settings(BaseModel):
    model: str = Field(default="gpt-3.5-turbo")
    temperature: int = Field(default=1)
    top_p: int = Field(default=1)
    max_tokens: int = Field(default=500)
    presence_penalty: int = Field(default=0)
    frequency_penalty: int = Field(default=0)
    #logit_bias: Optional[Dict[int, Any]] = None  # Noneかfloatの辞書

#item class
class Prompts(BaseModel):
    title: str = Field(default="Prompt Name")
    description: str= Field(default="Prompt Description")
    texts: List[Dict[str, str]] = Field(default={"system":"You are a helpful assistant.And you need to advise about the {things}.","user": "Hello!"})
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
    result = await get_prompts_list()
    global_values.prompt_list = result
    title_list = [d["title"] for d in result]
    if prompt.title in title_list:
        raise HTTPException(status_code=400, detail="Title already exists.")
    
    print(f"{prompt.title}\n{prompt.description}\n{prompt.texts}\n{prompt.setting}")
    
    await Create_or_add_json_data(prompt.title, prompt.description, prompt.texts, prompt.setting)

@app.get("/prompts-get/list", tags=["Prompts"])
async def get_prompt_list():
    result = await get_prompts_list()
    global_values.prompt_list = result
    return result

@app.get("/prompts-get/names", tags=["Prompts"])
async def get_prompt_name():
    result = await get_prompts_list()
    global_values.prompt_list = result
    new_dict = {}
    for item in result:
        title = item.get('title')
        description = item.get('description')
        if title and description:  # titleとdescriptionが存在する場合のみ追加
            new_dict[title] = description
    return new_dict

@app.get("/prompts-get/history/{prompt_name}", tags=["Prompts"])
async def get_history(prompt_name: str):
    result = await get_history(prompt_name)
    return result

@app.post("/requst/openai/{prompt_name}", tags=["OpenAI"])
async def OpenAI_request(prompt_name: str, item: variables_dict = None):
    if prompt_name == "template":
        raise HTTPException(status_code=400, detail="Editing Template.json is prohibited")
    responce = await GPT_request_API(prompt_name, item.variables)
    return responce

@app.post("/requst/openai/stream/{prompt_name}", tags=["OpenAI"])
async def OpenAI_request(prompt_name: str, item: variables_dict = None):
    if prompt_name == "template":
        raise HTTPException(status_code=400, detail="Editing Template.json is prohibited")
    responce = await GPT_request_API(prompt_name, item.variables,global_values.stream_queue)
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
        
        for key, value in prompt_text.items():
            if isinstance(value, str):  # この例ではstr型だけを対象としています
                placeholders = re.findall(r'{(.*?)}', value)
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
        json.dump(json_data, json_file, indent=4,ensure_ascii=False)

#GPTに問い合わせ実施
async def GPT_request_API(name,values,queue=None):

    prompt_list = global_values.prompt_list
    filtered_list = [item for item in prompt_list if name.lower() == item['title'].lower()]
    if len(filtered_list) == 0:
        prompt_list=await get_prompts_list()
        global_values.prompt_list = prompt_list
        filtered_list = [item for item in prompt_list if name.lower() == item['title'].lower()]
    if len(filtered_list) == 0:
        return None
    filtered_list = filtered_list[0]

    text = []
    for key, value in filtered_list['text'].items():
        if isinstance(value, str):
            # 文字列内のプレースホルダー（{xxx}）を見つける
            placeholders = re.findall(r'{(.*?)}', value)
            
            # values がすべてのプレースホルダーに対応するキーを持っているか確認
            if all(placeholder in values for placeholder in placeholders):
                if values:
                    value = value.format(**values)
            else:
                print(f"Warning: Missing keys for placeholders in '{value}'")

            text.append({key: value})
    if queue is None:
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
    else:
        timestamp=filtered_list['title']+" - "+str(time.time())
        response = await GPT_request().GPT_request_stream(queue,
                                                timestamp,
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