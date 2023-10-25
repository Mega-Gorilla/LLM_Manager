#API.py
from module.GPT_request import GPT_request
from module.rich_desgin import error
from rich import print
from fastapi import FastAPI, Request,HTTPException,Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder
from typing import Optional, Dict, List,Any,Union
from datetime import datetime,date
import os, json, shutil, re, csv
import asyncio
import time

openai_key = os.getenv("OPENAI_API_KEY")
prompt_folder_path = "data"
app = FastAPI()

class Settings(BaseModel):
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=1.00)
    top_p: float = Field(default=1)
    max_tokens: int = Field(default=500)
    presence_penalty: float = Field(default=0.00)
    frequency_penalty: float = Field(default=0.00)
    #logit_bias: Optional[Dict[int, Any]] = None  # Noneかfloatの辞書

#item class
class Prompts(BaseModel):
    title: str = Field(default="Prompt Name")
    description: str= Field(default="Prompt Description")
    texts: dict = Field(default={"system":"You are a helpful assistant.And you need to advise about the {things}."})
    setting: Settings
    other: dict

class variables_dict(BaseModel):
    user_assistant_prompt: List[Dict[str, str]] = Field(default=[{"user": "こんにちわ!みらい"}])
    variables: dict = Field(default={})

class GlobalValues:
    prompt_list = []
    stream_queue= asyncio.Queue()
    custom_value: List[Dict] = []
    custom_value_unity: List[Dict] = []
    
@app.exception_handler(404)
async def not_found_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=404,
        content={"detail": "The resource you are looking for was not found."},
    )

@app.post("/prompts-post/add_edit_prompt", tags=["Prompts"])
async def add_new_prompt(prompt: Prompts):
    result = await get_prompts_list()
    GlobalValues.prompt_list = result
    title_list = [d["title"] for d in result]
    if prompt.title in title_list:
        print("A prompt with the same name already exists. Overwriting process has been executed.")
        #raise HTTPException(status_code=400, detail="Title already exists.")
    print(prompt)
    await Create_or_add_json_data(prompt.title, prompt.description, prompt.texts, prompt.setting, other=prompt.other)

@app.get("/prompts-get/get_prompt_metadata", tags=["Prompts"])
async def get_all_prompts_data():
    result = await get_prompts_list()
    GlobalValues.prompt_list = result
    print("PromptsData:")
    print(result)
    return result

@app.get("/prompts-get/get_prompt_details", tags=["Prompts"])
async def get_all_prompts_names():
    result = await get_prompts_list()
    GlobalValues.prompt_list = result
    new_dict = {}
    for item in result:
        title = item.get('title')
        description = item.get('description')
        if title and description:  # titleとdescriptionが存在する場合のみ追加
            new_dict[title] = description
    print("{Name: Description}:")
    print(new_dict)
    return new_dict

@app.get("/prompts-get/lookup_prompt_by_name", tags=["Prompts"])
async def get_prompt_data_by_name(prompt_name: str):
    result = await get_prompts_list(prompt_name)
    GlobalValues.prompt_list = result
    print(result)
    return result

@app.get("/prompts-get/history/{prompt_name}", tags=["Prompts"])
async def get_history(prompt_name: str):
    result = await get_history(prompt_name)
    print(result)
    return result

@app.get("/cost-get/day/", tags=["Cost"])
async def get_cost_day(day: date=Query(default=datetime.now().strftime("%Y-%m-%d"))):
    # 指定された日付のtotal_tokensの合計値
    model_summary = {}

    with open("data/cost.csv", "r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            timestamp_date = timestamp.strftime("%Y-%m-%d")
            model_name = row["model_name"]

            # 指定された日付と一致するレコードの場合、各トークンを加算
            if timestamp_date == str(day):
                if model_name not in model_summary:
                    model_summary[model_name] = {"prompt_tokens_sum": 0, "completion_tokens_sum": 0}

                model_summary[model_name]["prompt_tokens_sum"] += int(row["prompt_tokens"])
                model_summary[model_name]["completion_tokens_sum"] += int(row["completion_tokens"])

    print(f"{day}: {model_summary}")
    return {"day": str(day), "model_summary": model_summary}

@app.post("/request/openai-post/{prompt_name}", tags=["OpenAI"])
async def OpenAI_request(prompt_name: str, value: variables_dict = None, stream: bool=False):
    if prompt_name == "template":
        raise HTTPException(status_code=400, detail="Editing Template.json is prohibited.\nPlease enter another json file.")
    
    if stream:
        responce = await GPT_request_API(prompt_name, value.user_assistant_prompt,value.variables, GlobalValues.stream_queue)
    else:
        responce = await GPT_request_API(prompt_name, value.user_assistant_prompt,value.variables)
    print(responce)
    return responce

@app.get("/requst/openai-get/queue", tags=["OpenAI"])
async def get_queue():
    if GlobalValues.stream_queue.qsize() != 0:
        try:
            result = await GlobalValues.stream_queue.get()
            print(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An error occurred while fetching from queue: {str(e)}")
        return result
    else:
        raise HTTPException(status_code=404, detail="No data available in the stream queue.")

@app.post("/custom/add/A",tags=["Custom"])
async def add_Dict(item: Dict):
    GlobalValues.custom_value = item
    return item

@app.get("/custom/get_data/A",tags=["Custom"])
async def get_Dict():
    return GlobalValues.custom_value

@app.post("/custom/add/B",tags=["Custom"])
async def add_Dict(item: Dict):
    GlobalValues.custom_value_unity = item
    return item

@app.get("/custom/get_data/B",tags=["Custom"])
async def get_Dict():
    return GlobalValues.custom_value_unity

# GPTのクエリをCSVに保存する関数
async def log_gpt_query_to_csv(prompt, model, prompt_tokens, completion_tokens, total_tokens):
    """
    GPTのクエリデータをCSVコストファイルに保存します。

    Parameters:
    - prompt: GPTに入力されたプロンプト
    - model: 使用されたGPTのモデル名
    - prompt_tokens: プロンプトのトークン数
    - completion_tokens: GPTの応答のトークン数
    - total_tokens: 全体のトークン数
    """
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    # データをリストとして格納
    data_row = [timestamp,model, prompt, prompt_tokens, completion_tokens, total_tokens]
    
    # 'data' ディレクトリが存在しない場合は作成
    if not os.path.exists('data'):
        os.makedirs('data')

    # CSVファイルが存在しない場合は、新規作成してヘッダーを書き出す
    if not os.path.exists('data/cost.csv'):
        header = ['timestamp','model_name', 'prompt', 'prompt_tokens', 'completion_tokens', 'total_tokens']
        with open('data/cost.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(header)

    # ファイルを開いてデータを最終行に追記
    with open('data/cost.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(data_row)

# jsonデータをリストで取得する関数
def get_file_list(search_query=None):
    """
    'data' ディレクトリ内のJSONファイルのリストを返します。
    オプションで特定のクエリを含むファイルだけをフィルタリングすることができます。

    Parameters:
    - search_query: フィルタリングするための文字列（省略可能）

    Returns:
    - json_files: JSONファイルのリスト
    """

    all_files = os.listdir("data")
    json_files = [f for f in all_files if f.endswith('.json')]
    
    # search_queryが存在する場合、該当するファイル名だけをフィルタします。
    if search_query:
        json_files = [f for f in json_files if search_query in f]
        
    return json_files

# プロンプトリストの取得関数
async def get_prompts_list(search_query=None):
    """
    指定されたクエリにマッチするJSONファイルから、プロンプトのリストを取得します。

    Parameters:
    - search_query: フィルタリングするための文字列（省略可能）

    Returns:
    - result: プロンプトデータのリスト
    """
    
    json_file_list = get_file_list(search_query)
    result = []

    for json_file in json_file_list:
        with open(f"data/{json_file}", 'r', encoding='utf-8') as f:
            data = json.load(f)
            result.append(data)
    return result

# プロンプト履歴の取得関数
async def get_history(name):
    """
    指定された名前のJSONファイルから、プロンプトの履歴を取得します。

    Parameters:
    - name: JSONファイルの名前

    Returns:
    - result: プロンプトの履歴データ
    """

    result = None
    with open(f"data/{name}.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'history' in data:
                result = data['history']
    return result

# Jsonデータを作成or編集する関数
async def Create_or_add_json_data(title, description=None, prompt_text=None, settings=None, history=None,other=None):
    """
    JSONファイルを新規作成するか、既存のものにデータを追加/編集します。
    
    Parameters:
    - title: JSONファイルのタイトル（ファイル名）
    - description: 説明文（省略可能）
    - prompt_text: プロンプトのテキスト情報（省略可能）
    - settings: 設定の情報（省略可能）
    - history: 履歴の情報（省略可能）

    Returns:
    なし
    """
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
        settings_json = json.dumps(settings, default=pydantic_encoder)
        settings_dict = json.loads(settings_json)
        
        for key, value in settings_dict.items():
            json_data['setting'][key] = value
    
    if other is not None:
        json_data['other']=other
    
    if history is not None:
        json_data['history'].append(history)
    
    with open(json_file_path, "w", encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4,ensure_ascii=False)

# GPTに問い合わせを実施する関数
async def GPT_request_API(name, user_prompts=[], values={}, queue=None):
    """
    GPT APIへの問い合わせを行い、返ってきたレスポンスを処理します。
    
    Parameters:
    - name: 問い合わせに使用するJSONファイルの名前
    - user_prompts: ユーザーからの追加のプロンプト (省略可能)
    - values: プレースホルダーを置き換えるためのキーと値のマッピング (省略可能)
    - queue: キュー処理に関連する変数 (省略可能)

    Returns:
    GPTからのレスポンスの内容
    """
    #jsonデータの検索
    prompt_list = GlobalValues.prompt_list
    filtered_list = [item for item in prompt_list if name.lower() == item['title'].lower()]
    if len(filtered_list) == 0:
        prompt_list=await get_prompts_list()
        GlobalValues.prompt_list = prompt_list
        filtered_list = [item for item in prompt_list if name.lower() == item['title'].lower()]
    if len(filtered_list) == 0:
        raise HTTPException(status_code=404, detail="The specified prompt could not be found.")
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
    if user_prompts != []:
        text+=user_prompts
    if queue is None:
        response = await GPT_request().GPT_request(filtered_list['title'],
                                openai_key,
                                text,
                                filtered_list['setting']['temperature'],
                                filtered_list['setting']['max_tokens'],
                                filtered_list['setting']['model'])
    else:
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        timestamp_name =name+" - "+ timestamp
        response = await GPT_request().GPT_request_stream(queue,
                                                timestamp_name,
                                                openai_key,
                                                text,
                                                filtered_list['setting']['temperature'],
                                                filtered_list['setting']['max_tokens'],
                                                filtered_list['setting']['model'])
    
    #データ追加
    response['variables']= values
    response['prompt']= text
    
    #レスポンスをロギング
    await Create_or_add_json_data(name,history=response)
    await log_gpt_query_to_csv(name,filtered_list['setting']['model'],response["usage"]['prompt_tokens'],response["usage"]['completion_tokens'],response["usage"]['total_tokens'])

    return response["choices"][0]["message"]["content"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)