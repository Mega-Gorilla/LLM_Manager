#API.py
from module.rich_desgin import error
from fastapi import FastAPI, BackgroundTasks,Query
from pydantic import BaseModel, Field
from pydantic.json import pydantic_encoder
from typing import List,Any
from datetime import datetime,date
import os, json, shutil, re, csv, time
import asyncio
import openai
import tiktoken
import requests
from openai import AsyncOpenAI,OpenAI
import google.generativeai as genai

app = FastAPI(title='GPT Manger API',version='β2.0')

class config:
    prompts_folder_path = "data"
    
class LLM_request:
    chat_completion_object:List[Any]=[]
    chat_completion_chank_object:List[Any]=[]

#Prompt Json Settngs
class Settings(BaseModel):
    model: str = Field(default="gpt-3.5-turbo")
    temperature: float = Field(default=1.00)
    top_p: float = Field(default=1)
    max_tokens: int = Field(default=500)
    presence_penalty: float = Field(default=0.00)
    frequency_penalty: float = Field(default=0.00)
    #logit_bias: Optional[Dict[int, Any]] = None  # Noneかfloatの辞書

class Prompts(BaseModel):
    title: str = "Prompt Name"
    description: str= "Prompt Description"
    texts: dict = {"system":"You are a helpful assistant.And you need to advise about the {things}."}
    setting: Settings
    variables : dict
    other: dict

class variables_dict(BaseModel):
    user_assistant_prompt: dict = {}
    variables: dict = {}

class openai_config:
    #openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    aclient = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class gemini_config:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    # https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini?hl=ja

    safety_settings_NONE=[
        { "category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE" },
        { "category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
    ]

@app.post("/prompts-post/add_edit_prompt", tags=["Prompts"])
def add_new_prompt(prompt: Prompts):
    """
    新しいプロンプトを追加または既存のプロンプトを編集します。
    - `prompt`: 'Prompts'モデルに基づいて、タイトル、説明、テキスト、設定、変数、その他の情報が含まれます。
    """
    result = get_prompts_list()
    title_list = [d["title"] for d in result]
    if prompt.title in title_list:
        print("A prompt with the same name already exists. Overwriting process has been executed.")
        #raise HTTPException(status_code=400, detail="Title already exists.")
    Create_or_add_json_data(title=prompt.title, description=prompt.description, prompt_text=prompt.texts, settings=prompt.setting,variables=prompt.variables, other=prompt.other)

@app.get("/prompts-get/get_prompt_metadata", tags=["Prompts"])
def get_all_prompts_data():
    """
    すべてのプロンプトデータを取得します。

    **取得データ例:**
    ```json
    [{
    "title": "",
    "description": "",
    "text": {
      "system": ""
    },
    "setting": {
      "model": "gpt-3.5-turbo",
      "temperature": 1,
      "top_p": 1,
      "max_tokens": 256,
      "presence_penalty": 0,
      "frequency_penalty": 1
    },
    "variables": {
      "character": "",
      "loc_time": "",
      "status": ""
    },
    "history": [],
    "other": {
      "translate_text": {
        "system": ""
      }
    }
    }]
    ```

    """
    result = get_prompts_list()
    return result

@app.get("/prompts-get/get_prompt_details", tags=["Prompts"])
def get_all_prompts_names():
    """
    プロンプト一覧を取得します。

    **取得データ:**
    - `title`: プロンプトのタイトル。
    - `description`: プロンプトの説明。

    **戻り値の例:**
    ```json
    {
      "Caracter-LLM Meta Prompt for Trainable Agents": "トレーニング可能なエージェントのためのメタプロンプト",
      "Character-LLM Meta Prompt for Baseline Instruction-following Models": "ベースライン指導フォローモデルのメタプロンプト"
    }
    ```

    このエンドポイントは、システムに登録されている全てのプロンプトの名前と説明の一覧を返します。
    """
    result = get_prompts_list()
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
def get_prompt_data_by_name(prompt_name: str):
    result = get_prompts_list(prompt_name)
    del result["history"]
    return result

@app.get("/prompts-get/history/", tags=["Prompts"])
def get_history(prompt_name: str):
    result = get_history(prompt_name)
    print(result)
    return result

@app.get("/cost-get/day/", tags=["Cost"])
def get_cost_day(day: date=Query(default=datetime.now().strftime("%Y-%m-%d"))):
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

@app.post("/LLM/request/",tags=["LLM"],summary="LLMモデルにリクエストを送る")
async def LLM_Chat_request(background_tasks: BackgroundTasks, prompt_name: str,request_id: str=None, value: variables_dict = None, stream_mode: bool=False):
    """
    LLMチャットモデルにリクエストします。
    
    **パラメータ:**
    - prompt_name: プロンプト名を設定してください。プロンプト名のJsonファイルからPromptの主要な情報は読み込まれます。
    - request_id: レスポンスデータのkeyとして用いられます。設定しなかった場合は、prompt_nameがkeyとなります。
    - variables_dict: プロンプトに{関数名}で囲まれた関数がある場合、辞書配列を入力することでプロンプト内の{}を設定した文字列に書き換えます。
        - "user_assistant_prompt": { "user": "こんにちわ!みらい"}
        - "variables": {"question":"human"}
    - stream_mode: Stream問合せを実行するかBoolで設定します。Stream時は、"/openai/get-chunk/"にて値を取得できます。
    """

    if request_id == None:
        request_id = prompt_name
    kwargs = LLM_request_API(prompt_name=prompt_name,
                               request_id=request_id,
                               user_prompts= value.user_assistant_prompt,
                               variables=value.variables)
    kwargs['stream_mode'] = stream_mode
    print(kwargs)

    if 'ok' in kwargs.keys():
        if kwargs['ok'] != True:
            return kwargs #プロンプト取得時に問題が発生した場合
    
    background_tasks.add_task(create_LLM_chat_completion,**kwargs)

    return{'ok':True,'message':'LLM Request sent in the background'}

@app.get("/LLM/get/", tags=["LLM"] ,summary='LLMリクエスト結果を取得する')
async def LLM_Chat_get(reset: bool =False,del_request_id:str = None):
    """
    LLMの結果を取得します。LLMで追加したタスクの結果を取得します。

    **パラメータ:**
    - reset: 関数を初期化します
    """
    return_data=LLM_request.chat_completion_object
    if reset:
        LLM_request.chat_completion_object=[]
    if del_request_id != None:
        return_data = [d for d in return_data if d['request_id'] == del_request_id]
        LLM_request.chat_completion_chank_object = [d for d in return_data if d['request_id'] != del_request_id]
    return return_data

@app.get("/LLM/get-chunk/", tags=["LLM"],summary='LLM Stream Chunk内容を取得する')
async def LLM_get_stream(reset_all: bool = False,del_request_id:str = None):
    """
    Stream時のchunk内容を取得します。
    **パラメータ:**
    - reset_all: 関数を初期化します
    - del_request_id: 指定されたリクエストidをchunkから消去し、消去した内容を返します。
    """
    return_data=LLM_request.chat_completion_chank_object
    if reset_all:
        LLM_request.chat_completion_chank_object=[]
    if del_request_id != None:
        return_data = [d for d in return_data if d['request_id'] == del_request_id]
        LLM_request.chat_completion_chank_object = [d for d in return_data if d['request_id'] != del_request_id]
    return return_data

# GPTのクエリをCSVに保存する関数
def log_gpt_query_to_csv(prompt_name, model, prompt_tokens, completion_tokens, total_tokens):
    """
    GPTのクエリデータをCSVコストファイルに保存します。

    Parameters:
    - prompt_name: ロギングするプロンプト名
    - model: 使用されたGPTのモデル名
    - prompt_tokens: プロンプトのトークン数
    - completion_tokens: GPTの応答のトークン数
    - total_tokens: 全体のトークン数
    """
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    # データをリストとして格納
    data_row = [timestamp,model, prompt_name, prompt_tokens, completion_tokens, total_tokens]
    
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
def get_file_list():
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
    
        
    return json_files

# プロンプトリストの取得関数
def get_prompts_list(search_query=None):
    """
    指定されたクエリにマッチするJSONファイルから、プロンプトのリストを取得します。

    Parameters:
    - search_query: フィルタリングするための文字列（省略可能）

    Returns:
    - result: プロンプトデータのリスト
    """
    
    json_file_list = get_file_list()
    result = []
    if search_query!=None:
        if f"{search_query}.json" in json_file_list:
            with open(f"data/{search_query}.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {"ok":False,"message":"Prompt Data Not Found."}
    else:
        for json_file in json_file_list:
            print(f"open:{json_file}")
            with open(f"data/{json_file}", 'r', encoding='utf-8') as f:
                data = json.load(f)
                result.append(data)
    return result

# プロンプト履歴の取得関数
def get_history(name):
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
def Create_or_add_json_data(title, description=None, prompt_text=None, settings=None,variables=None,history=None,other=None):
    """
    JSONファイルを新規作成するか、既存のものにデータを追加/編集します。
    
    Parameters:
    - title: JSONファイルのタイトル（ファイル名）
    - description: 説明文（省略可能）
    - prompt_text: プロンプトのテキスト情報（省略可能）
    - settings: 設定の情報（省略可能）
    - history: 履歴の情報（省略可能）

    """
    json_file_list = get_file_list()
    json_file_name = title + ".json"
    json_file_path = os.path.join(config.prompts_folder_path,json_file_name)

    if json_file_name not in json_file_list:
        #jsonファイルが存在しない場合新規作成する。
        tempfilepath=os.path.join(config.prompts_folder_path,"template.json")
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
                placeholders = re.findall(r'{([^{}"]+?)}', value)
                for placeholder in placeholders:
                    placeholder_dict[placeholder] = ""
    
    if variables is not None:
        if variables == {}:
            json_data['variables'] = placeholder_dict
        else:
            add_value_dict = {key: variables[key] for key in placeholder_dict if key in variables}
            json_data['variables'] = add_value_dict

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
def LLM_request_API(prompt_name,request_id, user_prompts={}, variables={}):
    """
    LLM APIへの問い合わせに必要な情報を返します
    
    Parameters:
    - name: 問い合わせに使用するJSONファイルの名前
    - user_prompts: ユーザーからの追加のプロンプト (省略可能)
    - variables: プレースホルダーを置き換えるためのキーと値のマッピング (省略可能)

    Returns:
    各モデルに基づいたLLM API問合せに必要なデータを返します。
    """
    #jsonデータの検索
    prompt_list = get_prompts_list(prompt_name)

    if 'ok' in prompt_list.keys():
        if prompt_list['ok'] != True:
            error("GPT Request Error","Prompt Data Not Found.",{"Prompt_name":prompt_name})
            return{"ok":False,"message":"Prompt Data Not Found."}

    request_text = []
    prompt_list['text'].update(user_prompts)
    
    #variablesに既存値を代入
    for key in prompt_list['variables']:
        if key not in variables:
            variables[key] = prompt_list['variables'][key]

    for key, value in prompt_list['text'].items():
        if isinstance(value, str):
            # 文字列内のプレースホルダー（{xxx}）を見つける
            prompt_variables_lsit = re.findall(r'{([^{}"]+?)}', value)
            prompt_variables_dict = {key: '' for key in prompt_variables_lsit}
            #プロンプトのvariablesと提供されたvariablesが一致しているか確認する
            if not all(key in variables for key in prompt_variables_dict):
                error("GPT Request Error","prompt variables error.",{"Request Variables":variables,"Prompt Variables":prompt_variables_dict,"keys":key,"Prompt":value})
                return{"ok":False,"message":f"prompt variables error."}
            #必要なkeyだけ取り出し
            for prompt_key in prompt_variables_dict:
                if prompt_key in variables:
                    prompt_variables_dict[prompt_key] = variables[prompt_key]
            #プロンプトの{}部分の置換処理
            for dict_key, dict_value in prompt_variables_dict.items():
                # 正規表現を用いて、キーに合致する部分だけを厳密に置換する
                value = re.sub(r'\{' + re.escape(dict_key) + r'\}', dict_value, value)
            request_text.append({key: value})
        else:
            error("GPT Request Error","prompt text format error.",prompt_list['text'])
            return{"ok":False,"message":f"prompt text format error. /{key}:{value}"}
    #print(f"request_text: {request_text}")
    
    request_kwargs = {
        'request_id':request_id,
        'prompt_name':prompt_name,
        'Prompt':request_text,
        'model':prompt_list['setting']['model'],
        'temp':prompt_list['setting']['temperature'],
        'tokens_max':prompt_list['setting']['max_tokens'],
        'top_p':prompt_list['setting']['top_p'],
        'frequency_penalty':prompt_list['setting']['presence_penalty'],
        'presence_penalty':prompt_list['setting']['frequency_penalty'],
        'add_responce_dict':{'variables':variables,'prompt':request_text}
    }
    try:
        responce_format = prompt_list['setting']['response_format']
        request_kwargs['response_format'] = responce_format
    except KeyError:
        request_kwargs['response_format'] = None
    return request_kwargs

def gpt_talknizer(texts):
        tokens = tiktoken.get_encoding('gpt2').encode(texts)
        return len(tokens)

def gemini_tokenizer(text):
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

async def create_GPT_chat_completion(request_id,prompt_name,Prompt,model,temp,tokens_max,top_p,frequency_penalty,presence_penalty,stream_mode,response_format,max_retries,add_responce_dict):
    process_time=time.time()
    asyclient=openai_config.aclient
    client=openai_config.client
    Prompts=[]
    for original_dict in Prompt:
        transformed_dict = {}
        for key, value in original_dict.items():
            transformed_dict["role"] = key
            transformed_dict["content"] = value
        Prompts.append(transformed_dict)
    gpt_error_mapping = GPT_error_list()
    retry_count = 0

    gpt_functions = {
        'model':model,
        'messages':Prompts,
        'temperature':temp,
        'max_tokens':tokens_max,
        'top_p':top_p,
        'frequency_penalty':frequency_penalty,
        'presence_penalty':presence_penalty,
        'stream':stream_mode,
    }

    if response_format != None:
        # json Mode等
        gpt_functions['response_format']={"type":response_format}

    while retry_count < max_retries:
        try:
            # Stream
            if stream_mode:
                fin_reason = None
                result_content = ""
                response = client.chat.completions.create(**gpt_functions)
                for event in response:
                    event_dict = event.dict()
                    if event_dict['choices'][0]['delta']['content'] != None:
                        content = event_dict['choices'][0]['delta']['content']
                    else:
                        content = ""
                    print(content,end='')
                    result_content += content
                    fin_reason = event_dict['choices'][0]['finish_reason']
                    LLM_request.chat_completion_chank_object.append({"request_id":request_id,
                                                                        "content": content,
                                                                        "finish_reason": fin_reason})
                #終了キーをPost
                LLM_request.chat_completion_chank_object.append({"request_id":request_id,
                                                                        "content": "",
                                                                        "finish_reason": "Done"})
                #トークン計算
                prompt_tokens=gpt_talknizer(''.join([item['content'] for item in Prompts]))
                completion_tokens=gpt_talknizer(result_content)
                total_tokens = prompt_tokens+completion_tokens
                
            else:
                # 通常モード
                response = await asyclient.chat.completions.create(**gpt_functions)
                response = response.dict()
                fin_reason = response['choices'][0]['finish_reason']
                result_content = response['choices'][0]['message']['content']
                fin_reason=response['choices'][0]['finish_reason']

                prompt_tokens=response['usage']['prompt_tokens']
                completion_tokens=response['usage']['completion_tokens']
                total_tokens=response['usage']['total_tokens']
                
                print(response['choices'][0]['message']['content'])
            
            # 辞書配列に結果を格納
            chat_completion_resp = {
                "request_id":request_id,
                "ok":True,
                "model":model,
                "content":result_content,
                "prompt": Prompt,
                "finish_reason":fin_reason,
                "completion_tokens":completion_tokens,
                "prompt_tokens":prompt_tokens,
                'total_tokens':total_tokens,
                'raw_data':response
            }
            #追加の辞書配列を入れる
            if add_responce_dict != None:
                chat_completion_resp.update(add_responce_dict)
            #実行時間を記録
            GPT_process_time = time.time()-process_time
            chat_completion_resp.update({"process_time":GPT_process_time})
            print(f"\n\nGPT Request Time: {GPT_process_time}\nFin Reason: {fin_reason}")

            #Logを記録
            Create_or_add_json_data(prompt_name,history=chat_completion_resp)
            #応答を配列に追加
            return chat_completion_resp    
        
        except Exception as e:
            #レスポンスエラー時
            retry_count+=1
            title, message, action, sleep_time = gpt_error_mapping.get(type(e), ("OpenAI Unknown Error", "不明なエラーです。", 'exit',1))
            error_log = ''
            for key,value in gpt_functions.items():
                error_log += f"{key} : {value}\n"
            e=f"{e}\n\n{error_log}"
            error(title, message, e if action == 'exit' else None)
            
            if action == 'exit':
                return {'request_id':request_id,'ok':False,'message':e} #エラーで終了した場合
            elif action == 'sleep':
                await asyncio.sleep(sleep_time)

    # リクエストID 結果を追加
    return {'request_id':request_id,'ok':False,'message':'Request Time out'} #3回問い合わせてレスポンスが返ってこなかった場合

async def create_gemini_chat_completion(request_id,prompt_name,Prompt,model,temp,tokens_max,top_p,frequency_penalty,presence_penalty,stream_mode,response_format,max_retries,add_responce_dict):
    process_time=time.time()
    gemini_name = model
    model = genai.GenerativeModel(model_name=model)
    chat = model.start_chat()
    prompt_text = Prompt[0]["system"]
    config = {
        "max_output_tokens": tokens_max,
        "temperature": temp,
        "top_p": top_p
    }
    retry_count = 0

    while retry_count < max_retries:
        try:
            response = None
            # Stream
            if stream_mode:
                result_content = ""
                response = chat.send_message(prompt_text,generation_config=config,safety_settings=gemini_config.safety_settings_NONE,stream=True)
                for chunk in response:
                    content = chunk.text
                    fin_reason = chunk.candidates[0].finish_reason.name
                    print(content,end='')
                    LLM_request.chat_completion_chank_object.append({"request_id":request_id,
                                                                        "content": content,
                                                                        "finish_reason": fin_reason})
                    result_content += content
                #終了キーをPost
                LLM_request.chat_completion_chank_object.append({"request_id":request_id,
                                                                        "content": "",
                                                                        "finish_reason": "Done"})
                
            else:
                # 通常モード
                response = chat.send_message(prompt_text,generation_config=config,safety_settings=gemini_config.safety_settings_NONE)
                result_content = response.text
                fin_reason = response.candidates[0].finish_reason.name
                print(result_content)

            prompt_tokens = gemini_tokenizer(prompt_text)
            completion_tokens = gemini_tokenizer(result_content)
            chat_completion_resp = {
                "request_id":request_id,
                "ok":True,
                "model":gemini_name,
                "content": result_content,
                "prompt": Prompt,
                "finish_reason":fin_reason,
                "completion_tokens":completion_tokens,
                "prompt_tokens":prompt_tokens,
                'total_tokens':completion_tokens+prompt_tokens,
            }
            if stream_mode==False:
                pass
                #chat_completion_resp.update({"raw_data":response.__dict__})
            if add_responce_dict != None:
                chat_completion_resp.update(add_responce_dict)
            #実行時間を記録
            LLM_process_time = time.time()-process_time
            chat_completion_resp.update({"process_time":LLM_process_time})
            print(f"\n\nGemini Request Time: {LLM_process_time}\nFin Reason: {fin_reason}")

            #Logを記録
            Create_or_add_json_data(prompt_name,history=chat_completion_resp)
            #応答を配列に追加
            return chat_completion_resp
        
        except Exception as e:
            #レスポンスエラー時
            retry_count+=1
            error(f"{gemini_name} Responce Error",e)
            await asyncio.sleep(3)

    # リクエストID 結果を追加
    return {'request_id':request_id,'ok':False,'message':'Request Time out'} #3回問い合わせてレスポンスが返ってこなかった場合

async def create_LLM_chat_completion(request_id,prompt_name,response_format={},Prompt=[{"system":"You are a helpful assistant."},{"user":"Hello!"}],model="gpt-4",temp=0,tokens_max=2000,top_p=1,frequency_penalty=0,presence_penalty=0,max_retries=3,add_responce_dict=None,stream_mode=False):
    result_data = {}
    if "gpt" in model:
        result_data = await create_GPT_chat_completion(request_id=request_id,prompt_name=prompt_name,Prompt=Prompt,model=model,temp=temp,tokens_max=tokens_max,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,stream_mode=stream_mode,response_format=response_format,max_retries=max_retries,add_responce_dict=add_responce_dict)
    elif "gemini" in model:
        result_data = await create_gemini_chat_completion(request_id=request_id,prompt_name=prompt_name,Prompt=Prompt,model=model,temp=temp,tokens_max=tokens_max,top_p=top_p,frequency_penalty=frequency_penalty,presence_penalty=presence_penalty,stream_mode=stream_mode,response_format=response_format,max_retries=max_retries,add_responce_dict=add_responce_dict)
    LLM_request.chat_completion_object.append(result_data)
    return 

def GPT_error_list():
    gpt_error_mapping = {
        openai.ConflictError:("OpenAI Conflict Error", """エラー内容:
                             openai.ConflictError は、APIがリクエストを処理できないことを示しています。これは、以下のような状況で発生する可能性があります：
                             リソースに対する変更が既に進行中で、新しいリクエストが競合している。
                             同じデータやリソースに対して一貫性のない変更を同時に行おうとした。
                             特定の操作が許可されていない状態でリクエストされた。
                             対応方法:
                             クエストの再試行: エラーが一時的な競合によるものであれば、リクエストを遅らせて再試行することが有効です。
                             同期の確認: 複数のプロセスまたはスレッドが同じリソースにアクセスしている場合は、適切な同期メカニズムを実装して競合を回避します。
                             リクエストの確認: 送信しているリクエストが正しいか確認し、競合を引き起こす可能性のあるリクエストがないか再検討します。
                             APIドキュメントの確認: ConflictError が発生した具体的なAPIエンドポイントのドキュメントを確認し、エラーの原因となっている競合条件について理解します。
                             エラーメッセージの詳細: エラーレスポンスに含まれる追加情報やメッセージを検証して、エラーの原因を特定します。""", 'sleep',3),
        openai.NotFoundError:("OpenAI NotFound Error", """エラー内容:
                             このエラーは、リクエストされたリソースが見つからない場合に発生します。たとえば、存在しないモデルやファイルを指定したリクエスト、またはIDが間違っているか削除されたオブジェクトに対するリクエストで遭遇する可能性があります。
                             対応方法:
                             リソースの確認: 指定したリソースのIDや名前が正しいことを確認してください。
                             削除の確認: リソースが削除されていないか、APIの変更履歴を確認してください。
                             URLの確認: エンドポイントのURLが正しいことを確認してください。
                             """, 'exit',1),
        openai.APIStatusError:("OpenAI APIStatus Error", """エラー内容:
                              APIサービス自体に問題が発生している場合、このエラーが返されます。サービスのダウンタイムや一時的な障害が原因である可能性があります。
                              対応方法:
                              ステータスチェック: OpenAIのステータスページを確認して、サービスの状態を確認してください。
                              再試行ポリシー: エラーが一時的である可能性があるため、時間を置いてから再試行してください。
                              サポートに連絡: 問題が継続する場合は、OpenAIのサポートに連絡することを検討してください。
                              """, 'sleep',3),
        openai.RateLimitError:("OpenAI Rate Limit Error", """エラー内容:
                              このエラーは、APIの利用制限を超えた場合に発生します。これはリクエスト数が短時間に多すぎるか、アカウントに設定されている使用量の上限に達したことを意味します。
                              対応方法:
                              利用状況の確認: 自分のAPI利用状況を確認して、レート制限の上限に達しているか確認してください。
                              リクエストの制限: リクエストの頻度を減らすか、必要に応じてレート制限を増やすようにリクエストしてください。
                              分散実行: リクエストを時間をかけて分散させ、レート制限に達しないようにしてください。
                              これらのエラーは、APIを使用する上でよく遭遇するものであり、それぞれに対する適切な対応が必要です。ユーザーはこれらの対応策を試して問題を解決できる可能性が高いですが、問題が解決しない場合はOpenAIのサポートチームに連絡することをお勧めします。
                              """, 'sleep',5),
        openai.APITimeoutError:("OpenAI API Timeout Error", """エラー内容:
                               APITimeoutErrorはAPIリクエストが指定されたタイムアウト期間内に完了しなかった場合に発生します。これは、OpenAIのサーバーが過負荷になっているか、ネットワーク接続が遅延している場合に起こりえます。
                               対応方法:
                               再試行: ネットワークの一時的な問題やサーバーの負荷が原因の場合、時間を置いてから再試行してください。
                               タイムアウトの延長: リクエストに対するタイムアウト設定を長くしてみてください。
                               ネットワークの確認: ネットワーク接続が安定していることを確認してください。""", 'sleep',1),
        openai.BadRequestError:("OpenAI Bad Request Error", """エラー内容:
                               BadRequestErrorは、APIへのリクエストが不正または不適切なフォーマットであるためにサーバーによって拒否された場合に発生します。これには、パラメータの誤り、不足、またはデータのフォーマットが不正な場合などが含まれます。
                               対応方法:
                               リクエストの検証: 送信したリクエストのパラメータとフォーマットが正しいかどうかを検証してください。
                               エラーメッセージの確認: エラーメッセージには、通常、何が問題であるかの手がかりが含まれています。メッセージの内容を注意深く読んで、指示に従ってください。
                               ドキュメントの確認: OpenAIのドキュメントを確認して、適切なリクエストフォーマットを確認してください。""", 'exit',1),
        openai.APIConnectionError:("OpenAI API Connection Error", """エラー内容:
                                  APIConnectionError は、APIへの接続を確立する際に何らかの問題が発生した場合に返されます。これは、ネットワーク障害、DNSの問題、またはOpenAIのサービスに到達できないことが原因である可能性があります。
                                  対応方法:
                                  ネットワークのトラブルシューティング: インターネット接続が正常に機能していることを確認し、必要に応じて修正してください。
                                  ファイアウォールの確認: ファイアウォールやセキュリティソフトウェアがOpenAIのAPIとの通信を妨げていないか確認してください。
                                  DNSの確認: DNS設定が正しいことを確認し、DNSキャッシュをクリアしてみてください。""", 'sleep',1),
        openai.AuthenticationError:("OpenAI Authentication Error", """エラー内容:
                                   AuthenticationError は、APIキーが無効、期限切れ、または不足している場合に発生します。これは、APIに対する認証が失敗したことを意味します。
                                   対応方法:
                                   APIキーの確認: 有効なAPIキーが使用されているか確認してください。
                                   期限の確認: APIキーの期限が切れていないか確認してください。
                                   キーの設定: APIキーがリクエストに正しく含まれていることを確認してください。""", 'exit',1),
        openai.InternalServerError:("OpenAI Internal Server Error", """エラー内容:
                                   InternalServerError は、OpenAIのサーバー内部でエラーが発生した場合に返されます。このエラーはクライアント側ではなく、サーバー側の問題を示しています。
                                   対応方法:
                                   再試行: 一時的なサーバーの問題が原因の場合は、時間を置いてからリクエストを再試行してください。
                                   ステータスチェック: OpenAIのステータスページを確認して、他に既知の問題がないか確認してください。
                                   サポートへの問い合わせ: 問題が継続する場合は、OpenAIサポートに報告してください。""", 'sleep',1),
        openai.PermissionDeniedError:("OpenAI Permission Denied Error", """エラー内容:
                                     PermissionDeniedError は、リクエストされた操作に対する適切なアクセス権がない場合に発生します。これは、リクエストが認証されたものの、実行しようとした操作に必要な権限がアカウントにないことを意味します。
                                     対応方法:
                                     権限の確認: アカウントに必要な権限があるか確認してください。
                                     ロールの確認: 適切なロールやアクセス権がアカウントに割り当てられているか確認してください。
                                     アクセスポリシーの確認: アカウントのアクセスポリシーがリクエストされた操作を許可していることを確認してください。""", 'exit',1),
        openai.UnprocessableEntityError:("OpenAI Unprocessable Entity Error", """エラー内容:
                                        UnprocessableEntityError は、リクエストが正しい形式で送信されたが、何らかの理由で処理できなかった場合に返されます。これは、リクエストの内容が無効であるか、リクエストに含まれるデータがサーバーの期待する形式と一致していないことを意味します。
                                        対応方法:
                                        リクエストのレビュー: リクエストがAPIの仕様に合致しているか確認してください。
                                        バリデーションエラーの詳細: エラーメッセージには、通常、リクエストが受け入れられなかった具体的な理由が含まれています。
                                        ドキュメントの確認: OpenAIのドキュメントを確認し、リクエストの形式が正しいかどうかを再確認してください。""", 'exit',1),
        openai.APIResponseValidationError:("OpenAI APIResponse Validation Error", """エラー内容:
                                          APIResponseValidationError は、APIからのレスポンスが期待するスキーマや形式と一致しない場合に発生するエラーです。これは、APIが無効なデータを返しているか、クライアント側のレスポンス処理が適切に構成されていないことを示しています。
                                          対応方法:
                                          レスポンスの確認: 受け取ったレスポンスデータを確認し、期待される形式に沿っているか検証してください。
                                          クライアントのアップデート: クライアントライブラリが最新か確認し、古い場合は更新してください。
                                          サポートへの報告: このエラーはAPI側の問題である可能性があるため、OpenAIに問題を報告することも検討してください。""", 'exit',1),
        openai._AmbiguousModuleClientUsageError:("OpenAI _Ambiguous Module Client Usage Error", """エラー内容:
                                                _AmbiguousModuleClientUsageError は、OpenAIライブラリ内でのクライアントの使用が不明確な場合に発生する内部エラーです。これは主に、APIの異なるバージョン間で混同が発生したり、ライブラリの使用方法が不適切である場合に発生する可能性があります。
                                                対応方法:
                                                方法の確認: OpenAIライブラリの使用方法を見直し、ガイドに沿っているか確認してください。
                                                クライアントの初期化: クライアントの初期化が適切に行われているか、ドキュメントと照らし合わせて確認してください。
                                                コードのリファクタリング: 必要に応じてコードをリファクタリングし、クライアントの使用方法を明確にしてください。""", 'exit',1),
    }
    return gpt_error_mapping

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("API:app", host="0.0.0.0", port=8000,reload=True)