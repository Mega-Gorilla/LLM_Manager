#API.py
from module.rich_desgin import error
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List,Any
from datetime import datetime,date
from aiohttp import ClientSession
import os, json, shutil, re, csv
import asyncio
import time
import openai
import tiktoken

app = FastAPI(title='GPT Manger API',version='β2.0')

class config:
    prompts_folder_path = "data"
    openai.api_key = os.getenv("OPENAI_API_KEY")

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
    user_assistant_prompt: dict = {"user": "こんにちわ!みらい"}
    variables: dict = {}

@app.on_event("startup")
async def startup_event():
    # アプリ起動時にセッションを設定
    openai.aiosession.set(ClientSession())

@app.on_event("shutdown")
async def shutdown_event():
    # アプリ終了時にセッションを閉じる
    print("Finished OpenAI aiosession")
    await openai.aiosession.get().close()

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

@app.post("/openai/request/", tags=["OpenAI"])
def OpenAI_request(background_tasks: BackgroundTasks, prompt_name: str,request_id: str=None, value: variables_dict = None, stream_mode: bool=False):
    """
    OpenAI APIにリクエストします。
    **パラメータ:**
    - prompt_name: プロンプト名を設定してください。
    - request_id: レスポンスデータのkeyとして用いられます。設定しなかった場合は、prompt_nameがkeyとなります。
    - variables_dict: プロンプトに{関数名}で囲まれた関数がある場合、辞書配列を入力することでプロンプト内の{}を設定した文字列に書き換えます。
    - stream_mode: Stream問合せを実行するかBoolで設定します。
    """
    if request_id == None:
        request_id = prompt_name
    kwargs = GPT_request_API(prompt_name=prompt_name,
                               request_id=request_id,
                               user_prompts= value.user_assistant_prompt,
                               variables=value.variables)
    kwargs['stream_mode']=stream_mode
    
    background_tasks.add_task(create_chat_completion,**kwargs)
    if 'ok' in kwargs.keys():
        if kwargs['ok'] != True:
            return kwargs
    return{'ok':True,'message':'GPT Request sent in the background'}

@app.get("/openai/get/", tags=["OpenAI"])
async def openai_get(reset: bool =False):
    return_data=LLM_request.chat_completion_object
    if reset:
        LLM_request.chat_completion_object=[]
    return return_data

@app.get("/openai/get-chunk/", tags=["OpenAI"])
async def openai_get(reset: bool =False):
    return_data=LLM_request.chat_completion_chank_object
    if reset:
        LLM_request.chat_completion_chank_object=[]
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
def GPT_request_API(prompt_name,request_id, user_prompts={}, variables={}):
    """
    GPT APIへの問い合わせを行い、返ってきたレスポンスを処理します。
    
    Parameters:
    - name: 問い合わせに使用するJSONファイルの名前
    - user_prompts: ユーザーからの追加のプロンプト (省略可能)
    - variables: プレースホルダーを置き換えるためのキーと値のマッピング (省略可能)

    Returns:
    GPTからのレスポンスの内容
    """
    #jsonデータの検索
    prompt_list = get_prompts_list(prompt_name)
    if 'ok' in prompt_list.keys():
        if prompt_list['ok'] != True:
            error("GPT Request Error","Prompt Data Not Found.",{"Prompt_name":prompt_name})
            return{"ok":False,"message":"Prompt Data Not Found."}

    request_text = []
    prompt_list['text'].update(user_prompts)
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
    return request_kwargs

def talknizer(texts):
        tokens = tiktoken.get_encoding('gpt2').encode(texts)
        return len(tokens)

async def create_chat_completion(request_id,prompt_name,Prompt=[{"system":"You are a helpful assistant."},{"user":"Hello!"}],model="gpt-4",temp=0,tokens_max=2000,top_p=1,frequency_penalty=0,presence_penalty=0,max_retries=3,add_responce_dict=None,stream_mode=False):
    
    Prompts=[]
    for original_dict in Prompt:
        transformed_dict = {}
        for key, value in original_dict.items():
            transformed_dict["role"] = key
            transformed_dict["content"] = value
        Prompts.append(transformed_dict)
    
    gpt_error_mapping = GPT_error_list()
    retry_count = 0
    result_content = ""
    while retry_count < max_retries:
        try:
            # 非同期でチャットの応答を取得
            chat_completion_resp = await openai.ChatCompletion.acreate(
                model=model,
                messages=Prompts,
                temperature=temp,
                max_tokens=tokens_max,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stream=stream_mode
            )

            if stream_mode:
                fin_reason = None
                async for chunk in chat_completion_resp:
                    content = chunk["choices"][0].get("delta", {}).get("content", "")
                    fin_reason = chunk["choices"][0].get("finish_reason")
                    print(content,end='')
                    result_content += content
                    LLM_request.chat_completion_chank_object.append({"request_id":request_id,
                                                                        'id':chunk["id"],
                                                                        "message": content,
                                                                        "index": chunk["choices"][0].get("index"),
                                                                        'object':chunk["object"],
                                                                        'object':chunk["object"],
                                                                        'created':chunk["created"],
                                                                        'model':chunk["model"],
                                                                        "finish_reason": fin_reason})
                    
                    await asyncio.sleep(0)
                print(f"\nFin Reason: {fin_reason}")
                prompt_tokens=talknizer(''.join([item['content'] for item in Prompts]))
                completion_tokens=talknizer(result_content)
                chat_completion_resp = {
                    "id": chunk["id"],
                    "object": chunk["object"],
                    "created": chunk["created"],
                    "model": chunk["model"],
                    "choices": [{
                        "index": chunk["choices"][0].get("index"),
                        "message": {
                        "role": "assistant",
                        "content": result_content,
                        },
                        "finish_reason": fin_reason
                        }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens+completion_tokens
                    }}

            # 辞書配列に結果を格納
            chat_completion_resp.update({'request_id':request_id,'ok':True})
            if add_responce_dict != None:
                chat_completion_resp.update(add_responce_dict)

            #応答を配列に追加
            LLM_request.chat_completion_object.append(chat_completion_resp)

            #データロギング
            Create_or_add_json_data(prompt_name,history=chat_completion_resp)
            log_gpt_query_to_csv(prompt_name,model,chat_completion_resp["usage"]['prompt_tokens'],chat_completion_resp["usage"]['completion_tokens'],chat_completion_resp["usage"]['total_tokens'])
            return
        except Exception as e:
            retry_count+=1
            title, message, action = gpt_error_mapping.get(type(e), ("OpenAI Unknown Error", "不明なエラーです。", 'exit'))
            print(e)
            e=str(e)+(f"\nmessages: {Prompts}\ntemperature: {temp}\nMax Tokens: {tokens_max}\ntop_p: {top_p}\nfrequency_penalty: {frequency_penalty}\npresence_penalty: {presence_penalty}")
            error(title, message, e if action == 'exit' else None)
            
            if action == 'exit':
                LLM_request.chat_completion_object.append({'request_id':request_id,'ok':False,'message':e})
                return
            elif action == 'sleep':
                await asyncio.sleep(1)

    LLM_request.chat_completion_object.append({'request_id':request_id,'ok':False,'message':'Request Time out'})
    return 

def GPT_error_list():
    gpt_error_mapping = {
        openai.error.APIError: ("OpenAI API Error", "しばらく時間をおいてからリクエストを再試行し、問題が解決しない場合は弊社までご連絡ください。", 'sleep'),
        openai.error.Timeout: ("OpenAI Timeout Error", "リクエストがタイムアウトしました。しばらく時間をおいてからリクエストを再試行し、問題が解決しない場合は弊社までご連絡ください。", 'sleep'),
        openai.error.RateLimitError: ("OpenAI Rate Limit Error", "リクエストのペースを上げてください。詳しくはレート制限ガイドをご覧ください。", 'exit'),
        openai.error.APIConnectionError: ("OpenAI API Connection Error", "ネットワーク設定、プロキシ設定、SSL証明書、またはファイアウォールルールを確認してください。", 'exit'),
        openai.error.InvalidRequestError: ("OpenAI API Invalid Request Error", "エラーメッセージは、具体的なエラーについてアドバイスするはずです。呼び出している特定のAPIメソッドのドキュメントを確認し、有効で完全なパラメータを送信していることを確認してください。また、リクエストデータのエンコーディング、フォーマット、サイズを確認する必要があるかもしれません。", 'exit'),
        openai.error.AuthenticationError: ("OpenAI Authentication Error", "APIキーまたはトークンを確認し、それが正しく、アクティブであることを確認してください。アカウントダッシュボードから新しいものを生成する必要があるかもしれません。", 'exit'),
        openai.error.ServiceUnavailableError: ("OpenAI Service Unavailable Error", "しばらく時間をおいてからリクエストを再試行し、問題が解決しない場合はお問い合わせください。ステータスページをご確認ください。", 'sleep')
    }
    return gpt_error_mapping

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)