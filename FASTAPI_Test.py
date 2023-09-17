from fastapi import FastAPI
from pydantic import BaseModel
import random
import threading
import time

app = FastAPI()

# データを格納するための簡単なリスト
dynamic_items = []

# BaseModelを継承したItemクラスを作成
class Item(BaseModel):
    value: int

# リストにランダムな数値を追加する関数
def add_random_item_periodically():
    while True:
        dynamic_items.append(random.randint(0, 100))
        time.sleep(5)  # 5秒ごとにランダムな数値を追加

# 別のスレッドでランダムな数値を追加
threading.Thread(target=add_random_item_periodically).start()

# GETリクエスト: リスト内のすべてのアイテムを返す
@app.get("/items/")
async def read_items():
    return dynamic_items

# GETリクエスト: 特定のアイテムを返す
@app.get("/items/{item_id}")
async def read_item(item_id: int):
    if item_id >= len(dynamic_items):
        return {"error": "Item not found"}
    return {"value": dynamic_items[item_id]}

# POSTリクエスト: アイテムを追加
@app.post("/items/")
async def create_item(item: Item):
    dynamic_items.append(item.value)
    return {"value": item.value}
