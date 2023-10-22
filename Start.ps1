# api.pyを実行
Start-Process powershell -ArgumentList "-NoExit","-Command & conda activate AI_Tuber; streamlit run .\GUI.py"

# デフォルトのブラウザでURLを開く
Start-Process "http://127.0.0.1:8000/docs"
# api.pyを実行
python .\api.py
Read-Host -Prompt "Press Enter to exit"
