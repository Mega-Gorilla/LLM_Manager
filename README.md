# GPT Manager

# Overview

# 🛠 Installation
#### 1.Bulid from source
    git clone https://github.com/Mega-Gorilla/gpt-prompt-editor.git
    cd gpt-prompt-editor
    pip install -r requirements.txt

#### 2.Setting up the OpenAI Key

To integrate OpenAI services into your application, follow these steps to set up your API key:

1. **Accessing the Windows Environment Variables:**
   - Open the start menu and search for 'Environment Variables', then choose "Edit the system environment variables".
   - In the System Properties window, click on the "Environment Variables" button.

2. **Adding a New Environment Variable:**
   - In the Environment Variables window, under the "User variables" section, click on the "New" button.
   - For the "Variable name", enter `OPENAI_API_KEY`.
   - For the "Variable value", enter your specific OpenAI API key.
   - Click "OK" to save the new environment variable.

3. Make sure to restart any services or applications that will use this key, so they can access the updated environment variable.

# 📦 Usage

#### 1. Launch API
     uvicorn API:app --reload

#### 2. Access Documentation
    127.0.0.1:8000/docs