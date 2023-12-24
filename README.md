# GPT Manager

# Overview

# 🛠 Installation
### 1.Bulid from source
    git clone https://github.com/Mega-Gorilla/gpt-prompt-editor.git
    cd gpt-prompt-editor
    pip install -r requirements.txt

### 2. Setting up the OpenAI and Gemini Pro API Keys

To integrate OpenAI services and Gemini Pro functionalities into your application, follow these steps to set up your API keys:

1. **Accessing the Windows Environment Variables:**
   - Open the start menu and search for 'Environment Variables', then choose "Edit the system environment variables".
   - In the System Properties window, click on the "Environment Variables" button.

2. **Adding New Environment Variables:**
   - In the Environment Variables window, under the "User variables" section, click on the "New" button.

   **For the OpenAI API Key:**
   - For the "Variable name", enter `OPENAI_API_KEY`.
   - For the "Variable value", enter your specific OpenAI API key.

   **For the Gemini Pro API Key:**
   - Click on the "New" button again to add another environment variable.
   - For the "Variable name", enter `GEMINI_API_KEY`.
   - For the "Variable value", enter your specific Gemini Pro API key.

   - Click "OK" to save each new environment variable.

3. Make sure to restart any services or applications that will use these keys, so they can access the updated environment variables.

# 📦 Usage

#### 1. Launch API
     uvicorn API:app

#### 2. Access Documentation
    127.0.0.1:8000/docs