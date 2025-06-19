# 🦅 Web-Scrapper-with-AI

A powerful, Streamlit-based web scraping tool that uses **LLM (LLaMA 3.2 via Ollama)** to extract meaningful insights from website content — with chatbot functionality included!

## 🔧 Features

- 🌐 Scrape any website using Selenium + BeautifulSoup
- 🧹 Clean and structure DOM content
- 🧠 Ask LLMs to parse & extract custom info from scraped text
- 🤖 Built-in chatbot with LLaMA 3.2 for general-purpose QA
- 💻 Streamlit interface — fast, simple, and clean

## 📁 Project Structure

Web-Scrapper-with-AI/
│
├── Firefoxdriver/             # Firefox geckodriver location
│   └── geckodriver.exe
│
├── __pycache__/               # Python bytecode cache
│
├── .env                       # Environment variables (add geckodriver path)
├── requirements.txt           # Python dependencies
├── main.py                    # Streamlit frontend logic
├── scrape.py                  # Website scraping and DOM cleaning
├── parse.py                   # Content parsing using LLaMA
├── OllamaChatBot.py           # Chatbot with LLaMA 3.2
├── eagle.bmp                  # App icon
├── image1.png                 # (Optional) UI preview or logo

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Rahul-18r/Web-Scrapper-with-AI.git
cd Web-Scrapper-with-AI
```

### 2. Set Up Your Environment

Create a `.env` file with your Firefox WebDriver path:

```env
SBR_WEBDRIVER=absolute/path/to/Firefoxdriver/geckodriver.exe
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama and the llama3 model are installed and running locally:

```bash
ollama run llama3
```

### 3. Run the App

```bash
streamlit run main.py
```

## ✨ How It Works

Home Tab:
- Input a website URL
- Scrape and clean content
- Describe what data you want to extract
- Let LLaMA parse and return results

Chatbot Tab:
- Enable and interact with the chatbot using LLaMA 3.2

## 🧪 Example Use Case

Example Prompt:  
"Extract all blog post titles and dates."  
"Summarize all product names and prices from this page."

## 🛠 Tech Stack

- Python
-  Streamlit
- Selenium
-  BeautifulSoup
- LangChain
-  Ollama (LLaMA 3.2)

## 🪪 License

Licensed under the MIT License.
