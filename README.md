# 🦅 Web-Scrapper-with-AI

A powerful, Streamlit-based web scraping tool that combines **Selenium**, **BeautifulSoup**, and **LLaMA 3.2 (via Ollama)** to extract meaningful insights from website content — with a built-in chatbot for interactive Q&A!

---

## 🔧 Features

- 🌐 Scrape any website using **Selenium + BeautifulSoup**
- 🧹 Clean and structure DOM content automatically
- 🧠 Ask **LLMs (LLaMA 3.2)** to parse and extract insights
- 💬 Built-in **chatbot interface** for natural language queries
- 💻 **Streamlit UI** — fast, simple, and user-friendly

---

## 📁 Project Structure

```
Web-Scrapper-with-AI/
│
├── Firefoxdriver/             # Firefox geckodriver directory
│   └── geckodriver.exe
│
├── __pycache__/               # Python cache
├── .env                       # Environment variables (set geckodriver path)
├── requirements.txt           # Python dependencies
├── main.py                    # Streamlit frontend
├── scrape.py                  # Scraper & DOM cleaning logic
├── parse.py                   # LLaMA-powered content parser
├── OllamaChatBot.py           # Chatbot using LLaMA 3.2
├── eagle.bmp                  # App icon
├── image1.png                 # Optional preview/logo
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Rahul-18r/Web-Scrapper-with-AI.git
cd Web-Scrapper-with-AI
```

### 2. Configure Environment

Create a `.env` file and specify the path to your `geckodriver.exe`:

```env
SBR_WEBDRIVER=absolute/path/to/Firefoxdriver/geckodriver.exe
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Ensure [Ollama](https://ollama.com/) and the `llama3` model are installed and running locally:

```bash
ollama run llama3
```

### 4. Launch the App

```bash
streamlit run main.py
```

---

## 🚀 How It Works

### 🔍 **Home Tab**
- Input the URL of any website
- Scrape and clean up raw HTML
- Specify what data you want (e.g., product names, blog titles)
- LLaMA processes and extracts your requested info

### 💬 **Chatbot Tab**
- Chat with the LLaMA 3.2 model directly
- Ask questions about scraped content or general topics

---

## 🧪 Example Use Cases

- 🔹 "Extract all blog post titles and dates from this page."
- 🔹 "Summarize all product names and prices."
- 🔹 "What is the main topic of this website?"

---

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **Selenium**
- **BeautifulSoup**
- **LangChain**
- **Ollama + LLaMA 3.2**

---

## 🪪 License

Licensed under the [MIT License](LICENSE).