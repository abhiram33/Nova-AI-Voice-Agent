# Nova AI Voice Agent

A production-ready voice-powered AI assistant built with Python, Streamlit, and Groq.

## Features

- Speech-to-text input using SpeechRecognition
- AI-powered responses via Groq LLM API
- Text-to-speech output using pyttsx3
- Interactive Streamlit UI

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Environment Variables

| Variable      | Description          |
|---------------|----------------------|
| `GROQ_API_KEY` | Your Groq API key   |

## License

MIT
