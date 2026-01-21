# 🛡️ Edu-Guardian: AI Socratic Tutor

**Edu-Guardian** is an autonomous, multi-agent educational platform built for **Conclave 4.0**. It uses a Socratic tutoring method to personalize learning for students like Krishna, turning complex topics into relatable analogies (like Cricket).

## 🚀 Key Features
- **Agentic Orchestration:** Built with **LangGraph** to coordinate specialized agents (Sentinel, Web Search, Tutor, Judge, and Quiz Master).
- **Self-Correction Loop:** An integrated "LLM-as-a-Judge" audits every lesson for hallucinations. If it fails, the Tutor Agent automatically regenerates the response.
- **Multimodal RAG:** Combines local knowledge with real-time web search and educational diagrams.
- **Privacy-First:** Redacts sensitive PII (emails/phones) before processing queries.

## 🛠️ Tech Stack
- **Framework:** LangGraph (Stateful Multi-Agent Orchestration)
- **LLM:** Llama 3.3 70B (via Groq for sub-second inference)
- **Search Engine:** Tavily AI (Optimized for RAG)
- **UI:** Streamlit

## 🏗️ Architecture
The system follows a cyclic graph workflow:
1. **Memory Agent:** Loads student preferences.
2. **Sentinel:** Checks for local context.
3. **Web Search:** Fetches real-time diagrams/data if local context is missing.
4. **Tutor:** Crafts a Socratic lesson with metaphors.
5. **Judge:** Scores the response for factual accuracy.
6. **Quiz Master:** Generates an interactive MCQ for knowledge check.



## 📥 Installation & Setup
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Set up environment variables in `.env` (or Streamlit Secrets):
   - `GROQ_API_KEY`
   - `TAVILY_API_KEY`
4. Run the app: `streamlit run app.py`.
