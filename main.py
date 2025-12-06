import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
import os

#API Key
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

app = FastAPI(title="Make Me Brave - Agentic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def log_performance(topic: str, score: int, reasoning: str):
    """
    Logs the candidate's performance for a specific topic.
    Args:
        topic: The technical topic being discussed (e.g., 'Python', 'SQL').
        score: A score from 1-10 based on the accuracy of the answer.
        reasoning: Brief explanation for the score.
    """
    print(f"\n[AGENT ACTION] Grading User...")
    print(f"   Topic: {topic}")
    print(f"   Score: {score}/10")
    print(f"   Reason: {reasoning}\n")
    return {"status": "success", "logged_score": score}

tools_list = [log_performance]

model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    tools=tools_list
)


chat_session = model.start_chat(enable_automatic_function_calling=True)

# Agent Persona
INITIAL_INSTRUCTION = """
You are "Make Me Brave", an Agentic AI Interviewer.
Your goal is to interview the candidate AND evaluate them in real-time.

PROTOCOL:
1. Ask one technical question at a time.
2. When the user answers, you MUST use the `log_performance` tool to grade them (1-10) before asking the next question.
3. If the score is low (<5), ask an easier follow-up.
4. If the score is high (>8), ask a harder follow-up.
5. Keep your spoken responses encouraging but short.

Start by asking them to introduce themselves.
"""

try:
    chat_session.send_message(INITIAL_INSTRUCTION)
    print("SUCCESS: Agentic AI Connected!")
except Exception as e:
    print(f"WARNING: API Error: {e}")

class StudentResponse(BaseModel):
    text: str

# --- API ENDPOINTS ---

@app.get("/")
def home():
    return FileResponse('index.html')

@app.post("/interview/chat")
def interview_chat(response: StudentResponse):
    try:

        gemini_response = chat_session.send_message(response.text)
        return {
            "interviewer_text": gemini_response.text,
            "status": "success"
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

@app.post("/interview/reset")
def reset_interview():
    global chat_session
    try:
        chat_session = model.start_chat(enable_automatic_function_calling=True)
        chat_session.send_message(INITIAL_INSTRUCTION)
        return {"message": "Interview reset. Agent ready."}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
