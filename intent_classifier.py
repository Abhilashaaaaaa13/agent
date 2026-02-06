import os
import json
import re
from dotenv import load_dotenv
from google.genai import Client

# Load environment variables
load_dotenv()

MODEL_NAME = "gemini-2.5-pro"

INTENTS = {
    "TASK_ASSIGNMENT",
    "VIEW_EMPLOYEE_PERFORMANCE",
    "VIEW_EMPLOYEES_UNDER_MANAGER",
    "UPDATE_TASK_STATUS",
    "VIEW_PENDING_TASKS",
    "ADD_USER",
    "DELETE_USER"
}

INTENT_CLASSIFIER_PROMPT = """
You are an Intent Classification Agent for a Task Management System.

Your task:
- Read the user's message (can be short, long, vague, or complex)
- Understand what the user is trying to do
- Identify the PRIMARY intent
- Return EXACTLY ONE intent from the allowed list

ALLOWED INTENTS (ONLY THESE):
TASK_ASSIGNMENT
VIEW_EMPLOYEE_PERFORMANCE
VIEW_EMPLOYEES_UNDER_MANAGER
UPDATE_TASK_STATUS
VIEW_PENDING_TASKS
ADD_USER
DELETE_USER

Important rules:
- Do NOT rely on keyword matching
- Focus on the meaning of the message
- If multiple actions are mentioned, choose the MAIN action
- Never invent a new intent

Return STRICT JSON only.
No markdown. No extra text.

Format:
{
  "intent": "<ONE_INTENT>",
  
  "reasoning": "short explanation"
}
"""

def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY missing")

    return Client(api_key=api_key)

def clean_json(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

def fallback_intent(user_message: str) -> str:
    msg = user_message.lower()

    if "close" in msg or "reopen" in msg or "status" in msg:
        return "UPDATE_TASK_STATUS"
    if "pending" in msg:
        return "VIEW_PENDING_TASKS"
    if "performance" in msg:
        return "VIEW_EMPLOYEE_PERFORMANCE"
    if "add" in msg:
        return "ADD_USER"
    if "delete" in msg or "remove" in msg:
        return "DELETE_USER"
    if "team" in msg or "employees" in msg:
        return "VIEW_EMPLOYEES_UNDER_MANAGER"

    return "TASK_ASSIGNMENT"

def intent_classifier(user_message: str) -> dict:
    client = init_gemini()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=f"{INTENT_CLASSIFIER_PROMPT}\n\nUser message:\n{user_message}"
    )

    cleaned = clean_json(response.text)

    try:
        result = json.loads(cleaned)
        intent = result.get("intent")
    except Exception:
        intent = None

    if intent not in INTENTS:
        return {
            "intent": fallback_intent(user_message),
            
            "reasoning": "Fallback rule-based classification"
        }

    return result

# -----------------------------
# REAL USER INPUT
# -----------------------------
if __name__ == "__main__":
    print("Task Manager Intent Classifier")
    print("Type 'exit' to quit\n")

    while True:
        user_query = input("User: ").strip()

        if user_query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        if not user_query:
          
            continue

        result = intent_classifier(user_query)
        print(json.dumps(result, indent=2))
        print()
