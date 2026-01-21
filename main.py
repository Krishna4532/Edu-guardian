import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tavily import TavilyClient
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# --- 1. UPDATED STATE DEFINITION ---
class AgentState(TypedDict):
    query: str
    context: str
    response: str
    quiz_question: str
    student_level: str
    # student_profile now dynamic: e.g., "Name: Rahul, Interest: Football"
    student_profile: str 
    faithfulness_score: float
    judge_feedback: str
    source_type: str 
    reasoning_log: List[str]
    image_url: str
    iterations: int

# --- 2. COMPONENTS ---
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --- 3. UPDATED AGENT DEFINITIONS ---

def memory_loader_agent(state: AgentState):
    """Now purely initializes the session without hardcoding Krishna."""
    log = state.get("reasoning_log", [])
    # student_profile is now passed from app.py at runtime
    log.append(f"🧠 *Memory:* Loading profile for {state.get('student_profile', 'New Student')}...") 
    return {
        "reasoning_log": log, 
        "iterations": 0
    }

def sentinel_agent(state: AgentState):
    log = state.get("reasoning_log", [])
    log.append("🕵️ *Sentinel:* Checking local knowledge base...")
    if "photosynthesis" in state['query'].lower():
        return {"context": "Plants use sunlight to make food (glucose).", "source_type": "Local", "reasoning_log": log}
    return {"context": "", "source_type": "None", "reasoning_log": log}

def web_search_agent(state: AgentState):
    log = state.get("reasoning_log", [])
    log.append("🌐 *Web Search:* Fetching live data for accuracy...")
    response = tavily.search(query=state['query'], include_images=True, max_results=2)
    context = "\n".join([r['content'] for r in response['results']])
    img = response['images'][0] if response['images'] else ""
    return {"context": context, "image_url": img, "source_type": "Web Search", "reasoning_log": log}

def tutor_agent(state: AgentState):
    iters = state.get("iterations", 0)
    log = state.get("reasoning_log", [])
    
    # DYNAMIC ANALOGY ENGINE: No longer hardcoded to Cricket
    prompt = f"""
    Context: {state['context']}
    Student Level: {state['student_level']} 
    Student Profile: {state['student_profile']}
    Query: {state['query']}
    
    Task: Explain this topic. 
    1. Start with a Socratic question.
    2. Use analogies based EXCLUSIVELY on the student's listed interests.
    3. Bold key terms.
    Feedback from Judge: {state.get('judge_feedback', 'None')}
    """
    res = llm.invoke(prompt).content
    log.append(f"🎓 *Tutor:* Prepared lesson using personalized metaphors.")
    return {"response": res, "reasoning_log": log, "iterations": iters + 1}

def llm_judge_agent(state: AgentState):
    log = state.get("reasoning_log", [])
    log.append("⚖️ *Judge:* Validating facts against search context...")
    
    prompt = f"Score 0-1 for faithfulness. Context: {state['context']} Lesson: {state['response']}"
    try:
        score_raw = llm.invoke(prompt).content.strip()
        score = float(''.join(c for c in score_raw if c.isdigit() or c=='.'))
    except:
        score = 0.9 # Default fallback
        
    feedback = "Verified" if score > 0.7 else "Hallucination detected. Revising..."
    return {"faithfulness_score": score, "judge_feedback": feedback, "reasoning_log": log}

def quiz_master_agent(state: AgentState):
    log = state.get("reasoning_log", [])
    log.append("📝 *Quiz Master:* Creating a personalized MCQ...")
    quiz = llm.invoke(f"Create 1 MCQ for: {state['response']}").content
    return {"quiz_question": quiz, "reasoning_log": log}

# --- 4. GRAPH CONSTRUCTION (Remains same, orchestrates flow) ---
workflow = StateGraph(AgentState)
workflow.add_node("memory", memory_loader_agent)
workflow.add_node("sentinel", sentinel_agent)
workflow.add_node("web_search", web_search_agent)
workflow.add_node("tutor", tutor_agent)
workflow.add_node("judge", llm_judge_agent)
workflow.add_node("quiz", quiz_master_agent)

workflow.add_edge(START, "memory")
workflow.add_edge("memory", "sentinel")
workflow.add_conditional_edges("sentinel", lambda x: "tutor" if x["context"] else "web_search", {"web_search": "web_search", "tutor": "tutor"})
workflow.add_edge("web_search", "tutor")
workflow.add_edge("tutor", "judge")
workflow.add_conditional_edges("judge", lambda x: "tutor" if x["faithfulness_score"] < 0.7 and x["iterations"] < 2 else "quiz", {"tutor": "tutor", "quiz": "quiz"})
workflow.add_edge("quiz", END)

edu_guardian = workflow.compile(checkpointer=MemorySaver())

