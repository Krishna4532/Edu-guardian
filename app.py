import streamlit as st
import os
from main import edu_guardian

st.set_page_config(page_title="Edu-Guardian: Multi-Account", layout="wide", page_icon="🛡️")

# --- 1. SIDEBAR: DYNAMIC USER PROFILES ---
with st.sidebar:
    st.header("👤 User Account")
    # user_name acts as the 'Account ID' for thread isolation
    user_name = st.text_input("Student Name", value="Krishna")
    
    # Custom interests allow the AI to move past just 'Cricket' analogies
    user_interest = st.text_input("Interests/Hobbies", value="Cricket")
    level = st.selectbox("Education Level", ["Primary", "Secondary", "University"])
    
    st.divider()
    if st.button("🗑️ Clear My History"):
        # Reset local session state
        st.session_state.messages = []
        st.rerun()

# --- 2. INITIALIZE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. UI HEADER ---
st.title(f"🛡️ Edu-Guardian: {user_name}'s Tutor")
st.markdown(f"**Mode:** Socratic | **Persona:** {user_interest} Enthusiast")

# --- 4. DISPLAY PERSISTENT CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "image" in msg:
            st.image(msg["image"], use_container_width=True)
        st.markdown(msg["content"])
        if "quiz" in msg:
            st.info(msg["quiz"])

# --- 5. NEW INPUT HANDLING ---
if prompt := st.chat_input(f"Hi {user_name}, what shall we learn today?"):
    # Append user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.status("🧠 Agents Collaborating...", expanded=True) as status:
        # CRITICAL: thread_id is now the user_name, ensuring isolated histories
        config = {"configurable": {"thread_id": user_name}}
        
        # Pass the dynamic profile to your LangGraph agents
        result = edu_guardian.invoke({
            "query": prompt, 
            "student_level": level, 
            "student_profile": f"Name: {user_name}, Interests: {user_interest}",
            "iterations": 0
        }, config)
        status.update(label="✅ Lesson Prepared!", state="complete")

    with st.chat_message("assistant"):
        response_data = {"role": "assistant", "content": result["response"]}
        
        if result.get("image_url"):
            st.image(result["image_url"], use_container_width=True)
            response_data["image"] = result["image_url"]
        
        st.markdown(result["response"])
        st.divider()
        st.info(result["quiz_question"])
        response_data["quiz"] = result["quiz_question"]
        
        # Save to history
        st.session_state.messages.append(response_data)
        
        # Optional interactive element
        if st.button("Submit Answer"):
            st.confetti()
            st.success(f"Great job, {user_name}!")
