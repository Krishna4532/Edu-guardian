import streamlit as st
import os
from main import edu_guardian

st.set_page_config(page_title="Edu-Guardian: Conclave 4.0", layout="wide", page_icon="🛡️")

# --- 1. INITIALIZE CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. UI HEADER ---
st.title("🛡️ Edu-Guardian: AI Socratic Tutor")
st.markdown("### *Socratic | Multi-Agent | Personalized*")

with st.sidebar:
    st.header("Student Profile")
    level = st.selectbox("Education Level", ["Primary", "Secondary", "University"])
    st.info("**User:** Krishna\n**Interests:** Cricket, Visual Learning")
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# --- 3. DISPLAY PERSISTENT CHAT HISTORY ---
# This loop ensures old messages don't fade away
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "image" in msg:
            st.image(msg["image"], use_container_width=True)
        st.markdown(msg["content"])
        if "quiz" in msg:
            st.info(msg["quiz"])

# --- 4. NEW INPUT HANDLING ---
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with Agents
    with st.status("🧠 Agents Collaborating...", expanded=True) as status:
        config = {"configurable": {"thread_id": "conclave_final"}}
        result = edu_guardian.invoke({"query": prompt, "student_level": level, "iterations": 0}, config)
        status.update(label="✅ Lesson Prepared!", state="complete")

    with st.chat_message("assistant"):
        response_data = {"role": "assistant", "content": result["response"]}
        
        # Display Image
        if result.get("image_url"):
            st.image(result["image_url"], use_container_width=True)
            response_data["image"] = result["image_url"]
        
        # Display Lesson
        st.markdown(result["response"])
        
        # Display Quiz
        st.divider()
        st.info(result["quiz_question"])
        response_data["quiz"] = result["quiz_question"]
        
        # Save to history so it persists on next run
        st.session_state.messages.append(response_data)
        
        if st.button("Submit Answer"):
            st.confetti()
            st.success("Great job engaging with the lesson!")