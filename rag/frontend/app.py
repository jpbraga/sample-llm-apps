from typing import Any, Dict, List
import streamlit as st
from retrievers import retriever

def get_reasoning_logs(reasoning_steps: Dict[str, Any]) -> List[str]:
    logs = []
    intermediate_steps = reasoning_steps.get('intermediate_steps', [])
    for step in intermediate_steps:
        agent_action = step[0]
        logs.append(agent_action.log)

    return logs

st.title("RAG Chatbot")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'reasoning' not in st.session_state:
    st.session_state['reasoning'] = []

user_input = st.text_input("You:", "")

intermediate_steps = []
if st.button("Send") and user_input.strip() != "":
    with st.spinner("Agent is thinking..."):
        try:
            response = retriever.retrieve(user_input, st.session_state['messages'])
            # intermediate_steps = get_reasoning_logs(response)
            st.session_state['messages'].append(("user", user_input))
            st.session_state['messages'].append(("assistant", response))
        except Exception as e:
            st.error(f"Error: {e}")

# Conversa
print(st.session_state['messages'])
for sender, message in st.session_state['messages']:
    if sender == "user":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Agent:** {message}")

# # Reasoning
# st.subheader("Reasoning Steps")
# for actions in intermediate_steps:
#     st.text(actions)


