import streamlit as st
from typing import TypedDict, Annotated
from operator import add
from src.initial_chain import classify_intent
from src.generate_code import generate_code
from src.explain_code import explain_code
from langgraph.graph import StateGraph, END

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Russo+One&display=swap');

    body {
        background-color: #001f3f;
        color: #ffffff;
        font-family: 'Russo One', sans-serif;
    }

    .main-title {
        font-size: 48px;
        color: #a8dadc;  /* soft baby blue */
        margin-bottom: 0px;
        font-family: 'Russo One';
    }

    .subtitle {
        font-size: 18px;
        color: #ffffff;
        margin-top: 5px;
    }

    .result-section {
        background-color: #012a4a;
        padding: 15px;
        border-radius: 10px;
        margin-top: 10px;
    }

    .input-area {
        background-color: #012a4a;
        border-radius: 10px;
        padding: 10px;
    }

    .header-column {
        padding: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([2, 1])  # Title gets more space, image smaller
with col1:
    st.markdown('<div class="main-title">PyCoder</div>', unsafe_allow_html=True)
    st.markdown(
    '<div class="subtitle">'
    "Your favourite chatbot for coding.<br>"
    "Don't know how to write it? Generate with PyCoder!<br>"
    "Don't understand code? Let PyCoder explain it simply the way you won't need to ask twice."
    '</div>',
    unsafe_allow_html=True
    )

class State(TypedDict):
    messages: Annotated[list[str], add]      # Allow appending messages
    user_input: str
    intent: str
    code_output: Annotated[str, add]        # Accumulate if needed
    explanation: Annotated[str, add]        # Accumulate if needed

def route_by_intent(state):
    """Determines the next node based on the 'intent' in the state."""
    if state["intent"] == "generate_code":
        return {"next_node": "generate", **state}
    elif state["intent"] == "explain_code":
        return {"next_node": "explain", **state}
    else:
        return {"next_node": END, **state}
    
graph = StateGraph(State)

graph.add_node("classify", classify_intent)
graph.add_node("route", route_by_intent)
graph.add_node("generate", generate_code)
graph.add_node("explain", explain_code)

# --- Add edges (unconditional) ---
graph.add_edge("classify", "route")
graph.add_edge("route", "generate")
graph.add_edge("route", "explain")
graph.add_edge("generate", END)
graph.add_edge("explain", END)

graph.set_entry_point("classify")

# --- Compile the graph ---
app_graph = graph.compile()

# --- Stream    lit input ---
user_input = st.text_area("Enter your command or code:", height=150, key="input_area")

if st.button("Run"):
    if user_input.strip():
        # Initialize state
        state = {
            "messages": [],
            "user_input": user_input,
            "intent": "",
            "code_output": "",
            "explanation": ""
        }
        # Invoke the graph
        result = app_graph.invoke(state)

        st.markdown('<div class="result-section">', unsafe_allow_html=True)
        # Display only relevant outputs
        if result["intent"] == "generate_code":
            st.subheader("Generated Code:")
            st.code(result["code_output"], language="python")
        elif result["intent"] == "explain_code":
            st.subheader("Explanation:")
            st.markdown(result["explanation"])
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some input.")