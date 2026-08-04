"""
graph.py — the LangGraph agent that ties classify_flower, retrieve_knowledge,
and get_weather together, using Gemini (via langchain-google-genai) to decide
which tool(s) to call and how to combine their outputs into a final answer.

Using Gemini here specifically because Google AI Studio has a genuine free
API tier (no credit card, no expiration) — makes sense for a student/portfolio
project where you don't want ongoing API costs during development.

This is a standard ReAct-style loop:
    agent (Gemini decides what to do)
       -> if it wants a tool: tools node executes it, result goes back to agent
       -> if it has enough info: produces a final answer, graph ends

Example flow for "My sunflower leaves are yellow, what should I do?" (image attached):
    1. agent sees image path in the message -> calls classify_flower
    2. tool result comes back: "sunflower", confidence 0.91
    3. agent calls retrieve_knowledge(query="yellow leaves disease", species="sunflower")
    4. tool result comes back with relevant passages
    5. agent synthesizes a grounded final answer citing what it found
"""
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

from tools import ALL_TOOLS

SYSTEM_PROMPT = """You are FloraSense AI, a botanical research assistant.

You have three tools:
- classify_flower: identify a flower species from an uploaded image
- retrieve_knowledge: search a curated botanical knowledge base (care, diseases, pollinators, uses, native range)
- get_weather: check current weather/conditions for a location

Guidelines:
- If the user mentions or attaches an image, use classify_flower first to identify the species before answering.
- Always ground factual claims (diseases, care, pollinators, uses) in retrieve_knowledge rather than
  relying on your own general knowledge alone — cite the source when you use retrieved information.
- If classify_flower returns low confidence, tell the user the identification is uncertain rather
  than stating it as fact.
- If a question is about a specific location (e.g. "can this grow in Punjab?"), use get_weather to
  check current conditions, and reason about climate suitability using what the knowledge base says
  about the plant's native range and requirements.
- Be concise and practical — the user often just wants a clear next step (e.g. "treat with X" /
  "yes, but protect from Y").
"""


def build_agent(model_name: str = "gemini-flash-lite-latest", api_key: str = None):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        max_output_tokens=1024,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: MessagesState):
        messages = state["messages"]
        # Prepend the system prompt fresh each turn rather than storing it in
        # state — keeps it from being duplicated as the conversation grows.
        response = llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(ALL_TOOLS))

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # after a tool runs, let the agent decide the next step

    return graph.compile()
