from typing import Literal

from config import *
from database import open_db, index_init
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langchain.tools import tool


@tool
def get_context_from_db(query: str, k: int = 4) -> list[str]:
    """Возвращает top-k релевантных фрагментов.
    query: запрос пользователя
    k: количество фрагментов (1..10)
    Использовать если точно не уверен в ответе
    """
    index_init()
    safe_k = max(1, min(int(k), 10))
    db = open_db()
    try:
        data = db.similarity_search(query, k=safe_k)
        data = [doc.page_content for doc in data]
        data.append(f"Запрос пользователя: {query}")
        return data
    finally:
        db._connection.close()


@tool
def get_all_context_from_db() -> list[str]:
    """Возвращает весь контекст из базы.
    Использовать в крайних случаях если контекст утерян.
    Использовать только если get_context_from_db не дал ответа.
    Никогда не использовать для приветствий/общих вопросов.
    """
    index_init()
    db = open_db()
    try:
        rows = db._connection.execute(
            f"SELECT text FROM {db._table} ORDER BY rowid"
        ).fetchall()
        return [row["text"] for row in rows]
    finally:
        db._connection.close()


tools = [get_all_context_from_db, get_context_from_db]

model = init_chat_model(
    model=f"{PROVIDER}:{MODEL}",
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=TEMPERATURE,
).bind_tools(tools, tool_choice="auto")


def llm_call(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END

def base_load(state: MessagesState):
    for msg in state["messages"]:
        if isinstance(msg, SystemMessage) and "Предзагруженная база знаний:" in msg.content:
            return {}

    all_ctx = get_all_context_from_db.invoke({})

    return {
        "messages": [SystemMessage(content="Предзагруженная база знаний:" + "\n\n" + "\n\n".join(all_ctx))]
    }

def default_state() -> None:
    global state
    state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
        ]
    }

tool_node = ToolNode(tools)
agent_builder = StateGraph(MessagesState)

agent_builder.add_node("base_load", base_load)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

agent_builder.add_edge(START, "base_load")
agent_builder.add_edge("base_load", "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue)
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()

state: MessagesState = {}
default_state()


def chat_once(user_text: str) -> str:
    global state
    state["messages"].append(HumanMessage(user_text))
    result = agent.invoke(state)
    state = result

    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str) and content.strip():
                return content

    return ""


if __name__ == "__main__":
    while True:
        content = input(">> ")
        print(chat_once(content))
