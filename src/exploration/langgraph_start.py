from dotenv import load_dotenv
import os

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.tools.tavily_search import TavilySearchResults

import json

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import MemorySaver

from typing import Literal
from langchain_core.messages import AIMessage, ToolMessage

from pydantic import BaseModel

class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str

# Load environment variables from .env file
load_dotenv()

# Access the API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"]="pr-sg-tutorial"

class State(TypedDict):
    messages: Annotated[list, add_messages]
    # This flag is new
    ask_human: bool


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str


tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatGoogleGenerativeAI(model="gemini-pro")
# We can bind the llm to a tool definition, a pydantic model, or a json schema
llm_with_tools = llm.bind_tools(tools + [RequestAssistance])


def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == RequestAssistance.__name__
    ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=[tool]))


def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )


def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):
        # Typically, the user will have updated the state during the interrupt.
        # If they choose not to, we will include a placeholder ToolMessage to
        # let the LLM continue.
        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        # Append the new messages
        "messages": new_messages,
        # Unset the flag
        "ask_human": False,
    }


graph_builder.add_node("human", human_node)


def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    # Otherwise, we can route as before
    return tools_condition(state)


graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", "__end__": "__end__"},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.set_entry_point("chatbot")
memory = MemorySaver()
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["human"],
)


try:
    output_path = "./graph_output.png"
    graph.get_graph().draw_mermaid_png(output_file_path=output_path)

    print(f"Graph saved to {output_path}")
except Exception:
    # This requires some extra dependencies and is optional
    pass


# def stream_graph_updates(user_input: str):
#     for event in graph.stream({"messages": [("user", user_input)]}):
#         for value in event.values():
#             print("Assistant:", value["messages"][-1].content)


# while True:
#     try:
#         user_input = input("User: ")
#         if user_input.lower() in ["quit", "exit", "q"]:
#             print("Goodbye!")
#             break

#         stream_graph_updates(user_input)
#     except:
#         # fallback if input() is not available
#         user_input = "What do you know about LangGraph?"
#         print("User: " + user_input)
#         stream_graph_updates(user_input)
#         break

# config = {"configurable": {"thread_id": "1"}}

# user_input = "Hi there! My name is Will."

# # The config is the **second positional argument** to stream() or invoke()!
# events = graph.stream(
#     {"messages": [("user", user_input)]}, config, stream_mode="values"
# )
# for event in events:
#     event["messages"][-1].pretty_print()

# user_input = "Remember my name?"

# # The config is the **second positional argument** to stream() or invoke()!
# events = graph.stream(
#     {"messages": [("user", user_input)]}, config, stream_mode="values"
# )
# for event in events:
#     event["messages"][-1].pretty_print()

# user_input = "I'm learning LangGraph. Could you do some research on it for me?"
# config = {"configurable": {"thread_id": "1"}}
# # The config is the **second positional argument** to stream() or invoke()!
# events = graph.stream(
#     {"messages": [("user", user_input)]}, config, stream_mode="values"
# )
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()

# snapshot = graph.get_state(config)
# existing_message = snapshot.values["messages"][-1]
# existing_message.pretty_print()

# answer = (
#     "LangGraph is a library for building stateful, multi-actor applications with LLMs."
# )
# new_messages = [
#     # The LLM API expects some ToolMessage to match its tool call. We'll satisfy that here.
#     ToolMessage(content=answer, tool_call_id=existing_message.tool_calls[0]["id"]),
#     # And then directly "put words in the LLM's mouth" by populating its response.
#     AIMessage(content=answer),
# ]

# new_messages[-1].pretty_print()
# graph.update_state(
#     # Which state to update
#     config,
#     # The updated values to provide. The messages in our `State` are "append-only", meaning this will be appended
#     # to the existing state. We will review how to update existing messages in the next section!
#     {"messages": new_messages},
# )

# graph.update_state(
#     config,
#     {"messages": [AIMessage(content="I'm an AI expert!")]},
#     # Which node for this function to act as. It will automatically continue
#     # processing as if this node just ran.
#     as_node="chatbot",
# )

# snapshot = graph.get_state(config)
# print(snapshot.values["messages"][-3:])
# print(snapshot.next)

# user_input = "I'm learning LangGraph. Could you do some research on it for me?"
# config = {"configurable": {"thread_id": "2"}}  # we'll use thread_id = 2 here
# events = graph.stream(
#     {"messages": [("user", user_input)]}, config, stream_mode="values"
# )
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()

# snapshot = graph.get_state(config)
# existing_message = snapshot.values["messages"][-1]
# print("Original")
# print("Message ID", existing_message.id)
# print(existing_message.tool_calls[0])
# new_tool_call = existing_message.tool_calls[0].copy()
# new_tool_call["args"]["query"] = "LangGraph human-in-the-loop workflow"
# new_message = AIMessage(
#     content=existing_message.content,
#     tool_calls=[new_tool_call],
#     # Important! The ID is how LangGraph knows to REPLACE the message in the state rather than APPEND this messages
#     id=existing_message.id,
# )

# print("Updated")
# print(new_message.tool_calls[0])
# print("Message ID", new_message.id)
# graph.update_state(config, {"messages": [new_message]})

# print("\n\nTool calls")
# graph.get_state(config).values["messages"][-1].tool_calls

# events = graph.stream(None, config, stream_mode="values")
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()

# events = graph.stream(
#     {
#         "messages": (
#             "user",
#             "Remember what I'm learning about?",
#         )
#     },
#     config,
#     stream_mode="values",
# )
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()

config = {"configurable": {"thread_id": "1"}}
events = graph.stream(
    {
        "messages": [
            ("user", "I'm learning LangGraph. Could you do some research on it for me?")
        ]
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

events = graph.stream(
    {
        "messages": [
            ("user", "Ya that's helpful. Maybe I'll build an autonomous agent with it!")
        ]
    },
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()

to_replay = None
for state in graph.get_state_history(config):
    print("Num Messages: ", len(state.values["messages"]), "Next: ", state.next)
    print("-" * 80)
    if len(state.values["messages"]) == 6:
        # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
        to_replay = state

        print(to_replay.next)
        print(to_replay.config)     

# The `checkpoint_id` in the `to_replay.config` corresponds to a state we've persisted to our checkpointer.
for event in graph.stream(None, to_replay.config, stream_mode="values"):
    if "messages" in event:
        event["messages"][-1].pretty_print()   