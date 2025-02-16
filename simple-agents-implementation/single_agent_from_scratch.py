from openai import OpenAI
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
from langchain_community.tools import DuckDuckGoSearchResults
import numpy as np
import json
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM API keys
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL")

# Google Search API credentials
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

# Memory storage
short_term_memory = []  # Stores the last 3 dialogues
long_term_memory = []  # Stores all dialogue vectors

# Sentence embedding model
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# LLM client
client = OpenAI(
    api_key=deepseek_api_key, 
    base_url=deepseek_base_url,
)

# Utility tools
tools = {
    "calculator": lambda expression: eval(expression),  # Evaluates mathematical expressions
    "search": lambda query: search(query),  # Calls the search function
}

def build_prompt(q):
    """
    Build a prompt that explicitly asks the model to respond with tool calls in JSON format.
    """
    prompt = "Recent chat:\n" + "\n".join([f"Q: {q}\nA: {a}\n" for q, a in short_term_memory[-3:]])

    # Retrieve the most relevant memory using dot-product similarity
    if long_term_memory:
        q_vec = encoder.encode(q)
        scores = [np.dot(q_vec, m) for m in long_term_memory]
        best_idx = np.argmax(scores)
        prompt += f"\nRelated memory:\n{long_term_memory[best_idx]}"

    # Instruction to enforce structured responses
    prompt += """
    Please reason step by step. 
    1. The "thinking" phase: You think carefully about the user's problem
    2. Tool Invocation Phase: Select the tool that can be invoked and output the parameters required by the corresponding tool
    If you think a tool (like 'calculator' or 'search') is needed, respond with a JSON object:
    {
        "tool": "<tool_name>",
        "params": "<parameters>"
    } 
    <tool_name> only has calculator and search. You don't generate anything else
    For example, if you need to calculate, respond with something like:
    {"tool": "calculator", "params": "3 + 4"}
    """
    return prompt + f"\nNew question: {q}"

def get_api_response(prompt):
    """
    Sends the prompt to the LLM and returns the response.
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )
    return response.choices[0].message.content

def update_memories(q, response):
    """
    Updates memory with the latest conversation.
    """
    short_term_memory.append((q, response))
    long_term_memory.append(encoder.encode(f"Q: {q}, A: {response}"))

    # Maintain short-term memory size limit
    if len(short_term_memory) > 3:
        short_term_memory.pop(0)

def extract_tool_request(response):
    """
    Attempts to parse the LLM response as a JSON tool request.
    """
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group(0))  # Expecting a JSON tool request
    except json.JSONDecodeError:
        return None  # If no JSON found, return None


def call_tool(tool_name, params):
    """
    Call the appropriate tool based on the tool name and parameters.
    """
    if tool_name in tools:
        # Call the tool and return its result
        return tools[tool_name](params)
    return None

def search(query):
    """
    Executes a search and returns the top results.
    """
    # # Use Google Search API
    # service = build("customsearch", "v1", developerKey=google_api_key)
    # res = service.cse().list(q=query, cx=google_cse_id, num=5).execute()

    # if "items" in res:
    #     return "\n".join(f"{item['title']} - {item['link']}" for item in res["items"])
    # return "No results found."

    # Use DuckDuckGo Search API
    search_tool = DuckDuckGoSearchResults()
    result = search_tool.invoke(query)
    return result

def agent(q):
    """
    Main agent function to process the input and either respond directly or call tools.
    """
    prompt = build_prompt(q)
    resp = get_api_response(prompt)
    # print(f"LLM response: {resp}")  # Debugging step to see what the model returns

    tool_request = extract_tool_request(resp)

    # If a tool request was found, call the tool and return the result
    if tool_request:
        tool_name = tool_request.get("tool")
        params = tool_request.get("params")
        if tool_name and params:
            tool_result = call_tool(tool_name, params)
            return f"Tool result: {tool_result}"
            resp = get_api_response(f"Organize the tool results: {tool_result} according to the question: {q}.")

    # If no tool was requested, just return the model's response
    update_memories(q, resp)
    return resp

if __name__ == "__main__":
    while True:
        q = input("🧑: ")
        # q = "Who won the latest FIFA Best Men's Player award?"
        if q.lower() == "quit": break
        response = agent(q)
        print(f"🤖: {response}")

