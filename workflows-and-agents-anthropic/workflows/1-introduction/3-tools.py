import os
import json
import requests
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

"""
docs: https://platform.openai.com/docs/guides/function-calling
"""

# Define the tool (function) that we want to call
def get_weather(latitude, longitude):
    """This is a publically available API that returns the weather for a given location."""
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
    )
    data = response.json()
    return data["current"]

# Step 1: Call model with get_weather tool defined
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current temperature for provided coordinates in celsius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }
]

system_prompt = "You are a helpful weather assistant."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "What's the weather like in Paris today?"},
]

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

# Step 2: Model decides to call function(s)
completion.model_dump()

# Step 3: Execute get_weather function
def call_function(name, args):
    if name == "get_weather":
        return get_weather(**args)
    else:
        raise ValueError(f"Unknown function: {name}")

for tool_call in completion.choices[0].message.tool_calls:
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    messages.append(completion.choices[0].message)

    result = call_function(function_name, function_args)
    messages.append(
        {
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id,
        }
    )

# Step 4: Supply result and call model again
class WeatherResponse(BaseModel):
    temperature: float = Field(description="The current temperature in celsius for the given location.")
    response: str = Field(description="A natural language response to the user's question.")

completion_2 = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=messages,
    response_format=WeatherResponse,
    tools=tools,
)

# Step 5: Check model response
final_response = completion_2.choices[0].message.parsed
print(final_response.temperature)
print(final_response.response)
