import os
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# Step 1: Define the response format in a Pydantic model
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

# Step 2: Call the model
completion = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You're a helpful assistant."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    response_format=CalendarEvent,
)

# Step 3: Parse the response
event = completion.choices[0].message.parsed
print(event.name)
print(event.date)
print(event.participants)