import os

from openai import OpenAI

key = 'Post you key here'


os.environ["OPENAI_API_KEY"] = key

class CGPClient:
    def __init__(self):
        self.client = OpenAI(api_key=key)


    def get_boolean_response(self, prompt):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=300,
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content == "True"