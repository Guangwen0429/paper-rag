from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "用一句话解释什么是 Transformer"}
    ]
)

print(response.choices[0].message.content)