from openai import OpenAI

client = OpenAI(api_key="xxx", base_url="http://65.108.33.75:18000/v1")
response = client.completions.create(
    model="gpt-4o-mini",
    prompt="What is the capital of France?",
    max_tokens=50
)
print(response.choices[0].text)