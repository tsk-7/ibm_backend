import os
import httpx
from dotenv import load_dotenv

# Load .env
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = os.getenv("OPENROUTER_BASE_URL")
model = os.getenv("OPENROUTER_MODEL")

print("========================================")
print("OpenRouter + NVIDIA Nemotron Test")
print("========================================")

print("API KEY:", "LOADED" if api_key else "MISSING")
print("BASE URL:", base_url)
print("MODEL:", model)

# Check configuration
if not api_key:
    raise SystemExit("ERROR: OPENROUTER_API_KEY is missing")

if not base_url:
    raise SystemExit("ERROR: OPENROUTER_BASE_URL is missing")

if not model:
    raise SystemExit("ERROR: OPENROUTER_MODEL is missing")

print("\nSending request to NVIDIA Nemotron...")
print("Please wait...")

try:
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Say hello in one short sentence."
                }
            ],
            "max_tokens": 20,
        },
        timeout=120,
    )

    print("\nSTATUS:", response.status_code)

    if response.status_code == 200:

        data = response.json()

        answer = data["choices"][0]["message"]["content"]

        print("\nNVIDIA RESPONSE:")
        print("----------------------------------------")
        print(answer)
        print("----------------------------------------")

        print("\nSUCCESS!")
        print("OpenRouter + NVIDIA Nemotron is working.")

    else:

        print("\nOPENROUTER ERROR:")
        print("----------------------------------------")
        print(response.text)
        print("----------------------------------------")

except httpx.TimeoutException:

    print("\nERROR: REQUEST TIMED OUT")
    print("The NVIDIA model did not respond within 120 seconds.")

except httpx.RequestError as e:

    print("\nERROR: NETWORK REQUEST FAILED")
    print(str(e))

except Exception as e:

    print("\nERROR:")
    print(type(e).__name__)
    print(str(e))