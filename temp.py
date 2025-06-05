import google.generativeai as genai
import os

# Ensure your API key is set as an environment variable or uncomment and replace directly
# genai.configure(api_key="YOUR_API_KEY") 
# If you're setting it as an environment variable, make sure it's accessible here.
# For example, if you set GOOGLE_API_KEY in your shell, you might need:
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
else:
    print("GOOGLE_API_KEY environment variable not set. Please set it or pass the key directly.")
    exit() # Exit if key is not set to prevent further errors

print("Available Gemini models:")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"- {m.name} (Supports generateContent)")