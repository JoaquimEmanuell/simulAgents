import google.generativeai as genai
import os

genai.configure(api_key="AIzaSyADKhqHD-ofKyh1mGXTANih4KD9ocq4fXw")

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)