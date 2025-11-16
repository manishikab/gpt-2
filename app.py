# app.py
import streamlit as st
import torch
import torch.nn.functional as F
import tiktoken
from gpt2 import GPT

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT.from_pretrained("gpt2").to(device).eval()
enc = tiktoken.get_encoding("gpt2")

def generate(prompt, max_length=100):
    tokens = enc.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1)
            xcol = torch.gather(topk_indices, dim=-1, index=ix)
            x = torch.cat((x, xcol), dim=1)
    return enc.decode(x[0].tolist())

# UI
st.title("Mini GPT-2 Chatbot 🤖")
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    reply = generate(user_input, max_length=80)
    st.session_state.messages.append({"role": "assistant", "content": reply})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
