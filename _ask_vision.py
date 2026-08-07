import base64, json, os, sys, urllib.request

img_path = sys.argv[1]
question = sys.argv[2] if len(sys.argv) > 2 else "Describe this figure."
with open(img_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
payload = {
    "model": "kimi-k3",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
                {"type": "text", "text": question},
            ],
        }
    ],
    "max_tokens": 2000,
}
req = urllib.request.Request(
    "https://api.moonshot.ai/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['MOONSHOT_API_KEY']}",
    },
)
with urllib.request.urlopen(req, timeout=180) as resp:
    data = json.loads(resp.read().decode())
msg = data["choices"][0]["message"]
print(msg.get("content") or msg.get("reasoning_content") or "(no content)")
