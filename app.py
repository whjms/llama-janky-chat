from generator_factory import get_generator

MAX_CONTEXT = 2048
generator = get_generator("7B", "tokenizer.model", MAX_CONTEXT)

from flask import Flask, request
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    payload = request.get_json()
    try:
        prompt = payload["prompt"]
        max_gen_len = int(payload["max_gen_len"])
        temp = float(payload["temp"])
        top_p = float(payload["top_p"])
    except (KeyError, ValueError) as e:
        return { "error": repr(e) }, 400

    if len(prompt) > MAX_CONTEXT:
        return { "error": "Prompt is too long" }, 400

    print(payload)
    def on_gen(decoded: str):
        print(decoded)
    results = generator.generate([prompt],
        max_gen_len=max_gen_len,
        temperature=temp,
        top_p=top_p,
        gen_callback=on_gen)
    return f'"{results[0]}"'
