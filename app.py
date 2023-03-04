from generator_factory import get_generator
generator = get_generator("7B", "tokenizer.model", 2048)

from flask import Flask
app = Flask(__name__)

@app.route("/generate")
def hello_world():
    prompts = ["The capital of Germany is the city of"]
    def on_gen(decoded: str):
        print(decoded)
    results = generator.generate(prompts,
        max_gen_len=256,
        temperature=0.8,
        top_p=0.95,
        gen_callback=on_gen)
    return f"<p>{results}</p>"
