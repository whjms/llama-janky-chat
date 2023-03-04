from logging.config import dictConfig
from threading import Lock
import time
import os
from flask import Flask, request, Response, render_template

from generator_factory import get_generator
from message_announcer import MessageAnnouncer

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '%(levelname)s %(asctime)s [%(module)s] %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

def load_generator():
    MAX_CONTEXT = 2048
    def env_var_with_default(name: str, default: str) -> str:
        try:
            return os.environ[name]
        except ValueError:
            return default

    checkpoint_dir = env_var_with_default("CHECKPOINT_DIR", "7B")
    tokenizer_path = env_var_with_default("TOKENIZER_PATH", "tokenizer.model")
    return get_generator(checkpoint_dir, tokenizer_path, MAX_CONTEXT)

MAX_CONTEXT = 2048
generation_lock = Lock()
generator = get_generator("7B", "tokenizer.model", MAX_CONTEXT)

app = Flask(__name__)
sse_publisher = MessageAnnouncer()

@app.route("/generate", methods=["POST"])
def generate():
    payload = request.get_json()
    try:
        prompt = payload["prompt"]
        max_gen_len = int(payload["maxGenTokens"])
        temp = float(payload["temp"])
        top_p = float(payload["topP"])
    except (KeyError, ValueError) as e:
        return { "error": repr(e) }, 400

    if len(prompt) > MAX_CONTEXT:
        return { "error": "Prompt is too long" }, 400

    acquired = generation_lock.acquire(blocking=False)
    if not acquired:
        app.logger.info("ignoring concurrent request %s", payload)
        return { "error": "Another client is still generating text. Try refreshing the page to see it." }, 400

    try:
        app.logger.info("got generation request: %s", payload)
        def on_gen(decoded: str):
            sse_publisher.announce(decoded, "partial")

        t0 = time.time()
        result = generator.generate([prompt],
            max_gen_len=max_gen_len,
            temperature=temp,
            top_p=top_p,
            gen_callback=on_gen)[0]

        sse_publisher.announce(result, "complete")
        app.logger.info("finished generation request (%.2fs): %s", time.time() - t0, payload)
        return f'{result}'
    finally:
        generation_lock.release()

# https://maxhalford.github.io/blog/flask-sse-no-deps/
@app.route('/listen', methods=['GET'])
def listen():
    def stream():
        messages = sse_publisher.listen()  # returns a queue.Queue
        while True:
            msg = messages.get()  # blocks until a new message arrives
            yield msg

    return Response(stream(), mimetype='text/event-stream')

@app.route("/")
def index():
    return render_template("index.html")