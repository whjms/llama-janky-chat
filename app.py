from logging.config import dictConfig
from threading import Lock
import time
import json
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

class GenerationCancelled(Exception):
    pass

def load_generator():
    def env_var_with_default(name: str, default: str) -> str:
        try:
            return os.environ[name]
        except KeyError:
            return default

    checkpoint_dir = env_var_with_default("CHECKPOINT_DIR", "7B")
    tokenizer_path = env_var_with_default("TOKENIZER_PATH", "tokenizer.model")
    max_context = int(env_var_with_default("CONTEXT_LEN", 768))
    return get_generator(checkpoint_dir, tokenizer_path, max_context), max_context

signal_generation_cancelled = Lock()
generation_lock = Lock()
generator, max_context = load_generator()

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

    if len(prompt) > max_context:
        return { "error": "Prompt is too long" }, 400

    acquired = generation_lock.acquire(blocking=False)
    if not acquired:
        app.logger.info("ignoring concurrent request %s", payload)
        return { "error": "Another client is still generating text. Try refreshing the page to see it." }, 400

    try:
        if signal_generation_cancelled.locked():
            signal_generation_cancelled.release()

        app.logger.info("got generation request: %s", payload)
        def on_gen(decoded: str):
            msg = {
                "prompt": prompt,
                "generated": decoded,
            }
            sse_publisher.announce(json.dumps(msg), "partial")

            if signal_generation_cancelled.locked():
                raise GenerationCancelled()

        t0 = time.time()
        try:
            result = generator.generate([prompt],
                max_gen_len=max_gen_len,
                temperature=temp,
                top_p=top_p,
                gen_callback=on_gen)[0]
        except GenerationCancelled:
            app.logger.info("cancelled generation request (%.2fs): %s", time.time() - t0, payload)
            sse_publisher.announce("<cancelled>", "complete")
            signal_generation_cancelled.release()
            return '{"error": "Generation was cancelled" }', 400

        msg = {
            "prompt": prompt,
            "generated": result,
        }
        sse_publisher.announce(json.dumps(msg), "complete")
        app.logger.info("finished generation request (%.2fs): %s", time.time() - t0, payload)
        return msg
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

@app.route("/cancel", methods=["POST"])
def cancel():
    signal_generation_cancelled.acquire(False)
    return "", 200

@app.route("/")
def index():
    return render_template("index.html")