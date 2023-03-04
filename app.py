from logging.config import dictConfig
import time
from flask import Flask, request, Response

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

MAX_CONTEXT = 2048
generator = get_generator("7B", "tokenizer.model", MAX_CONTEXT)

app = Flask(__name__)
sse_publisher = MessageAnnouncer()

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
    return f'"{result}"'

# https://maxhalford.github.io/blog/flask-sse-no-deps/
@app.route('/listen', methods=['GET'])
def listen():
    def stream():
        messages = sse_publisher.listen()  # returns a queue.Queue
        while True:
            msg = messages.get()  # blocks until a new message arrives
            yield msg

    return Response(stream(), mimetype='text/event-stream')
