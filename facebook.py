import requests
import json
from flask import Flask, request

# FB messenger credentials
ACCESS_TOKEN = "EAANCs3pJ868BAD7W7LfxF3MOAfAB6w0R1eWK4LwR6AXW1gN5RNhfX0n92YnIFm0VNrUX4WFd91BUbIXXXjx2frqiw502M9pJFcUFnfZAuTbgzx8ZAaOa1FY7J8Oes7JNZAtAaDYnzPZALW6wI9Jm8kDNb9hasZCVtlgD0G6ImHuCU1wm4vlrC"


app = Flask(__name__)

@app.route('/', methods=['GET'])
def verify():
    # our endpoint echos back the 'hub.challenge' value specified when we setup the webhook
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token") == 'foo':
            return "Verification token mismatch", 403
        return request.args["hub.challenge"], 200

    return 'Hello World (from Flask!)', 200

def reply(user_id, msg):
    data = {
        "recipient": {"id": user_id},
        "message": {"text": msg}
    }
    resp = requests.post("https://graph.facebook.com/v2.6/me/messages?access_token=" + ACCESS_TOKEN, json=data)
    print(resp.content)


@app.route('/', methods=['POST'])
def handle_incoming_messages():
    data = request.json
    sender = data['entry'][0]['messaging'][0]['sender']['id']
    message = data['entry'][0]['messaging'][0]['message']['text']




    #responsestr = api_response.read().decode('utf-8')
    #Add response
        #reply(sender, response)

    return "ok"

if __name__ == '__main__':
    app.run(debug=True)
