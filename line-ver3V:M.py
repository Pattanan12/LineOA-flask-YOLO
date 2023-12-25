from __future__ import unicode_literals
import errno
import os
from linebot import LineBotApi
import tempfile
from flask import Flask, request, abort, send_from_directory, make_response
from linebot.models import ImageSendMessage
from werkzeug.middleware.proxy_fix import ProxyFix
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import Configuration, ApiClient, MessagingApi
from linebot.v3.webhooks import MessageEvent, ImageMessageContent
from linebot.v3.messaging.api.messaging_api_blob import MessagingApiBlob
import cv2
import numpy as np

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)

# Environment variables
channel_secret = os.getenv('LINE_CHANNEL_SECRET', '71b1d3aed54fba8b572c5b64d0318c89')
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', 'JELc63Iy6RD21WuajBWof71GtOaIuuCJJKaX6Syhrc+hCKDm4jojliQOHcmZxw9nlxrlUWh7iYQplPqzwYIgcJXibNtUaJIxS/GCyvn2KtgPsvRzWiLXTFW5nGAyvnxWZgHyukBPcXon8I2K6H3SggdB04t89/1O/w1cDnyilFU=')

configuration = Configuration(access_token=channel_access_token)
handler = WebhookHandler(channel_secret)

# YOLOv3 configuration
yolo_config_path = '/home/waaris_m/Desktop/Senior/Line_OA/LineOA-flask-YOLO/model-v3/yolov3_custom.cfg'
yolo_weights_path = "/home/waaris_m/Desktop/Senior/Line_OA/LineOA-flask-YOLO/model-v3/yolov3_custom_final.weights"
classes_path = '/home/waaris_m/Desktop/Senior/Line_OA/LineOA-flask-YOLO/model-v3/classes.txt'

# Load YOLOv3 model
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

# Load class names
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')

# Function to get the output layers of the YOLO network
def get_output_layers(net):
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()

    # If the function returns an array of arrays (e.g., [[1], [2], [3]]), flatten it
    if isinstance(out_layers, list) and all(isinstance(item, np.ndarray) for item in out_layers):
        out_layers = [item[0] for item in out_layers]

    return [layer_names[i - 1] for i in out_layers]

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

def handle_image_message(event):
    # Initialize the API client with the correct configuration
    api_client = ApiClient(configuration)
    message_content = MessagingApiBlob(api_client).get_message_content(event.message.id)

    with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix='jpg-', delete=False) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name

    dist_path = tempfile_path + '.jpg'
    os.rename(tempfile_path, dist_path)

    # YOLOv3 model processing
    image = cv2.imread(dist_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w // 2
                y = center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(classes[class_ids[i]]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    result_path = os.path.join(static_tmp_path, 'result.jpg')
    cv2.imwrite(result_path, image)

    url = request.url_root + 'static/tmp/result.jpg'
    configuration.reply_message(
        event.reply_token, [
            ImageSendMessage(original_content_url=url, preview_image_url=url)
        ])

    response = make_response('OK', 200)
    return response

def get_output_layers(net):
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()

    # If the function returns an array of arrays (e.g., [[1], [2], [3]]), flatten it
    if isinstance(out_layers, list) and all(isinstance(item, np.ndarray) for item in out_layers):
        out_layers = [item[0] for item in out_layers]

    return [layer_names[i - 1] for i in out_layers]


@app.route('/static/<path:path>')
def send_static_content(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    if not os.path.exists(static_tmp_path):
        os.makedirs(static_tmp_path)
    app.run(host="0.0.0.0", port=8000, debug=True)
