from __future__ import unicode_literals
import errno
import os
import sys
import tempfile
from flask import Flask, make_response, request, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (LineBotApiError, InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage, SourceUser, ImageMessage, ImageSendMessage)
import cv2
import numpy as np

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)

# Load environment variables
channel_secret = os.getenv('LINE_CHANNEL_SECRET', '71b1d3aed54fba8b572c5b64d0318c89')
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', 'JELc63Iy6RD21WuajBWof71GtOaIuuCJJKaX6Syhrc+hCKDm4jojliQOHcmZxw9nlxrlUWh7iYQplPqzwYIgcJXibNtUaJIxS/GCyvn2KtgPsvRzWiLXTFW5nGAyvnxWZgHyukBPcXon8I2K6H3SggdB04t89/1O/w1cDnyilFU=')
if channel_secret is None or channel_access_token is None:
    #print('Specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# Load YOLOv3 model
yolo_config_path = '/Users/pattanankorkiattrakool/Desktop/Senior/pipeline_api/line-yolo-api-flask/v3/yolov3_custom.cfg'
yolo_weights_path = '/Users/pattanankorkiattrakool/Desktop/Senior/pipeline_api/line-yolo-api-flask/v3/yolov3_custom_final.weights'
net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

# Load class names
classes_path = '/Users/pattanankorkiattrakool/Desktop/Senior/pipeline_api/line-yolo-api-flask/v3/classes.txt'
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')

def load_yolo_model():
    try:
        net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)
        return net
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        raise

def load_classes():
    try:
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    except Exception as e:
        print(f"Error loading classes: {e}")
        raise

def get_output_layers(net):
    layer_names = net.getLayerNames()
    return [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

@app.route("/", methods=['GET'])
def home():
    return "Object Detection API"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    print('reached post stage')

    try:
        handler.handle(body, signature)
    except LineBotApiError as e:
        print("Got exception from LINE Messaging API: %s\n" % e.message)
        for m in e.error.details:
            print("  %s: %s" % (m.property, m.message))
        print("\n")
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    text = event.message.text
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=event.message.text))

@handler.add(MessageEvent, message=ImageMessage)
def handle_content_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix='jpg-', delete=False) as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        tempfile_path = tf.name

    dist_path = tempfile_path + '.jpg'
    os.rename(tempfile_path, dist_path)

    image = cv2.imread(dist_path)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
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
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv2.rectangle(image, (round(x), round(y)), (round(x+w), round(y+h)), (255,0,0), 2)
        cv2.putText(image, str(classes[class_ids[i]]) + ":" + str(confidences[i]), (round(x-10), round(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

    save_path = os.path.join(static_tmp_path, 'result.jpg')
    cv2.imwrite(save_path, image)

    line_bot_api.reply_message(
        event.reply_token, [
            TextSendMessage(text='Object detection result:'),
            ImageSendMessage(os.path.join('static', 'tmp', 'result.jpg'), os.path.join('static', 'tmp', 'result.jpg'))
        ])

@app.route('/static/<path:path>')
def send_static_content(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    try:
        
        if not os.path.exists(static_tmp_path):
            os.makedirs(static_tmp_path)
        app.run(host="0.0.0.0", port=8000, debug=True)
    except Exception as e:
        print(f"Error")