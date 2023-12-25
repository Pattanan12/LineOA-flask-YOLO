# from __future__ import unicode_literals

# import errno
# import os
# import sys
# import tempfile
# from dotenv import load_dotenv

# from flask import Flask, make_response, request, abort, send_from_directory
# from werkzeug.middleware.proxy_fix import ProxyFix

# from linebot import (
#     LineBotApi, WebhookHandler
# )
# from linebot.exceptions import (
#     LineBotApiError, InvalidSignatureError
# )
# from linebot.models import (
#     MessageEvent, TextMessage, TextSendMessage,
#     SourceUser, PostbackEvent, StickerMessage, StickerSendMessage, 
#     LocationMessage, LocationSendMessage, ImageMessage, ImageSendMessage)

# import time
# from pathlib import Path

# import cv2
# import torch
# from utils.plots import Annotator, colors

# app = Flask(__name__)
# app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)

# # reads the key-value pair from .env file and adds them to environment variable.
# load_dotenv()

# # get channel_secret and channel_access_token from your environment variable
# channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
# channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
# if channel_secret is None or channel_access_token is None:
#     print('Specify LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN as environment variables.')
#     sys.exit(1)

# line_bot_api = LineBotApi(channel_access_token)
# handler = WebhookHandler(channel_secret)

# static_tmp_path = os.path.join(os.path.dirname(__file__), 'static', 'tmp')


# ### YOLOv5 ###
# # Setup
# #yolov5s.pt = model
# weights, view_img, save_txt, imgsz = 'yolov5s.pt', False, False, 640
# conf_thres = 0.25
# iou_thres = 0.45
# classes = None
# agnostic_nms = False
# save_conf = False
# save_img = True
# line_thickness = 3

# # Directories
# save_dir = 'static/tmp/'

# # Load model
# model = torch.hub.load('./', 'custom', path='yolov5s.pt', source='local', force_reload=True)
# # model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# # function for create tmp dir for download content
# def make_static_tmp_dir():
#     try:
#         os.makedirs(static_tmp_path)
#     except OSError as exc:
#         if exc.errno == errno.EEXIST and os.path.isdir(static_tmp_path):
#             pass
#         else:
#             raise

# @app.route("/", methods=['GET'])
# def home():
#     return "Object Detection API"

# @app.route("/callback", methods=['POST'])
# def callback():
#     # get X-Line-Signature header value
#     signature = request.headers['X-Line-Signature']

#     # get request body as text
#     body = request.get_data(as_text=True)
#     app.logger.info("Request body: " + body)

#     # handle webhook body
#     try:
#         handler.handle(body, signature)
#     except LineBotApiError as e:
#         print("Got exception from LINE Messaging API: %s\n" % e.message)
#         for m in e.error.details:
#             print("  %s: %s" % (m.property, m.message))
#         print("\n")
#     except InvalidSignatureError:
#         abort(400)

#     return 'OK'


# @handler.add(MessageEvent, message=TextMessage)
# def handle_text_message(event):
#     text = event.message.text

#     if text == 'profile':
#         if isinstance(event.source, SourceUser):
#             profile = line_bot_api.get_profile(event.source.user_id)
#             line_bot_api.reply_message(
#                 event.reply_token, [
#                     TextSendMessage(text='Display name: ' + profile.display_name),
#                 ]
#             )
#         else:
#             line_bot_api.reply_message(
#                 event.reply_token,
#                 TextSendMessage(text="Bot can't use profile API without user ID"))
#     else:
#         line_bot_api.reply_message(
#             event.reply_token, TextSendMessage(text=event.message.text))


# @handler.add(MessageEvent, message=LocationMessage)
# def handle_location_message(event):
#     line_bot_api.reply_message(
#         event.reply_token,
#         LocationSendMessage(
#             title='Location', address=event.message.address,
#             latitude=event.message.latitude, longitude=event.message.longitude
#         )
#     )


# @handler.add(MessageEvent, message=StickerMessage)
# def handle_sticker_message(event):
#     line_bot_api.reply_message(
#         event.reply_token,
#         StickerSendMessage(
#             package_id=event.message.package_id,
#             sticker_id=event.message.sticker_id)
#     )


# # Other Message Type
# @handler.add(MessageEvent, message=(ImageMessage))
# def handle_content_message(event):
#     if isinstance(event.message, ImageMessage):
#         ext = 'jpg'
#     else:
#         return

#     message_content = line_bot_api.get_message_content(event.message.id)
#     with tempfile.NamedTemporaryFile(dir=static_tmp_path, prefix=ext + '-', delete=False) as tf:
#         for chunk in message_content.iter_content():
#             tf.write(chunk)
#         tempfile_path = tf.name

#     dist_path = tempfile_path + '.' + ext
#     os.rename(tempfile_path, dist_path)

#     im_file = open(dist_path, "rb")
#     im = cv2.imread(im_file)
#     im0 = im.copy()

#     results = model(im, size=640)  # reduce size=320 for faster inference
#     print(results)
#     annotator = Annotator(im0, line_width=line_thickness)
#     # Write results 
#     df = results.pandas().xyxy[0]
#     for idx, r in df.iterrows():
#         c = int(r['class'])  # integer class
#         name = r['name']
#         label = f'{name} {r.confidence:.2f}'
#         annotator.box_label((r.xmin, r.ymin, r.xmax, r.ymax), label, color=colors(c, True))

#     save_path = str(save_dir + os.path.basename(tempfile_path) + '_result.' + ext) 
#     cv2.imwrite(save_path, im0)

#     url = request.url_root + '/' + save_path

#     line_bot_api.reply_message(
#         event.reply_token, [
#             TextSendMessage(text='Object detection result:'),
#             ImageSendMessage(url,url)
#         ])

#     response = make_response('OK', 200)
#     return response

# @app.route('/static/<path:path>')
# def send_static_content(path):
#     return send_from_directory('static', path)

# # create tmp dir for download content
# make_static_tmp_dir()

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000, debug=True)

from __future__ import unicode_literals
import errno
import os
import sys
import tempfile
from flask import Flask, make_response, request, abort, send_from_directory
from werkzeug.middleware.proxy_fix import ProxyFix
from linebot.exceptions import (LineBotApiError, InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage, SourceUser, ImageMessage, ImageSendMessage)
from linebot import LineBotApi
from linebot.webhook import WebhookHandler
# from linebot.v3 import LineBotApi
# from linebot.v3.webhook import WebhookHandler
import cv2
import numpy as np

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_host=1, x_proto=1)

channel_secret = os.getenv('LINE_CHANNEL_SECRET', '71b1d3aed54fba8b572c5b64d0318c89')
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', '90uqD9Sbp92bkbxhUSn9x7E9bLr4LWifgInbEr6Qmv9Fp0AmCuQRkwonLwtAR3ePlxrlUWh7iYQplPqzwYIgcJXibNtUaJIxS/GCyvn2KtgNLogRg2M7Eqr0uXlWogRSPkbL6/AnCWXxstbdQF53QgdB04t89/1O/w1cDnyilFU=')
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

# def get_output_layers(net):
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#     # Check if output_layers is a valid list
#     if not isinstance(output_layers, list):
#         raise ValueError("Output layers are not a list")
#     return output_layers
def get_output_layers(net):
    try:
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        if not isinstance(output_layers, list):
            raise ValueError("Output layers are not a list")
        return output_layers
    except Exception as e:
        print(f"Error getting output layers: {e}")
        raise


@app.route("/", methods=['GET'])
def home():
    return "Object Detection API"

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

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
    try:
        net = load_yolo_model()
        classes = load_classes()

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
            cv2.rectangle(image, (round(x), round(y)), (round(x+w), round(y+h)), (255, 0, 0), 2)
            cv2.putText(image, str(classes[class_ids[i]]) + ":" + str(confidences[i]),
                        (round(x-10), round(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        save_path = os.path.join(static_tmp_path, 'result.jpg')
        cv2.imwrite(save_path, image)

        line_bot_api.reply_message(
            event.reply_token, [
                TextSendMessage(text='Object detection result:'),
                ImageSendMessage(os.path.join('static', 'tmp', 'result.jpg'), os.path.join('static', 'tmp', 'result.jpg'))
            ])

    except Exception as e:
        print(f"Error processing content message: {e}")
        return

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

