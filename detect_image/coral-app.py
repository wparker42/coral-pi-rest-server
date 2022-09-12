# Start the server:
#   python3 coral-app.py
# Submit a request via cURL:
#   curl -X POST -F image=@images/test-image3.jpg 'http://localhost:5000/v1/vision/detection'
#   curl -X POST -F image=@images/test-image3.jpg 'http://raspi.local:5000/v1/vision/detection'

import argparse
import os
import time
import pathlib
import logging
import flask
import numpy as np
from PIL import Image, ImageDraw
from pycoral.adapters import classify, detect, common
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

app = flask.Flask(__name__)

LOGFORMAT = "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
logging.basicConfig(filename="coral.log", level=logging.DEBUG, format=LOGFORMAT)
stderrLogger=logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderrLogger)


DEFAULT_MODELS_DIRECTORY = "../models"
DEFAULT_MODEL = "tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
DEFAULT_LABELS = "coco_labels.txt"
ROOT_URL = "/v1/vision/detection"
DRAW_URL = "/v1/vision/processing"
IMAGE_SAVE_PATH = "../saved_images/output.jpg"
IMAGE_SAVE_QUALITY = 95
COUNT = 5


def draw_objects(image, objs, labels):
    """Draws the bounding box and label for each object."""
    draw = ImageDraw.Draw(image)
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')
    image.save(IMAGE_SAVE_PATH, quality=IMAGE_SAVE_QUALITY)
    return None

@app.route("/")
def info():
    info_str = "Flask app exposing tensorflow lite model {}".format(model_file)
    return info_str

@app.route(DRAW_URL, methods=['POST', 'GET'])
def send_image():
    print(IMAGE_SAVE_PATH)
    return flask.send_file(IMAGE_SAVE_PATH, mimetype='image/gif')

@app.route(ROOT_URL, methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        print('\n ------- POST --------')

        if flask.request.files.get("image"):
            # Init the interpreter
            print("\n Initialising interpreter")
            interpreter = make_interpreter(model_file)
            print("\n Initialised interpreter with model : {}".format(model_file))
            interpreter.allocate_tensors()
            print("\n Allocated tensors")

            # Process the image
            image_file = flask.request.files["image"]
            image = Image.open(image_file)
            # image = Image.open("../images/people_car.jpg")
            # Transform image if required
            size = common.input_size(interpreter)
            params = common.input_details(interpreter, 'quantization_parameters')
            _, scale = common.set_resized_input(
                interpreter,
                image.size,
                lambda size: image.resize(size, Image.ANTIALIAS))
           
            # Run the inference
            if flask.request.args.get("thresh"):
                score_min = float(flask.request.args.get("thresh"))
            else:
                score_min = 0.45
            print(f'\n Score threshold: {score_min}')
            print('\n ---- INFERENCE TIME ----')
            print('\n Note: The first inference on Edge TPU is slow because it includes',
            'loading the model into Edge TPU memory.')
            for _ in range(COUNT):
                start = time.perf_counter()
                interpreter.invoke()
                inference_time = time.perf_counter() - start
                objs = detect.get_objects(interpreter, score_min, scale)
                print('%.1fms' % (inference_time * 1000))
            data["duration"] = inference_time

            # Process the results
            print('\n ------- RESULTS --------')
            if objs:
                inferd_objects = []
                for obj in objs:
                    print('%s: %.5f' % (labels.get(obj.id, obj.id), obj.score))
                    inferd_objects.append(
                        {
                            "confidence": float(obj.score),
                            "label": labels[obj.id],
                            "y_min": int(obj.bbox[1]),
                            "x_min": int(obj.bbox[0]),
                            "y_max": int(obj.bbox[3]),
                            "x_max": int(obj.bbox[2]),
                        })
                data["predictions"] = inferd_objects
                draw_objects(image, objs, labels)
            else:
                print('\n No objects detected')
                data["predictions"] = []

    # return the data dictionary as a JSON response
    data["success"] = True
    return flask.jsonify(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Flask app exposing coral USB stick",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--models_directory",
        default=DEFAULT_MODELS_DIRECTORY,
        help="the directory containing the model & labels files",)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="model file")
    parser.add_argument("--labels", default=DEFAULT_LABELS,
                        help="labels file of model")
    parser.add_argument("-p", "--port", type=int, default=5000,
                        help="port number")
    args = parser.parse_args()

    global model_file
    model_file = os.path.join(args.models_directory, args.model)
    assert os.path.isfile(model_file)
    print(f'\n Using tensorflow lite model: {model_file}')

    labels_file = os.path.join(args.models_directory, args.labels)
    assert os.path.isfile(labels_file)
    global labels
    labels = read_label_file(labels_file)
    print(f'\n Using labels: {labels_file}\n')

    app.run(host="0.0.0.0", debug=True, port=args.port)
