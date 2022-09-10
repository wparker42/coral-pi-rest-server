# Start the server:
# 	python3 coral-app.py
# Submit a request via cURL:
# 	curl -X POST -F image=@images/test-image3.jpg 'http://localhost:5080/v1/vision/detection'
#   curl -X POST -F image=@images/test-image3.jpg 'http://raspi.local:5080/v1/vision/detection'

import argparse
import io
import os
import time
import pathlib
import logging
import flask
import numpy as np
from PIL import Image
from pycoral.adapters import classify, detect, common
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.dataset import read_label_file

app = flask.Flask(__name__)

LOGFORMAT = "%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s"
logging.basicConfig(filename="coral.log", level=logging.DEBUG, format=LOGFORMAT)
stderrLogger=logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderrLogger)


DEFAULT_MODELS_DIRECTORY = "models"
DEFAULT_MODEL = "tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
DEFAULT_LABELS = "coco_labels.txt"
INPUT_MEAN = 128.0  # should get from args
INPUT_STD = 128.0  # should get from args
ROOT_URL = "/v1/vision/detection"


@app.route("/")
def info():
    info_str = "Flask app exposing tensorflow lite model {}".format(model_file)
    return info_str


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
            size = common.input_size(interpreter)
            image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)
            params = common.input_details(interpreter, 'quantization_parameters')
            scale = params['scales']
            zero_point = params['zero_points']
            mean = INPUT_MEAN
            std = INPUT_STD
            if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
                # Input data does not require preprocessing.
                common.set_input(interpreter, image)
            else:
                # Input data requires preprocessing
                normalized_input = (np.asarray(image) - mean) / (std * scale) + zero_point
                np.clip(normalized_input, 0, 255, out=normalized_input)
                common.set_input(interpreter, normalized_input.astype(np.uint8))

            # Run the inference
            if flask.request.args.get("thresh"):
                score_min = float(flask.request.args.get("thresh"))
            else:
                score_min = 0.45
            print(f'\n Score threshold: {score_min}')
            print('\n ---- INFERENCE TIME ----')
            print('\n Note: The first inference on Edge TPU is slow because it includes',
            'loading the model into Edge TPU memory.')
            start = time.perf_counter()
            interpreter.invoke()
            inference_time = time.perf_counter() - start
            objs = detect.get_objects(interpreter, score_min, (scale, scale))
            print('%.1fms' % (inference_time * 1000))

            # Process the results
            if objs:
                data["success"] = True
                print('\n ------- RESULTS --------')
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
            else:
                print('\n No objects detected')

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Flask app exposing coral USB stick",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--models_directory",
        default=DEFAULT_MODELS_DIRECTORY,
        help="the directory containing the model & labels files",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model file",
    )
    parser.add_argument(
        "--labels",
        default=DEFAULT_LABELS,
        help="labels file of model"
    )
    parser.add_argument(
        "--port",
        default=5080,
        type=int,
        help="port number"
    )
    args = parser.parse_args()

    global model_file
    model_file = os.path.join(args.models_directory, args.model)
    assert os.path.isfile(model_file)
    print(f'\n -----Using tensorflow lite model: {model_file}')

    labels_file = os.path.join(args.models_directory, args.labels)
    assert os.path.isfile(labels_file)
    global labels
    labels = read_label_file(labels_file)
    print(f'\n -----Using labels: {labels_file}\n')

    app.run(host="0.0.0.0", debug=True, port=args.port)
