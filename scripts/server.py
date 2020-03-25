import flask
from flask import request, jsonify
from flask_cors import CORS
from CNER_ProcessClinicalText import process_text

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/', methods=['GET'])
def home():
    return "<h1>Distant Reading Archive</h1><p>This site is a prototype API for distant reading of science fiction novels.</p>"


def all_caps(data):
    return process_text(data)


@app.route('/api/runModel', methods=['POST'])
def run_my_model():
    text = request.json['myData']
    text.replace("\n","<br>")
    res = {
        'return': 'success'
    }
    return jsonify(process_text(text))

app.run()