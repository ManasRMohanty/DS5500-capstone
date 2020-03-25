import flask
from flask import request, jsonify, render_template
from flask_cors import CORS
from CNER_ProcessClinicalText import process_text

app = flask.Flask(__name__)
app.config["DEBUG"] = True
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/')
def home():
    return render_template('index.html')


def all_caps(data):
    return process_text(data)


@app.route('/api/runModel', methods=['POST'])
def run_my_model():
    text = request.json['myData']
    text.replace("\n","<br>")
    print("data from client")
    print(text)
    res = {
        'return': 'success'
    }
    return jsonify(process_text(text))

app.run(debug=True, host='0.0.0.0')