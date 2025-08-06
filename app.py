from flask import Flask, render_template, request, jsonify
from predict_skin_disease import predict_skin_disease
from chatbot import get_bot_response
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_skin_disease(filepath)
            return render_template('index.html', prediction=prediction, image_path=filepath)
    return render_template('index.html')


@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        return render_template('chat.html')

    if request.method == 'POST':
        user_message = request.json.get('message')
        reply = get_bot_response(user_message)
        return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
