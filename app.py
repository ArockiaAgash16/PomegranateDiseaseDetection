from flask import Flask, render_template, request, send_file
import io
import os
import warnings
import json
from prediction import run_prediction
from chatbot import generate_llm_response
from generate_pdf_report import generate_pdf_report


# ✅ Suppress TensorFlow INFO logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded."

    file = request.files['image']
    if file.filename == '':
        return "No file selected."

    # ✅ Save the uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    filepath = filepath.replace("\\", "/")  # Normalize for Windows/Linux

    # ✅ Step 1: Run prediction
    try:
        print("🟡 Running prediction...")
        prediction = run_prediction(filepath)
        print("✅ Prediction completed.")
    except Exception as e:
        print(f"[ERROR] Prediction failed: {str(e)}")
        return "Prediction Failed. Check server logs.", 500

    # ✅ Extract prediction values
    disease = prediction.get('predicted_class', 'Unknown')
    severity = prediction.get('severity', 'N/A')
    severity_pct = prediction.get('severity_pct', 0.0)
    gradcam_image = prediction.get('full_gradcam_image', None)

    # ✅ Step 2: Generate chatbot response
    if disease == "Healthy":
        chatbot_response = (
            "🟢 This pomegranate fruit is healthy. No treatment is required. "
            "Keep monitoring regularly and maintain hygiene to prevent disease outbreaks."
        )
    else:
        chatbot_query = f"Give treatment plan for {severity} severity {disease} in pomegranate"
        try:
            print(f"🟡 Generating LLM response for {disease} ({severity})...")
            chat_history = []
            chatbot_response = generate_llm_response(disease, severity, chatbot_query, chat_history)
            print("✅ Chatbot response generated.")
        except Exception as e:
            print(f"[ERROR] Chatbot failed: {str(e)}")
            chatbot_response = "No response from chatbot."

    # ✅ Step 3: Render result page
    return render_template(
        'result.html',
        uploaded_image=file.filename,
        disease=disease,
        severity=severity,
        severity_pct=severity_pct,
        chatbot_response=chatbot_response,
        full_gradcam_image=gradcam_image
    )

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get("question")
    disease = data.get("disease")
    severity = data.get("severity")
    chat_history = data.get("history", [])

    try:
        print(f"🧠 Follow-up question: {question}")
        response = generate_llm_response(disease, severity, question, chat_history)
        print("✅ Follow-up answer generated.")
        return json.dumps({"response": response}), 200
    except Exception as e:
        print(f"[ERROR] Follow-up chatbot failed: {str(e)}")
        return json.dumps({"error": str(e)}), 500
    
@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    data = request.json
    disease = data.get('disease', 'N/A')
    severity = data.get('severity', 'N/A')
    severity_pct = data.get('severity_pct', 'N/A')
    chatbot_response = data.get('chatbot_response', 'N/A')
    history = data.get('history', [])
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], data.get('uploaded_image'))

    output_path = "static/reports/generated_report.pdf"
    generate_pdf_report(
        image_path=image_path,
        disease=disease,
        severity=severity,
        severity_pct=float(severity_pct),
        treatment=chatbot_response,
        chat_history=history,
        output_path=output_path
    )

    return send_file(output_path, as_attachment=True, download_name="pomegranate_report.pdf", mimetype='application/pdf')


if __name__ == "__main__":
    print("🚀 Starting Flask app at http://127.0.0.1:5000/")
    app.run(debug=True)