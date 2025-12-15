# flask_app/app.py

import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from google.genai import Client
import joblib

# optional: markdown converter
import markdown as md

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env (place GEMINI_API_KEY=... in .env)")

# MODEL_PATH: change if your joblib file is somewhere else
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "stack_ensemble_aqi.joblib")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}\nPut your joblib model at this path or update MODEL_PATH.")

# Load ML model
model = joblib.load(MODEL_PATH)

# Initialize Gemini client (reads GEMINI_API_KEY automatically)
client = Client()

app = Flask(__name__, template_folder="templates")


def ask_gemini(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    """
    Call Gemini and return the generated text. Use simple call (no max tokens to avoid SDK mismatches).
    Returns a string (or error message).
    """
    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        text = getattr(resp, "text", None)
        if text is None:
            try:
                text = resp.generations[0].text
            except Exception:
                text = str(resp)
        return text
    except Exception as e:
        return f"Error calling Gemini: {e}"


def make_explain_prompt(pred_label: str, features: dict) -> str:
    """
    Build a clear prompt for Gemini to explain the prediction step-by-step.
    """
    lines = [
        f"The AQI classification model predicted: {pred_label}.",
        "Below are the numeric feature values used for the prediction:"
    ]
    for k, v in features.items():
        lines.append(f"- {k}: {v}")
    lines.append(
        "\nTask: Provide a short, step-by-step explanation suitable for a school report.\n"
        "1) For each feature, write 1 sentence on how it influences air quality.\n"
        "2) Give a short overall conclusion (1-2 sentences).\n"
        "3) Use bullet points or numbered steps and simple language."
    )
    return "\n".join(lines)


@app.route("/", methods=["GET"])
def home():
    # show empty form on GET
    return render_template("index.html", prediction=None, explanation=None, explanation_html=None, values=None)


@app.route("/predict", methods=["POST"])
def predict():
    # Read form values (strings) and convert safely to floats
    try:
        co = float(request.form.get("co", "").strip() or 0.0)
        no2 = float(request.form.get("no2", "").strip() or 0.0)
        pt08 = float(request.form.get("pt08", "").strip() or 0.0)
        temperature = float(request.form.get("temperature", "").strip() or 0.0)
        relative_humidity = float(request.form.get("relative_humidity", "").strip() or 0.0)
        absolute_humidity = float(request.form.get("absolute_humidity", "").strip() or 0.0)
    except ValueError:
        return render_template("index.html",
                               prediction=None,
                               explanation="Please provide valid numeric values for all fields.",
                               explanation_html=None,
                               values=request.form)

    # Prepare features in expected order (ensure matches training)
    features = [[co, no2, pt08, temperature, relative_humidity, absolute_humidity]]

    # Predict with the model
    try:
        pred = model.predict(features)[0]
    except Exception as e:
        return render_template("index.html",
                               prediction=None,
                               explanation=f"Error running model.predict: {e}",
                               explanation_html=None,
                               values=request.form)

    # Map numeric prediction to label (update mapping if your model uses different encoding)
    label_map = {0: "Good", 1: "Moderate", 2: "Poor"}
    prediction_label = label_map.get(int(pred), str(pred))

    # features dict for prompt and display
    features_dict = {
        "CO": co,
        "NO2": no2,
        "PT08S1": pt08,
        "Temperature": temperature,
        "Relative Humidity": relative_humidity,
        "Absolute Humidity": absolute_humidity,
    }

    # Build prompt and call Gemini
    explain_prompt = make_explain_prompt(prediction_label, features_dict)
    explanation = ask_gemini(explain_prompt)

    # Convert Gemini's markdown (if any) to HTML for nice rendering
    try:
        explanation_html = md.markdown(explanation or "", extensions=["extra", "sane_lists", "nl2br"])
    except Exception:
        explanation_html = "<pre style='white-space:pre-wrap;'>%s</pre>" % (explanation or "No explanation returned")

    # Optional: save history
    try:
        history_path = os.path.join(os.path.dirname(__file__), "..", "history.log")
        with open(history_path, "a", encoding="utf-8") as f:
            f.write("FEATURES: " + str(features_dict) + "\n")
            f.write("PREDICTION: " + prediction_label + "\n")
            f.write("EXPLANATION: " + explanation + "\n\n")
    except Exception:
        pass  # ignore logging errors

    # Render template with prediction + explanation_html (safe)
    return render_template(
        "index.html",
        prediction=prediction_label,
        explanation=explanation,
        explanation_html=explanation_html,
        values=features_dict
    )


if __name__ == "__main__":
    # Run on port 8000
    app.run(debug=True, port=8000)
