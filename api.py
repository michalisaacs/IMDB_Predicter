from flask import Flask, request, jsonify, send_file
import pandas as pd
import joblib
import re

from assets_data_prep import prepare_data


app = Flask(__name__)

# טעינת המודל
artifacts = joblib.load("random_forest.pkl")

if isinstance(artifacts, dict):
    model = artifacts["model"]
    mappings = artifacts.get("mappings", None)
else:
    model = artifacts
    mappings = None


@app.route("/", methods=["GET"])
def home():
    return send_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # 1. קריאת הנתונים מהטופס
        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "No JSON data was received"
            }), 400

        # 2. בדיקת שדות חובה
        required_fields = [
            "tconst",
            "startYear",
            "runtimeMinutes",
            "genres",
            "Language",
            "lead_actors_ids"
        ]

        for field in required_fields:
            if field not in data or str(data[field]).strip() == "":
                return jsonify({
                    "error": f"Missing required field: {field}"
                }), 400

        # 3. בדיקת תקינות tconst
        if not re.fullmatch(r"tt\d+", data["tconst"]):
            return jsonify({
                "error": "Invalid tconst format. Expected format: tt1234567"
            }), 400
        
        # 4. בדיקת שדות מספריים
        try:
            data["startYear"] = int(data["startYear"])
            data["runtimeMinutes"] = int(data["runtimeMinutes"])
        except ValueError:
            return jsonify({
                "error": "startYear and runtimeMinutes must be numeric values"
            }), 400

        # 5. בניית DataFrame
        input_df = pd.DataFrame([data])

        # 6. הפעלת prepare_data
        if mappings is not None:

            processed_df = prepare_data(
                input_df,
                crew_lookup_df=None,
                is_train=False,
                mappings=mappings
            )

        else:

            processed_df = prepare_data(input_df)

        # 7. הפעלת המודל
        prediction = model.predict(processed_df)[0]

        # 8. החזרת התוצאה
        return jsonify({
            "predicted_rating": round(float(prediction), 2)
        })

    except Exception as e:

        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
