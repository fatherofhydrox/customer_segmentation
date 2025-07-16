from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model/kmeans_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# Expected columns from training (same order)
expected_columns = [
    'ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income', 'Kidhome', 'Teenhome',
    'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
    'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
    'NumWebVisitsMonth', 'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
    'Complain', 'Response', 'day', 'month', 'year'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    if not file:
        return "⚠️ No file uploaded."

    try:
        # Load uploaded CSV
        df = pd.read_csv(file)

        # Encode categorical (object) columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = pd.factorize(df[col])[0]

        # Ensure column order matches model training
        df = df[expected_columns]

        # Convert to NumPy array to avoid feature name mismatch error
        df_scaled = scaler.transform(df.to_numpy())

        # Predict customer segments
        predictions = model.predict(df_scaled)
        df["Segment"] = predictions

        # Convert DataFrame to HTML table
        table_html = df.to_html(classes="table table-striped", index=False)

        return render_template("index.html", table=table_html)

    except Exception as e:
        return f"⚠️ Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
