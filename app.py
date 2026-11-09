from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained house price model
with open("house_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    bedrooms = int(request.form["bedrooms"])
    size = float(request.form["size"])

    prediction = model.predict([[bedrooms, size]])[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
