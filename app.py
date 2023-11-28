from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("mRNA_htrrbp.csv")


train_data = data.iloc[:-200]
eval_data = data.iloc[-200:]

X_train = train_data.drop(columns=["group"])
y_train = train_data["group"]

X_eval = eval_data.drop(columns=["group"])
y_eval = eval_data["group"]

# Chuẩn hóa dữ liệu sử dụng StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_eval_scaled = scaler.transform(X_eval)

loaded_model = tf.keras.models.load_model("HTR_model")

eval_loss, eval_acc = loaded_model.evaluate(X_eval_scaled, y_eval)
print(
    f"\nPrecision: {eval_acc * 100:.2f}%")

# ---------------APP-------------------- #
app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Get user input from the web form
        CD44 = float(request.form["CD44"])
        TOP2A = int(request.form["TOP2A"])
        SAMSN1 = float(
            request.form["SAMSN1"])
        RBM47 = int(request.form["RBM47"])
        CORO1A = float(request.form["CORO1A"])
        ZC3HAV1 = int(request.form["ZC3HAV1"])
        HLAA = float(request.form["HLAA"])
        

        # Create a DataFrame from user input
        input_data = pd.DataFrame({
            "CD44": [CD44],
            "TOP2A": [TOP2A],
            "SAMSN1": [SAMSN1],
            "RBM47": [RBM47],
            "CORO1A": [CORO1A],
            "ZC3HAV1": [ZC3HAV1],
            "HLAA": [HLAA]
            })

        # Scale the input data using the pre-trained scaler
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(input_data_scaled)
        predicted_classe = (prediction > 0.5).astype(int)
        print(input_data)
        print(prediction)
        print(predicted_classe)
        # Chuyển đổi giá trị prediction và predicted_classe thành chuỗi
        predicted_classe_str = str(predicted_classe[0][0])
        prediction_str = "{:.2f}%".format(prediction[0][0]*100)

        # Determine the result (group)
        result = "Rejection" if predicted_classe[0][0] == 1 else "Non-rejection"

        return render_template("result.html", result=result, input_data=input_data, prediction=prediction_str, predicted_classe=predicted_classe_str)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
