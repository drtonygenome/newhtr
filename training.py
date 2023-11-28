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

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu',
                          input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10, restore_best_weights=True)

history = model.fit(X_train_scaled, y_train, validation_data=(X_eval_scaled, y_eval),
                    epochs=10000, callbacks=[early_stopping], verbose=1)

# Dừng training khi độ chính xác > 95%
for i in range(len(history.history['val_accuracy'])):
    if history.history['val_accuracy'][i] > 0.99:
        model.stop_training = True
        break

model.save("HTR_model")

loaded_model = tf.keras.models.load_model("HTR_model")

eval_loss, eval_acc = loaded_model.evaluate(X_eval_scaled, y_eval)
print(f"\nprecision: {eval_acc * 100:.2f}%")

