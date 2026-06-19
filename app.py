import streamlit as st
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model import ANN_IDS

# --- Constants ---
SEQUENCE_LENGTH = 10
FEATURE_DIM = 30  # ✅ Update this based on your trained model
NUM_CLASSES = 19  # ✅ Number of label classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model ---
@st.cache_resource
def load_model(model_path):
    model = ANN_IDS(input_size=FEATURE_DIM, num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

model = load_model("models/best_ann_model_final.pth")

# --- Load Scaler from npz ---
scaler_data = np.load("scaler_data.npz")
scaler = StandardScaler()
scaler.mean_ = scaler_data["mean"]
scaler.scale_ = scaler_data["scale"]
scaler.var_ = scaler_data["var"]
scaler.n_samples_seen_ = scaler_data["n_samples_seen"]
scaler.n_features_in_ = scaler_data["n_features_in"]

# --- Load label classes from .txt (NO pickling) ---
with open("models/label_classes.txt", "r") as f:
    class_list = [line.strip() for line in f.readlines()]
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(class_list)

# --- Streamlit UI ---
st.title("🔐 ANN Intrusion Detection System")

st.markdown("""
This app demonstrates a trained ANN hybrid model for network intrusion detection.
Click the button below to generate a **synthetic traffic sample** and view the predicted attack type.
""")

if st.button("🔄 Generate Random Traffic Sample"):
    # 1. Random feature sequence
    random_features = np.random.rand(SEQUENCE_LENGTH, FEATURE_DIM).astype(np.float32)

    # Display raw features to user
    st.subheader("📊 Generated Random Traffic Features")
    st.dataframe(random_features, use_container_width=True)

    # 2. Scale features
    scaled = scaler.transform(random_features)

    # Optional: Display scaled features
    # st.subheader("📊 Scaled Feature Input to Model")
    # st.dataframe(scaled, use_container_width=True)

    # 3. Model input shape (1, seq_len, features)
    input_tensor = torch.tensor(scaled.reshape(1, SEQUENCE_LENGTH, FEATURE_DIM), dtype=torch.float32).to(DEVICE)

    # 4. Run model
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_index = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_index].item()
        predicted_label = label_encoder.inverse_transform([pred_index])[0]

    # 5. Display prediction
    st.subheader("🧠 Prediction Result")
    st.success(f"**Predicted Attack Type:** `{predicted_label}`")
    st.info(f"**Confidence Score:** `{confidence:.4f}`")
    st.caption("Note: This is a randomly generated input, not real network traffic.")
