import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -----------------------------
# Load and prepare dataset
# -----------------------------
@st.cache_data
def load_data():
    # Prefer new dataset if present
    if os.path.exists("Laptop_price_2025_fullrange.csv"):
        df = pd.read_csv("Laptop_price_2025_fullrange.csv")
    else:
        df = pd.read_csv("Laptop_price.csv")

    # Clean numeric & categorical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


@st.cache_resource
def train_model(df):
    df = df.copy()

    le_brand = LabelEncoder()
    df["Brand"] = le_brand.fit_transform(df["Brand"])

    feature_cols = ["Brand", "Processor_Speed", "RAM_Size",
                    "Storage_Capacity", "Screen_Size", "Weight"]

    X = df[feature_cols]
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return model, le_brand, feature_cols, metrics


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Laptop Price Predictor",
                       page_icon="https://cdn-icons-png.freepik.com/256/14601/14601752.png?semt=ais_white_label",
                       layout="wide")

    # ========== SIDEBAR ==========
    st.sidebar.image(
        "https://img.pikbest.com/ai/illus_our/20230427/a374ccc763d521f967cd4d9b14fd9e8c.jpg!w700wp",
        use_container_width=True
    )
    st.sidebar.markdown("### Laptop Price Predictor")
    st.sidebar.write("Adjust the specs and see the predicted price in real-time.")

    # Team Members
    with st.sidebar.expander("Team Members"):
        st.write("- Abhivav K")
        st.write("- Devaj TN")
        st.write("- George Pramod Thomas")
        st.write("- Razik Rahman M S")
        st.write("- Sahil Shahanas")

    # ========== TOP IMAGES ==========
    col_banner1, col_banner2 = st.columns([2, 1])

    with col_banner1:
        st.image(
            "https://img.freepik.com/premium-photo/silver-macbook-pro-laptop-with-blue-screen-that-says-windows-screen_862335-23577.jpg?semt=ais_hybrid&w=740&q=80",
            width=350,
            caption="Work smarter with data-driven pricing"
        )

    with col_banner2:
        st.image(
            "https://www.trentonsystems.com/hs-fs/hubfs/Machine_Learning%20.jpeg?width=4041&name=Machine_Learning%20.jpeg",
            use_container_width=True,
            caption="Machine Learning in action"
        )

    st.title("Where specifications meet intelligence")
    st.write("This app predicts laptop prices using a **Linear Regression model** trained on your dataset.")

    # Load dataset
    df = load_data()

    # Train model
    model, le_brand, feature_cols, metrics = train_model(df)

    # ========== METRICS ==========
    mcol1, mcol2 = st.columns([2, 1])

    with mcol1:
        st.subheader("Model Performance")
        st.write(f"**R² Score:** {metrics['r2']:.4f}")
        st.write(f"**MAE:** {metrics['mae']:.2f}")
        st.write(f"**RMSE:** {metrics['rmse']:.2f}")

    with mcol2:
        st.image(
            "https://nextgeninvent.com/wp-content/uploads/2023/04/AI-and-ML.jpg.webp",
            use_container_width=True,
            caption="Predict before you buy"
        )

    st.markdown("---")
    st.subheader("Enter Laptop Specifications")

    # Dropdown values from dataset
    brand_options = sorted(df["Brand"].unique().tolist())
    brand_names = sorted(df["Brand"].astype(str).unique())

    # Range values from dataset
    processor_min, processor_max = df["Processor_Speed"].min(), df["Processor_Speed"].max()
    ram_min, ram_max = df["RAM_Size"].min(), df["RAM_Size"].max()
    storage_min, storage_max = df["Storage_Capacity"].min(), df["Storage_Capacity"].max()
    screen_min, screen_max = df["Screen_Size"].min(), df["Screen_Size"].max()
    weight_min, weight_max = df["Weight"].min(), df["Weight"].max()

    col1, col2 = st.columns(2)

    with col1:
        brand_input = st.selectbox("Brand", brand_names)
        processor_speed = st.slider("Processor Speed (GHz)",
                                    min_value=float(processor_min),
                                    max_value=float(processor_max),
                                    value=float(processor_min + processor_max) / 2,
                                    step=0.01)
        ram_size = st.slider("RAM Size (GB)", int(ram_min), int(ram_max), 8)

    with col2:
        storage_capacity = st.slider("Storage (GB)", int(storage_min), int(storage_max), 512)
        screen_size = st.slider("Screen Size (inches)",
                                float(screen_min), float(screen_max),
                                float((screen_min + screen_max) / 2))
        weight = st.slider("Weight (kg)",
                           float(weight_min), float(weight_max),
                           float((weight_min + weight_max) / 2))

    # Predict Price
    if st.button("Predict Price"):
        brand_encoded = le_brand.transform([brand_input])[0]

        input_data = pd.DataFrame([{
            "Brand": brand_encoded,
            "Processor_Speed": processor_speed,
            "RAM_Size": ram_size,
            "Storage_Capacity": storage_capacity,
            "Screen_Size": screen_size,
            "Weight": weight
        }])[feature_cols]

        predicted_price = model.predict(input_data)[0]

        st.success(f"Estimated Laptop Price: **₹ {predicted_price:,.2f}**")

        st.image(
            "https://images.unsplash.com/photo-1517244861144-72a5b18b77c8",
            use_container_width=True,
            caption="Prediction generated based on your inputs"
        )

    st.markdown("---")
    st.subheader("Sample Dataset")
    st.dataframe(df.head())


if __name__ == "__main__":
    main()
