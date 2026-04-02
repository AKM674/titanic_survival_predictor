import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---- Load and Train Model ----
@st.cache_resource
def train_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df = df.dropna(subset=["Age"])
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    X = df[features]
    y = df["Survived"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# ---- App UI ----
st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival!")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.slider("Siblings/Spouses aboard", 0, 8, 0)
parch = st.slider("Parents/Children aboard", 0, 6, 0)
fare = st.slider("Fare paid", 0, 512, 32)

# ---- Predict ----
if st.button("Predict Survival"):
    sex_encoded = 1 if sex == "Female" else 0
    passenger = pd.DataFrame([[pclass, sex_encoded, age, sibsp, parch, fare]],
                             columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"])
    result = model.predict(passenger)
    
    if result[0] == 1:
        st.success("✅ This passenger SURVIVED!")
    else:
        st.error("❌ This passenger DIED.")
