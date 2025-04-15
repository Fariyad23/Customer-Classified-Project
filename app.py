import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------------
# Simulate Customer Data
# ------------------------------
@st.cache_data
def generate_data(n=500):
    np.random.seed(42)
    age = np.random.randint(18, 70, size=n)
    income = np.random.normal(50000, 15000, size=n)
    spending_score = np.random.randint(1, 100, size=n)

    # Classify customers based on synthetic logic
    labels = []
    for i in range(n):
        if income[i] > 70000 and spending_score[i] > 60:
            labels.append('High-Value')
        elif income[i] < 30000 and spending_score[i] < 40:
            labels.append('Low-Value')
        else:
            labels.append('Mid-Value')

    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'SpendingScore': spending_score,
        'Segment': labels
    })
    return df

# ------------------------------
# Streamlit App
# ------------------------------
def main():
    st.title("🧠 Customer Classification App")
    st.markdown("Classifying customers into segments using Random Forest.")

    # Load data
    df = generate_data()
    st.subheader("Sample Customer Data")
    st.dataframe(df.head())

    # Visualize
    st.subheader("Income vs Spending Score")
    fig, ax = plt.subplots()
    for label in df['Segment'].unique():
        subset = df[df['Segment'] == label]
        ax.scatter(subset['Income'], subset['SpendingScore'], label=label)
    ax.set_xlabel("Income")
    ax.set_ylabel("Spending Score")
    ax.legend()
    st.pyplot(fig)

    # Preprocessing
    X = df[['Age', 'Income', 'SpendingScore']]
    y = df['Segment']

    # Encode target labels
    y_encoded = pd.factorize(y)[0]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    # Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred = clf.predict(X_test)

    st.subheader("Model Evaluation")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))

    # Prediction input
    st.subheader("Try Predicting a New Customer Segment")
    age = st.slider("Age", 18, 70, 30)
    income = st.slider("Income", 10000, 100000, 50000)
    score = st.slider("Spending Score", 1, 100, 50)

    input_data = np.array([[age, income, score]])
    pred = clf.predict(input_data)[0]
    label_map = dict(enumerate(pd.factorize(df['Segment'])[1]))
    st.success(f"Predicted Segment: **{label_map[pred]}**")

if __name__ == "__main__":
    main()

