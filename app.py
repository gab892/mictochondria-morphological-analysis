import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# --- Streamlit UI ---
st.title("Mitochondria Morphology Classifier")

# Input widgets for parameters (example)
fragmented_area_mean = st.number_input("Fragmented Area Mean", value=20.0, min_value=0.0, max_value=200.0)
tubular_area_mean = st.number_input("Tubular Area Mean", value=80.0, min_value=0.0, max_value=200.0)
# Add more input widgets for other parameters...

# --- Data Generation (modified to use Streamlit inputs) ---
features = {
    'Mitochondria_Area': {
        'Fragmented': {'mean': fragmented_area_mean, 'std': 5},
        'Tubular': {'mean': tubular_area_mean, 'std': 15},
        'Intermediate': {'mean': 50, 'std': 10},
        'Hypertubular': {'mean': 150, 'std': 25}
    },
    # ... (rest of the features dictionary)
}

n_samples = 100

data = []
for morphology in ['Fragmented', 'Tubular', 'Intermediate', 'Hypertubular']:
    for _ in range(n_samples):
        sample = {'Morphology': morphology}
        for feature_name, feature_params in features.items():
            mean = feature_params[morphology]['mean']
            std = feature_params[morphology]['std']
            sample[feature_name] = np.random.normal(mean, std)
        data.append(sample)

df = pd.DataFrame(data)

# --- Model Training and Evaluation ---
X = df.drop('Morphology', axis=1)
y = df['Morphology']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True) #Get report as dictionary

# --- Display Results ---
st.write(f"Accuracy: {accuracy}")
report_df = pd.DataFrame(report).transpose()  #Convert to DataFrame
st.dataframe(report_df)  #Display as DataFrame