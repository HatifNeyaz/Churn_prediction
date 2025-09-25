# ==============================================================================
# Step 3: API Creation with Flask
# ==============================================================================
#
# This script creates a REST API to serve the trained churn prediction model.
#
# The API has one endpoint: /predict
#   - Method: POST
#   - Input: A JSON object with a single customer's data.
#   - Output: A JSON object with the churn prediction and probability.
#
# This script must be in the same directory as 'churn_model.pth' and
# 'preprocessor.joblib'.
# ==============================================================================

# --- 1. Import Necessary Libraries ---
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import pandas as pd
import joblib
import numpy as np

print("Libraries imported successfully.")

# --- 2. Initialize Flask App ---
app = Flask(__name__)

# --- 3. Define the PyTorch Model Architecture ---
# This MUST be the same architecture as the one used for training.
# We need to define the class so we can load the saved state_dict into it.
class ChurnClassifier(nn.Module):
    def __init__(self, input_features):
        super(ChurnClassifier, self).__init__()
        self.layer_1 = nn.Linear(input_features, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.sigmoid(self.output_layer(x))
        return x

# --- 4. Load the Preprocessor and Model ---
try:
    # Load the preprocessing pipeline
    preprocessor = joblib.load('preprocessor.joblib')
    print("Preprocessor loaded successfully.")

    # Determine the input size from the preprocessor
    # This is crucial for initializing the model with the correct architecture.
    # It's the number of numerical features + the number of one-hot encoded categories.
    num_numerical_features = len(preprocessor.transformers_[0][2])
    num_categorical_features = len(preprocessor.named_transformers_['cat'].get_feature_names_out())
    INPUT_SIZE = num_numerical_features + num_categorical_features
    
    print(f"Model input size determined from preprocessor: {INPUT_SIZE}")

    # Initialize the model architecture
    model = ChurnClassifier(input_features=INPUT_SIZE)
    
    # Load the trained model weights
    model.load_state_dict(torch.load('churn_model.pth'))
    
    # Set the model to evaluation mode
    model.eval()
    print("PyTorch model loaded and set to evaluation mode.")

except Exception as e:
    print(f"Error loading model or preprocessor: {e}")
    model = None
    preprocessor = None

# --- 5. Create the Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests to the /predict endpoint.
    Takes customer data in JSON format and returns a churn prediction.
    """
    if not model or not preprocessor:
        return jsonify({"error": "Model or preprocessor not loaded properly. Check server logs."}), 500

    try:
        # Get the JSON data from the request
        json_data = request.get_json()
        if not json_data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert the single JSON object into a pandas DataFrame
        # The preprocessor expects a DataFrame as input
        input_df = pd.DataFrame([json_data])
        
        # Ensure correct data types for columns that might be misinterpreted
        # For example, TotalCharges can sometimes be missing or need conversion
        input_df['TotalCharges'] = pd.to_numeric(input_df.get('TotalCharges'), errors='coerce').fillna(0)
        input_df['tenure'] = pd.to_numeric(input_df.get('tenure'), errors='coerce').fillna(0)
        input_df['MonthlyCharges'] = pd.to_numeric(input_df.get('MonthlyCharges'), errors='coerce').fillna(0)


        # Apply the same preprocessing pipeline
        processed_data = preprocessor.transform(input_df)
        
        # Convert to a dense array if it's sparse
        processed_data = processed_data.toarray() if hasattr(processed_data, "toarray") else processed_data
        
        # Convert to a PyTorch tensor
        input_tensor = torch.tensor(processed_data, dtype=torch.float32)

        # Make a prediction (no gradients needed)
        with torch.no_grad():
            prediction_prob = model(input_tensor)
        
        # Get the probability value
        prob_value = prediction_prob.item()
        
        # Determine the class based on a 0.5 threshold
        prediction_class = "Churn" if prob_value > 0.5 else "No Churn"

        # Return the result as JSON
        return jsonify({
            "prediction": prediction_class,
            "probability": f"{prob_value:.4f}"
        })

    except Exception as e:
        # Log the error for debugging
        print(f"An error occurred during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction. Please check the input data format."}), 500

# --- 6. Run the Flask App ---
if __name__ == '__main__':
    # Running on 0.0.0.0 makes the API accessible from other devices on the network
    # and is necessary for Docker.
    app.run(host='0.0.0.0', port=5000)
