# Air-Quality-Prediction

### 🌍 AQI Prediction Using Deep Learning, Machine Learning, and Deep Reinforcement Learning

This project aims to predict Air Quality Index (AQI) using various models, including:

- **Deep Learning Models** (LSTM, Transformer, GRU, TCN)  
- **Machine Learning Model** (XGBoost)  
- **Deep Reinforcement Learning Model** (PPO - Proximal Policy Optimization)  

The project follows a structured pipeline, ensuring proper data preprocessing, model training, and evaluation.

---

## 📂 Project Structure

```
├── data_preprocessing.py
├── lstm_model.py
├── transformer_model.py
├── gru_model.py
├── xgboost_model.py
├── tcn_model.py
├── deep_rl_model.py
├── main.py
├── requirements.txt
└── README.md
```

### **File Descriptions**
- **`data_preprocessing.py`** — Data loading, preprocessing, and feature engineering with rolling mean features and sinusoidal encoding for time.
- **`lstm_model.py`** — LSTM model with optimal hyperparameters for improved AQI prediction.
- **`transformer_model.py`** — Transformer model optimized for temporal sequences with customized configurations.
- **`gru_model.py`** — GRU model with tuned hyperparameters for better generalization.
- **`xgboost_model.py`** — XGBoost model optimized for fast and efficient AQI prediction.
- **`tcn_model.py`** — Temporal Convolutional Network model designed for sequence modeling tasks.
- **`deep_rl_model.py`** — PPO (Proximal Policy Optimization) model with a custom AQI environment using OpenAI Gym.
- **`main.py`** — The primary script that integrates all models for training and evaluation.
- **`requirements.txt`** — List of required libraries for smooth execution.

---

## 🚀 Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/PratikChakrabortyaz/Air-Quality-Prediction.git
cd Air-Quality-Prediction
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the Project**
```bash
python main.py
```

---

## 📊 Results

| **Model**            | **R² Score** | **RMSE** |
|----------------------|---------------|------------|
| **LSTM**              | 0.9882         | 0.0032      |
| **Transformer**       | 0.9820         | 0.0040      |
| **GRU**               | 0.9873         | 0.0034      |
| **XGBoost**           | 0.9524         | 0.0110      |
| **TCN**               | 0.9896         | 0.0029      |
| **Deep RL (PPO)**     | -1.7167        | 0.0493      |

---

## 🛠️ Key Features

👉 Data preprocessing with feature engineering (Rolling Mean Features + Sinusoidal Encoding)  
👉 Optimized deep learning models with tuned hyperparameters  
👉 Custom-built PPO model for AQI prediction using OpenAI Gym  
👉 Structured and modular code for better readability and maintenance  
👉 Comprehensive evaluation of all models using R² Score and RMSE  

---

