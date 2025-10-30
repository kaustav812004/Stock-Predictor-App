# 📈 Stock Price Prediction using LSTM

An AI-powered stock price predictor built using **TensorFlow (Keras)** and **Streamlit**, which forecasts the future closing prices of a selected stock using a **Long Short-Term Memory (LSTM)** neural network.

---

## 🚀 Overview

This project uses a **deep learning model (LSTM)** trained on historical stock data to predict future prices.  
The model is capable of learning temporal dependencies and long-term patterns from past closing prices.

You can visualize:
- Historical prices
- Moving averages (MA50, MA100, MA200)
- Actual vs Predicted price curves

---

## 🧠 Model Architecture

The LSTM network has been built and trained using the following architecture:

| Layer Type | Units | Activation | Additional Info |
|-------------|--------|-------------|------------------|
| LSTM | 50 | tanh | return_sequences=True |
| Dropout | 0.2 | — | Prevents overfitting |
| LSTM | 60 | tanh | return_sequences=True |
| Dropout | 0.3 | — | — |
| LSTM | 80 | tanh | return_sequences=True |
| Dropout | 0.4 | — | — |
| LSTM | 120 | tanh | — |
| Dropout | 0.5 | — | — |
| Dense | 1 | — | Output layer |

**Optimizer:** Adam  
**Loss:** Mean Squared Error (MSE)  
**Metric:** Mean Absolute Error (MAE)

---

## 📊 Model Performance

| Metric | Value | Meaning |
|--------|--------|----------|
| **MAE** | 0.03 (normalized) | On average, predictions differ by 3% of price range |
| **Real-scale MAE** | ≈ ₹25–₹30 | Average prediction error in rupees |
| **Verdict** | ✅ Good accuracy for stock time-series forecasting |

> 💡 The model effectively captures market trends, though some short-term volatility may still remain unpredictable.

---

## 🧩 How It Works

1. **Data Collection:**  
   Stock data is downloaded via the [Yahoo Finance API](https://pypi.org/project/yfinance/).

2. **Preprocessing:**  
   - Only the `Close` price is used.  
   - Data is normalized using `MinMaxScaler` (0–1 range).  
   - 80% of data used for training, 20% for testing.

3. **Model Prediction:**  
   - The last 100 days of stock data are used to predict the next day’s price.  
   - Predictions are scaled back to real values for visualization.

4. **Visualization (Streamlit):**  
   - Moving averages: MA50, MA100, MA200  
   - Actual vs Predicted closing prices

---

## 🧰 Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy, Pandas, Matplotlib**
- **Scikit-learn**
- **Streamlit**
- **Yahoo Finance API (`yfinance`)**

---

## 🖥️ Streamlit Web App

### Run Locally:
```bash
# 1️⃣ Clone the repository
git clone https://github.com/<your-username>/Stock-Prediction-LSTM.git
cd Stock-Prediction-LSTM

# 2️⃣ Create a virtual environment
python -m venv myenv
source myenv/bin/activate   # on Windows: myenv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Streamlit app
streamlit run app.py
