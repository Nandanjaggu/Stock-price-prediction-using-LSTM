Overview

This project uses Long Short-Term Memory (LSTM) neural networks to predict stock prices based on historical data. It processes stock market data, trains an LSTM model, and visualizes predictions.

Features

Load and preprocess stock data

Normalize data using MinMaxScaler

Train an LSTM model with historical stock prices

Predict future stock prices

Visualize stock price trends and predictions

Installation

Clone the repository:

git [clone https://github.com/nandanjaggu/stock-price-prediction.git
cd stock-price-prediction](https://github.com/Nandanjaggu/Stock-price-prediction-using-LSTM/tree/main/Stock-Price-Prediction-Project-Code)

Install dependencies:

pip install -r requirements.txt

Dependencies

Ensure you have the following Python libraries installed:

pandas

numpy

matplotlib

scikit-learn

tensorflow

keras

You can install them using:

pip install pandas numpy matplotlib scikit-learn tensorflow keras

Usage

Prepare the dataset (NSE-TATA.csv in this example) and place it in the project directory.

Run the stock_pred.py script:

python stock_pred.py

The trained LSTM model will be saved as saved_lstm_model.h5.

The script will generate visualizations of stock price trends and predictions.

File Structure

stock_pred.py: Main script for training and predicting stock prices using LSTM.

stock_app.py: Additional script for data handling or visualization.

NSE-TATA.csv: Stock data file (ensure it's placed in the directory before running the script).

saved_lstm_model.h5: Trained LSTM model.

Notes

Ensure you replace NSE-TATA.csv with your own dataset if using different stock data.

Modify the script parameters (epochs, batch size, etc.) to optimize performance.

License

This project is licensed under the MIT License.

Author

Your Name - nandan gowda
