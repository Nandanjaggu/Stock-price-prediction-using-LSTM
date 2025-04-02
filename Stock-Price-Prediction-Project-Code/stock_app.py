import dash
from dash import dcc, html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# Initialize Dash app
app = dash.Dash()
server = app.server

scaler = MinMaxScaler(feature_range=(0, 1))

# Load dataset if available
nse_file = "./NSE-TATA.csv"
if os.path.exists(nse_file):
    df_nse = pd.read_csv(nse_file)
else:
    raise FileNotFoundError("NSE-TATA.csv not found!")

# Convert date column to datetime format
df_nse["Date"] = pd.to_datetime(df_nse.Date, format="%Y-%m-%d")
df_nse.set_index("Date", inplace=True)

# Prepare data for model
new_data = df_nse[["Close"]].copy()
dataset = new_data.values
train, valid = dataset[:987, :], dataset[987:, :]
scaled_data = scaler.fit_transform(dataset)

# Prepare training data
x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Load the pre-trained LSTM model
model = load_model("saved_model.h5")

# Prepare test data
inputs = new_data[-(len(valid) + 60):].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = np.array([inputs[i - 60:i, 0] for i in range(60, inputs.shape[0])])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

valid = new_data.iloc[987:].copy()
valid['Predictions'] = closing_price

# Load stock data
stock_file = "./stock_data.csv"
if os.path.exists(stock_file):
    df = pd.read_csv(stock_file)
else:
    raise FileNotFoundError("stock_data.csv not found!")

# Dash Layout
app.layout = html.Div([
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='NSE-TATAGLOBAL Stock Data', children=[
            html.Div([
                html.H2("Actual Closing Price", style={"textAlign": "center"}),
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=valid.index, y=valid["Close"], mode='markers', name='Actual')],
                        layout=go.Layout(title='Actual Closing Price', xaxis={'title': 'Date'}, yaxis={'title': 'Price'})
                    )
                ),
                html.H2("LSTM Predicted Closing Price", style={"textAlign": "center"}),
                dcc.Graph(
                    figure=go.Figure(
                        data=[go.Scatter(x=valid.index, y=valid["Predictions"], mode='markers', name='Predicted')],
                        layout=go.Layout(title='LSTM Predicted Price', xaxis={'title': 'Date'}, yaxis={'title': 'Price'})
                    )
                )
            ])
        ]),
        
        dcc.Tab(label='Stock Data Comparison', children=[
            html.Div([
                html.H2("Stock High vs Low Prices", style={'textAlign': 'center'}),
                dcc.Dropdown(id='dropdown-highlow',
                             options=[{'label': stock, 'value': stock} for stock in df["Stock"].unique()],
                             multi=True, value=['FB'], style={"width": "60%", "margin": "auto"}),
                dcc.Graph(id='highlow'),
                
                html.H2("Stock Market Volume", style={'textAlign': 'center'}),
                dcc.Dropdown(id='dropdown-volume',
                             options=[{'label': stock, 'value': stock} for stock in df["Stock"].unique()],
                             multi=True, value=['FB'], style={"width": "60%", "margin": "auto"}),
                dcc.Graph(id='volume')
            ])
        ])
    ])
])

# Callbacks
@app.callback(Output('highlow', 'figure'), [Input('dropdown-highlow', 'value')])
def update_highlow(selected_stocks):
    traces = [
        go.Scatter(x=df[df["Stock"] == stock]["Date"], y=df[df["Stock"] == stock]["High"], mode='lines', name=f'{stock} High')
        for stock in selected_stocks
    ] + [
        go.Scatter(x=df[df["Stock"] == stock]["Date"], y=df[df["Stock"] == stock]["Low"], mode='lines', name=f'{stock} Low')
        for stock in selected_stocks
    ]
    return go.Figure(data=traces, layout=go.Layout(title="High vs Low Prices", xaxis={'title': 'Date'}, yaxis={'title': 'Price'}))

@app.callback(Output('volume', 'figure'), [Input('dropdown-volume', 'value')])
def update_volume(selected_stocks):
    traces = [
        go.Scatter(x=df[df["Stock"] == stock]["Date"], y=df[df["Stock"] == stock]["Volume"], mode='lines', name=f'{stock} Volume')
        for stock in selected_stocks
    ]
    return go.Figure(data=traces, layout=go.Layout(title="Stock Market Volume", xaxis={'title': 'Date'}, yaxis={'title': 'Transactions'}))

# Run server
if __name__ == '__main__':
    app.run(debug=True)

