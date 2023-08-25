from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
app = Flask(__name__)

# Load the saved ARIMA model
with open('arima_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Extract input data from the request
#         input_date = pd.to_datetime(request.form['input_date'])
        
#         # Make a prediction using the model
#         forecast = model.get_prediction(start=input_date, dynamic=False)
#         pred_ci = forecast.conf_int()

#         # Extract relevant data for rendering in HTML
#         predicted_mean = forecast.predicted_mean
#         confidence_interval_lower = pred_ci.iloc[0, 0]
#         confidence_interval_upper = pred_ci.iloc[0, 1]

#         return render_template('result.html',
#                                input_date=input_date,
#                                predicted_mean=predicted_mean,
#                                lower_bound=confidence_interval_lower,
#                                upper_bound=confidence_interval_upper)
def predict():
    if request.method == 'POST':
        # ... (prediction code)
        # Extract input data from the request
        input_date = pd.to_datetime(request.form['input_date'])
        
        # Make a prediction using the model
        forecast = model.get_prediction(start=input_date, dynamic=False)
        pred_ci = forecast.conf_int()

        # Extract relevant data for rendering in HTML
        predicted_mean = forecast.predicted_mean
        confidence_interval_lower = pred_ci.iloc[0, 0]
        confidence_interval_upper = pred_ci.iloc[0, 1]        
    
        # Generate a Matplotlib plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Predicted Sales')
        ax.fill_between(forecast.predicted_mean.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Furniture Sales')
        ax.legend()

        # Save the plot as an image
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_url = base64.b64encode(img_buf.read()).decode('utf-8')
        fig.savefig('static/plot.png', bbox_inches='tight')

        return render_template('result.html',
                               input_date=input_date,
                               predicted_mean=predicted_mean,
                               lower_bound=confidence_interval_lower,
                               upper_bound=confidence_interval_upper,
                               plot_url=img_url)

if __name__ == '__main__':
    app.run(debug=True)
