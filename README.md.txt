# Stock Market Copilot

This project demonstrates a basic example of predicting stock movement using Machine Learning and deploying it with FastAPI.

## Deployment Code

### Libraries and Software:

| Library/Software | Version     | Purpose                                             |
|------------------|-------------|-----------------------------------------------------|
| Python           | 3.10 or higher | Programming language                              |
| IPython          | 8.14.0      | Provides interactive computing capabilities         |
| IPython.display  | 8.14.0      | Provides tools for displaying rich content in Jupyter notebooks |
| pandas           | 2.0.3       | Data manipulation and analysis                     |
| NumPy            | 1.25.2      | Numerical operations                               |
| joblib           | 1.3.2       | Loading and saving models                          |
| Flask            | 2.3.3       | Web framework for building APIs                    |
| json             | (built-in)  | Working with JSON data                             |
| requests         | 2.31.0      | Making HTTP requests                               |
| wandb            | 0.15.12     | Weights and Biases for experiment tracking         |
| FastAPI          | 0.103.1     | Web framework for building APIs                    |
| pydantic         | 2.4.2       | Data validation and parsing                        |
| multiprocessing  | (built-in)  | Running the API in a separate process              |
| threading        | (built-in)  | Running the API in a separate thread               |
| logging          | (built-in)  | Logging information                                |
| datetime         | (built-in)  | Working with dates and times                       |
| matplotlib       | 3.8.0       | Creating plots                                     |
| scikit-learn     | 1.3.2       | Machine learning library                           |
| yfinance         | 0.2.31      | Fetching financial data                            |
| ta               | 0.10.2      | Technical analysis library                         |
| itertools        | (built-in)  | Working with iterators                             |
| re               | (built-in)  | Working with regular expressions                   |
| TensorFlow       | 2.14.0      | Deep learning library                              |
| warnings         | (built-in)  | Controlling warning messages                       |
| Ngrok            | latest      | Exposing local server to the internet              |


### Dependencies:

Install the necessary libraries using:
`pip install -r requirements.txt`

### Weights & Biases:

Create a Weights & Biases account, obtain an API key, and set it as an environment variable:

`WANDB_API_KEY="your_api_key"`

### Data Source:

The project fetches stock data from Yahoo Finance.  Update the `ticker`, `range`, `n_days` and `date`  variables in the code to use different stock symbols and prediction periods.
Here,
- `ticker`: ticker code of the stock selected 
- `range`: Prediction period selected (2 weeks, 1 month, 3 months)
- `n_days`: No. of working days for which prediction is carried out ['2 weeks': 10 days, '1 month': 20 days, '3 months': 60 days]
- `date`: Date from which onwards the prediction is to be carried out.

### Model Training:

The model was trained using LSTM Neural Network architecture on a dataset of daily stock price data. You can find more details about the model and training process in the Notebook. Version 0 of the models - `stock_lstm_2w_model`, `stock_lstm_1mo_model` and `stock_lstm_3mo_model` from the project mavely-jerry-indian-school-of-business/stock-Price-lstm were used in this code. 

- `stock_lstm_2w_model`: Best model for predicting closing price for 2 weeks of timeline
- `stock_lstm_1mo_model`: Best model for predicting closing price for 1 month of timeline
- `stock_lstm_3mo_model`: Best model for predicting closing price for 3 months of timeline

### API Endpoints:

- `/predict`: Predicts the stock movement for the given ticker and period.
- `payload = {
     "Date": "2025-04-20",
     "n_days": 20,
     "range": "1 month",
     "stock": "AAPL (AAPL)"
	}`: Format of the payload accepted by the API.

### Deployment:

For local development, you can use Ngrok to expose the API:

`bash ngrok http 8000`

### Disclaimer:

This code is for educational and demonstration purposes only. It should not be used for making real-world financial decisions.



## UI - Streamlit Code

### Libraries and Software:


| Library/Software | Version                      | Purpose                            | Installation/Setup            | Important Factors                                      |
|------------------|------------------------------|------------------------------------|-------------------------------|--------------------------------------------------------|
| Python           | 3.10 or higher (Recommended) | Base programming language          | Pre-installed in Google Colab | Ensure compatibility with other libraries              |
| Streamlit        | 1.25.0 or higher (Recommended)| Building interactive web apps      | pip install streamlit         | Updates may introduce new features/changes             |
| Altair           | 5.0.1 or higher (Recommended) | Statistical visualizations         | pip install altair            | Used for interactive charts (optional)                 |
| Uvicorn          | 0.22.0 or higher (Recommended)| ASGI server for web apps           | pip install uvicorn           | Required for running FastAPI (if applicable)           |
| Pyngrok          | 5.2.1 or higher (Recommended) | Secure public URLs for local servers | pip install pyngrok         | Requires an auth token for extended usage              |
| Scikit-learn     | 1.3.2                        | Machine learning library           | pip install scikit-learn==1.3.2 | Used for model training and predictions              |
| Pandas           | 2.0.3 or higher (Recommended) | Data manipulation and analysis     | pip install pandas            | Updates may affect data handling                       |
| NumPy            | 1.24.3 or higher (Recommended)| Numerical computing                | pip install numpy             | Core library for scientific computing                  |
| Joblib           | 1.3.2 or higher (Recommended) | Model persistence                  | pip install joblib            | Used for loading/saving models (optional)              |
| Plotly           | 5.15.0 or higher (Recommended)| Interactive visualizations         | pip install plotly            | Used for creating charts                               |
| Seaborn          | 0.12.2 or higher (Recommended)| Statistical data visualization     | pip install seaborn           | Used for creating KDE plots                            |
| Matplotlib       | 3.7.2 or higher (Recommended) | Static plotting library            | pip install matplotlib         | Used for basic plotting (optional)                     |
| Google Colab     | N/A                          | Cloud-based notebook environment   | Access through a web browser  | Requires a Google account                              |
| Ngrok            | N/A                          | Secure tunneling service           | Requires account and auth token| Used for public URL exposure                           |
| FastAPI          | N/A                          | Web framework (optional)           | pip install fastapi           | Used for building APIs (if applicable)                 |
| Requests         | 2.31.0 or higher (Recommended)| HTTP requests for API interaction  | pip install requests          | Essential for API communication                        |
| Datetime         | N/A                          | Date and time manipulation         | Part of Python standard library | Used for date formatting                             |


### Features:

* **Predictive Analytics:** Utilizes a pre-trained machine learning model to forecast stock prices for a selected period.
* **Interactive UI:** Built with Streamlit for an intuitive user experience, allowing users to select stocks, prediction dates, and periods.
* **Visualization:** Presents predictions and historical data using interactive charts and graphs for clear insights.
* **Data Export:** Offers the option to download prediction data in CSV format for further analysis.
* **Drift Detection (Developer Mode):** Provides tools for monitoring data drift, ensuring model accuracy over time.


### Run the Application:

`bash streamlit run app.py`

### Usage

1. **Launch the application** by following the instructions in the Installation and Setup section.
2. **Select a stock** from the dropdown menu.
3. **Choose a prediction start date** using the date picker.
4. **Select a prediction period** (2 weeks, 1 month, or 3 months).
5. **Click the "Generate Predictions" button** to initiate the prediction process.
6. **View the results** on the interactive chart and data table.
7. **Download predictions** as a CSV file (optional).
8. **Switch to "Developer" mode** (using the profile toggle) to access drift detection tools.


## Drift Check Code

### Libraries and Software:

| Library/Software         | Version (if specified) | Purpose                           | Other Important Factors                          |
|--------------------------|------------------------|-----------------------------------|--------------------------------------------------|
| Python                   | 3.10 or higher         | Core programming language         |                                                  |
| IPython                  |                        | Interactive computing environment |                                                  |
| IPython.display          |                        | Displaying rich content in notebooks |                                              |
| pandas                   |                        | Data manipulation and analysis    |                                                  |
| numpy                    |                        | Numerical computing               |                                                  |
| matplotlib               |                        | Data visualization                |                                                  |
| seaborn                  |                        | Statistical data visualization    |                                                  |
| alibi                    |                        | Drift detection library           |                                                  |
| alibi_detect             |                        | Drift detection algorithms        |                                                  |
| fastapi                  |                        | Building APIs                     |                                                  |
| uvicorn                  |                        | ASGI server for FastAPI           |                                                  |
| pyngrok                  |                        | Secure tunnels to localhost       |                                                  |
| yfinance                 |                        | Downloading financial data        |                                                  |
| ta (Technical Analysis)  |                        | Technical indicators calculation  |                                                  |
| joblib                   |                        | Persistence of Python objects     |                                                  |
| sklearn (scikit-learn)   |                        | Machine learning                  |                                                  |
| requests                 |                        | HTTP requests                     |                                                  |
| tensorflow               |                        | Deep learning                     | Not directly used but included as a dependency   |
| logging                  |                        | Logging events                    |                                                  |
| re                       |                        | Regular expressions               |                                                  |
| datetime                 |                        | Date and time manipulation        |                                                  |


### Set up Ngrok authentication:

`bash ngrok authtoken`

### Usage

1. Run the Colab notebook to start the API server.
2. Access the API endpoint using the public URL provided by Ngrok.
3. Send a POST request to `/checkDrift` with the required parameters.

### Dataset

Stock data is sourced from Yahoo Finance.

### Model Monitoring

Drift detection is performed using Chi Square and K-S tests to identify changes in data distribution.
