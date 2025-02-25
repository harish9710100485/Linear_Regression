```markdown
# Sales Forecasting & Data Analysis System

## Overview
This project is a **Sales Forecasting & Data Analysis System** built using **Flask, Pandas, NumPy, and Scikit-Learn**. It processes sales data, performs data cleaning, and applies **Linear Regression** to predict future sales trends.

## Features
- **Automatic Data Cleaning & Preprocessing**
- **Currency Standardization (INR, USD, EUR conversion)**
- **Predictive Analysis using Linear Regression**
- **Error Handling & Data Validation**
- **Customer-based Sales Forecasting**

## Installation
### Prerequisites
Ensure you have **Python 3.x** installed along with the required dependencies.

### Install Dependencies
```bash
pip install flask pandas numpy scikit-learn
```

## Usage
### Running the Application
1. **Update** the `INPUT_FILE_PATH` variable with your sales data file.
2. Open a terminal and run:
   ```bash
   python script.py
   ```
3. The predictions will be saved in `predictions.txt`.

### API Endpoint (Optional)
You can integrate this with a Flask API:
```python
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    predictions = process_dataframe(df)
    return jsonify(predictions)
```

## File Structure
```
├── script.py  # Main script for data processing & prediction
├── predictions.txt  # Output file storing forecasted results
├── requirements.txt  # List of dependencies
└── README.md  # Documentation
```

## Future Enhancements
- Implement **Time-Series Forecasting (ARIMA, LSTM)**
- Build an **Interactive Dashboard**
- Add **User Authentication & Web Interface**

## Author:Harish
## Intern:Minervasoft


