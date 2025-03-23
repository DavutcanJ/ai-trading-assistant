# AI Trading Assistant

An advanced AI-powered trading assistant that combines sentiment analysis, technical indicators, and deep learning for stock market predictions.

## Features

- **Real-time Stock Data**: Fetches live stock data using yfinance
- **News Sentiment Analysis**: 
  - Uses FinBERT for financial news sentiment analysis
  - Integrates Alpha Vantage News API for real-time market news
  - Combines multiple sentiment sources for better accuracy
- **Technical Analysis**:
  - Calculates EMA-9 and EMA-21 for trend detection
  - Implements RSI indicator
  - Tracks volume changes
- **Deep Learning Predictions**:
  - Custom TensorFlow Transformer model for time series prediction
  - Predicts price movements for next 1-4 hours
  - Combines technical indicators with sentiment analysis
- **Automated Trading Signals**:
  - Generates LONG/SHORT recommendations
  - Considers multiple factors:
    - News sentiment
    - Technical indicators
    - ML predictions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DavutcanJ/ai-trading-assistant.git
cd ai-trading-assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a .env file with your Alpha Vantage API key:
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## Usage

Run the main script:
```bash
python trading_assistant.py
```

The program will:
1. Fetch real-time stock data
2. Analyze latest news sentiment
3. Calculate technical indicators
4. Make price predictions
5. Generate trading recommendations
6. Save results to CSV file

## Configuration

Edit the following variables in `trading_assistant.py`:

- `HISSELER`: List of stock symbols to monitor
- `gun_sayisi`: Number of days of historical data to use
- `guncelleme_suresi`: Update interval in seconds

## Model Architecture

The project uses a custom Transformer architecture:
- Multi-head self-attention layers
- Positional encoding for time series
- Dense output layer for multi-step prediction

Input features:
- Price data
- Volume
- Technical indicators (RSI, EMAs)
- Sentiment scores

## Output Format

The program generates a CSV file with:
- Stock symbol
- Latest news
- Sentiment analysis (FinBERT + Alpha Vantage)
- Technical indicators
- Price predictions (1-4 hours)
- Trading recommendations

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Do not use it for actual trading without proper validation and risk management. 