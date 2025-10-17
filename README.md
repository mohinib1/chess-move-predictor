# Chess Move Predictor

A machine learning project that predicts the next chess move based on board features (FEN) using logistic regression.

## Features
- Parses PGN files into (position, move) pairs
- Extracts material + piece counts
- Trains a multinomial logistic regression model
- Outputs accuracy and sample predictions

## Run Locally
```bash
pip install -r requirements.txt
python chess_move_predictor.py
