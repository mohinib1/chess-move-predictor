###############################################################################
# CHESS MOVE PREDICTOR USING LOGISTIC REGRESSION
# Goal: Predict the next move from a given chess position (FEN)
# Author: Mohith Baskaran
###############################################################################

import chess
import chess.pgn
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


###############################################################################
# STEP 1: LOAD PGN DATA AND EXTRACT (POSITION, NEXT MOVE) PAIRS
###############################################################################

def extract_positions_from_pgn(pgn_path, max_games=300):
    positions, moves = [], []

    with open(pgn_path) as pgn:
        for _ in tqdm(range(max_games), desc="Reading games"):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            board = game.board()
            for move in game.mainline_moves():
                positions.append(board.fen())
                moves.append(board.san(move))
                board.push(move)

    df = pd.DataFrame({"fen": positions, "move": moves})
    print(f"\nExtracted {len(df)} positions from {max_games} games.")
    return df


###############################################################################
# STEP 2: FEATURE EXTRACTION FROM FEN
###############################################################################

def fen_to_features(fen):
    board = chess.Board(fen)
    pieces = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    feats = []

    # Count pieces for both sides
    for p in pieces:
        feats.append(len(board.pieces(p, chess.WHITE)))  # white
        feats.append(len(board.pieces(p, chess.BLACK)))  # black

    # Material balance (weighted)
    piece_values = [1, 3, 3, 5, 9]
    material_balance = sum(
        (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK))) * val
        for p, val in zip(pieces, piece_values)
    )
    feats.append(material_balance)

    # Whose turn it is (1 = white, 0 = black)
    feats.append(1 if board.turn == chess.WHITE else 0)

    return np.array(feats, dtype=float)


###############################################################################
# STEP 3: LOAD DATA + ENCODE LABELS
###############################################################################

def prepare_data(df, top_n_moves=20):
    # Keep only top N most frequent moves
    top_moves = df["move"].value_counts().nlargest(top_n_moves).index
    df = df[df["move"].isin(top_moves)]

    # Convert FENs to numerical features
    X = np.vstack(df["fen"].apply(fen_to_features))
    le = LabelEncoder()
    y = le.fit_transform(df["move"])

    print(f"Using top {top_n_moves} moves for prediction.")
    return X, y, le


###############################################################################
# STEP 4: TRAIN MODEL
###############################################################################

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000, multi_class="multinomial")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred))
    return model


###############################################################################
# STEP 5: TEST MODEL ON RANDOM SAMPLE
###############################################################################

def demo_prediction(df, model, le):
    idx = np.random.randint(0, len(df))
    sample_fen = df.iloc[idx]["fen"]
    true_move = df.iloc[idx]["move"]

    x = fen_to_features(sample_fen).reshape(1, -1)
    pred = model.predict(x)[0]
    pred_move = le.inverse_transform([pred])[0]

    print("-------------------------------------------------------")
    print("Board Position (FEN):", sample_fen)
    print("True Move:", true_move)
    print("Predicted Move:", pred_move)
    print("-------------------------------------------------------")


###############################################################################
# MAIN SCRIPT
###############################################################################

def main():
    pgn_path = "lichess_db_standard_rated_2013-07.pgn"  # <- change this if needed

    print("\n=== Chess Move Prediction Project ===\n")
    df = extract_positions_from_pgn(pgn_path, max_games=300)
    X, y, le = prepare_data(df, top_n_moves=20)
    model = train_model(X, y)
    demo_prediction(df, model, le)


if __name__ == "__main__":
    main()
