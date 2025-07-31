import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def generate_weight_combinations(total=6, num_weights=4, num_samples=50):
    combinations = []
    for _ in range(num_samples):
        rand = sorted([0] + random.sample(range(1, total * 100), num_weights - 1) + [total * 100])
        sample = [(rand[i+1] - rand[i])/100 for i in range(num_weights)]
        combinations.append(sample)
    return combinations

Tk().withdraw()
filename = askopenfilename(filetypes=[("CSV files", "*.csv")])
df = pd.read_csv(filename)
df = df[df['angle'].between(0, 80)]

X = df[['w1', 'w2', 'w3', 'w4', 'angle']].values
y = df['score'].values

model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(5,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=500, batch_size=32, verbose=1)

def predict_best_weights(model, angles, total=6, num_weights=4, num_samples=500):
    results = []
    for angle in angles:
        weight_sets = generate_weight_combinations(total, num_weights, num_samples)
        X_test = np.array([ws + [angle] for ws in weight_sets])
        scores = model.predict(X_test, verbose=0).flatten()
        best_idx = np.argmax(scores)
        best_weight = weight_sets[best_idx]
        best_score = scores[best_idx]
        results.append({'angle': angle, 'weights': best_weight, 'predicted_score': best_score})
    return pd.DataFrame(results)

angle_list = list(range(0, 81, 10))
predicted_df = predict_best_weights(model, angle_list)
predicted_df.to_csv("predicted_best_weights.csv", index=False)