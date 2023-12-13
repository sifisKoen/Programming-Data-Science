import numpy as np
import pandas as pd


predictions_df = pd.read_csv("15.txt", header=None)
assert predictions_df.shape == (51077, 1)
assert np.all((predictions_df.values >= 0) & (predictions_df.values <= 1))