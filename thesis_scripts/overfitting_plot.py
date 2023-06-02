import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_path = "/home/aleks/magister/audio-noise-reduction-using-nn/speech_denoising_wavenet/sessions/default_5_batch_1e-4lr/history_default_5_batch_1e-4lr.csv"

df = pd.read_csv(csv_path)
loss = df[["loss"]].values.flatten().tolist()
val_loss = df[["val_loss"]].values.flatten().tolist()

epochs = list(range(0, len(loss), 1))

fig, ax = plt.subplots()
# plt.plot(epochs, loss, color="orange")
# plt.xlabel("Epoka")
# plt.ylabel("MSE")

plt.plot(epochs, val_loss, color="dodgerblue", marker="o")
plt.xlabel("Epoka")
plt.ylabel("MSE")
# plt.legend(["Zbiór walidacyjny"])

fig, ax = plt.subplots()
plt.plot(epochs, loss, color="orange", marker="o")
plt.xlabel("Epoka")
plt.ylabel("MSE")


plt.show()