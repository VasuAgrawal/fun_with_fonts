#!/usr/bin/env python3

import argparse
import pathlib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path", type=pathlib.Path)
args = parser.parse_args()

data = np.genfromtxt(args.path, delimiter=",", dtype=int, skip_header=1)
best_params = []
for i in sorted(set(data[:, 0])):
    rows = data[data[:, 0] == i]
    tossed = min(rows[:, -2])
    params = rows[rows[:, -2] == tossed, :]
    best_params.append(params[0, :])

params = np.vstack(best_params)
print("\n".join([",".join(map(str, row)) for row in params]))

import matplotlib.pyplot as plt

plt.plot(params[:, 0], 1 - (params[:, -2] / params[:, -4]))
plt.ylim(0, 1)
plt.show()
