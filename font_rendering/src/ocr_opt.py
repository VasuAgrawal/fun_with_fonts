#!/usr/bin/env python3

from bayes_opt import BayesianOptimization
import subprocess
import pandas as pd
import io

def run(dpi, point, padding):
    output = subprocess.run(["../build/OcrStats",
        "--font-dir", "../../by_extension",
        "--atlas", "the quick brown fox jumps over the lazy dog",
        "--dpi", str(dpi),
        "--point", str(point),
        "--padding", str(padding),
        "--thread_count", str(36),
        "--count", str(1000),
        "--lowercase", "true",
        "--csv", "true",
    ], capture_output = True)

    csv = pd.read_csv(io.StringIO(output.stdout.decode('utf-8')))
    if csv.iloc[0][0] == '0':
        return csv.iloc[0][1]
    else:
        return 0


def run_wrapped(dpi, point, padding = 100):
    # Convert the various floating point values to ints 
    try:
        return run(int(dpi), int(point), int(padding))
    except:
        return 0


def main():
    pbounds = {
            "dpi" : (100, 1000),
            "point" : (10, 100),
            #  "padding": (0, 500),
            }

    optimizer = BayesianOptimization(
            f = run_wrapped, 
            pbounds = pbounds, 
            verbose = 2,
            random_state = 1
    ) 

    optimizer.maximize(
        init_points = 10,
        n_iter = 100,
    )

    print(optimizer.max)


if __name__ == "__main__":
    main()

