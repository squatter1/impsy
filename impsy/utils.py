import numpy as np
import pandas as pd
import random


# MDRNN config 

SIZE_TO_PARAMETERS = {
    "xxs": {
        "units": 16,
        "mixes": 5,
        "layers": 2,
    },
    "xs": {
        "units": 32,
        "mixes": 5,
        "layers": 2,
    },
    "s": {"units": 64, "mixes": 5, "layers": 2},
    "m": {"units": 128, "mixes": 5, "layers": 2},
    "l": {"units": 256, "mixes": 5, "layers": 2},
    "xl": {"units": 512, "mixes": 5, "layers": 3},
    "default": {"units": 128, "mixes": 5, "layers": 2},
}


def mdrnn_config(size: str):
    """Get a config dictionary from a size string as used in the IMPS command line interface."""
    return SIZE_TO_PARAMETERS[size]


# Manages Training Data for the Musical MDN and can generate fake datsets for testing.


def batch_generator(seq_len, batch_size, dim, corpus):
    """Returns a generator to cut up datasets into
    batches of features and labels."""
    # generator = batch_generator(SEQ_LEN, BATCH_SIZE, 3, corpus)
    batch_X = np.zeros((batch_size, seq_len, dim))
    batch_y = np.zeros((batch_size, dim))
    while True:
        for i in range(batch_size):
            # choose random example
            l = random.choice(corpus)
            last_index = len(l) - seq_len - 1
            start_index = np.random.randint(0, high=last_index)
            batch_X[i] = l[start_index : start_index + seq_len]
            batch_y[i] = l[
                start_index + 1 : start_index + seq_len + 1
            ]  # .reshape(1,dim)
        yield batch_X, batch_y


def generate_data(samples=50000):
    """Generating some Slightly fuzzy sine wave data."""
    NSAMPLE = samples
    print("Generating", str(NSAMPLE), "toy data samples.")
    t_data = np.float32(np.array(range(NSAMPLE)) / 10.0)
    t_interval = t_data[1] - t_data[0]
    t_r_data = np.random.normal(0, t_interval / 20.0, size=NSAMPLE)
    t_data = t_data + t_r_data
    r_data = np.random.normal(size=NSAMPLE)
    x_data = np.sin(t_data) * 1.0 + (r_data * 0.05)
    df = pd.DataFrame({"t": t_data, "x": x_data})
    df.t = df.t.diff()
    df.t = df.t.fillna(1e-4)
    print(df.describe())
    return np.array(df)


def generate_synthetic_3D_data():
    """
    Generates some slightly fuzzy sine wave data
    in two dimensions (plus time).
    """
    NSAMPLE = 50000
    print("Generating", str(NSAMPLE), "toy data samples.")
    t_data = np.float32(np.array(range(NSAMPLE)) / 10.0)
    t_interval = t_data[1] - t_data[0]
    t_r_data = np.random.normal(0, t_interval / 20.0, size=NSAMPLE)
    t_data = t_data + t_r_data
    r_data = np.random.normal(size=NSAMPLE)
    x_data = (np.sin(t_data) + (r_data / 10.0) + 1) / 2.0
    y_data = (np.sin(t_data * 3.0) + (r_data / 10.0) + 1) / 2.0
    df = pd.DataFrame({"a": x_data, "b": y_data, "t": t_data})
    df.t = df.t.diff()
    df.t = df.t.fillna(1e-4)
    print(df.describe())
    return np.array(df)

