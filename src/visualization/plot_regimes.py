import numpy as np
import matplotlib.pyplot as plt
import os

def main():

    regimes = np.load("outputs/regimes/test_regime_labels.npy")
    shifts = np.load("outputs/regimes/test_shift_indices.npy")

    # binary shift signal
    shift_signal = np.zeros(len(regimes))
    shift_signal[shifts] = 1

    plt.figure(figsize=(14,3))

    plt.plot(
        shift_signal,
        color="red",
        linewidth=2
    )

    plt.title("Detected Distribution Shifts")
    plt.xlabel("Window Index")
    plt.ylabel("Shift Occurrence")

    plt.grid(True)

    os.makedirs("outputs/plots", exist_ok=True)

    plt.savefig(
        "outputs/plots/regime_shifts_v2.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

if __name__ == "__main__":
    main()