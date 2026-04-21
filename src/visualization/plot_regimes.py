import numpy as np
import matplotlib.pyplot as plt


def main():
    regime_labels = np.load("outputs/regimes/test_regime_labels.npy")
    shift_indices = np.load("outputs/regimes/test_shift_indices.npy")

    plt.figure(figsize=(14, 4))
    plt.plot(regime_labels, drawstyle="steps-mid", label="Regime Label")

    for idx in shift_indices:
        plt.axvline(idx, color="red", alpha=0.08)

    plt.title("Regime Timeline with Shift Detections")
    plt.xlabel("Window Index")
    plt.ylabel("Regime Label")
    plt.yticks(sorted(np.unique(regime_labels)))
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()