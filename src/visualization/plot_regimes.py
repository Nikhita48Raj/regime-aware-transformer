import numpy as np
import matplotlib.pyplot as plt
import os


def main():

    regimes = np.load("outputs/regimes/test_regime_labels.npy")
    shifts = np.load("outputs/regimes/test_shift_indices.npy")

    plt.figure(figsize=(14, 3))

    # plot regime timeline
    plt.plot(regimes, drawstyle="steps-mid", linewidth=2)

    # plot vertical lines for shifts
    for s in shifts:
        plt.axvline(s, color="red", alpha=0.25, linewidth=1)

    plt.title("Regime Timeline with Shift Detections")
    plt.xlabel("Window Index")
    plt.ylabel("Regime Label")

    plt.grid(True)

    os.makedirs("outputs/plots", exist_ok=True)

    plt.savefig(
        "outputs/plots/regime_shifts.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


if __name__ == "__main__":
    main()