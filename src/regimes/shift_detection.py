import os
import numpy as np


def detect_regime_shifts(regime_labels):
    shift_indices = []
    for i in range(1, len(regime_labels)):
        if regime_labels[i] != regime_labels[i - 1]:
            shift_indices.append(i)
    return np.array(shift_indices)


def main():
    os.makedirs("outputs/regimes", exist_ok=True)

    regime_labels = np.load("outputs/regimes/test_regime_labels.npy")
    shift_indices = detect_regime_shifts(regime_labels)

    np.save("outputs/regimes/test_shift_indices.npy", shift_indices)

    print(f"Total windows: {len(regime_labels)}")
    print(f"Total detected shifts: {len(shift_indices)}")
    print("First 20 shift indices:")
    print(shift_indices[:20])


if __name__ == "__main__":
    main()