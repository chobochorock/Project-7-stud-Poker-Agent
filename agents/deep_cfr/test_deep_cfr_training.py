"""Small regression checks for the Deep CFR trainer."""

import numpy as np

from train_deep_cfr_7th import ACTIONS, DIMENSIONS, RECORD_DTYPE, Reservoir


def main() -> None:
    records = np.zeros(10, dtype=RECORD_DTYPE)
    records["iteration"] = 1
    records["state"][:, -ACTIONS:] = 1
    records["state"][:1, 2] = 1
    records["state"][1:, 4] = 1
    reservoir = Reservoir(10, 5)
    reservoir.add(records, np.random.default_rng(7))
    assert reservoir.size == 10
    assert reservoir.seen == 10
    assert reservoir.sizes == [1, 0, 9]
    states, targets, iterations = reservoir.batch(4, np.random.default_rng(8))
    assert states.shape == (4, DIMENSIONS)
    assert targets.shape == (4, ACTIONS)
    assert np.all(iterations == 1)
    print('{"deep_cfr_training_self_test":"ok"}')


if __name__ == "__main__":
    main()
