"""Drive2Win — the shared package your iterations grow into.

What's here on day one:
    normalize.py    input/output scaling. Single source of truth.
    nn.py           MLP forward pass + Adam + reference backprop.
    eval.py         run_policy + score_runs (used by benchmark.py).
    viz.py          every plot the iteration loop needs.

What you'll add as you iterate (suggested names, not required):
    cnn.py          a PyTorch CNN/hybrid model (expose make_policy).
    pipeline.py     a sklearn Pipeline wrapping normalize + model.
    augment.py      data-augmentation helpers (heading-noise, flips).
    agent.py        the model wrapper you'll point benchmark.py at.
"""
