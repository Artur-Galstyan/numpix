import jax.numpy as jnp
import numpy as np
import torch

from numpix import pix

pix(np.linspace(0, 1, 100).reshape(10, 10), use_kitty_protocol=False)

board = np.indices((8, 8)).sum(axis=0) % 2
pix(board.astype(float))


x = np.linspace(0, 4 * np.pi, 100)
y = np.linspace(0, 4 * np.pi, 100)
pix(np.sin(x[:, None] + y[None, :]))


img = np.zeros((28, 28))
img[4:24, 8:12] = 1.0
img[4:8, 8:20] = 1.0
img[12:16, 8:20] = 1.0
pix(img)


pix((np.arange(50)[:, None] + np.arange(50)[None, :]) % 5 / 4.0)


sparse = np.zeros((50, 50))
sparse[np.random.randint(0, 50, 100), np.random.randint(0, 50, 100)] = 1.0
pix(sparse)


pix(np.sin(np.linspace(0, 2 * np.pi, 100)))


batch = np.stack(
    [
        np.random.rand(28, 28),
        np.eye(28),
        np.outer(np.sin(np.linspace(0, np.pi, 28)), np.ones(28)),
    ]
)
pix(batch)


pix(jnp.array(batch))
pix(torch.from_numpy(batch))

print("Multi Array 1:")

# Multi-array: each gets its own color range
a = np.sin(np.linspace(0, 4 * np.pi, 100)).reshape(10, 10)
b = np.random.rand(10, 10) * 100
pix(a, b)

# Multi-array with shared range: same color scale across all
print("Multi Array 2 with shared range:")
pix(a, b, shared_range=True)

# Arrays with -inf (e.g. masked log-space values)
masked = np.random.randn(8, 8)
masked[np.tril_indices(8, -1)] = -float("inf")
pix(masked)
