import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(h, w, maxit=20, r=2):
    x = np.linspace(-2.5, 1.5, 4*h + 1)
    y = np.linspace(-1.5, 1.5, 3*w + 1)
    a, b = np.meshgrid(x, y)
    c = a + b * 1j
    z = np.zeros_like(c)
    divTime = maxit + np.zeros(z.shape, dtype=int)

    for i in range(maxit):
        z = z ** 2 + c
        diverge = abs(z) > r  # who is diverging
        div_now = diverge & (divTime == maxit)  # who is diverging now
        divTime[div_now] = i  # note when
        z[diverge] = r  # avoid diverging too much

    return divTime


plt.imshow(mandelbrot(400, 400))
plt.show()

