from scipy import misc
from numpy import linalg
import matplotlib.pyplot as plt

img = misc.face()

img_array = img / 255
img_gray = img_array @ [0.2126, 0.7152, 0.0722]

plt.imshow(img_gray, cmap='gray')
plt.show()
