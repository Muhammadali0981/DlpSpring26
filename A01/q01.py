import numpy as np
from PIL import Image


img = Image.open("dog.jpg")
img_array = np.array(img)
np.save("dog.npy", img_array)
print("dog.jpg converted to dog.npy")

image = np.load("dog.npy").astype(np.float32)
mean = 0
std_dev = 15
noise = np.random.normal(mean, std_dev, image.shape)
noisy_image = image + noise
noisy_image = np.clip(noisy_image, 0, 255)
noisy_image = noisy_image.astype(np.uint8)

np.save("dog_noisy.npy", noisy_image)
print("Gaussian noise added and saved as dog_noisy.npy")


noisy_pil = Image.fromarray(noisy_image)
noisy_pil.save("dog_noisy.jpg")

print("dog_noisy.npy converted to dog_noisy.jpg")