# %% [markdown]
# # Tugas 3 | Deteksi Tepi
# 

# %% [markdown]
# Nama: Ronggo Widjoyo<br>
# NIM: 220411100061<br>
# Kelas: PCD A<br>
# 

# %%
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Gradient
# 

# %%
img = cv.imread('test_cropped.jpg', cv.IMREAD_GRAYSCALE)

kernelx = np.array([
	[-1, 1],
	[-1, 1]
])
kernely = np.array([
	[1, 1],
	[-1, -1]
])


gx = cv.filter2D(img, cv.CV_16S, kernelx)
gy = cv.filter2D(img, cv.CV_16S, kernely)
gradient_img = np.sqrt(gx**2 + gy**2)

fig, ax = plt.subplots(2,2, figsize=(5,5))
ax[0,0].imshow(img, cmap='gray')
ax[0,0].set_axis_off()
ax[0,0].set_title('Original Image')

ax[0,1].imshow(gx, cmap='gray')
ax[0,1].set_axis_off()
ax[0,1].set_title('Kernel x Image')

ax[1,0].imshow(gradient_img, cmap='gray')
ax[1,0].set_axis_off()
ax[1,0].set_title('Gradient Image')

ax[1,1].imshow(gy, cmap='gray')
ax[1,1].set_axis_off()
ax[1,1].set_title('Kernel y Image')

# %% [markdown]
# ## Gradient dengan Threshold
# 

# %%
_, th50 = cv.threshold(gradient_img, 50, 255, cv.THRESH_BINARY)
_, th70 = cv.threshold(gradient_img, 70, 255, cv.THRESH_BINARY)
_, th80 = cv.threshold(gradient_img, 80, 255, cv.THRESH_BINARY)

fig, ax = plt.subplots(2,2, figsize=(5,5))
ax[0,0].imshow(gradient_img, cmap='gray')
ax[0,0].set_axis_off()
ax[0,0].set_title('Gradient Image')

ax[0,1].imshow(th50, cmap='gray')
ax[0,1].set_axis_off()
ax[0,1].set_title('Threshold 50')

ax[1,0].imshow(th70, cmap='gray')
ax[1,0].set_axis_off()
ax[1,0].set_title('Threshold 75')

ax[1,1].imshow(th80, cmap='gray')
ax[1,1].set_axis_off()
ax[1,1].set_title('Threshold 80')

# %% [markdown]
# ## Sobel
# 

# %%
img = cv.imread('test_cropped.jpg', cv.IMREAD_GRAYSCALE)

kernelx = np.array([
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]
])
kernely = np.array([
	[1, 2, 1],
	[0, 0, 0],
	[-1, -2, -1]
])

gx = cv.filter2D(img, cv.CV_16S, kernelx)
gy = cv.filter2D(img, cv.CV_16S, kernely)
sobel_img = np.sqrt(gx**2 + gy**2)

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(img, cmap='gray')
ax[0,0].set_axis_off()
ax[0,0].set_title('Original Image')

ax[0,1].imshow(gx, cmap='gray')
ax[0,1].set_axis_off()
ax[0,1].set_title('Kernel x Image')

ax[1,0].imshow(sobel_img, cmap='gray')
ax[1,0].set_axis_off()
ax[1,0].set_title('Sobel Image')

ax[1,1].imshow(gy, cmap='gray')
ax[1,1].set_axis_off()
ax[1,1].set_title('Kernel y Image')

# %% [markdown]
# ## Sobel dengan Threshold
# 

# %%
_, th50 = cv.threshold(sobel_img, 50, 255, cv.THRESH_BINARY)
_, th70 = cv.threshold(sobel_img, 70, 255, cv.THRESH_BINARY)
_, th80 = cv.threshold(sobel_img, 80, 255, cv.THRESH_BINARY)

fig, ax = plt.subplots(2,2, figsize=(5,5))
ax[0,0].imshow(sobel_img, cmap='gray')
ax[0,0].set_axis_off()
ax[0,0].set_title('Sobel Image')

ax[0,1].imshow(th50, cmap='gray')
ax[0,1].set_axis_off()
ax[0,1].set_title('Threshold 50')

ax[1,0].imshow(th70, cmap='gray')
ax[1,0].set_axis_off()
ax[1,0].set_title('Threshold 75')

ax[1,1].imshow(th80, cmap='gray')
ax[1,1].set_axis_off()
ax[1,1].set_title('Threshold 80')

# %% [markdown]
# ## Robert's Cross
# 

# %%
img = cv.imread('test_cropped.jpg', cv.IMREAD_GRAYSCALE)

kernelx = np.array([
	[1, 0],
	[0, -1]
])
kernely = np.array([
	[0, 1],
	[-1, 0]
])

gx = cv.filter2D(img, cv.CV_16S, kernelx)
gy = cv.filter2D(img, cv.CV_16S, kernely)
robert_img = np.sqrt(gx**2 + gy**2)


fig, ax = plt.subplots(2,2, figsize=(5,5))
ax[0,0].imshow(img, cmap='gray')
ax[0,0].set_axis_off()
ax[0,0].set_title('Original Image')

ax[0,1].imshow(gx, cmap='gray')
ax[0,1].set_axis_off()
ax[0,1].set_title('Kernel x Image')

ax[1,0].imshow(robert_img, cmap='gray')
ax[1,0].set_axis_off()
ax[1,0].set_title('Robert\'s Cross Image')

ax[1,1].imshow(gy, cmap='gray')
ax[1,1].set_axis_off()
ax[1,1].set_title('Kernel y Image')

# %% [markdown]
# ## Robert's Cross dengan Threshold
# 

# %%
_, th50 = cv.threshold(robert_img, 50, 255, cv.THRESH_BINARY)
_, th70 = cv.threshold(robert_img, 70, 255, cv.THRESH_BINARY)
_, th80 = cv.threshold(robert_img, 80, 255, cv.THRESH_BINARY)

fig, ax = plt.subplots(2,2, figsize=(5,5))
ax[0,0].imshow(robert_img, cmap='gray')
ax[0,0].set_axis_off()
ax[0,0].set_title('Robert\'s Cross Image')

ax[0,1].imshow(th50, cmap='gray')
ax[0,1].set_axis_off()
ax[0,1].set_title('Threshold 50')

ax[1,0].imshow(th70, cmap='gray')
ax[1,0].set_axis_off()
ax[1,0].set_title('Threshold 75')

ax[1,1].imshow(th80, cmap='gray')
ax[1,1].set_axis_off()
ax[1,1].set_title('Threshold 80')

# %% [markdown]
# ## Prewit
# 

# %%
img = cv.imread('test_cropped.jpg', cv.IMREAD_GRAYSCALE)

kernelx = np.array([
	[1, 1, 1],
	[0, 0, 0],
	[-1, -1, -1]
])
kernely = np.array([
	[-1, 0, 1],
	[-1, 0, 1],
	[-1, 0, 1]
])

gx = cv.filter2D(img, cv.CV_16S, kernelx)
gy = cv.filter2D(img, cv.CV_16S, kernely)
prewit_img = np.sqrt(gx**2 + gy**2)


fig, ax = plt.subplots(2,2, figsize=(5,5))
ax[0,0].imshow(img, cmap='gray')
ax[0,0].set_axis_off()
ax[0,0].set_title('Original Image')

ax[0,1].imshow(gx, cmap='gray')
ax[0,1].set_axis_off()
ax[0,1].set_title('Kernel x Image')

ax[1,0].imshow(prewit_img, cmap='gray')
ax[1,0].set_axis_off()
ax[1,0].set_title('Prewit Image')

ax[1,1].imshow(gy, cmap='gray')
ax[1,1].set_axis_off()
ax[1,1].set_title('Kernel y Image')

# %% [markdown]
# ## Prewit dengan Threshold
# 

# %%
_, th50 = cv.threshold(prewit_img, 50, 255, cv.THRESH_BINARY)
_, th70 = cv.threshold(prewit_img, 70, 255, cv.THRESH_BINARY)
_, th80 = cv.threshold(prewit_img, 80, 255, cv.THRESH_BINARY)

fig, ax = plt.subplots(2,2, figsize=(5,5))
ax[0,0].imshow(prewit_img, cmap='gray')
ax[0,0].set_axis_off()
ax[0,0].set_title('Prewit Image')

ax[0,1].imshow(th50, cmap='gray')
ax[0,1].set_axis_off()
ax[0,1].set_title('Threshold 50')

ax[1,0].imshow(th70, cmap='gray')
ax[1,0].set_axis_off()
ax[1,0].set_title('Threshold 75')

ax[1,1].imshow(th80, cmap='gray')
ax[1,1].set_axis_off()
ax[1,1].set_title('Threshold 80')

# %% [markdown]
# ## Laplacian of Gaussian
# 

# %%
img = cv.imread('test_cropped.jpg', cv.IMREAD_GRAYSCALE)

def create_LOG_kernel(size, sigma):
    R = size // 2
    y, x = np.ogrid[-R:R+1, -R:R+1]

    kernel = (1/sigma**2) * ((x**2 + y**2) / (sigma**2) - 2) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel - np.mean(kernel)

def apply_LOG_filter(img, kernel_size=5, sigma=1.0):
    kernel = create_LOG_kernel(kernel_size, sigma)
    
    kernel = kernel / np.sum(np.abs(kernel))
    
    filtered_img = cv.filter2D(img, -1, kernel)
    return filtered_img


result = apply_LOG_filter(img, kernel_size=17, sigma=1.2)

fig, ax = plt.subplots(1,2, figsize=(10,10))

ax[0].imshow(img, cmap='gray')
ax[0].set_axis_off()
ax[0].set_title('Original Image')

ax[1].imshow(result, cmap='gray')
ax[1].set_axis_off()
ax[1].set_title('LoG Image')

# %% [markdown]
# ## Hasil Akhir
# 

# %% [markdown]
# Dari beberapa percobaan menggunakan berbagai macam algoritma deteksi tepi, dengan menggunakan gambar yang ada, didapat dari hasil yang ditunjukkan oleh algoritma **Laplacian of Gaussian** merupakan algoritma deteksi tepi yang cocok digunakan pada gambar tersebut. Dapat dilihat, hasil dari gelap terang tepian Laplacian yang dihasilkan memiliki tingkat presisi yang bagus diimbangi dengan Gaussian agar mengurangi noise yang ada pada hasil, sehingga menghasilkan hasil yang sesuai dibandingkan dengan algoritma yang lain yang menghasilkan deteksi tepi yang kurang sesuai.
# 


