import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import imageio

from init_centroids import init_centroids
from k_means import k_means
from closest_centroids import closest_centroids

if __name__ == '__main__':

    print('='*18, "Beginning", '='*18, '\n')

    # 24-bit (3 Byte) colour representation of an image (16.78m colours)
    # Each pixel represented by three 8-bit unsigned integers (0-255), specifying the RGB intensity
        # 8-bit (1 Byte) - Green
        # 8-bit (1 Byte) - Red
        # 8-bit (1 Byte) - Blue

    # We want to compress the image. Therefore we reduce the number of colours to 16
    # Each pixel requires only 4 bits (2^bits = 16 colours)

    # ========================= Part 1: K-Means Clustering on Image Pixels ========================
    print('='*10, 'Part 1: K-Means Clustering on Image Pixels', '='*10)
    print('\nRunning K-Means clustering on pixels from image...\n')

    # Load an image of a bird
    A = imageio.imread(r'C:\Users\mackt\Python\Machine Learning\Data\12. Toucan_Picture.png') # (255, 255, 3)
    # Get the RGB colour saturation for each pixel
    A = A.astype(float)/255

    # Size of the image
    img_size = A.shape # (128, 128, 3)

    # Reshape the image into an Nx3 matrix where N = number of pixels (16384).
    X = A.reshape([img_size[0] * img_size[1], img_size[2]]) # (16384, 3)

    K = 16
    max_iters = 10

    # When using K-Means, it is important the initialize the centroids randomly.
    initial_centroids = init_centroids(X, K)

    # Run K-Means
    centroids, idx = k_means(X, initial_centroids, max_iters)

    # ========================= Part 2: Image Compression ========================
    print('\n', '='*10, 'Part 2: Image Compression', '='*10)
    print('\nApplying K-Means to compress image...')

    idx = closest_centroids(X, centroids)

    # Recover the image from the indices (idx) by mapping each pixel (specified by its index in idx) to the centroid value.
    X_recovered = centroids[idx.astype(int), :]

    # Reshape the recovered image into proper dimensions
    X_recovered = X_recovered.reshape(img_size)

    fig = plt.figure()
    # Display the original image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(A)
    ax1.set_title('Original - 16.8m colours')
    # Display compressed image side by side
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(X_recovered)
    ax2.set_title('Compressed - {} colours.'.format(K))
    plt.show()

    print('\n','='*22, "End", '='*22)