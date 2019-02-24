import cv2
import numpy as np
import pathlib
import os
import multiprocessing
import time

input_path = pathlib.Path.cwd().joinpath("input")
output_path = pathlib.Path.cwd().joinpath("output")

def flip_image(image):
    img = np.fliplr(image)

    return img

def rotate_image(image):
    (rows, cols) = image.shape[:2]
    r_matrix = cv2.getRotationMatrix2D((cols // 2, rows // 2), 45, 1)
    r_img = cv2.warpAffine(image, r_matrix, (cols, rows))

    return r_img

# def crop_image(image):
#     c_img = image[300:1500, 250:1700] # [Y_start:Y_end, X_start:X_end]
#     return c_img

# def translate_image(image):
#     t_matrix = np.float32([[1,0,300], [0,1,-100]])
#     t_img = cv2.warpAffine(image, t_matrix, (image.shape[1], image.shape[0]))

#     return t_img

# def gaussian_noise(image):
#     #row, col, ch = image.shape
#     mean = (10, 12, 50)
#     #var = 0.1
#     sigma = (1, 5, 30) #var ** 0.5
#     #gauss = np.random.normal(mean, sigma, (row, col, ch))
#     gauss = cv2.randn(image, mean, sigma)
#     gauss = gauss.reshape(row, col, ch)
#     g_img = image + gauss

#     return g_img

def transform_flip(filename):
    img_input_path = pathlib.Path.cwd().joinpath(input_path, filename)
    img = cv2.imread(str(img_input_path), cv2.IMREAD_UNCHANGED)

    img_out = flip_image(img)

    img_output_path = pathlib.Path.cwd().joinpath(output_path, "f_" + filename)
    cv2.imwrite(str(img_output_path), img_out)

    print(f"Flip image: {filename}")

def transform_rotate(filename):
    img_input_path = pathlib.Path.cwd().joinpath(input_path, filename)
    img = cv2.imread(str(img_input_path), cv2.IMREAD_UNCHANGED)

    img_out = rotate_image(img)

    img_output_path = pathlib.Path.cwd().joinpath(output_path, "r_" + filename)
    cv2.imwrite(str(img_output_path), img_out)

    print(f"Rotate image: {filename}")

# def transform_gaussian(filename):
#     img_input_path = pathlib.Path.cwd().joinpath(input_path, filename)
#     img = cv2.imread(str(img_input_path), cv2.IMREAD_UNCHANGED)

#     img_out = gaussian_noise(img)

#     img_output_path = pathlib.Path.cwd().joinpath(output_path, "g_" + filename)
#     cv2.imwrite(str(img_output_path), img_out)

#     print(f"Gaussian image: {filename}")


if __name__ == '__main__':
    t1 = time.time()

    filenames = os.listdir(str(input_path))

    p = multiprocessing.Pool(os.cpu_count() - 1)
    
    p.map_async(transform_flip, filenames)
    p.map_async(transform_rotate, filenames)
    # p.map_async(transform_gaussian, filenames)
    
    p.close()
    p.join()

    print(time.time() - t1)

