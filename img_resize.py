import pathlib
import os
import cv2
import multiprocessing
import time

input_path = pathlib.Path.cwd().joinpath('input')
output_path = pathlib.Path.cwd().joinpath('output')
IMAGE_SIZE = 240

# https://stackoverflow.com/a/44659589
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is not None and height is not None:
        dim = (width, height)
    elif width is None:
        ratio = height / float(h)
        dim = (int(w * ratio), height)
    else:
        ratio = width / float(w)
        dim = (width, int(h * ratio))
    
    resized = cv2.resize(image, dim, interpolation = inter)

    #RGB_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    return resized


def resizer(filename):
    img_input_path = pathlib.Path.cwd().joinpath(input_path, filename)
    image = cv2.imread(str(img_input_path), cv2.IMREAD_UNCHANGED)

    result = []
    if image.shape[0] > image.shape[1]:
       result = image_resize(image, width=IMAGE_SIZE) # Vertical image
    else:
       result = image_resize(image, height=IMAGE_SIZE) # Horizontal image

    #result = image_resize(image, width=28, height=28)

    img_output_path = pathlib.Path.cwd().joinpath(output_path, filename)
    cv2.imwrite(str(img_output_path), result)
    
    print(f"{filename}: {image.shape} --> {result.shape}")

if __name__ == '__main__':
    t1 =time.time()

    filenames = os.listdir(str(input_path))

    p = multiprocessing.Pool(os.cpu_count() - 1)
    p.map(resizer, filenames)
    p.close()

    print(time.time() - t1)