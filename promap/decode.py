import promap
import cv2
import numpy as np

class DecodeError(promap.PromapError):
    pass

def decode_gray_code(a):
    a = a ^ (a >> 16)
    a = a ^ (a >> 8)
    a = a ^ (a >> 4)
    a = a ^ (a >> 2)
    a = a ^ (a >> 1)
    return a

def threshold_images(images, threshold=0.5, threshold_bw=0.5):
    images = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images]

    (black_frame, white_frame) = images[0:2]
    images = images[2:]

    mask = np.maximum(white_frame, black_frame) - black_frame
    brightest = np.amax(mask)
    (ret, mask) = cv2.threshold(mask, threshold_bw * brightest, 255, cv2.THRESH_BINARY)

    thresh_frames = []
    for (i, im) in enumerate(images):
        im = (np.minimum(np.maximum(im, black_frame), white_frame) - black_frame) / np.maximum(white_frame - black_frame, 1)
        (ret, im) = cv2.threshold(im, threshold, 1, cv2.THRESH_BINARY)
        im = im * mask # apply mask
        thresh_frames.append(im)

    return (mask, thresh_frames)

def decode_gray_images(w, h, images):
    n_vertical_bars = int(np.ceil(np.log2(w)))
    n_horizontal_bars = int(np.ceil(np.log2(h)))

    expected_n_images = n_vertical_bars + n_horizontal_bars
    if expected_n_images != len(images):
        raise DecodeError("Wrong number of images (got {}, expected {})".format(len(images), expected_n_images))

    x_gray = np.zeros(images[0].shape, dtype="uint32")
    for i in range(n_vertical_bars):
        image = images[i]
        x_gray += (image != 0).astype("uint32") << (n_vertical_bars - i - 1)

    y_gray = np.zeros(images[0].shape, dtype="uint32")
    for i in range(n_horizontal_bars):
        image = images[n_vertical_bars + i]
        y_gray += (image != 0).astype("uint32") << (n_horizontal_bars - i - 1)

    x = decode_gray_code(x_gray)
    y = decode_gray_code(y_gray)
    return (x, y)
