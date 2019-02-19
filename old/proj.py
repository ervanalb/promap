import numpy as np
import cv2
import time
import threading

width=1920
height=1080

def gen_gray_code(n):
    # Always use 32-bit storage
    n_bits = int(np.ceil(np.log2(n)))
    binary_code = np.arange(n, dtype="uint32")
    gray_code = binary_code ^ (binary_code >> 1)
    gray_code_bytes = np.vstack((
        (gray_code >> 24).astype("uint8"),
        ((gray_code >> 16) & 255).astype("uint8"),
        ((gray_code >> 8) & 255).astype("uint8"),
        (gray_code & 255).astype("uint8"),
    )).T.reshape(-1)
    gray_code_bits = np.unpackbits(gray_code_bytes).reshape(-1, 32)[:,-n_bits:]
    return gray_code_bits.T

def gen_gray_images(w, h):
    vertical_bars = gen_gray_code(w)
    n = len(vertical_bars)
    vertical_bars = np.repeat(vertical_bars[:,None,:], h, axis=1) * 255

    horizontal_bars = gen_gray_code(h)
    n = len(horizontal_bars)
    horizontal_bars = np.repeat(horizontal_bars[:,:,None], w, axis=2) * 255

    return list(vertical_bars) + list(horizontal_bars)

# Generate gray code
#graycode = cv2.structured_light.GrayCodePattern_create(width, height)
#(retval, pattern) = graycode.generate()
#assert retval, "graycode.generate failed"
#(black, white) = graycode.getImagesForShadowMasks(None, None)

pattern = gen_gray_images(width, height)
print(len(pattern))
black = np.full((height, width), 0, dtype="uint8")
white = np.full((height, width), 255, dtype="uint8")
pattern = [pattern[-1], black, white, black] + pattern + [white, black]

# Open camera
cap = cv2.VideoCapture("/dev/video2")
cap.set(cv2.CAP_PROP_SETTINGS, 1);
#2304x1536
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2304);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536);

cv2.startWindowThread()
cv2.namedWindow("pattern", cv2.WINDOW_NORMAL);
cv2.resizeWindow("pattern", width, height);
cv2.moveWindow("pattern", 0, 0);
#cv2.setWindowProperty("pattern", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

frames = []
times = []

timestep = 1

run = True

def capture_thread():
    while run:
        (retval, frame) = cap.read()
        frames.append(frame)
        times.append(time.time())

t = threading.Thread(target=capture_thread, daemon=True)
t.start()

STARTUP_TIME = 3
MAX_LATENCY = 1
cv2.imshow("pattern", pattern[0])
time.sleep(STARTUP_TIME)

start_time = time.time() + MAX_LATENCY
for (i, im) in enumerate(pattern):
    while time.time() < start_time + i * timestep:
        time.sleep(0.005)
    print(i, "/", len(pattern))
    cv2.imshow("pattern", im)

time.sleep(1)
run = False
t.join()

cap.release()

print("Got", len(frames), "images in", times[-1] - times[0], "seconds")

averages = [cv2.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))[0] for frame in frames]
start_time = times[0]
prev = None
for (t, a) in zip(times, averages):
    print(a)
    if t - start_time > STARTUP_TIME and a > prev * 1.05:
        start_time = t
        break
    prev = a
else:
    assert False, "No flash detected"

i = 0
gray_frames = []
EXTRA_FRAMES = 2
for (t, frame) in zip(times, frames):
    if t - start_time > i * timestep + 0.3 * timestep:
        gray_frames.append(frame)
        i += 1
        if i >= len(pattern) - EXTRA_FRAMES:
            break
else:
    assert False, "Flash detected too late"

gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in gray_frames]

white_frame = gray_frames[0]
black_frame = gray_frames[1]
mask = np.maximum(white_frame, black_frame) - black_frame
brightest = np.amax(mask)
(ret, mask) = cv2.threshold(mask, 0.5 * brightest, 255, cv2.THRESH_BINARY)
cv2.imshow("pattern", white_frame)
time.sleep(2)
cv2.imshow("pattern", black_frame)
time.sleep(2)
cv2.imshow("pattern", mask)
time.sleep(6)

thresh_frames = []

for (i, im) in enumerate(gray_frames):
    im = (np.minimum(np.maximum(im, black_frame), white_frame) - black_frame) / (white_frame - black_frame)
    (ret, im) = cv2.threshold(im, 0.5, 1, cv2.THRESH_BINARY)
    im = im * mask # apply mask
    thresh_frames.append(im)
    cv2.imshow("pattern", im)
    #time.sleep(0.5)

thresh_frames = thresh_frames[2:-2] # remove head and tail

def decode_gray_code(a):
    a = a ^ (a >> 16)
    a = a ^ (a >> 8)
    a = a ^ (a >> 4)
    a = a ^ (a >> 2)
    a = a ^ (a >> 1)
    return a

def decode_gray_images(w, h, images):
    n_vertical_bars = int(np.ceil(np.log2(w)))
    n_horizontal_bars = int(np.ceil(np.log2(h)))

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

(x, y) = decode_gray_images(width, height, thresh_frames)
x = np.minimum(x, width)
y = np.minimum(y, height)
np.save("x.npy", x)
np.save("y.npy", y)

# Convert to grayscale for visualizing
x = (x * 255 / width).astype("uint8")
y = (y * 255 / height).astype("uint8")
cv2.imshow("pattern", x)
time.sleep(6)
cv2.imshow("pattern", y)
time.sleep(6)

cv2.destroyAllWindows()
