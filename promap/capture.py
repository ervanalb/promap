import promap
import cv2
import logging
import threading
import time

class CaptureError(promap.PromapError):
    pass

def open_camera(camera):
    if camera is None:
        cap = cv2.VideoCapture()
        if not cap.isOpened():
            raise CaptureError("Could not open default capture device") 
    else:
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise CaptureError("Could not open {}".format(camera))
    return cap

def get_camera_size(camera):
    cap = open_camera(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 65535)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 65535)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = int(w)
    h = int(h)
    logger = logging.getLogger(__name__)
    logger.info("Queried camera for size and got {} x {}".format(w, h))
 
    cap.release()
    return (w, h)

def perform_capture(cap):
    logger = logging.getLogger(__name__)
    run = True
    cur_frame = None
    def _worker():
        nonlocal cur_frame
        nonlocal run
        while run:
            (retval, frame) = cap.read()
            cur_frame = frame

    def _cur_frame():
        # locking?
        logger.info("Captured frame")
        return cur_frame

    t = threading.Thread(target=_worker)
    t.daemon = True

    def _stop():
        nonlocal run
        run = False
        t.join()

    t.start()

    return (_cur_frame, _stop)

def capture(camera, width, height):
    # Open camera
    cap = open_camera(camera)
    cap.set(cv2.CAP_PROP_SETTINGS, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if w != width or h != height:
        w = int(w)
        h = int(h)
        raise CaptureError("Could not set camera resolution to {} x {} (camera suggests {} x {})".format(width, height, w, h))

    (get_frame, stop) = perform_capture(cap)

    frames = []

    def _capture():
        frame = get_frame()
        frames.append(frame)
        return frame

    def _stop():
        stop()
        cap.release()
        return frames

    return (_capture, _stop)
