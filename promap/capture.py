import cv2
import logging

class CaptureError(Exception):
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
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(w, h)
 
    cap.close()

def perform_capture(cap):
    run = True
    cur_frame = None
    def _worker():
        nonlocal cur_frame
        while run:
            (retval, frame) = cap.read()
            cur_frame = frame

    def _cur_frame():
        # locking?
        return cur_frame

    t = threading.Thread(target=_worker)
    t.daemon = True

    def _stop():
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

    (get_frame, stop) = perform_capture(cap)
    time.sleep(3)
    stop()

    cap.close()
