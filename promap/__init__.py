import argparse
import logging
import itertools

class ArgumentError(Exception):
    pass

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser(prog="promap")

    def size(s):
        dims = s.lower().split("x")
        if len(dims) != 2:
            raise ValueError("Size must be in format: w x h")
        try:
            (w, h) = [int(d.strip()) for d in dims]
        except ValueError as e:
            raise ValueError("Width and height must be integers in base-10") from e
        return (w, h)

    # Main operations
    parser.add_argument("-g", "--gray", action="store_true", help="Generate a set of gray code patterns for projection")
    parser.add_argument("-p", "--project", action="store_true", help="Project gray code patterns")
    parser.add_argument("-c", "--capture", action="store_true", help="Capture the gray code projection with the camera")
    parser.add_argument("-d", "--decode", action="store_true", help="Decode a series of gray code images into a lookup image that goes from camera to projector space")
    parser.add_argument("-i", "--invert", action="store_true", help="Invert a lookup image (so that it goes from projector to camera space)")
    parser.add_argument("-l", "--lookup", action="store_true", help="Convert an image from camera space to projector space using a lookup image")
    parser.add_argument("-a", "--all", action="store_true", help="Do all of these operations")

    # Global parameters
    parser.add_argument("--camera-size", type=size, help="The camera resolution")
    parser.add_argument("--projector-size", type=size, help="The projector resolution")

    # Gray code / project
    parser.add_argument("--gray-file", type=str, help="The file to save / load the gray code patterns to / from")

    # Project
    parser.add_argument("--screen", type=str, help="The name of the screen output connected to the projector")

    # Project / Capture
    parser.add_argument("--startup_delay", type=float, help="How long to wait for camera settings to stabilize before displaying the first frame in seconds",
                        default=5)
    parser.add_argument("--period", type=float, help="How long to display each frame for in seconds",
                        default=2)

    # Capture
    parser.add_argument("--camera", type=str, help="The name of the camera device to open")

    # Capture / decode
    parser.add_argument("--capture-file", type=str, help="The file to save / load the captured images to / from")

    # Decode
    parser.add_argument("--threshold-file", type=str, help="The file to save the captured images to after thresholding")

    # Decode / invert
    parser.add_argument("--decoded-file", type=str, help="The file to save / load the decoded image to / from")
    parser.add_argument("--unnormalized", dest="normalized", action="store_false", help="Don't normalize the color values in the decoded file")

    args = parser.parse_args()

    # Internal state
    args.gray_code_images = None
    args.captured_images = None
    args.decoded_image = None

    ops = ("gray",
        "project",
        "capture",
        "decode",
        "invert",
        "lookup")

    if args.all:
        for op in ops:
            setattr(args, op, True)

    ops_bool = [getattr(args, op) for op in ops]

    if not any(ops_bool):
        logger.warning("No operations specified--not doing anything")
        return

    ops_indices = [i for (i, b) in enumerate(ops_bool) if b]
    first = ops_indices[0]
    last = ops_indices[-1]
    if ops_indices != list(range(first, last + 1)):
        missing = [op for (i, op) in enumerate(ops) if i in range(first, last + 1) and i not in ops_indices]
        raise ArgumentError("Non-contiguous operations specified. Missing: " + ", ".join(missing))

    if args.gray:
        op_gray(args)
    if args.project:
        op_project(args)
    if args.capture and not args.project: # capturing while projecting is handled by op_project
        op_capture(args)
    if args.decode:
        op_decode(args)
    if args.invert:
        op_invert(args)
    if args.lookup:
        op_lookup(args)

def filename2format(fn, places=3):
    """Converts a filename to a format string capable of adding an index after the basename"""
    index = "{:0" + str(places) + "d}" if places else "{}"
    last_dot = fn.rfind(".")
    if last_dot < 0: # not found
        return fn + index
    else:
        return fn[0:last_dot] + index + fn[last_dot:]

def op_gray(args):
    import promap.gray

    logger = logging.getLogger(__name__)

    if not args.projector_size:
        if args.project:
            project_get_projector_size(args)
            logger.info("Projector size not specified, querying screen gave {}x{}".format(*args.projector_size))
        else:
            raise ArgumentError("Unknown projector size")

    args.gray_code_images = promap.gray.generate_images(*args.projector_size)

    if not args.project or args.gray_file:
        # Save the image if we are not going to project it
        import cv2
        filename_format = filename2format(args.gray_file if args.gray_file else "gray.png")
        filenames = [filename_format.format(i) for i in range(len(args.gray_code_images))]
        for (fn, im) in zip(filenames, args.gray_code_images):
            cv2.imwrite(fn, im)

def project_get_projector_size(args):
    import promap.project
    args.projector_size = promap.project.get_size(args.screen)

def op_project(args):
    import promap.project

    logger = logging.getLogger(__name__)

    if not args.gray_code_images:
        # Load the gray code from the given files
        import cv2
        filename_format = filename2format(args.gray_file if args.gray_file else "gray.png")
        images = []
        for i in itertools.count():
            fn = filename_format.format(i)
            im = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            if im is None:
                break
            if not args.projector_size:
                args.projector_size = (im.shape[1], im.shape[0])
            else:
                if (im.shape[1], im.shape[0]) != args.projector_size:
                    raise ArgumentError("Image {} does not match projector size {}x{}".format(fn, args.projector_size[0], args.projector_size[1]))
            images.append(im)
        if not images:
            raise ArgumentError("No gray codes to project")
        args.gray_code_images = images

    if not args.projector_size:
        raise ArgumentError("Unknown projector size")

    if args.capture:
        project_and_capture(args) 
    else:
        promap.project.project(args.gray_code_images, args.startup_delay, args.period, args.screen)

def project_and_capture(args):
    logger = logging.getLogger(__name__)
    import promap.capture

    if not args.camera_size:
        args.camera_size = promap.capture.get_camera_size(args.camera)
    (capture, stop) = promap.capture.capture(args.camera, *args.camera_size)

    i = 0
    filename_format = None
    if not args.decode or args.capture_file:
        filename_format = filename2format(args.capture_file if args.capture_file else "cap.png")

    def _capture_callback():
        nonlocal i
        im = capture()
        if filename_format is not None:
            import cv2
            fn = filename_format.format(i)
            cv2.imwrite(fn, im)
        i += 1

    promap.project.project(args.gray_code_images, args.startup_delay, args.period, args.screen, _capture_callback)
    args.captured_images = stop()

def op_capture(args):
    logger = logging.getLogger(__name__)
    import promap.capture

    if not args.camera_size:
        args.camera_size = promap.capture.get_camera_size(args.camera)

    (capture, stop) = promap.capture.capture(args.camera, *args.camera_size)
    i = 0
    filename_format = None
    if not args.decode or args.capture_file:
        filename_format = filename2format(args.capture_file if args.capture_file else "cap.png")
    try:
        print("Press Ctrl-C to finish capturing.")
        while True:
            input("Press [Enter] to capture image {:03d}...".format(i))
            im = capture()
            if filename_format is not None:
                import cv2
                fn = filename_format.format(i)
                cv2.imwrite(fn, im)
            i += 1
    except KeyboardInterrupt:
        pass
    args.captured_images = stop()

def op_decode(args):
    import promap.decode
    import numpy as np
    logger = logging.getLogger(__name__)

    if not args.captured_images:
        # Load the captured images from the given files
        import cv2
        filename_format = filename2format(args.gray_file if args.gray_file else "cap.png")
        images = []
        for i in itertools.count():
            fn = filename_format.format(i)
            im = cv2.imread(fn)
            if im is None:
                break
            if not args.camera_size:
                args.camera_size = (im.shape[1], im.shape[0])
            else:
                if (im.shape[1], im.shape[0]) != args.camera_size:
                    raise ArgumentError("Image {} does not match camera size {}x{}".format(fn, args.camera_size[0], args.camera_size[1]))
            images.append(im)
        if not images:
            raise ArgumentError("No images to decode")
        args.captured_images = images

    if not args.projector_size:
        raise ArgumentError("Unknown projector size")

    (mask, thresh_images) = promap.decode.threshold_images(args.captured_images)
    if args.threshold_file:
        import cv2
        filename_format = filename2format(args.threshold_file)
        cv2.imwrite(filename2format(args.threshold_file, places=None).format("mask"), mask)
        for (i, im) in enumerate(thresh_images):
            fn = filename_format.format(i)
            cv2.imwrite(fn, im)

    (x, y) = promap.decode.decode_gray_images(args.projector_size[0], args.projector_size[1], thresh_images)
    args.decoded_image = np.dstack((x, y))
    if not args.invert or args.decoded_file:
        import cv2
        fn = args.decoded_file if args.decoded_file else "decoded.png"
        im = np.dstack((np.zeros(x.shape), y, x))
        if args.normalized:
            im[:,:,2] /= args.projector_size[0]
            im[:,:,1] /= args.projector_size[1]
            maxval = (1 << 16) - 1
            im = np.round(np.maximum(np.minimum(im * maxval, maxval), 0)).astype(np.uint16)
        cv2.imwrite(fn, im)

def op_invert(args):
    import promap.reproject
    import numpy as np
    logger = logging.getLogger(__name__)

    if not args.projector_size:
        raise ArgumentError("Unknown projector size")

    if not args.decoded_image:
        # Load the decoded image from the given file
        import cv2
        fn = args.decoded_file if args.decoded_file else "decoded.png"
        im = cv2.imread(fn)
        x = im[:,:,2]
        y = im[:,:,1]
        if args.normalized:
            if x.dtype == np.uint8:
                bits = 8
            elif x.dtype == np.uint16:
                bits=16
            else:
                raise ArgumentError("Decoded image has unrecognized bit depth")
            maxval = (1 << bits) - 1
            x = np.round(x / maxval * args.projector_size[0]).astype(np.int)
            y = np.round(y / maxval * args.projector_size[1]).astype(np.int)
    else:
        x = args.decoded_image[0]
        y = args.decoded_image[1]

    (camx, camy) = promap.reproject.compute_inverse_and_disparity(x, y, args.projector_size[0], args.projector_size[1])
    import matplotlib.pyplot as plt
    plt.imshow(camx)
    plt.show()
    plt.imshow(camy)
    plt.show()

def op_lookup(args):
    logger = logging.getLogger(__name__)
    logger.warning("lookup not implemented")

