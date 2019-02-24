import argparse
import logging
import itertools
import os
import numpy as np
import cv2

class PromapError(Exception):
    pass

class ArgumentError(PromapError):
    pass

class FileWriteError(PromapError):
    pass

class FileReadError(PromapError):
    pass

def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig()

    def size(s):
        dims = s.lower().split("x")
        if len(dims) != 2:
            raise ValueError("Size must be in format: w x h")
        try:
            (w, h) = [int(d.strip()) for d in dims]
        except ValueError as e:
            raise ValueError("Width and height must be integers in base-10") from e
        return (w, h)

    description = """promap is a command-line tool that you can use in conjunction with your digital video workflow to do projection mapping.
promap uses a projector and a single camera to compute the physical scene as viewed from the projector, as well as a coarse disparity (depth) map.
You can import the maps that it generates straight into your digital video workflow, or you can post-process them into masks and uvmaps first using image processing tools.
"""

    parser = argparse.ArgumentParser(prog="promap", description=description)

    # Main operations
    group = parser.add_argument_group("Operations", "Specify what operations are to be performed. Must select a contiguous set.")
    group.add_argument("-g", "--gray", action="store_true", help="Generate a set of gray code patterns for projection")
    group.add_argument("-p", "--project", action="store_true", help="Project gray code patterns")
    group.add_argument("-c", "--capture", action="store_true", help="Capture the gray code projection with the camera")
    group.add_argument("-d", "--decode", action="store_true", help="Decode a series of gray code images into a lookup image that goes from camera to projector space")
    group.add_argument("-i", "--invert", action="store_true", help="Invert a lookup image (so that it goes from projector to camera space)")
    group.add_argument("-r", "--reproject", action="store_true", help="Reproject an image from camera space to projector space using a lookup image")
    group.add_argument("-a", "--all", action="store_true", help="Do all of these operations")

    # Global parameters
    group = parser.add_argument_group("Globals", "These global parameters affect all operations.")
    group.add_argument("-f", "--all-files", action="store_true", help="Store all intermediate results into files")
    group.add_argument("-w", "--working-directory", type=str, help="Save and load files in this directory", default="")
    group.add_argument("--camera-size", type=size, help="The camera resolution")
    group.add_argument("--projector-size", type=size, help="The projector resolution")
    group.add_argument("--unnormalized", dest="normalized", action="store_false", help="Don't normalize UV coordinates (use integer pixel coordinates)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Print out extra debugging information")
    group.add_argument("-q", "--quiet", dest="quiet", action="store_true", help="Only print out warnings and errors")

    # Gray code / project
    group = parser.add_argument_group("Gray code & project")
    group.add_argument("--gray-file", type=str, help="The file to save / load the gray code patterns to / from")

    # Project
    group = parser.add_argument_group("Project")
    group.add_argument("--screen", type=str, help="The name of the screen output connected to the projector")

    # Project / Capture
    group = parser.add_argument_group("Project & capture")
    group.add_argument("--startup-delay", type=float, help="How long to wait for camera settings to stabilize before displaying the first frame in seconds",
                       default=5)
    group.add_argument("--period", type=float, help="How long to display each frame for in seconds",
                       default=2)

    # Capture
    group = parser.add_argument_group("Capture")
    group.add_argument("--camera", type=str, help="The name of the camera device to open")

    # Capture / decode
    group = parser.add_argument_group("Capture & decode")
    group.add_argument("--capture-file", type=str, help="The file to save / load the captured images to / from")

    # Decode
    group = parser.add_argument_group("Decode")
    group.add_argument("--threshold-file", type=str, help="The file to save the captured images to after thresholding")

    # Decode / invert
    group = parser.add_argument_group("Decode & invert")
    group.add_argument("--decoded-file", type=str, help="The file to save / load the decoded image to / from")

    # Invert
    group = parser.add_argument_group("Invert")
    group.add_argument("--disparity-file", type=str, help="The file to save the disparity to")
    group.add_argument("--quantile", type=float, help="What fraction of the data to keep for least-squares fit",
                       default=.7)
    group.add_argument("--z-score", type=float, help="How many standard deviations away from the mean constitutes an outlier in the disparity map",
                       default=4)

    # Invert / reproject
    group = parser.add_argument_group("Invert & reproject")
    group.add_argument("--lookup-file", type=str, help="The file to save / load the lookup table image to / from")

    # Reproject
    group = parser.add_argument_group("Reproject")
    group.add_argument("--scene", type=str, help="The file to load an extra picture of the scene from")
    group.add_argument("--reprojected-file", type=str, help="The file to save the extra picture of the scene to, after reprojecting")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    # Internal state
    args.gray_code_images = None
    args.captured_images = None
    args.decoded_image = None
    args.lookup_image = None

    try:
        ops = ("gray",
            "project",
            "capture",
            "decode",
            "invert",
            "reproject")

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
        if args.reproject:
            op_reproject(args)
    except PromapError as e:
        logger.error("{}: {}".format(str(e.__class__.__name__), str(e)))

def check_imwrite(fn, *args):
    logger = logging.getLogger(__name__)
    logger.debug("Writing file {}".format(fn))
    if not cv2.imwrite(fn, *args):
        raise FileWriteError("Could not write {}".format(fn))

def imread(fn, *args):
    logger = logging.getLogger(__name__)
    logger.debug("Reading file {}".format(fn))
    return cv2.imread(fn, *args)

def check_imread(fn, *args):
    im = imread(fn, *args)
    if im is None:
        raise FileReadError("Could not read {}".format(fn))
    return im

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
    logger.info("Start operation: generate gray codes")

    if not args.projector_size:
        if args.project:
            project_get_projector_size(args)
            logger.info("Projector size not specified, querying screen gave {}x{}".format(*args.projector_size))
        else:
            raise ArgumentError("Unknown projector size")

    args.gray_code_images = promap.gray.generate_images(*args.projector_size)
    logger.debug("Generated {} gray code images".format(len(args.gray_code_images)))

    if not args.project or args.gray_file or args.all_files:
        # Save the image if we are not going to project it
        filename_format = filename2format(args.gray_file if args.gray_file else "gray.png")
        filenames = [filename_format.format(i) for i in range(len(args.gray_code_images))]
        for (fn, im) in zip(filenames, args.gray_code_images):
            check_imwrite(os.path.join(args.working_directory, fn), im)

def project_get_projector_size(args):
    import promap.project
    args.projector_size = promap.project.get_size(args.screen)

def op_project(args):
    import promap.project

    logger = logging.getLogger(__name__)
    logger.info("Start operation: project gray codes")

    if not args.gray_code_images:
        # Load the gray code from the given files
        filename_format = filename2format(args.gray_file if args.gray_file else "gray.png")
        images = []
        for i in itertools.count():
            fn = filename_format.format(i)
            im = imread(os.path.join(args.working_directory, fn), cv2.IMREAD_GRAYSCALE)
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
    import promap.capture

    logger = logging.getLogger(__name__)
    logger.info("Projecting and capturing gray codes")

    if not args.camera_size:
        args.camera_size = promap.capture.get_camera_size(args.camera)
    (capture, stop) = promap.capture.capture(args.camera, *args.camera_size)

    i = 0
    filename_format = None
    if not args.decode or args.capture_file or args.all_files:
        filename_format = filename2format(args.capture_file if args.capture_file else "cap.png")

    def _capture_callback():
        nonlocal i
        im = capture()
        if filename_format is not None:
            fn = filename_format.format(i)
            check_imwrite(os.path.join(args.working_directory, fn), im)
        i += 1

    promap.project.project(args.gray_code_images, args.startup_delay, args.period, args.screen, _capture_callback)
    args.captured_images = stop()

def op_capture(args):
    import promap.capture
    logger = logging.getLogger(__name__)
    logger.info("Start operation: capture gray codes")

    if not args.camera_size:
        args.camera_size = promap.capture.get_camera_size(args.camera)
        logger.info("Camera size not specified, querying camera gave {}x{}".format(*args.camera_size))

    (capture, stop) = promap.capture.capture(args.camera, *args.camera_size)
    i = 0
    filename_format = None
    if not args.decode or args.capture_file or args.all_files:
        filename_format = filename2format(args.capture_file if args.capture_file else "cap.png")
    try:
        print("Press Ctrl-C to finish capturing.")
        while True:
            input("Press [Enter] to capture image {:03d}...".format(i))
            im = capture()
            if filename_format is not None:
                fn = filename_format.format(i)
                check_imwrite(os.path.join(args.working_directory, fn), im)
            i += 1
    except KeyboardInterrupt:
        pass
    args.captured_images = stop()

def float_to_int(a, dtype=np.uint16):
    minval = np.iinfo(dtype).min
    maxval = np.iinfo(dtype).max
    return np.round(np.maximum(np.minimum(minval + (a * (maxval - minval)), maxval), minval)).astype(dtype)

def int_to_float(a):
    minval = np.iinfo(a.dtype).min
    maxval = np.iinfo(a.dtype).max
    return (a - minval) / (maxval - minval)

def op_decode(args):
    import promap.decode
    logger = logging.getLogger(__name__)
    logger.info("Start operation: decode gray codes")

    if not args.projector_size:
        raise ArgumentError("Unknown projector size")

    if not args.captured_images:
        logger.debug("No captured images in memory, reading from files")
        # Load the captured images from the given files
        filename_format = filename2format(args.capture_file if args.capture_file else "cap.png")
        images = []
        for i in itertools.count():
            fn = filename_format.format(i)
            im = imread(os.path.join(args.working_directory, fn))
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

    logger.debug("Thresholding images")
    (mask, thresh_images) = promap.decode.threshold_images(args.captured_images)
    if args.threshold_file or args.all_files:
        logger.debug("Writing thresholded images to files")
        filename = args.threshold_file if args.threshold_file else "thresh.png"
        filename_format = filename2format(filename)
        fn = filename2format(filename, places=None).format("mask")
        check_imwrite(os.path.join(args.working_directory, fn), mask)
        for (i, im) in enumerate(thresh_images):
            fn = filename_format.format(i)
            check_imwrite(os.path.join(args.working_directory, fn), im)

    logger.debug("Gray code decoding")
    (x, y) = promap.decode.decode_gray_images(args.projector_size[0], args.projector_size[1], thresh_images)
    args.decoded_image = np.dstack((x, y))
    if not args.invert or args.decoded_file or args.all_files:
        fn = args.decoded_file if args.decoded_file else "decoded.png"
        im = np.dstack((np.zeros(x.shape), y, x))
        if args.normalized:
            im[:,:,2] /= args.projector_size[0]
            im[:,:,1] /= args.projector_size[1]
            im = float_to_int(im)
        check_imwrite(os.path.join(args.working_directory, fn), im)

def op_invert(args):
    import promap.reproject
    logger = logging.getLogger(__name__)
    logger.info("Start operation: invert decoded map")

    if not args.projector_size:
        raise ArgumentError("Unknown projector size")

    if args.decoded_image is None:
        logger.debug("No decoded image in memory, reading from file")
        # Load the decoded image from the given file
        fn = args.decoded_file if args.decoded_file else "decoded.png"
        im = check_imread(os.path.join(args.working_directory, fn))
        if args.normalized:
            im = int_to_float(im)
            x = im[:,:,2]
            y = im[:,:,1]
            x = np.round(x * args.projector_size[0]).astype(np.int)
            y = np.round(y * args.projector_size[1]).astype(np.int)
        else:
            x = im[:,:,2]
            y = im[:,:,1]
    else:
        x = args.decoded_image[:,:,0]
        y = args.decoded_image[:,:,1]

    if not args.camera_size:
        args.camera_size = (x.shape[1], x.shape[0])
    else:
        if (x.shape[1], x.shape[0]) != args.camera_size:
            raise ArgumentError("Decoded image does not match camera size {}x{}".format(args.camera_size[0], args.camera_size[1]))

    ((camx, camy), disparity) = promap.reproject.compute_inverse_and_disparity(x, y, args.projector_size[0], args.projector_size[1],
        args.quantile, args.z_score)
    args.lookup_image = np.dstack((camx, camy))
    if not args.reproject or args.lookup_file or args.all_files:
        fn = args.lookup_file if args.lookup_file else "lookup.png"
        im = np.dstack((np.zeros(camx.shape), camy, camx))
        if args.normalized:
            im[:,:,2] /= args.camera_size[0]
            im[:,:,1] /= args.camera_size[1]
            im = float_to_int(im)
        check_imwrite(os.path.join(args.working_directory, fn), im)

    # Write out disparity
    fn = args.disparity_file if args.disparity_file else "disparity.png"
    max_disparity = np.amax(disparity)
    im = disparity / max_disparity
    im = float_to_int(im)
    check_imwrite(os.path.join(args.working_directory, fn), im)

def op_reproject(args):
    import promap.reproject
    logger = logging.getLogger(__name__)
    logger.info("Start operation: reproject images")

    scenes = []
    fns = []
    if args.scene:
        logger.debug("Only reprojecting given scene file, not captured images")
        fn = args.scene
        im = check_imread(os.path.join(args.working_directory, fn))
        new_fn = args.reprojected_file if args.reprojected_file else "reprojected.png"
        if not args.camera_size:
            args.camera_size = (im.shape[1], im.shape[0])
        else:
            if (im.shape[1], im.shape[0]) != args.camera_size:
                raise ArgumentError("Scene image {} does not match camera size {}x{}".format(args.scene, args.camera_size[0], args.camera_size[1]))
        scenes = [im]
        fns = [new_fn]
    else:
        logger.debug("No scene file specified, reprojecting the first two captured images")
        if not args.captured_images:
            logger.debug("No captured images in memory, reading from files")
            # Load the captured images from the given files
            filename_format = filename2format(args.capture_file if args.capture_file else "cap.png")
            for i in range(2):
                fn = filename_format.format(i)
                im = imread(os.path.join(args.working_directory, fn))
                if im is None:
                    break
                if not args.camera_size:
                    args.camera_size = (im.shape[1], im.shape[0])
                else:
                    if (im.shape[1], im.shape[0]) != args.camera_size:
                        raise ArgumentError("Scene image {} does not match camera size {}x{}".format(fn, args.camera_size[0], args.camera_size[1]))
                scenes.append(im)
        else:
            scenes = args.captured_images[0:2]
        fns = ["dark.png", "light.png"][0:len(scenes)]

    if len(scenes) == 0:
        raise ArgumentError("No scene to reproject")

    if args.lookup_image is None:
        logger.debug("No lookup image in memory, reading from file")
        # Load the lookup image from the given file
        fn = args.lookup_file if args.lookup_file else "lookup.png"
        im = check_imread(os.path.join(args.working_directory, fn))
        if args.normalized:
            im = int_to_float(im)
            x = im[:,:,2]
            y = im[:,:,1]
            x = np.round(x * args.camera_size[0]).astype(np.int)
            y = np.round(y * args.camera_size[1]).astype(np.int)
            args.lookup_image = np.dstack((x, y))
        else:
            args.lookup_image = np.dstack((im[:,:,2], im[:,:,1]))

    if not args.projector_size:
        args.projector_size = (x.shape[1], x.shape[0])
    else:
        if (args.lookup_image.shape[1], args.lookup_image.shape[0]) != args.projector_size:
            raise ArgumentError("Lookup image does not match projector size {}x{}".format(args.projector_size[0], args.projector_size[1]))

    for (fn, scene) in zip(fns, scenes):
        im = promap.reproject.reproject(args.lookup_image, scene)
        check_imwrite(os.path.join(args.working_directory, fn), im)
