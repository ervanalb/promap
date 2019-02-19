import argparse
import logging

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
    parser.add_argument("-y", "--disparity", action="store_true", help="Compute disparity of a lookup image")
    parser.add_argument("-a", "--all", action="store_true", help="Do all of these operations")

    # Global parameters
    parser.add_argument("--camera-size", type=size, help="The camera resolution")
    parser.add_argument("--projector-size", type=size, help="The projector resolution")

    # Gray code / project
    parser.add_argument("--gray-file", type=str, help="The file to save / load the gray code patterns from")

    # Project
    parser.add_argument("--screen", type=str, help="The name of the screen output connected to the projector")

    args = parser.parse_args()

    ops = ("gray",
        "project",
        "capture",
        "decode",
        "invert",
        "lookup",
        "disparity")

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

    for op in ops:
        if getattr(args, op):
            globals()["op_" + op](args)

def filename2format(fn, places=3):
    """Converts a filename to a format string capable of adding an index after the basename"""
    index = "{:0" + str(places) + "d}"
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

    if not args.projector_size:
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

    if not args.projector_size:
        if args.project:
            project_get_projector_size(args)

    if not args.projector_size:
        raise ArgumentError("Unknown projector size")

    promap.project.project(args.gray_code_images)

def op_capture(args):
    logger = logging.getLogger(__name__)
    logger.warning("capture not implemented")

def op_decode(args):
    logger = logging.getLogger(__name__)
    logger.warning("decode not implemented")

def op_invert(args):
    logger = logging.getLogger(__name__)
    logger.warning("invert not implemented")

def op_lookup(args):
    logger = logging.getLogger(__name__)
    logger.warning("lookup not implemented")

def op_disparity(args):
    logger = logging.getLogger(__name__)
    logger.warning("disparity not implemented")
