import numpy as np

def generate_code(n):
    """Generates a gray code up to the given size"""
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

def generate_images(w, h):
    """Generate a set of gray code images for the given width and height."""

    vertical_bars = generate_code(w)
    n = len(vertical_bars)
    vertical_bars = np.repeat(vertical_bars[:,None,:], h, axis=1) * 255

    horizontal_bars = generate_code(h)
    n = len(horizontal_bars)
    horizontal_bars = np.repeat(horizontal_bars[:,:,None], w, axis=2) * 255

    bw = black_white(w, h)

    return list(bw) + list(vertical_bars) + list(horizontal_bars)

def black_white(w, h):
    black = np.full((h, w), 0, dtype="uint8")
    white = np.full((h, w), 255, dtype="uint8")
    return (black, white)
