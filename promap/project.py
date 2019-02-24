import promap
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QGridLayout
from PyQt5.QtCore import QTimer
import logging

app = QApplication([])

class ProjectError(promap.PromapError):
    pass

def get_screen(screen_name=None):
    t = QTimer()
    screens = []
    def getScreensAndQuit():
        nonlocal screens
        screens = QApplication.screens()
        app.quit()
    t.timeout.connect(getScreensAndQuit)
    t.setSingleShot(True)
    t.start()
    app.exec()

    if not screens:
        raise ProjectError("Could not enumerate screens")

    if screen_name is None:
        return screens[-1]

    for s in screens:
        if s.name().lower().strip() == screen_name.lower().strip():
            return s

    raise ProjectError("Could not find {} (available screens are: {})".format(screen_name, ", ".join(s.name() for s in screens)))

def get_size(screen_name=None):
    s = get_screen(screen_name)
    return (s.size().width(), s.size().height())

def project(images, startup_delay=5, period=2, screen_name=None, capture_callback=None):
    logger = logging.getLogger(__name__)

    s = get_screen(screen_name)

    win = QWidget()
    win.show()
    win.windowHandle().setScreen(s)
    win.setGeometry(s.geometry())
    win.showFullScreen()

    layout = QGridLayout()
    layout.setSpacing(0)
    layout.setContentsMargins(0, 0, 0, 0)
    win.setLayout(layout)

    l = QLabel(win)
    layout.addWidget(l)
    l.show()

    def show_image(im):
        i = QImage(im.data, im.shape[1], im.shape[0],
                   QImage.Format_Grayscale8)
        pm = QPixmap()
        pm.convertFromImage(i)
        l.setPixmap(pm)

    t = QTimer()
    current_image = -1
    def advance():
        nonlocal current_image
        if current_image >= 0 and capture_callback is not None:
            capture_callback()
        current_image += 1
        if current_image >= len(images):
            app.quit()
            return
        show_image(images[current_image])
        logger.info("Showing image {}".format(current_image))
        t.start(int(period * 1000))
    t.timeout.connect(advance)
    t.setSingleShot(True)
    t.start(int(startup_delay * 1000))
    show_image(images[-1]) # this image is usually a good neutral gray
    logger.info("Showing last image so camera can stabilize")
    app.exec()
