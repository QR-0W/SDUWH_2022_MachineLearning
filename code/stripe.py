import pystripe
import cv2


def destripe(img):
    # filter a single image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return pystripe.filter_streaks(img, sigma=[128, 256], level=1, wavelet='db2')
