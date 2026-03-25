import cv2
import numpy as np
from MyFunctions import cup_threshold, show_grayscale_comparison, contrast_enhance
# import training data #
# normal eye
img1 = cv2.imread("Training_set_normal/images/drishtiGS_008.png", 0);
img1_s = cv2.imread("Training_set_normal/softmaps/drishtiGS_008_cupsegSoftmap.png", 0);
img2 = cv2.imread("Training_set_normal/images/drishtiGS_017.png", 0);
img2_s = cv2.imread("Training_set_normal/softmaps/drishtiGS_017_cupsegSoftmap.png", 0);
img3 = cv2.imread("Training_set_normal/images/drishtiGS_035.png", 0);
img3_s = cv2.imread("Training_set_normal/softmaps/drishtiGS_035_cupsegSoftmap.png", 0);

# glaucoma eye
img4 = cv2.imread("Training_set_glaucoma/images/drishtiGS_002.png", 0);
img4_s = cv2.imread("Training_set_glaucoma/softmaps/drishtiGS_002_cupsegSoftmap.png", 0);
img5 = cv2.imread("Training_set_glaucoma/images/drishtiGS_015.png", 0);
img5_s = cv2.imread("Training_set_glaucoma/softmaps/drishtiGS_015_cupsegSoftmap.png", 0);
img6 = cv2.imread("Training_set_glaucoma/images/drishtiGS_032.png", 0);
img6_s = cv2.imread("Training_set_glaucoma/softmaps/drishtiGS_032_cupsegSoftmap.png", 0);

# group images
normal_images = [img1, img2, img3];
normal_softmaps = [img1_s, img2_s, img3_s];

glaucoma_images = [img4, img5, img6];
glaucoma_softmaps = [img4_s, img5_s, img6_s];

# TEST: apply threshold and compare with softmap #
# normal eye
for i in range(3):
    processed = cup_threshold(normal_images[i]);
    show_grayscale_comparison(processed, normal_softmaps[i], f"Normal Eye {i + 1} Comparison");

# glaucoma eye
for i in range(3):
    processed = cup_threshold(glaucoma_images[i]);
    show_grayscale_comparison(processed, glaucoma_softmaps[i], f"Glaucoma Eye {i + 1} Comparison");

"""CONCLUSION:
    V = {
        mean value + 40;    mean value < 20, power = 2
        mean value + 90;    mean value < 60, power = 3
        mean value + 60;    otherwise, power = 2

    }
"""