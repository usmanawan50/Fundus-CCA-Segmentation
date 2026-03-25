import cv2
import numpy as np
from MyFunctions import (show_grayscale_comparison, contrast_enhance,
                         OD_threshold, cup_threshold, noise_reduce,
                         save_pixels_to_file, show_on_white_background,
                         compute_dice_coefficient, two_pass_labelling)

# import testing data #
normal = cv2.imread("Test_set/drishtiGS_085.png", 0);
normal_OD_s = cv2.imread("Test_set/drishtiGS_085_ODsegSoftmap.png", 0);
normal_cup_s = cv2.imread("Test_set/drishtiGS_085_cupsegSoftmap.png", 0);

glaucoma = cv2.imread("Test_set/drishtiGS_087.png", 0);
glaucoma_OD_s = cv2.imread("Test_set/drishtiGS_087_ODsegSoftmap.png", 0);
glaucoma_cup_s = cv2.imread("Test_set/drishtiGS_087_cupsegSoftmap.png", 0);

# OD separation #
# normal eye OD
normal_OD_bin = OD_threshold(normal);
show_grayscale_comparison(normal_OD_bin, normal_OD_s, "Normal OD vs Softmap");

# glaucoma eye OD
glaucoma_OD_bin = OD_threshold(glaucoma);
show_grayscale_comparison(glaucoma_OD_bin, glaucoma_OD_s, "Glaucoma OD vs Softmap");

# Cup separation #
# normal eye Cup
normal_cup_bin = cup_threshold(normal);
show_grayscale_comparison(normal_cup_bin, normal_cup_s, "Normal Cup vs Softmap");

# glaucoma eye Cup
glaucoma_cup_bin = cup_threshold(glaucoma);
show_grayscale_comparison(glaucoma_cup_bin, glaucoma_cup_s, "Glaucoma Cup vs Softmap");

# CCL and noise reduction (OD) #
# normal eye OD
normal_OD_labels = two_pass_labelling(normal_OD_bin);
normal_OD_clean = noise_reduce(normal_OD_labels);
save_pixels_to_file(normal_OD_clean, "Normal_eye_OD_Pixels.txt");

# glaucoma eye OD
glaucoma_OD_labels = two_pass_labelling(glaucoma_OD_bin);
glaucoma_OD_clean = noise_reduce(glaucoma_OD_labels);
save_pixels_to_file(glaucoma_OD_clean, "Glaucoma_eye_OD_Pixels.txt");

# CCL and noise reduction (Cup) #
# normal eye Cup
normal_cup_labels = two_pass_labelling(normal_cup_bin);
normal_cup_clean = noise_reduce(normal_cup_labels);
save_pixels_to_file(normal_cup_clean, "Normal_eye_Cup_Pixels.txt");

# glaucoma eye Cup
glaucoma_cup_labels = two_pass_labelling(glaucoma_cup_bin);
glaucoma_cup_clean = noise_reduce(glaucoma_cup_labels);
save_pixels_to_file(glaucoma_cup_clean, "Glaucoma_eye_Cup_Pixels.txt");

# Show results #
show_on_white_background(normal, normal_OD_clean, "Normal Eye OD");
show_on_white_background(glaucoma, glaucoma_OD_clean, "Glaucoma Eye OD");
show_on_white_background(normal, normal_cup_clean, "Normal Eye Cup");
show_on_white_background(glaucoma, glaucoma_cup_clean, "Glaucoma Eye Cup");
cv2.destroyAllWindows();

# compute dice for normal eye #
print("\n\n- Normal Eye Results -");
# OD
dice_OD_normal = compute_dice_coefficient(normal_OD_clean, normal_OD_s);
print("OD Dice:", dice_OD_normal);

# Cup
dice_Cup_normal = compute_dice_coefficient(normal_cup_clean, normal_cup_s);
print("Cup Dice:", dice_Cup_normal);

# background (Inverse of OD)
bg_pred_normal = 1 - (normal_OD_clean > 0).astype(np.uint8);
bg_gt_normal = 255 - normal_OD_s;
dice_BG_normal = compute_dice_coefficient(bg_pred_normal, bg_gt_normal);
print("Background Dice:", dice_BG_normal);

# compute dice for glaucoma eye #
print("\n\n- Glaucoma Eye Results -");
# OD
dice_OD_glaucoma = compute_dice_coefficient(glaucoma_OD_clean, glaucoma_OD_s);
print("OD Dice:", dice_OD_glaucoma);

# Cup
dice_Cup_glaucoma = compute_dice_coefficient(glaucoma_cup_clean, glaucoma_cup_s);
print("Cup Dice:", dice_Cup_glaucoma);

# background (Inverse of OD)
bg_pred_glaucoma = 1 - (glaucoma_OD_clean > 0).astype(np.uint8);
bg_gt_glaucoma = 255 - glaucoma_OD_s;
dice_BG_glaucoma = compute_dice_coefficient(bg_pred_glaucoma, bg_gt_glaucoma);
print("Background Dice:", dice_BG_glaucoma);