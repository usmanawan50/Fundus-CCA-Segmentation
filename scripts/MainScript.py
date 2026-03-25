import numpy as np
import os
import cv2
from MyFunctions import (show_grayscale_comparison, contrast_enhance,
                         OD_threshold, cup_threshold, noise_reduce,
                         save_pixels_to_file, show_on_white_background,
                         compute_dice_coefficient)

def two_pass_labelling(binary_img):
    """
    Robust Two-Pass Connected Components Labeling
    """
    rows, cols = binary_img.shape
    labels = np.zeros((rows, cols), dtype=int)
    label_counter = 1
    equivalence = {}  # Using a dict prevents IndexError

    def find(label):
        # Path compression for efficiency
        path = []
        while label != equivalence.get(label, label):
            path.append(label)
            label = equivalence.get(label, label)
        for p in path:
            equivalence[p] = label
        return label

    def union(l1, l2):
        root1 = find(l1)
        root2 = find(l2)
        if root1 != root2:
            # Union by ID (smaller ID becomes parent)
            if root1 < root2:
                equivalence[root2] = root1
            else:
                equivalence[root1] = root2

    # --- First Pass ---
    for r in range(rows):
        for c in range(cols):
            if binary_img[r, c] > 0:  # Foreground
                neighbors = []
                # Check North
                if r > 0 and labels[r - 1, c] > 0:
                    neighbors.append(labels[r - 1, c])
                # Check West
                if c > 0 and labels[r, c - 1] > 0:
                    neighbors.append(labels[r, c - 1])

                if not neighbors:
                    labels[r, c] = label_counter
                    equivalence[label_counter] = label_counter
                    label_counter += 1
                else:
                    min_label = min(neighbors)
                    labels[r, c] = min_label
                    for neighbor in neighbors:
                        if neighbor != min_label:
                            union(min_label, neighbor)

    # --- Second Pass ---
    # Flatten labels to canonical roots
    for r in range(rows):
        for c in range(cols):
            if labels[r, c] > 0:
                labels[r, c] = find(labels[r, c])

    return labels;

# --- 1. SETUP: Define Image List ---
# List of all images to process
image_files = [
    "Test_set/all/drishtiGS_001.png", "Test_set/all/drishtiGS_003.png",
    "Test_set/all/drishtiGS_005.png", "Test_set/all/drishtiGS_006.png",
    "Test_set/all/drishtiGS_007.png", "Test_set/all/drishtiGS_009.png",
    "Test_set/all/drishtiGS_011.png", "Test_set/all/drishtiGS_013.png",
    "Test_set/all/drishtiGS_014.png", "Test_set/all/drishtiGS_020.png",
    "Test_set/all/drishtiGS_021.png", "Test_set/all/drishtiGS_023.png",
    "Test_set/all/drishtiGS_025.png", "Test_set/all/drishtiGS_027.png",
    "Test_set/all/drishtiGS_028.png", "Test_set/all/drishtiGS_029.png",
    "Test_set/all/drishtiGS_030.png", "Test_set/all/drishtiGS_034.png",
    "Test_set/all/drishtiGS_039.png", "Test_set/all/drishtiGS_043.png",
    "Test_set/all/drishtiGS_048.png", "Test_set/all/drishtiGS_050.png",
    "Test_set/all/drishtiGS_052.png", "Test_set/all/drishtiGS_053.png",
    "Test_set/all/drishtiGS_054.png", "Test_set/all/drishtiGS_055.png",
    "Test_set/all/drishtiGS_056.png", "Test_set/all/drishtiGS_059.png",
    "Test_set/all/drishtiGS_065.png", "Test_set/all/drishtiGS_067.png",
    "Test_set/all/drishtiGS_070.png", "Test_set/all/drishtiGS_071.png",
    "Test_set/all/drishtiGS_072.png", "Test_set/all/drishtiGS_073.png",
    "Test_set/all/drishtiGS_074.png", "Test_set/all/drishtiGS_077.png",
    "Test_set/all/drishtiGS_078.png", "Test_set/all/drishtiGS_079.png",
    "Test_set/all/drishtiGS_082.png", "Test_set/all/drishtiGS_083.png",
    "Test_set/all/drishtiGS_085.png", "Test_set/all/drishtiGS_086.png",
    "Test_set/all/drishtiGS_087.png"
]

# Create output directory if it doesn't exist
if not os.path.exists('Processed_Images'):
    os.makedirs('Processed_Images')

# Open results file
with open("dice_results.txt", "w") as result_file:
    result_file.write("Filename, OD_Dice, Cup_Dice\n")

    # --- 2. MAIN PROCESSING LOOP ---
    for file_path in image_files:
        filename = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing: {filename}")

        # Load Image
        img1 = cv2.imread(file_path, 0)
        if img1 is None:
            print(f"  Error: Could not load {file_path}")
            continue

        # --- OD Processing ---
        # 1. Threshold
        img_OD_thresh = OD_threshold(img1)

        # 2. Save Threshold Image (Fix: Scale to 255 for visibility)
        cv2.imwrite(f"Processed_Images/{filename}_OD.png", (img_OD_thresh * 255).astype(np.uint8))

        # 3. Labelling & Noise Reduction
        img_OD_labels = two_pass_labelling(img_OD_thresh)
        img_OD_clean = noise_reduce(img_OD_labels)
        save_pixels_to_file(img_OD_clean, f"Processed_Images/{filename}_OD_Pixels.txt")

        # --- Cup Processing ---
        # 1. Threshold
        img_Cup_thresh = cup_threshold(img1)

        # 2. Save Threshold Image (Fix: Scale to 255 for visibility)
        cv2.imwrite(f"Processed_Images/{filename}_Cup.png", (img_Cup_thresh * 255).astype(np.uint8))

        # 3. Labelling & Noise Reduction
        img_Cup_labels = two_pass_labelling(img_Cup_thresh)
        img_Cup_clean = noise_reduce(img_Cup_labels)
        save_pixels_to_file(img_Cup_clean, f"Processed_Images/{filename}_Cup_Pixels.txt")

        # --- Calculate Dice Coefficients ---
        dice_OD = compute_dice_coefficient(img_OD_clean, img_OD_labels)
        dice_Cup = compute_dice_coefficient(img_Cup_clean, img_Cup_labels)

        print(f"  > OD Dice: {dice_OD:.4f} | Cup Dice: {dice_Cup:.4f}")
        result_file.write(f"{filename}, {dice_OD:.4f}, {dice_Cup:.4f}\n")

print("\nProcessing Complete. Results saved to 'dice_results.txt'.")
