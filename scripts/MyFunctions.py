import cv2
import numpy as np

# comparison function
def show_grayscale_comparison(img1, img2, window_name):
    # resize both images to 512x512
    img1_res = cv2.resize(img1, (512, 512));
    img2_res = cv2.resize(img2, (512, 512));

    # combine images
    combined = np.hstack((img1_res, img2_res));
    cv2.imshow(window_name, combined);
    cv2.waitKey();

# contrast enhancement function
def contrast_enhance(image, power):
    e = image.copy();  # copy image
    e = e.astype(np.float32);  # convert type to apply law
    e = np.power(e, power);  # apply power law
    e = (e / np.max(e)) * 255;  # scale back to range: 0-255
    e = e.astype(np.uint8);  # convert type back to int
    return e;

# define OD thresholding function
def OD_threshold(image):
    # set threshold
    mean_val = np.mean(image);
    if mean_val < 20:
        power = 0.9;
        offset = 35;
    elif mean_val < 60:
        power = 1.2;
        offset = 90;
    else:
        power = 1.5;
        offset = 45;

    # enhance image
    image = contrast_enhance(image, power);

    # recalculate mean of the enhanced image
    new_mean = np.mean(image);
    threshold_val = new_mean + offset;
    if threshold_val > 255:
        threshold_val = 255

    # convert to binary
    _, img = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY);
    return img;

# define Cup thresholding function
def cup_threshold(image):
    # set threshold
    mean_val = np.mean(image);
    if mean_val < 20:
        power = 2;
        offset = 40;
    elif mean_val < 60:
        power = 3;
        offset = 90;
    else:
        power = 2;
        offset = 60;

    # enhance image
    image = contrast_enhance(image, power);

    # recalculate mean of the enhanced image
    new_mean = np.mean(image);
    threshold_val = new_mean + offset;
    if threshold_val > 255:
        threshold_val = 255

    # convert to binary
    _, img = cv2.threshold(image, threshold_val, 255, cv2.THRESH_BINARY);
    return img;

# define noise reduction function
def noise_reduce(labels):
    # get all unique labels and count their number of pixels
    unique_labels, counts = np.unique(labels, return_counts=True);

    # iterate through every label
    for label, count in zip(unique_labels, counts):
        # skip background
        if label == 0:
            continue;

        # if the object is smaller than 25 pixels, remove it/set to 0
        if count < 25:
            labels[labels == label] = 0;
    return labels;

# save pixels of objects
def save_pixels_to_file(labels, filename):
    with open(filename, "w") as f:
        # get coordinates of all pixels that are part of an object
        coords = np.argwhere(labels > 0);   # labels = 0 is background
        for r, c in coords:
            f.write(f"{r} {c}\n");
    print(f"Saved pixels to {filename}");

# define function to show object as is, but with white background
def show_on_white_background(original, mask, window_name):
    # create background
    original_res = cv2.resize(original, (512, 512));
    white_bg = np.full(original_res.shape, 255, dtype=np.uint8);

    # use INTER_NEAREST for masks to keep edges sharp and binary
    mask_res = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST);

    # where mask > 0 use original, otherwise use white background
    result = np.where(mask_res > 0, original_res, white_bg);

    # show output
    cv2.imshow(window_name, result);
    cv2.waitKey();

# define dice doefficient dalculation function
def compute_dice_coefficient(img_pred, img_gt):
    # resize ground truth to match prediction
    h, w = img_pred.shape;
    img_gt_resized = cv2.resize(img_gt, (w, h), interpolation = cv2.INTER_NEAREST);

    # binarize arrrays (0 or 1)
    pred = (img_pred > 0).astype(np.uint8);
    gt = (img_gt_resized > 0).astype(np.uint8);

    # count true and false pixels
    true_pixels = np.sum(pred & gt);    # intersection (overlap)
    false_pixels = np.sum(pred != gt);  # difference (non-overlap)

    # get total true pixels in original mask
    total_original = np.sum(gt);

    # prevent division by zero
    if total_original == 0:
        return 0.0;

    # normalize by dividing with total number of true pixels in original mask
    dice = true_pixels / total_original;

    return dice;

# define CCL-8 connectivity function
def two_pass_labelling(binary_img):
    # ensure background is 0 and OD is 1 for the logic
    binary = (binary_img > 0).astype(np.int32);

    h, w = binary.shape;
    labels = np.zeros((h, w), dtype = np.int32);
    equivalence = list(range(10000));  # table to track linked labels
    k = 0;

    # --- PASS 1: Assign Temporary Labels ---
    for i in range(h):
        for j in range(w):
            if binary[i, j] == 1:
                # get neighbors (Top and Left)
                # we check bounds so we don't crash at the edges
                neighbors = [];
                if i > 0 and labels[i - 1, j] > 0:
                    neighbors.append(labels[i - 1, j]);
                if j > 0 and labels[i, j - 1] > 0:
                    neighbors.append(labels[i, j - 1]);
                if i > 0 and j > 0 and labels[i - 1, j - 1] > 0:
                    neighbors.append(labels[i - 1, j - 1]);
                if i > 0 and j < w - 1 and labels[i - 1, j + 1] > 0:
                    neighbors.append(labels[i - 1, j + 1]);

                if not neighbors:
                    # case 1: No labeled neighbors (Step 6-8 in your image)
                    k += 1;
                    labels[i, j] = k;
                else:
                    # case 2: Neighbors exist (Step 10-14 in your image)
                    min_label = min(neighbors);
                    labels[i, j] = min_label;

                    # update equivalence table for all neighbors found
                    for label in neighbors:
                        # find the "root" of this label and link it to the min_label
                        root = label;
                        while equivalence[root] != root:
                            root = equivalence[root];
                        equivalence[root] = min(equivalence[root], min_label);

    # --- RESOLVE EQUIVALENCE: Flatten the table ---
    # this ensures every label points directly to its smallest "ancestor"
    for i in range(1, k + 1):
        root = i;
        while equivalence[root] != root:
            root = equivalence[root];
        equivalence[i] = root;

    # --- PASS 2: Replace labels with their roots (Step 20 in your image) ---
    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                labels[i, j] = equivalence[labels[i, j]];

    return labels;


