import cv2
import matplotlib.pyplot as plt

def blur_bbox_regions(image_path, bboxes, rate=15):
    image = cv2.imread(image_path)

    # Make a copy of the image to avoid modifying the original
    output_image = image.copy()

    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox.int().tolist()

        if (y_max - y_min) // rate == 0 or (x_max - x_min) // rate == 0: # if the region is too small
            rate = 2
            
        # Extract the region of interest (ROI)
        roi = output_image[y_min:y_max, x_min:x_max]
        
        # Apply the blur to the ROI
        blurred_roi = cv2.resize(roi, ((x_max - x_min) // rate, (y_max - y_min) // rate))
        blurred_roi = cv2.resize(blurred_roi, (x_max - x_min, y_max - y_min), interpolation=cv2.INTER_AREA)
        
        # Place the blurred ROI back into the image
        output_image[y_min:y_max, x_min:x_max] = blurred_roi
            
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    # Blurred image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()
    return output_image
