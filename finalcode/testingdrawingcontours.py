import cv2
import numpy as np

# Example contour
contour = np.array([[[1919, 1040]],
                    [[1809,  369]],
                    [[1919,  165]],
                    [[ 159,  158]],
                    [[ 696,  324]],
                    [[   0,  319]],
[[   0,  544]],
[[ 631,  537]],
[[ 144,  804]],
[[ 664,  768]],
[[ 685, 1079]],
[[1573, 1079]],
[[1381,  599]],
[[1566,  675]],
[[1668, 1072]]], dtype=np.int32)

# Make a blank image (adjust size as needed)
img = np.zeros((1200, 2000, 3), dtype=np.uint8)

# Draw the contour
cv2.polylines(img, [contour], isClosed=True, color=(0, 255, 0), thickness=2)

# Show the image
cv2.imshow("Contour", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
