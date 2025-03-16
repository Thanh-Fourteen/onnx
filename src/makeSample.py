import cv2
import numpy as np

if __name__ == "__main__":
    for i in range(1,10):
        img = np.zeros((28, 28), dtype=np.uint8)
        cv2.putText(img, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255), 2)
        cv2.imwrite(f"sample/sample_digit{str(i)}.png", img)
