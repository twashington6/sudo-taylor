# collect handwritten digit samples
# the output feeds directly into train_on_data.py.

# imports
import cv2
# draws the canvas window, handles mouse events

import numpy as np
# creates the blank canvas (array of zeros = black)

import os
# creates the folder structure for each digit


# WHY 28x28: MNIST is 28x28. Training data must match
# the format the model was trained on. This is non-negotiable --
# mismatched input shapes will crash the model or produce garbage.

# MNIST digits are clean, centered, machine-scanned.
# The sudoku app will receive messy, real handwriting from a canvas.
# The gap between "MNIST handwriting" and "ACTUAL handwriting"
# is exactly what fine-tuning bridges.
# The more samples collected, the better the model can learn handwriting style.


# creates folders data/0, data/1, ... data/9
for i in range(10):
    os.makedirs(f'backend/data/handwriting/{i}', exist_ok=True)

def collect_digit(digit):
    """Draw a digit, press S to save, R to retry, Q to quit"""
    canvas = np.zeros((280, 280), dtype=np.uint8)
    drawing = False
    
    # mouse callback to draw on the canvas
    def draw(event, x, y, flags, param):
        nonlocal drawing
        if event == cv2.EVENT_LBUTTONDOWN: 
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.circle(canvas, (x, y), 12, 255, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    cv2.namedWindow(f'Draw digit: {digit}')
    cv2.setMouseCallback(f'Draw digit: {digit}', draw)
    
    # find next available filename
    existing = os.listdir(f'backend/data/handwriting/{digit}')
    count = max([int(f.split('.')[0]) for f in existing if f.endswith('.png')], default=0) + 1

    while True:
        # display the canvas
        cv2.imshow(f'Draw digit: {digit}', canvas)
        key = cv2.waitKey(1) & 0xFF

        # S=save, R=retry, Q=next digit
        if key == ord('s'):  # save
            img = cv2.resize(canvas, (28, 28))
            path = f'backend/data/handwriting/{digit}/{count}.png'
            cv2.imwrite(path, img)
            count += 1
            canvas[:] = 0  # clear for next drawing
            print(f"Saved sample {count} for digit {digit}")

        elif key == ord('r'):  # retry / clear
            canvas[:] = 0

        elif key == ord('q'):  # move to next digit
            break

    cv2.destroyAllWindows()

# collect samples for each digit
for digit in range(10):
    print(f"\n--- Draw digit {digit} --- (S=save, R=retry, Q=next digit)")
    print("Aim for at least 30 samples per digit")
    collect_digit(digit)

print("\nDone collecting! Check backend/data/handwriting/")