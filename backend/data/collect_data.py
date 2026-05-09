import cv2
import numpy as np
import os

# creates folders data/0, data/1, ... data/9
for i in range(10):
    os.makedirs(f'backend/data/my_handwriting/{i}', exist_ok=True)

def collect_digit(digit):
    """Draw a digit, press S to save, R to retry, Q to quit"""
    canvas = np.zeros((280, 280), dtype=np.uint8)
    drawing = False
    
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
    count = len(os.listdir(f'backend/data/my_handwriting/{digit}'))

    while True:
        cv2.imshow(f'Draw digit: {digit}', canvas)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # save
            img = cv2.resize(canvas, (28, 28))
            path = f'backend/data/my_handwriting/{digit}/{count}.png'
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

print("\nDone collecting! Check backend/data/my_handwriting/")