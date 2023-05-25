import cv2
import numpy as np

def main():
    image = cv2.imread('01_h.jpg')

    # Rozmycie obrazu
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Wyostrzenie obrazu
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)

    # Normalizacja histogramu
    normalized = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    normalized[:, :, 0] = cv2.equalizeHist(normalized[:, :, 0])
    normalized = cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

    # Wyświetlenie obrazów
    cv2.imshow('Original', image)
    cv2.imshow('Blurred', blurred)
    cv2.imshow('Sharpened', sharpened)
    cv2.imshow('Normalized', normalized)


if __name__ == '__main__':
    main()


