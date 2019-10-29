from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255

# Параметры детектора Харриса
# Размер окна детектора Харриса
# https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#gac1fc3598018010880e370e2f709b4345
blockSize = 2
# Размер ядра оператора Собеля (вычисляет градиенты пикселей)
# https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
apertureSize = 3
# Эмпирический коэффициент, советуют выбирать 0.04-0.06
k = 0.04


def corner_harris(val):
    # В результате получим на каждый пиксель число от 0 до 255,
    # чем оно больше, тем более вероятно, что здесь угол. Отмечаем
    # те точки, в которых значение больше thresh
    thresh = val

    # Вызываем сам детектор Харриса, ищем углы
    dst = cv.cornerHarris(src_gray, blockSize, apertureSize, k)

    # В результате получим матрицу чисел. "Растянем" их на отрезке
    # от 0 до 255 (нормализуем), чтобы проще было с ними работать
    # и чтобы их можно было показать как простую картинку
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    dst_norm_scaled = cv.convertScaleAbs(dst_norm)

    # Рисуем кружок вокруг тех точек, где порог превысил значение,
    # указанное пользователем
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > thresh:
                cv.circle(dst_norm_scaled, (j, i), 5, 0, 2)

    # Рисуем картинку в окне результата
    cv.namedWindow(corners_window, cv.WINDOW_NORMAL)
    cv.imshow(corners_window, dst_norm_scaled)


# Загружаем картинку, указанную в параметре --input
parser = argparse.ArgumentParser(description='Code for Harris corner detector tutorial.')
parser.add_argument('--input', help='Path to input image.')
parser.add_argument('--sharpen', help='Sharpen the image before processing', default=False, action='store_true')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Конвертируем картинку в градации серого
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

# Если задана опция sharpen, немного увеличиваем резкость
# с помощью простого фильтра (так лучше работает на картинках UW1-UW5)
if args.sharpen:
    sharp_kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ])
    src_gray = cv.filter2D(src_gray, -1, sharp_kernel)

# Создаем окно и полоску выбора порога
cv.namedWindow(source_window, cv.WINDOW_NORMAL)
thresh = 200
cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, corner_harris)

# Запускаем детектор Харриса
cv.imshow(source_window, src)
corner_harris(thresh)
cv.waitKey()
