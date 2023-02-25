import csv
import os
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from constants import GRAY_SCALE_BLACK, GRAY_SCALE_WHITE
from constants import LSTM_CSV_DIR_PATH, INFO_FILE_PATH, IMAGE_DIR_PATH
from constants import EXTENSION_CSV, EXTENSION_PNG


def calc_body_center(max_contour: np.ndarray) -> np.ndarray:
    dic = {}
    for vector_list in max_contour:
        vector = vector_list[0]
        if vector[0] not in dic:
            dic[vector[0]] = [vector[1]]
        else:
            dic[vector[0]].append(vector[1])

    max_wise = 0
    body_center_vector = np.zeros(2, dtype=np.int64)
    for x_coordinate, values in dic.items():
        y_top = min(values)
        y_bottom = max(values)
        wise = y_bottom - y_top
        if wise > max_wise:
            max_wise = wise
            body_center_vector[0] = x_coordinate
            body_center_vector[1] = int((y_top + y_bottom) / 2)

    return body_center_vector


def main():
    os.makedirs(LSTM_CSV_DIR_PATH, exist_ok=True)
    columns = ["head_x", "head_y", "movement_distance_head", "tail_x", "tail_y", "movement_distance_tail",
               "center_x", "center_y", "movement_distance_center", "img_name"]

    df = pd.read_csv(INFO_FILE_PATH).astype(str)
    for state in df['state'].unique():
        os.makedirs(os.path.join(LSTM_CSV_DIR_PATH, state), exist_ok=True)

    for _, row in df.iterrows():
        state = row['state']
        created_datetime = row['created_datetime']
        analysis_csv = os.path.join(LSTM_CSV_DIR_PATH, state, f'{created_datetime}{EXTENSION_CSV}')

        try:
            with open(analysis_csv, mode='x') as f:
                writer = csv.writer(f)
                writer.writerow(columns)

                read_dir = os.path.join(IMAGE_DIR_PATH, created_datetime)
                img_paths = list(sorted(Path(read_dir).glob(f"*{EXTENSION_PNG}")))
                for idx, path in enumerate(img_paths):
                    bgr = cv2.imread(str(path))
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                    saturation = cv2.split(hsv)[1]
                    th_saturation = cv2.threshold(saturation, GRAY_SCALE_BLACK, GRAY_SCALE_WHITE, cv2.THRESH_OTSU)[1]
                    reversal_th_saturation = cv2.bitwise_not(th_saturation)
                    contours = cv2.findContours(reversal_th_saturation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                    max_contour = max(contours, key=lambda cnt: cv2.contourArea(cnt))

                    head = min(enumerate(max_contour), key=lambda xy: xy[1][0][0])[1][0]
                    tail = max(enumerate(max_contour), key=lambda xy: xy[1][0][0])[1][0]
                    center = calc_body_center(max_contour)

                    if idx == 0:
                        movement_distance_head = 0
                        movement_distance_tail = 0
                        movement_distance_center = 0
                    else:
                        movement_distance_head = np.linalg.norm(previous_head - head)
                        movement_distance_tail = np.linalg.norm(previous_tail - tail)
                        movement_distance_center = np.linalg.norm(previous_center - center)

                    previous_head = head
                    previous_tail = tail
                    previous_center = center

                    row = [head[0], head[1], movement_distance_head, tail[0], tail[1], movement_distance_tail,
                           center[0], center[1], movement_distance_center, path.name]
                    writer.writerow(row)
        except FileExistsError:
            pass


if __name__ == '__main__':
    main()
    quit(0)

