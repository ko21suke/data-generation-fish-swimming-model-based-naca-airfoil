import math
import csv
import os
import cv2
import numpy as np

from datetime import datetime
from skimage.draw import line
from pathlib import Path
from constants import BLACK, WHITE, GREEN
from constants import AMBERJACK, IMAGE_DIR_PATH, INFO_FILE_PATH, PARAMS_DIR_PATH
from constants import EXTENSION_CSV, EXTENSION_PNG

# Coefficients
COEFFICIENT_VIBRATIONAL_DISTRIBUTION_1 = 0.648
COEFFICIENT_VIBRATIONAL_DISTRIBUTION_2 = 0.0195
COEFFICIENT_VIBRATIONAL_DISTRIBUTION_HEAD = 0.25
COEFFICIENT_MOVEMENT_DISTANCE = 0.7380
WAVE_LENGTH = 2.5

# Path
OUTPUT_DIR_NAME = IMAGE_DIR_PATH
BASE_IMG_NAME = AMBERJACK

# Boolean for having noise.
HAS_NOISE = [False, True]


def calc_base_body_thickness(x_coordinate: int, base_body_length: float, thickness_ratio: float) -> float:
    """
    Caution: 2.2 exponent is not original in NACA formular. Original value is 2.0.
    Reference -> http://airfoiltools.com/airfoil/naca4digit
    """
    return 10 * thickness_ratio * base_body_length * (0.29690 * np.sqrt(x_coordinate / base_body_length)
                                                      - 0.12600 * (x_coordinate / base_body_length)
                                                      - 0.35160 * ((x_coordinate / base_body_length) ** 2.2)
                                                      + 0.28430 * ((x_coordinate / base_body_length) ** 3)
                                                      - 0.10360 * ((x_coordinate / base_body_length) ** 4))


def calc_vibrational_distribution_tail_side(x_coordinate: int, body_length: float) -> float:
    half_body_length = body_length / 2
    distance_from_center = x_coordinate - half_body_length
    return (COEFFICIENT_VIBRATIONAL_DISTRIBUTION_1 * (distance_from_center ** 2)
            - COEFFICIENT_VIBRATIONAL_DISTRIBUTION_2 * distance_from_center)


def calc_vibrational_distribution_head_side(x_coordinate: int, body_length: float) -> float:
    half_body_length = body_length / 2
    distance_from_center = half_body_length - x_coordinate
    return COEFFICIENT_VIBRATIONAL_DISTRIBUTION_HEAD * (
            COEFFICIENT_VIBRATIONAL_DISTRIBUTION_1 * (distance_from_center ** 2)
            - COEFFICIENT_VIBRATIONAL_DISTRIBUTION_2 * distance_from_center)


def calc_motion_state_tail_side(x_coordinate: int, periodic_time: float, period: float, body_length: float) -> float:
    """
    Reference -> https://www.jstage.jst.go.jp/article/kikaib1979/76/764/76_KJ00006254100/_pdf/-char/ja
    """
    angular_frequency = 2 * math.pi / period
    angular_wave_number = 2 * math.pi / WAVE_LENGTH
    half_body_length = body_length / 2
    return math.sin(angular_frequency * periodic_time - angular_wave_number * (x_coordinate - half_body_length))


def calc_motion_state_head_side(x_coordinate: int, periodic_time: float, period: float, body_length: float) -> float:
    """
    Reference -> https://www.jstage.jst.go.jp/article/kikaib1979/76/764/76_KJ00006254100/_pdf/-char/ja
    """
    angular_frequency = 2 * math.pi / period
    angular_wave_number = 2 * math.pi / WAVE_LENGTH
    half_body_length = body_length / 2
    return math.sin(angular_frequency * periodic_time - angular_wave_number * (half_body_length - x_coordinate))


def calc_variation_tail_side(x_coordinate: int, periodic_time: float, deformation_period: float, body_length: float,
                             tail_magnification_right: float, tail_magnification_left: float, ) -> float:
    vibrational_distribution = calc_vibrational_distribution_tail_side(x_coordinate, body_length)
    motion_state = calc_motion_state_tail_side(x_coordinate, periodic_time, deformation_period, body_length)
    if is_motion_direction_right(motion_state):
        return tail_magnification_right * vibrational_distribution * motion_state
    elif is_motion_direction_left(motion_state):
        return tail_magnification_left * vibrational_distribution * motion_state
    else:
        # return 0
        return vibrational_distribution * motion_state


def calc_variation_head_side(x_coordinate: int, periodic_time: float, deformation_period: float, body_length: float,
                             head_magnification_right: float, head_magnification_left: float, ) -> float:
    vibrational_distribution = calc_vibrational_distribution_head_side(x_coordinate, body_length)
    motion_state = calc_motion_state_head_side(x_coordinate, periodic_time, deformation_period, body_length)
    if is_motion_direction_right(motion_state):
        return head_magnification_right * vibrational_distribution * motion_state
    elif is_motion_direction_left(motion_state):
        return head_magnification_left * vibrational_distribution * motion_state
    else:
        # return 0
        return vibrational_distribution * motion_state


def is_motion_direction_left(motion_state) -> bool:
    return motion_state > 0


def is_motion_direction_right(motion_state) -> bool:
    return motion_state < 0


def is_vector_over_img(outline: np.ndarray, width: int, height: int) -> bool:
    is_x_over_width = np.any(outline > width, axis=0)[0]
    is_y_over_height = np.any(outline > height, axis=0)[1]
    is_xy_under_zero = np.any(outline < 0)
    return is_x_over_width or is_y_over_height or is_xy_under_zero


def get_swimming_speed_coefficient(head_magnification_right: float, head_magnification_left: float,
                                   tail_magnification_right: float, tail_magnification_left: float) -> float:
    sum_magnification = head_magnification_right + head_magnification_left \
                        + tail_magnification_right + tail_magnification_left

    if sum_magnification >= 1.75:
        return 0.9
    elif sum_magnification >= 1.50:
        return 0.8
    elif sum_magnification >= 1.25:
        return 0.7
    else:
        return 0.6


def get_current_datetime_str() -> str:
    now = str(datetime.now())
    return f'{now[0:4]}{now[5:7]}{now[8:10]}{now[11:13]}{now[14:16]}{now[17:19]}'


def rotate_vector(target_x: float, target_y: float, axis_x: float, axis_y: float, rotation_angle: float,
                  remaining_target_x: float = 0, remaining_target_y: float = 0,
                  remaining_axis_x: float = 0, remaining_axis_y: float = 0) -> tuple:
    rad = math.radians(rotation_angle)
    x = target_x + remaining_target_x
    a = axis_x + remaining_axis_x
    y = target_y + remaining_target_y
    b = axis_y + remaining_axis_y

    dest_x = (x - a) * math.cos(rad) - (y - b) * math.sin(rad) + a
    dest_y = (x - a) * math.sin(rad) + (y - b) * math.cos(rad) + b
    remaining_x, next_x = math.modf(dest_x)
    remaining_y, next_y = math.modf(dest_y)
    return int(next_x), int(next_y), remaining_x, remaining_y


def get_base_img_name(frame: int) -> str:
    return f"{BASE_IMG_NAME}_{str(frame).zfill(4)}{EXTENSION_PNG}"


def create_fish_cage(height: int, width: int, color: tuple) -> np.ndarray:
    return np.full((height, width, 3), np.array(color, dtype='uint8'))


def has_noise(num_has_noise: int) -> bool:
    return HAS_NOISE[num_has_noise]


def draw_salt_pepper_noise(num_noise: int, image: np.array) -> None:
    height, width, _ = image.shape
    x_coordinates_white = np.random.randint(0, width - 1, num_noise)
    y_coordinates_white = np.random.randint(0, height - 1, num_noise)
    image[(y_coordinates_white, x_coordinates_white)] = WHITE

    x_coordinates_black = np.random.randint(0, width - 1, num_noise)
    y_coordinates_black = np.random.randint(0, height - 1, num_noise)
    image[(y_coordinates_black, x_coordinates_black)] = BLACK


def get_header() -> list[str]:
    return [
        "sea_blue", "sea_green", "sea_red", "fish_blue", "fish_green", "fish_red", "fish_angle", "img_width",
        "img_height", "fish_length",
        "head_x", "head_y", "thickness_ratio", "num_fps", "frames", "has_noise", "num_noise", "variation_angle",
        "deformation_period_sec", "head_magnification_right", "head_magnification_left", "tail_magnification_right",
        "tail_magnification_left", "state", "created_datetime"
    ]


def main():
    os.makedirs(OUTPUT_DIR_NAME, exist_ok=True)
    info_file = open(INFO_FILE_PATH, mode='a+', encoding='utf-8')
    header = get_header()
    csv_writer = csv.DictWriter(info_file, fieldnames=header)
    csv_writer.writeheader()
    params_paths = Path(PARAMS_DIR_PATH).glob(f'*{EXTENSION_CSV}')
    for path in params_paths:
        csv_file = open(str(path), mode='r', encoding='utf-8')
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            dt_dirname = get_current_datetime_str()
            dt_dir_path = os.path.join(OUTPUT_DIR_NAME, dt_dirname)
            os.makedirs(dt_dir_path, exist_ok=True)

            """Define initial head and tail coordinates"""
            head_x = int(row['head_x'])
            head_y = int(row['head_y'])
            body_length = int(row['fish_length'])
            fish_angle = float(row['fish_angle'])
            tail_x, tail_y, remaining_tail_x, remaining_tail_y = rotate_vector(head_x + body_length - 1, head_y,
                                                                               head_x, head_y, fish_angle)
            remaining_head_x = 0
            remaining_head_y = 0

            """Get other params to create fish swimming images"""
            sea_blue = int(row['sea_blue'])
            sea_green = int(row['sea_green'])
            sea_red = int(row['sea_red'])
            img_width = int(row['img_width'])
            img_height = int(row['img_height'])
            thickness_ratio = float(row['thickness_ratio'])
            base_body_length = 1.0
            center = base_body_length / 2
            variation_angle = float(row['variation_angle'])
            deformation_period_sec = float(row['deformation_period_sec'])
            head_magnification_right = float(row['head_magnification_right'])
            head_magnification_left = float(row['head_magnification_left'])
            tail_magnification_right = float(row['tail_magnification_right'])
            tail_magnification_left = float(row['tail_magnification_left'])
            num_has_noise = int(row['has_noise'])
            num_fps = int(row['num_fps'])
            frames = int(row['frames'])
            num_noise = int(row['num_noise'])
            fish_blue = int(row['fish_blue'])
            fish_green = int(row['fish_green'])
            fish_red = int(row['fish_red'])

            """Write rows for the created image info"""
            row = {
                'sea_blue': sea_blue,
                'sea_green': sea_green,
                'sea_red': sea_red,
                'fish_blue': fish_blue,
                'fish_green': fish_green,
                'fish_red': fish_red,
                'img_width': img_width,
                'img_height': img_height,
                'fish_length': body_length,
                'head_x': head_x,
                'head_y': head_y,
                'variation_angle': variation_angle,
                'fish_angle': fish_angle,
                'frames': frames,
                'num_fps': num_fps,
                'thickness_ratio': thickness_ratio,
                'deformation_period_sec': deformation_period_sec,
                'num_noise': num_noise,
                'head_magnification_right': head_magnification_right,
                'head_magnification_left': head_magnification_left,
                'tail_magnification_right': tail_magnification_right,
                'tail_magnification_left': tail_magnification_left,
                'state': row['state'],
                'has_noise': num_has_noise,
                'created_datetime': dt_dirname,
            }
            csv_writer.writerow(row)
            for frame in range(frames):
                body_axis_x_coordinates, body_axis_y_coordinates = line(head_x, head_y, tail_x, tail_y)
                base_body_axis = np.linspace(0, base_body_length, len(body_axis_x_coordinates))
                thicknesses = [body_length * calc_base_body_thickness(x, base_body_length, thickness_ratio)
                               for x in base_body_axis]

                top_vectors = [(x, y + thickness / 2) for thickness, x, y in zip(thicknesses,
                                                                                 body_axis_x_coordinates,
                                                                                 body_axis_y_coordinates)]
                btm_vectors = [(x, y - thickness / 2) for thickness, x, y in zip(thicknesses,
                                                                                 body_axis_x_coordinates,
                                                                                 body_axis_y_coordinates)]

                time_sec = frame / num_fps
                periodic_time = time_sec % deformation_period_sec
                poly_top_vectors = []
                poly_btm_vectors = []
                for x, top_vector, btm_vector in zip(base_body_axis, top_vectors, btm_vectors):
                    if x > center:
                        variation = calc_variation_tail_side(x, periodic_time, deformation_period_sec, base_body_length,
                                                             tail_magnification_right, tail_magnification_left)
                    else:
                        variation = calc_variation_head_side(x, periodic_time, deformation_period_sec, base_body_length,
                                                             head_magnification_right, head_magnification_left)

                    poly_top_vectors.append((top_vector[0], top_vector[1] + int(variation * body_length)))
                    poly_btm_vectors.append((btm_vector[0], btm_vector[1] + int(variation * body_length)))

                rotated_top_vectors = []
                rotated_btm_vectors = []
                for top, btm, axis_x, axis_y in zip(poly_top_vectors, poly_btm_vectors,
                                                    body_axis_x_coordinates, body_axis_y_coordinates):
                    rotated_top_x, rotated_top_y, _, _ = rotate_vector(top[0], top[1], axis_x, axis_y, fish_angle)
                    rotated_btm_x, rotated_btm_y, _, _ = rotate_vector(btm[0], btm[1], axis_x, axis_y, fish_angle)
                    rotated_top_vectors.append((rotated_top_x, rotated_top_y))
                    rotated_btm_vectors.append((rotated_btm_x, rotated_btm_y))

                if is_vector_over_img(np.array(rotated_top_vectors), img_width, img_height):
                    "Stop creating image"
                    break
                if is_vector_over_img(np.array(rotated_btm_vectors), img_width, img_height):
                    "Stop creating image"
                    break

                cage = create_fish_cage(img_height, img_width, (sea_blue, sea_green, sea_red))
                cv2.polylines(cage, [np.array(rotated_top_vectors), np.array(rotated_btm_vectors)],
                              False, GREEN, thickness=2)

                gray = cv2.cvtColor(cage, cv2.COLOR_BGR2GRAY)
                ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cv2.contourArea(contours[0]) > (th.shape[0] * th.shape[1]) * 0.005:
                    cv2.fillPoly(cage, [contours[0][:, 0, :]], (fish_blue, fish_green, fish_red), lineType=cv2.LINE_8)

                rotated_head_x, rotated_head_y, remaining_rotated_head_x, remaining_rotated_head_y = \
                    rotate_vector(head_x, head_y, tail_x, tail_y, variation_angle,
                                  remaining_head_x, remaining_head_y, remaining_tail_x, remaining_tail_y)

                swimming_speed_coefficient = get_swimming_speed_coefficient(head_magnification_right,
                                                                            head_magnification_left,
                                                                            tail_magnification_right,
                                                                            tail_magnification_left)
                body_movement_distance = \
                    swimming_speed_coefficient * (
                                COEFFICIENT_MOVEMENT_DISTANCE / num_fps) * body_length / deformation_period_sec

                moved_head_x = (rotated_head_x + remaining_rotated_head_x) - body_movement_distance
                moved_tail_x = (tail_x + remaining_tail_x) - body_movement_distance

                head_x, head_y, remaining_head_x, remaining_head_y = \
                    rotate_vector(moved_head_x, rotated_head_y, rotated_head_x, rotated_head_y, fish_angle,
                                  remaining_target_y=remaining_rotated_head_y,
                                  remaining_axis_x=remaining_head_x,
                                  remaining_axis_y=remaining_head_y)

                tail_x, tail_y, remaining_tail_x, remaining_tail_y = \
                    rotate_vector(moved_tail_x, tail_y, tail_x, tail_y, fish_angle,
                                  remaining_target_y=remaining_tail_y,
                                  remaining_axis_x=remaining_tail_x,
                                  remaining_axis_y=remaining_tail_y)

                fish_angle = fish_angle + variation_angle

                if has_noise(num_has_noise):
                    draw_salt_pepper_noise(num_noise, cage)

                output_path = os.path.join(dt_dir_path, get_base_img_name(frame))
                cv2.imwrite(output_path, cage)
        csv_file.close()
    info_file.close()


if __name__ == '__main__':
    main()
    quit(0)
