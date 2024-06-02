import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import ultralytics
from ultralytics import YOLO



class BRIGHTNESS_LEVEL:
    HIGH=2
    MEDIUM=1
    LOW=0

class BRIGHTNESS_THRESHOLD:
    HIGH=127
    LOW=117


class ImageUtils:
    @staticmethod
    def render_image(image):
        plt.figure(figsize=(10, 10))
        plt.grid(True)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.show()

    @staticmethod
    def crop_image(image, point, offset=0):
        image = image.copy()
        cropped_image = image[point[1] + offset:, :]
        return cropped_image

    @staticmethod
    def get_distance(point1, point2, decimal_places=0):
        x1, y1 = point1
        x2, y2 = point2
        distance = round(((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5, decimal_places)
        return distance

    @staticmethod
    def get_lists_intersection(list1, list2):
        return list(set(map(tuple, list1)).intersection(set(map(tuple, list2))))

    @staticmethod
    def cartesian_to_polar(x, y, offset):
        radius = np.sqrt((x - offset) ** 2 + y ** 2)
        theeta = np.arctan2(y, x - offset)
        return [radius, theeta]

    @staticmethod
    def polar_to_cartesian(radius, theeta, offset_x=0, offset_y=0):
        theeta = theeta * (np.pi / 180)
        x = round(radius * np.cos(theeta)) + offset_x
        y = round(radius * np.sin(theeta)) + offset_y
        return [x, y]

    @staticmethod
    def get_line_intersections(point1, point2, ring_points, divisions=1000):
        line_points = np.round(np.linspace(point1, point2, num=divisions)).tolist()
        unique_line_points = [list(pt) for pt in set(tuple(pt) for pt in line_points)]
        intersection_pts = ImageUtils.get_lists_intersection(ring_points, unique_line_points)
        return intersection_pts, len(intersection_pts)

    @staticmethod
    def rotate_line_ints(self, pivot_point, dynamic_point, region_intersection, max_angle=181, step_size=1):
        pivot_x, pivot_y = pivot_point
        x, y = dynamic_point
        radius = x - pivot_x
        num_intersections, intersections, angles = [], [], []
        for angle in range(0, max_angle):
            angle_changed = step_size * angle
            theeta = angle_changed * (np.pi / 180)
            x, y = radius * np.cos(theeta) + pivot_x, radius * np.sin(theeta) + pivot_y
            ints, num_ints = self.get_line_intersections([pivot_x, pivot_y], [x, y], region_intersection, radius)
            intersections.append(ints)
            num_intersections.append(num_ints)
            angles.append(angle_changed)
        return intersections, num_intersections, angles

    @staticmethod
    def compare_images(image, image2):
        # Load the images
        img1 = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(image2.copy(), cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to both images
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img1 = clahe.apply(img1)
        img2 = clahe.apply(img2)

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches
        # img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None,
        #                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Find Homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            return 0, 0

        # Decompose the homography matrix to find translation, rotation, and scaling
        _, _, trans, _ = cv2.decomposeHomographyMat(M, np.eye(3))
        return trans[0][0][0], trans[0][1][0]

    @staticmethod
    def classify_color(rgb):
        # Define the RGB values for the target colors
        colors = {
            "black": np.array([0, 0, 0]),
            "white": np.array([255, 255, 255]),
            "red": np.array([255, 0, 0]),
            "green": np.array([0, 255, 0])
        }
        # Convert the input RGB to a numpy array
        input_color = np.array(rgb)
        # Initialize variables to store the minimum distance and corresponding color
        min_distance = float('inf')
        closest_color = None
        # Adjustments: factors to lean towards green and red, away from white
        # bias_factors = {
        #     "black": 0.97,  # No change
        #     "white": 1.07,  # Increase distance to make white less likely
        #     "red": 1.00,  # Decrease distance to favor red
        #     "green": 0.93  # Decrease distance to favor green
        # }
        bias_factors = {
            "black": 1.00,  # No change
            "white": 1.00,  # Increase distance to make white less likely
            "red": 1.00,  # Decrease distance to favor red
            "green": 1.00  # Decrease distance to favor green
        }
        # Compute the Euclidean distance between the input color and each target color
        for color, value in colors.items():
            distance = np.linalg.norm(input_color - value) * bias_factors[color]
            if distance < min_distance:
                min_distance = distance
                closest_color = color
        # Return the name of the closest color
        return closest_color

    @staticmethod
    def color_of_pixel(image, dX, dY):
        image_px = image[dY][dX]
        rgb = [int(image_px[2]), int(image_px[1]), int(image_px[0])]

        return ImageUtils.classify_color(rgb)

    @staticmethod
    def color_of_pixel_normalized(image, red_mask, green_mask, dX, dY):
        color_of_px = ImageUtils.color_of_pixel(image, dX, dY)

        if color_of_px not in ["green", "black"]:
            return color_of_px

        if color_of_px == "black":
            point_img = np.zeros_like(image.copy())
            cv2.circle(point_img, (dX, dY), 0, (255, 255, 255), -1)
            if np.count_nonzero(cv2.bitwise_and(point_img, red_mask)) > 0:
                return "red"
            if np.count_nonzero(cv2.bitwise_and(point_img, green_mask)) > 0:
                return "green"

            return "black"

        if color_of_px == "green" and "black" in [ImageUtils.color_of_pixel(image, dX - 1, dY),
                                                  ImageUtils.color_of_pixel(image, dX + 1, dY),
                                                  ImageUtils.color_of_pixel(image, dX, dY - 1),
                                                  ImageUtils.color_of_pixel(image, dX, dY + 1)]:
            return "white"

        return color_of_px

    @staticmethod
    def above_below_pixel_with_nonwhite_color(image, red_mask, green_mask, dX, dY, limit=5):
        pxs = []
        for i in range(limit):
            px_color = ImageUtils.color_of_pixel_normalized(image, red_mask, green_mask, dX, (dY - i - 1))
            if px_color != "white":
                pxs.append({"x": dX, "y": (dY - i - 1), "color": px_color})
                break
            elif i == limit - 1:
                pxs.append({"x": dX, "y": (dY - i - 1), "color": px_color})
        for i in range(limit):
            px_color = ImageUtils.color_of_pixel_normalized(image, red_mask, green_mask, dX, (dY + i + 1))
            if px_color != "white":
                pxs.append({"x": dX, "y": (dY + i + 1), "color": px_color})
                break
            elif i == limit - 1:
                pxs.append({"x": dX, "y": (dY + i + 1), "color": px_color})
        return pxs

    @staticmethod
    def find_lowest_white_pixel(img):
        # Check if the image is loaded properly
        if img is None:
            print("Error: Image did not load.")
            return None

        # Initialize variables to track the lowest white pixel
        lowest_white_pixel = None
        img_height = img.shape[0]

        # Iterate over the image from top to bottom
        for y in range(img_height - 1, -1, -1):  # Start from the bottom of the image
            # Check if there's a white pixel in this row
            white_pixels_in_row = np.where(img[y] == 255)[0]
            if white_pixels_in_row.size > 0:
                lowest_white_pixel = (white_pixels_in_row[0], y)  # x, y position
                break  # Exit the loop once the lowest white pixel is found

        return lowest_white_pixel if lowest_white_pixel is not None else (0, 0)


class BoardCalculatorHelper:
    @staticmethod
    def create_point_image(image, position):
        point_image = np.zeros_like(image.copy())
        cv2.circle(point_image, position, 0, (255, 255, 255), -1)

        return point_image

    @staticmethod
    def detect_3x_multiplication_zone(board_map, dX, dY):
        board_img = board_map["IMAGE"]
        zone = board_map["MULT_3"]
        roi = board_map["ROI"]
        center = board_map["CENTER"]
        red_mask = board_map["RED_MASK"]
        green_mask = board_map["GREEN_MASK"]

        point_image = np.zeros_like(board_img.copy())
        cv2.circle(point_image, (dX, dY), 0, (255, 255, 255), -1)

        corrected_image_circle_above_1 = np.zeros_like(board_img.copy())
        cv2.circle(corrected_image_circle_above_1, (dX, dY - 1), 0, (255, 255, 255), -1)

        corrected_image_line_2 = np.zeros_like(point_image.copy())
        cv2.line(corrected_image_line_2, (dX, dY), (dX, dY - 2), (255, 255, 255), 1)

        if np.count_nonzero(cv2.bitwise_and(point_image.copy(), zone)) > 0:
            if np.count_nonzero(cv2.bitwise_and(corrected_image_circle_above_1.copy(), zone)) > 0:
                return {"is_detected": True, "confidence_boost": None}

        if BoardCalculatorHelper.is_multiplication_detected_by_color(board_img, zone, red_mask, green_mask, center[0], center[1], dX,
                                                        dY):
            return {"is_detected": True, "confidence_boost": None}

        if np.count_nonzero(cv2.bitwise_and(point_image.copy(), roi)) > 0:
            corrected_image_circle = np.zeros_like(point_image.copy())
            cv2.circle(corrected_image_circle, (dX, dY), 2, (255, 255, 255), -1)

            overlap_m3_2 = np.count_nonzero(cv2.bitwise_and(corrected_image_line_2, zone))
            overlap_m3_circle = np.count_nonzero(cv2.bitwise_and(corrected_image_circle, zone))

            if overlap_m3_2 > 0:
                return {"is_detected": True, "confidence_boost": 50}
            elif overlap_m3_circle > 0:
                return {"is_detected": True, "confidence_boost": 25}

        return {"is_detected": False, "confidence_boost": None}

    @staticmethod
    def is_in_2x_multiplication_zone(board_map, dX, dY):
        img = board_map["IMAGE"]
        zone = board_map["MULT_2"]
        center = board_map["CENTER"]

        point_image = np.zeros_like(img.copy())
        cv2.circle(point_image, (dX, dY), 0, (255, 255, 255), -1)

        corrected_image_1 = np.zeros_like(img.copy())
        cv2.circle(corrected_image_1, (dX, dY - 1), 0, (255, 255, 255), -1)

        if np.count_nonzero(cv2.bitwise_and(point_image.copy(), zone)) > 0:
            if np.count_nonzero(cv2.bitwise_and(corrected_image_1.copy(), zone)) > 0:
                return True

        if BoardCalculatorHelper.is_multiplication_detected_by_color(img, zone, board_map["RED_MASK"], board_map["GREEN_MASK"],
                                                        center[0], center[1], dX, dY):
            return True

        return False

    @staticmethod
    def is_multiplication_detected_by_color(board_img, multiplication_zone, red_mask, green_mask, cX, cY, dX,
                                                dY):
        above_pxs = ImageUtils.above_below_pixel_with_nonwhite_color(board_img, red_mask, green_mask, dX, dY, 5)

        if len(above_pxs) != 2:
            raise Exception("List of color must have 2 items.")

        print(above_pxs)
        if BoardCalculatorHelper.check_if_list_have_combination_of_two_colors(
                [above_pxs[0]["color"], above_pxs[1]["color"]], ["red", "green"]
        ) and (
                BoardCalculatorHelper.is_circle_in_zone(multiplication_zone, above_pxs[0]["x"], above_pxs[0]["y"], 0) or
                BoardCalculatorHelper.is_circle_in_zone(multiplication_zone, above_pxs[1]["x"], above_pxs[1]["y"], 0)
        ):
            return True
        else:
            if dX < cX:
                above_pxs = ImageUtils.above_below_pixel_with_nonwhite_color(board_img, red_mask, green_mask, dX - 2, dY,
                                                                           5)
            else:
                above_pxs = ImageUtils.above_below_pixel_with_nonwhite_color(board_img, red_mask, green_mask, dX + 2, dY,
                                                                           5)

            if (BoardCalculatorHelper.check_if_list_have_combination_of_two_colors(
                    [above_pxs[0]["color"], above_pxs[1]["color"]], ["black", "green"]
            ) or BoardCalculatorHelper.check_if_list_have_combination_of_two_colors(
                [above_pxs[0]["color"], above_pxs[1]["color"]], ["red", "green"]
            )) and (
                    BoardCalculatorHelper.is_circle_in_zone(multiplication_zone, above_pxs[0]["x"], above_pxs[0]["y"], 0) or
                    BoardCalculatorHelper.is_circle_in_zone(multiplication_zone, above_pxs[1]["x"], above_pxs[1]["y"], 0)
            ):
                return True

        pxs = []
        if BoardCalculatorHelper.is_circle_in_zone(multiplication_zone, dX, dY, 10):
            if dX < cX:
                pxs = [
                    ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX + 3, dY),
                    ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX + 4, dY),
                    ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX + 7, dY)
                ]
            else:  # right part
                pxs = [
                    ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX - 3, dY),
                    ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX - 4, dY),
                    ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX - 7, dY)
                ]

            if all(px == "red" for px in pxs) or all(px == "green" for px in pxs):
                return True

            color_of_px = ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY)
            color_of_px_below = ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY + 1)
            if color_of_px in ["red", "green"] or (color_of_px == "white" and color_of_px_below in ["red", "green"]):
                if color_of_px == "white":
                    color_of_px = color_of_px_below

                pxs = [
                    ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX - 1, dY),
                    ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX + 1, dY),
                    ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY - 1),
                    ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY + 1),
                ]

                matches = 0
                directions = [False] * 4
                for idx, px in enumerate(pxs):
                    if px == color_of_px:
                        directions[idx] = True
                        matches += 1

                if matches >= 2:
                    if directions[0] and all(pxs[0] == px for px in [
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX - 2, dY),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX - 3, dY),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX - 4, dY),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX - 5, dY),
                    ]):  # left
                        return True
                    if directions[1] and all(pxs[1] == px for px in [
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX + 2, dY),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX + 3, dY),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX + 4, dY),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX + 5, dY),
                    ]):  # right
                        return True
                    if directions[2] and all(pxs[2] == px for px in [
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY - 2),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY - 3),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY - 4),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY - 5),
                    ]):  # above
                        return True
                    if directions[3] and all(pxs[3] == px for px in [
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY + 2),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY + 3),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY + 4),
                        ImageUtils.color_of_pixel_normalized(board_img, red_mask, green_mask, dX, dY + 5),
                    ]):  # below
                        return True

        return False

    @staticmethod
    def is_circle_in_zone(zone, dX, dY, radius):
        img = np.zeros_like(zone.copy())
        cv2.circle(img, (dX, dY), radius, (255, 255, 255), -1)

        return np.count_nonzero(cv2.bitwise_and(img.copy(), zone)) > 0

    @staticmethod
    def check_if_list_have_combination_of_two_colors(color_list, combination):
        return ((color_list[0] == combination[0] and color_list[1] == combination[1]) or
                (color_list[0] == combination[1] and color_list[1] == combination[0]))

    @staticmethod
    def is_slope_between_sector_grads(slope, sector, slope_range_threshold=0.05):
        if "grad_in" not in sector or "grad_out" not in sector:
            return False

        grad_in = sector["grad_in"]
        grad_out = sector["grad_out"]

        min_grad = min(grad_in, grad_out)
        max_grad = max(grad_in, grad_out)

        return BoardCalculatorHelper.is_between(slope, min_grad - slope_range_threshold, max_grad + slope_range_threshold)

    @staticmethod
    def is_between(number, limit1, limit2):
        low_limit = min(limit1, limit2)
        upper_limit = max(limit1, limit2)

        return upper_limit >= number >= low_limit

    @staticmethod
    def __refine_tip_coordinates_to_lowest_point__(original_img, dart_img_path, initial_tip_x, initial_tip_y,
                                                   roi_size=10):
        #
        # if self.__color_of_pixel_normalized__(original_img, initial_tip_x, initial_tip_y) == "black":
        #     return initial_tip_x, initial_tip_y

        # Load the images
        original_img_grayscale = cv2.cvtColor(original_img.copy(), cv2.COLOR_BGR2GRAY)
        dart_img = cv2.imread(dart_img_path, cv2.IMREAD_GRAYSCALE)

        x1, y1 = max(0, initial_tip_x - roi_size), max(0, initial_tip_y - roi_size)
        x2, y2 = min(original_img.shape[1], initial_tip_x + roi_size), min(original_img.shape[0],
                                                                           initial_tip_y + roi_size)
        # Extract the ROIs from both images
        original_roi = original_img_grayscale[y1:y2, x1:x2]
        dart_roi = dart_img[y1:y2, x1:x2]
        # Compute the absolute difference between the two ROIs
        difference = cv2.absdiff(original_roi, dart_roi)

        threshold = 10
        brightness_level = get_brightness_level(original_img)
        if brightness_level == 2:
            threshold = 50
        elif brightness_level == 1:
            threshold = 15

        difference_full = cv2.absdiff(original_img_grayscale, dart_img)
        _, thresh = cv2.threshold(difference_full, threshold, 255, cv2.THRESH_BINARY)

        # Threshold the difference to convert it to a binary image
        _, thresholded_diff = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
        thresholded_diff = remove_small_contours_from_thresh_binary(thresholded_diff, 5)

        lowest_px_x, lowest_px_y = ImageUtils.__find_lowest_white_pixel__(thresholded_diff)

        if lowest_px_x == 0 or lowest_px_y == 0:
            return initial_tip_x, initial_tip_y

        return (x1 + lowest_px_x, y1 + lowest_px_y)



class PlotUtils:
    @staticmethod
    def plot_from_list(in_list, show=0):
        in_array = np.array(in_list)
        plt.scatter(in_array[:, 0], in_array[:, 1])
        if show:
            plt.show()

    @staticmethod
    def plot_hough_lines(hough_lines):
        for line in hough_lines:
            for x1, y1, x2, y2 in line:
                plt.plot([x1, x2], [y1, y2])

    @staticmethod
    def plot_from_angles(angles, reference_point, radius=500):
        ref_x, ref_y = reference_point
        for angle in angles:
            theeta = angle * (np.pi / 180)
            x, y = radius * np.cos(theeta) + ref_x, radius * np.sin(theeta) + ref_y
            plt.plot([ref_x, x], [ref_y, y], c='y')


def _get_line_combinations(points_list):
    start_points = [point for ind, point in enumerate(points_list) if ind % 2 == 0]
    end_points = [point for ind, point in enumerate(points_list) if ind % 2 == 1]
    return [[start_points[ind], end_points[ind]]
            for ind, pt in enumerate(start_points)]


def _get_line_intersection(line1, line2):
    [(x1, y1), (x2, y2)] = line1
    [(x3, y3), (x4, y4)] = line2

    determinent = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (determinent)
    Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (determinent)
    return (round(Px), round(Py))


def get_gradient_distribution(hough_lines):
    gradient_dict = {}
    for line in hough_lines:
        for x1, y1, x2, y2 in line:
            gradient = round(((y2 - y1) / (x2 - x1)), 2) if x1 != x2 else 10000
            if gradient != 0 and gradient != 10000:
                if gradient in gradient_dict.keys():
                    gradient_dict[gradient] += [(x1, y1), (x2, y2)]
                else:
                    gradient_dict[gradient] = [(x1, y1), (x2, y2)]
    return gradient_dict


from itertools import product

def is_segment_of_lines_intersect(seg1, seg2):
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def intersect_1D(a, b, c, d):
        return max(a, b) >= min(c, d) and max(c, d) >= min(a, b)

    A, B = seg1
    C, D = seg2
    return intersect_1D(A[0], B[0], C[0], D[0]) and intersect_1D(A[1], B[1], C[1], D[1]) and ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def distance_segment_segment(seg1, seg2):
    def dist_squared(v, w):
        return (v[0] - w[0]) ** 2 + (v[1] - w[1]) ** 2

    def dist_to_segment_squared(p, v, w):
        l2 = dist_squared(v, w)
        if l2 == 0:
            return dist_squared(p, v)
        t = max(0, min(1, ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2))
        projection = (v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]))
        return dist_squared(p, projection)

    A, B = seg1
    C, D = seg2
    return math.sqrt(min(
        min(dist_to_segment_squared(A, C, D), dist_to_segment_squared(B, C, D)),
        min(dist_to_segment_squared(C, A, B), dist_to_segment_squared(D, A, B))
    ))


def get_bullseye(hough_lines, image):
    image = image.copy()
    gradients = get_gradient_distribution(hough_lines)
    max_gradient, min_gradient = max(gradients.keys()), min(gradients.keys())
    pts_max_gradient, pts_min_gradient = gradients[max_gradient], gradients[min_gradient]
    max_lines_list, min_lines_list = _get_line_combinations(pts_max_gradient), \
        _get_line_combinations(pts_min_gradient)
    max_min_product_lines = list(product(max_lines_list, min_lines_list))
    intersection_pts = [_get_line_intersection(val[0], val[1]) for val in max_min_product_lines]
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    bullseye_image = cv2.circle(image, intersection_pts[0], 2, (255, 0, 0), -1)
    return bullseye_image, intersection_pts


def get_radial_lines(image):
    image = image.copy()
    grayscale_image = image.copy()  # cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_image, 100, 200, apertureSize=3)
    #     ImageUtils.render_image(edges)
    hough_lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, 80, 10)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # hough_lines = get_hough_lines(edges)
    # print(hough_lines)
    for line in hough_lines:
        for x1, y1, x2, y2 in line:
            gradient = round(((y2 - y1) / (x2 - x1)), 2) if x1 != x2 else 10000
            if gradient > 0:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #     ImageUtils.render_image(image)
    return hough_lines, image


def get_inner_outer_bullseye(image):
    img = image.copy()
    h, w, c = img.shape
    h = h // 2
    w = w // 2
    img = img[h - 220:h, w - 150:w + 150]  # Crop image to the approximate location of bullseye
  
    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(grayscale_image, 120, 255,
                                 cv2.THRESH_BINARY_INV)
    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    hough_lines, radial_lines = get_radial_lines(opening)
    _, bullseye = get_bullseye(hough_lines, opening)

    bullseye_x = bullseye[0][0]
    bullseye_y = bullseye[0][1]
    
    # Collect only the circle around the "bullseye" 
    radius = 70
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (bullseye_x, bullseye_y), radius, (255, 255, 255), -1)
    img = cv2.bitwise_and(img, mask)

    # Convert to HSV and detect red and green segments
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([15, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([165, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join masks
    mask_red = mask0 + mask1

    # set my output img to zero everywhere except the red mask
    output_img_red = img.copy()
    output_img_red[np.where(mask_red == 0)] = 0
    output_img_red[np.where(mask_red != 0)] = 255

    mask_green = get_green_mask(img)

    # set my output img to zero everywhere except the green mask
    output_img_green = img.copy()
    output_img_green[np.where(mask_green == 0)] = 0
    output_img_green[np.where(mask_green != 0)] = 255

    # Initialize output image to be all black
    output_image = np.zeros(image.shape, np.uint8)
    inner_bullseye = np.zeros_like(output_image)
    inner_bullseye[h - 220:h, w - 150:w + 150] = output_img_red
    outer_bullseye = np.zeros_like(output_image)
    outer_bullseye[h - 220:h, w - 150:w + 150] = output_img_green

    # Place the bullseye back in its original place
    output_image[h - 220:h, w - 150:w + 150] = cv2.add(output_img_red, output_img_green)

    return remove_small_contours(output_image, 25), remove_small_contours(inner_bullseye, 25), remove_small_contours(outer_bullseye, 25)

def get_inner_outer_bullseye_by_coordinates(image, x, y):
    img = image.copy()

    img = img[y-75:  y+75, x-100: x+100]

    grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(grayscale_image, 120, 255,
                                 cv2.THRESH_BINARY_INV)
    kernel = np.ones((4, 4), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

    hough_lines, radial_lines = get_radial_lines(opening)
    _, bullseye = get_bullseye(hough_lines, opening)

    bullseye_x = bullseye[0][0]
    bullseye_y = bullseye[0][1]

    # Collect only the circle around the "bullseye"
    radius = 100
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (bullseye_x, bullseye_y), radius, (255, 255, 255), -1)
    plt.imshow(mask)
    plt.show()
    img = cv2.bitwise_and(img, mask)

    # Convert to HSV and detect red and green segments
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([15, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([165, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join masks
    mask_red = mask0 + mask1

    # set my output img to zero everywhere except the red mask
    output_img_red = img.copy()
    output_img_red[np.where(mask_red == 0)] = 0
    output_img_red[np.where(mask_red != 0)] = 255

    mask_green = get_green_mask(img)

    # set my output img to zero everywhere except the green mask
    output_img_green = img.copy()
    output_img_green[np.where(mask_green == 0)] = 0
    output_img_green[np.where(mask_green != 0)] = 255

    # Initialize output image to be all black
    img = image.copy()
    output_image = np.zeros(img.shape, np.uint8)
    inner_bullseye = np.zeros_like(output_image)

    inner_bullseye[y-75:  y+75, x-100: x+100] = output_img_red
    outer_bullseye = np.zeros_like(output_image)
    outer_bullseye[y-75:  y+75, x-100: x+100] = output_img_green
    output_image[y-75:  y+75, x-100: x+100] = cv2.add(output_img_red, output_img_green)

    return  remove_small_contours(output_image, 25),  remove_small_contours(inner_bullseye, 25), remove_small_contours(outer_bullseye, 25)


def get_center_coordinates(bullseye_image):
    contours, _ = cv2.findContours(cv2.cvtColor(bullseye_image, cv2.COLOR_BGR2GRAY),
                                   cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_areas = [cv2.contourArea(contour, False) for contour in contours]
    max_area = max(contour_areas)
    contour_max_area = [contour for contour in contours if cv2.contourArea(contour, False) == max_area]
    M = cv2.moments(contour_max_area[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def get_green_mask(image):
    img = image.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:,:,1]
    # manually set threshold value
    _, th = cv2.threshold(a_channel, 105, 255, cv2.THRESH_BINARY_INV)
    # perform masking
    return cv2.bitwise_and(img, img, mask = th)

def get_red_green_masks(image):
    img = image.copy()
    # ImageUtils.render_image(img)

    # Convert to HSV and detect red and green segments
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([165, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join masks
    mask_red = mask0 + mask1

    # set my output img to zero everywhere except the red mask
    output_img_red = img.copy()
    output_img_red[np.where(mask_red == 0)] = 0
    # ImageUtils.render_image(output_img_red)

    # set my output img to zero everywhere except the green mask
    output_img_green = get_green_mask(img)
    # ImageUtils.render_image(output_img_green)

    return output_img_red, output_img_green


def crop_region(red_green_image, cY):
    rings_image = red_green_image.copy()
    rings_image[:cY, :] = np.zeros(rings_image.shape)[:cY, :]
    return rings_image


def remove_text(rings_image):
    img = rings_image
    # --- ensure image is of the type float ---
    img = img.astype(np.float32)

    # --- the following holds the square root of the sum of squares of the image dimensions ---
    # --- this is done so that the entire width/height of the original image is used to express the complete circular range of the resulting polar image ---
    value = np.sqrt(((img.shape[0] / 1.7) ** 2.0) + ((img.shape[1] / 1.7) ** 2.0))

    polar_image = cv2.linearPolar(img, (img.shape[0] / 1.7, img.shape[1] / 1.7), value, cv2.WARP_FILL_OUTLIERS)

    polar_image = polar_image.astype(np.uint8)
    # ImageUtils.render_image(polar_image)

    image_tmp = np.zeros(rings_image.shape, np.uint8)
    polar_image[10:400, :200] = image_tmp[10:400, :200]
    polar_image[:400, 200:] = image_tmp[:400, 200:]

    cartesian_image = cv2.linearPolar(polar_image, (polar_image.shape[0] / 1.7,
                                                    polar_image.shape[1] / 1.7),
                                      value, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    return cartesian_image

def remove_small_contours(image, min_area=400):
    image = image.copy()
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(grayscale_image, 50, 255,
                                 cv2.THRESH_BINARY)
    contours, hr = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    threshold_mask = threshold.copy()

    # Iterate through the contours and filter out small ones
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(threshold_mask, [cnt], -1, 0, thickness=cv2.FILLED)

    return cv2.bitwise_and(image, image, mask=threshold_mask)

def remove_small_contours_from_thresh_binary(image, min_area=400):
    image = image.copy()

    contours, hr = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    threshold_mask = image.copy()

    # Iterate through the contours and filter out small ones
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(threshold_mask, [cnt], -1, 0, thickness=cv2.FILLED)

    return cv2.bitwise_and(image, image, mask=threshold_mask)

def get_brightness_level(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    avg_brightness = np.mean(l_channel)


    result = BRIGHTNESS_LEVEL.LOW
    if avg_brightness > BRIGHTNESS_THRESHOLD.HIGH:
        result = BRIGHTNESS_LEVEL.HIGH
    elif BRIGHTNESS_THRESHOLD.HIGH > avg_brightness > BRIGHTNESS_THRESHOLD.LOW:
        result = BRIGHTNESS_LEVEL.MEDIUM

    return result

def get_outer_border(rings_image, image, center, threshold = 60):
    mask = np.zeros_like(image)

    cv2.circle(mask, center, 360, (255, 255, 255), -1)
    cv2.rectangle(mask, (0, 0), (image.shape[1], center[1]), (0, 0, 0), -1)

    masked_image = cv2.bitwise_and(image, mask)

    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    ret, th = cv2.threshold(gray, threshold, 255, 0)

    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = max(contours, key=cv2.contourArea)

    contour_mask = np.zeros_like(image)

    cv2.drawContours(contour_mask, [biggest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    cv2.drawContours(masked_image, [biggest_contour], -1, (0, 255, 0), 2)

    playing_area = cv2.bitwise_and(image.copy(), contour_mask)

    rings_image = cv2.cvtColor(rings_image.copy(), cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_and(playing_area, playing_area, mask=rings_image)

def get_inner_outer_rings(cartesian_image):
    grayscale_image = cv2.cvtColor(cartesian_image, cv2.COLOR_BGR2GRAY)
    # TODO: Change threshold
    _, threshold = cv2.threshold(grayscale_image, 50, 255,
                                 cv2.THRESH_BINARY)
    kernel = np.ones((3, 3))
    closing = cv2.dilate(threshold, kernel, iterations=5)
    closing = cv2.erode(closing, kernel, iterations=5)
    contours, hr = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    closing_coloured = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
    c_img = cv2.drawContours(closing_coloured.copy(), contours, -1, (255, 0, 0), thickness=1)  # cv2.FILLED)
    contour_areas = [cv2.contourArea(contour, False) for contour in contours]
    contours_sorted = sorted(contour_areas)
    outer_ring_area = contours_sorted[-1]
    inner_ring_area = contours_sorted[-2]
    outer_ring_cnt = [contour for contour in contours if cv2.contourArea(contour, False) == outer_ring_area]
    inner_ring_cnt = [contour for contour in contours if cv2.contourArea(contour, False) == inner_ring_area]
    outer_ring = cv2.drawContours(np.zeros(closing_coloured.shape, np.uint8), outer_ring_cnt, -1, (255, 255, 255),
                                  thickness=cv2.FILLED)
    inner_ring = cv2.drawContours(np.zeros(closing_coloured.shape, np.uint8), inner_ring_cnt, -1, (255, 255, 255),
                                  thickness=cv2.FILLED)

    return (inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt)


from math import acos
from math import sqrt
from math import pi


def slope(p1, p2):
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


def length(v):
    return sqrt(v[0] ** 2 + v[1] ** 2)


def dot_product(v, w):
    return v[0] * w[0] + v[1] * w[1]


def determinant(v, w):
    return v[0] * w[1] - v[1] * w[0]


def inner_angle(v,w):
    len_v = length(v)
    len_w = length(w)

    # Check if the vectors are non-zero
    if len_v == 0 or len_w == 0:
        # Handle the case where one or both vectors are zero
        return 0

    cosx=dot_product(v,w)/(length(v)*length(w))

    # Check if the cosine value is within the valid range
    if cosx < -1:
        cosx = -1
    elif cosx > 1:
        cosx = 1

    rad=acos(cosx) # in radians
    return rad*180/pi # returns degrees


def angle_clockwise(A, B):
    inner = inner_angle(A, B)
    det = determinant(A, B)
    if det < 0:  # this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else:  # if the det > 0 then A is immediately clockwise of B
        return 360 - inner


def findangle(x, c, y):
    if all(x == y):
        return 0
    v1 = [x[0] - c[0], x[1] - c[1]]
    v2 = [y[0] - c[0], y[1] - c[1]]
    return inner_angle(v1, v2)


def ROI_mask(ring, cnt, cnt_num, center):
    pg = ring.copy()
    ring_cnts = cnt[0]
    ring_cnts = ring_cnts.reshape(ring_cnts.shape[0], ring_cnts.shape[2])
    ymin = min(ring_cnts, key=lambda x: (x[1], x[0]))

    pg = cv2.line(pg, tuple(center), tuple(ymin), (0, 0, 255), 5)

    max_angle = 0
    max_angle_pt = ymin
    pt_angle = []
    for pt in ring_cnts:
        angle = findangle(pt, center, ymin)
        pt_angle.append((pt, angle))

    #     if angle >= max_angle:
    #         max_angle = angle
    #         max_angle_pt = pt

    max_angle_pt, max_angle = max(pt_angle, key=lambda x: (x[1], -1 * x[0][1]))
    #     print(max_angle_pt, max_angle)

    pg = cv2.line(pg, tuple(max_angle_pt), tuple(center), (0, 0, 255), 5)
    pg_gray = cv2.cvtColor(pg, cv2.COLOR_BGR2GRAY)
    contours, hr = cv2.findContours(pg_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(pg)
    cv2.fillPoly(mask, pts=[sorted(contours, key=cv2.contourArea)[cnt_num]], color=(255, 255, 255))

    return mask


def get_ROI_image(image, oi_mask, io_mask):
    tmp = image.copy()
    roi = cv2.bitwise_and(tmp, oi_mask)
    roi = cv2.bitwise_and(roi, cv2.bitwise_not(io_mask))
    return roi


def get_sector_gradients(roi, center, inner_ring_cnt, outer_ring_cnt, same_ring_thresh=80, image=None, i=0):
    brightness_level = get_brightness_level(roi)
    threshold = 200
    if brightness_level == BRIGHTNESS_LEVEL.HIGH:
        threshold = 250
    elif brightness_level == BRIGHTNESS_LEVEL.MEDIUM:
        threshold = 225

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(roi, 150, threshold)
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    linesDilatedP = cv2.HoughLinesP(dilated_edges, 1, np.pi / 180, 50, None, 50, 10)

    sectors = []
    for line in linesP:
        l = line[0]
        pt0 = (int(l[0]), int(l[1]))
        pt1 = (int(l[2]), int(l[3]))

        dist_p0_to_center = (pt0[0] - center[0]) ** 2 + (pt0[1] - center[1]) ** 2
        dist_p1_to_center = (pt1[0] - center[0]) ** 2 + (pt1[1] - center[1]) ** 2

        pt_in, pt_out = (pt0, pt1) if dist_p0_to_center < dist_p1_to_center else (pt1, pt0)

        dist_in = abs(cv2.pointPolygonTest(inner_ring_cnt[0], pt_in, True))
        dist_out = abs(cv2.pointPolygonTest(outer_ring_cnt[0], pt_out, True))

        if dist_in <= same_ring_thresh and dist_out <= same_ring_thresh:
            grad_in = (pt_in[1] - center[1]) / (pt_in[0] - center[0])
            grad_out = (pt_out[1] - center[1]) / (pt_out[0] - center[0])
            grad_avg = (grad_in + grad_out) / 2.0

            x1, y1 = pt_in
            x2, y2 = pt_out

            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            min_distance_limit = 5
            max_distance_limit = 8

            distance_limit = max_distance_limit
            if angle < 20 or angle > 160:
                distance_limit = min_distance_limit

            unique = True
            for sector in sectors:
                if (
                        ImageUtils.get_distance(pt_in, sector["pt_in"], 2) < distance_limit or
                        ImageUtils.get_distance(pt_out, sector["pt_out"], 2) < distance_limit
                ):
                    unique = False
                    break

            if not (grad_avg < 0 and angle < 0) and unique:
                    sectors.append(
                        {"pt_in": pt_in, "pt_out": pt_out, "angle": angle, "grad_avg": grad_avg, "grad_in": grad_in,
                         "grad_out": grad_out})

    for line in linesDilatedP:
        l = line[0]
        pt0 = (int(l[0]), int(l[1]))
        pt1 = (int(l[2]), int(l[3]))

        dist_p0_to_center = (pt0[0] - center[0]) ** 2 + (pt0[1] - center[1]) ** 2
        dist_p1_to_center = (pt1[0] - center[0]) ** 2 + (pt1[1] - center[1]) ** 2

        pt_in, pt_out = (pt0, pt1) if dist_p0_to_center < dist_p1_to_center else (pt1, pt0)

        dist_in = abs(cv2.pointPolygonTest(inner_ring_cnt[0], pt_in, True))
        dist_out = abs(cv2.pointPolygonTest(outer_ring_cnt[0], pt_out, True))

        if dist_in <= same_ring_thresh and dist_out <= same_ring_thresh:
            grad_in = (pt_in[1] - center[1]) / (pt_in[0] - center[0])
            grad_out = (pt_out[1] - center[1]) / (pt_out[0] - center[0])
            grad_avg = (grad_in + grad_out) / 2.0

            x1, y1 = pt_in
            x2, y2 = pt_out

            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            min_distance_limit = 5
            max_distance_limit = 8

            distance_limit = max_distance_limit
            if angle < 20 or angle > 160:
                distance_limit = min_distance_limit

            unique = True
            for sector in sectors:
                if (
                        ImageUtils.get_distance(pt_in, sector["pt_in"], 2) < distance_limit or
                        ImageUtils.get_distance(pt_out, sector["pt_out"], 2) < distance_limit
                ):
                    unique = False
                    break

            if not (grad_avg < 0 and angle < 0) and unique:
                    sectors.append(
                        {"pt_in": pt_in, "pt_out": pt_out, "angle": angle, "grad_avg": grad_avg, "grad_in": grad_in,
                         "grad_out": grad_out})

    median_center = center #calculate_center_as_median_intersection_point(sectors, center)

    angle_threshold_max = 6
    angle_threshold_min = 4
    filtered_sectors = []

    for sector in sectors:
        is_unique = True
        for unique_sector in filtered_sectors:
            if sector["angle"] < 30 or sector["angle"] > 150:
                if abs(sector["angle"] - unique_sector["angle"]) < angle_threshold_min:
                    is_unique = False
                    break
            else:
                if abs(sector["angle"] - unique_sector["angle"]) < angle_threshold_max:
                    is_unique = False
                    break

            if is_segment_of_lines_intersect((sector["pt_in"], sector["pt_out"]),
                                             (unique_sector["pt_in"], unique_sector["pt_out"])):
                is_unique = False
                break
            if distance_segment_segment((sector['pt_in'], sector['pt_out']),
                                        (unique_sector['pt_in'], unique_sector['pt_out'])) < 5:
                is_unique = False
                break

        if is_unique:
            grad_in = (sector["pt_in"][1] - median_center[1]) / (sector["pt_in"][0] - median_center[0])
            grad_out = (sector["pt_out"][1] - median_center[1]) / (sector["pt_out"][0] - median_center[0])
            grad_avg = (grad_in + grad_out) / 2.0

            x1, y1 = sector["pt_in"]
            x2, y2 = sector["pt_out"]

            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            filtered_sectors.append(
                {"pt_in": sector["pt_in"], "pt_out": sector["pt_out"], "angle": angle, "grad_avg": grad_avg,
                 "grad_in": grad_in, "grad_out": grad_out})

            cv2.circle(image, sector["pt_in"], 4, (255, 0, 0), -1)
            cv2.circle(image, sector["pt_out"], 4, (0, 0, 255), -1)
            cv2.line(image, sector["pt_in"], sector["pt_out"], (0, 255, 0),
                     1)  # Draw the line in green with thickness 2
            #cv2.imsave()
    # VPROROK TEST CODE
    # if i == 0:
    #     cv2.line(image, center, (587, 331), (255, 0, 0), 1)
    #     cv2.line(image, median_center, (587, 331), (0, 0, 255), 1)
    #     cv2.circle(image, (587, 331), 2, (0, 255, 0), -1)
    #     cv2.imwrite('/app/tmp/images/image_with_circles.jpg', image)
    #
    #     testSlope = slope(center, [587, 331])
    #     medianSlope = slope(median_center, [587, 331])
    #     file_path = f"/app/tmp/testSlope.json"
    #     with open(file_path, 'w') as json_file:
    #         json.dump({
    #             'slope': testSlope,
    #             'median_slope': medianSlope,
    #             'centerX': center[0],
    #             'centerY': center[1],
    #             'median_centerX': median_center[0],
    #             'median_centerY': median_center[1]
    #         }, json_file)

    return filtered_sectors

def calculate_center_as_median_intersection_point(sectors, initial_center):

    # Calculate intersection points of these lines
    intersection_points = []

    # remove first and last element since they contain "Infinity"
    sectors = sectors[1:-1]

    for line1 in sectors:
        for line2 in sectors:
            if line1 != line2:  # Ensure lines are different
                x1, y1 = line1[0]
                x2, y2 = line1[1]
                x3, y3 = line2[0]
                x4, y4 = line2[1]

                # Calculate intersection using parametric line equations
                denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denominator != 0:  # Check for non-parallel lines
                    intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
                    intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

                    intersection_points.append((intersection_x, intersection_y))

    # Sort intersection points based on x-coordinates (or y-coordinates)
    sorted_points = sorted(intersection_points,
                           key=lambda p: p[0])  # Sort by x-coordinates, change to p[1] for y-coordinates

    # Find the median point
    num_points = len(sorted_points)
    if num_points % 2 == 1:  # Odd number of points
        median_point = sorted_points[num_points // 2]
    else:  # Even number of points
        median_point = [
            (sorted_points[num_points // 2 - 1][0] + sorted_points[num_points // 2][0]) / 2,
            (sorted_points[num_points // 2 - 1][1] + sorted_points[num_points // 2][1]) / 2
        ]

    return [round((initial_center[0] + median_point[0]) / 2), round((initial_center[1] + median_point[1]) / 2)]


def map_pts(sectors, config):
    points_pos = config["POINTS_POS"]  # [20,5,12,9,14,11,8,16,7,19]
    points_neg = config["POINTS_NEG"]  # [1,18,4,13,6,10,15,2,17,3]

    sector_pos = [sector for sector in sectors if sector["grad_avg"] > 0]
    sector_neg = [sector for sector in sectors if sector["grad_avg"] <= 0]

    sector_range = [({"grad_avg": -np.inf}, sectors[0], points_pos[0]),
                    (sectors[-1], {"grad_avg": np.inf}, points_pos[0])]  # The 20 point sector
    for i in range(1, len(sector_neg)):
        sector_range.append((sector_neg[i - 1], sector_neg[i], points_neg[i - 1]))
    ind = 1
    for i in range(len(sector_pos) - 1, 0, -1):
        sector_range.append((sector_pos[i - 1], sector_pos[i], points_pos[ind]))
        ind += 1

    return sorted(sector_range, key=lambda x: (x[0]["grad_avg"], -x[1]["grad_avg"]))


def make_black(image, threshold=30):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to identify dark regions
    _, mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Convert the mask to a 3-channel image
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Set the pixels in the original image to black where the mask is white
    result_image = cv2.bitwise_and(image, mask)

    return result_image


 
def get_white_contour(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 175, 255, cv2.THRESH_BINARY)
    inverted_image = cv2.bitwise_not(binary_image)
    contours, hierarchy = cv2.findContours(inverted_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
    white_contours= []
    for i, contour in enumerate(sorted_contours[:-2]):
    # min area
    #print(cv2.contourArea(contour))
        if cv2.contourArea(contour) > 1200:
        
            white_contours.append(contour)
    return white_contours

def get_contours_centers(contours):
    centers =[]
    for contour in contours:
        # Calculate the moments of the contour
        M = cv2.moments(contour)
    
        # Calculate the centroid using the moments
        if M['m00'] != 0:
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
        else:
            # If the area of contour is 0, set center to (0, 0)
            center_x, center_y = 0, 0
            
        centers.append((center_x, center_y))
    return centers
def get_quadrant(point, center):
    x, y = point
    cx, cy = center
    
    if x >= cx and y >= cy:
        return 1
    elif x < cx and y >= cy:
        return 2
    elif x < cx and y < cy:
        return 3
    elif x >= cx and y < cy:
        return 4
    else:
        return None
def are_in_same_quadrant(point1, point2, center):
    quadrant1 = get_quadrant(point1, center)
    quadrant2 = get_quadrant(point2, center)
    
    return quadrant1 is not None and quadrant2 is not None and quadrant1 == quadrant2

def get_dartboard_roi(image, model_path=None):
    model = YOLO('D://dart/best.pt')
    results = model.predict(source=image, show=False,
                        hide_labels=False, 
                        hide_conf= False,
                        save_txt=False,
                        save_crop=False,  
                        conf=0.25 , 
                        save= False )

    #segmented_image = results[0].orig_img
    h, w, c =  image.shape
    blank_Im = np.zeros((h,w,1))
    polygons= results[0].masks.xy[0]
    ellipse = cv2.fitEllipse(polygons)
    ellipse_contour  = poly = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])), (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)), int(ellipse[2]), 0, 360, 5)
    ell_img= cv2.fillPoly(np.zeros_like(blank_Im), [ellipse_contour], 255)
    ell_img = np.uint8(ell_img)
    b, g, r = cv2.split(image)
    inverse_mask = cv2.bitwise_not(ell_img)
    # Subtract the mask from each channel
    b_result = cv2.subtract(b, inverse_mask)
    g_result = cv2.subtract(g, inverse_mask)
    r_result = cv2.subtract(r, inverse_mask)

    # Clip the resulting values to ensure they remain within the valid range
    b_result = np.clip(b_result, 0, 255)
    g_result = np.clip(g_result, 0, 255)
    r_result = np.clip(r_result, 0, 255)

    # Merge the channels back together
    result_image = cv2.merge((b_result, g_result, r_result))
    black_background_im = result_image.copy()
    result_image[inverse_mask != 0] = 255

    return black_background_im  , result_image

def subtract_roi(image , inner_ring, outer_ring , inner_be , outer_be , is_white):
    if is_white :
        color_mask = [255,255,255]
    else:
        color_mask = [0,0,0]
    white_pixels = np.all(inner_ring == [255, 255, 255], axis=-1)

   
    image = np.where(white_pixels[..., np.newaxis],color_mask, image)
    white_pixels = np.all(outer_ring == [255, 255, 255], axis=-1)

   
    image = np.where(white_pixels[..., np.newaxis],color_mask, image)
    white_pixels = np.all(inner_be == [255, 255, 255], axis=-1)

 
    image = np.where(white_pixels[..., np.newaxis], color_mask, image)
    white_pixels = np.all(outer_be == [255, 255, 255], axis=-1)

 
    image = np.where(white_pixels[..., np.newaxis], color_mask, image)

    return  image.astype(np.uint8)
def preproces_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h,w = gray_image.shape
                
    # Create a mask where 1 indicates the regions to be filtered and 0 indicates the regions to be unchanged
    mask = np.ones((h, w), dtype=np.uint8) * 255

    # Define the rectangular region at the center of the image
    # TODO use bulls eye here
    
    rect_start_row = (h - 200) // 2
    rect_end_row = rect_start_row + 200
    rect_start_col = (w - 300) // 2
    rect_end_col = rect_start_col + 300

    # Exclude the rectangular region from the mask
    cv2.rectangle(mask, (rect_start_col, rect_start_row), (rect_end_col, rect_end_row), 0, -1)
    # Apply median filter to the entire image, excluding the rectangular region
    filtered_image = cv2.medianBlur(gray_image, 7)

    # Restore the rectangular region in the filtered image
    gray_image = np.where(mask == 0, gray_image, filtered_image)
    return gray_image

def get_black_contour(gray_image, black_background_im):
    threshold =30
    areas = []
    all_thresholds = []
    while threshold <255 : 
        print(threshold)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY )
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
        rem_contours = []
        area = 0 
        for i, contour in enumerate(sorted_contours[:-1]):
            if cv2.contourArea(contour) >1000:
                rem_contours.append(contour)
                area+= cv2.contourArea(contour)
                
        
        
        if len(rem_contours)== 20:
            areas.append(area)
            all_thresholds.append(threshold)
        threshold+=1
      

    if len(areas)>0: 
        areas=np.array(areas)
        idx= np.argmax(areas)
        threshold = all_thresholds[idx]
        print(threshold)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY )
        
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
        black_contours = []

        heatmap_normalized = np.zeros_like(gray_image,dtype=np.float32 )
        heatmap_unnormalized = np.zeros_like(gray_image,dtype=np.float32 )

        for i, contour in enumerate(sorted_contours[:-1]):
            if cv2.contourArea(contour) >1000:
                black_contours.append(contour)
                mask = np.zeros_like(gray_image)
                cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
                distance_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                heatmap_unnormalized += distance_transform
                distance_transform = cv2.normalize(distance_transform, None, 0, 1.0, cv2.NORM_MINMAX)
                heatmap_normalized += distance_transform

        #for white hereee      making pixels black for white transform
        out_im = black_background_im.copy()
        black_background_im = cv2.drawContours(out_im, black_contours, -1, (0,0,0),
                                    thickness=cv2.FILLED)     
        return black_background_im , black_contours, heatmap_unnormalized, heatmap_normalized
    return None
def get_white_contour(gray_image ):
    areas = []
    all_thresholds= [] 
    for threshold in range(70, 245):
        print(threshold)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY )
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
        rem_contours = []
        area = 0 
        for i, contour in enumerate(sorted_contours[:]):
            if cv2.contourArea(contour) >800:
                rem_contours.append(contour)
                area+= cv2.contourArea(contour)
              
        if len(rem_contours)== 20:
            areas.append(area)
            all_thresholds.append(threshold)

    if len(areas)>0: 
        areas=np.array(areas)
        idx= np.argmax(areas)
        threshold = all_thresholds[idx]
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY )
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=False)
        white_contours = []
    
        for i, contour in enumerate(sorted_contours[:]):
            if cv2.contourArea(contour) >800:
                white_contours.append(contour)
            
        return white_contours
    return None
        
def get_scoremap_with_contour(gradients, contours, scoremap, bulls_eye_coord , centers):
    scoremap_with_contour = []
    for i , obj in enumerate(scoremap[:-1]): 
        new_ob = obj
        if i > 0 : 
            low_grad = obj[0]['grad_avg']
            high_grad = obj[1]['grad_avg']
           
            for idx, g in enumerate(gradients):
                if low_grad <= g <= high_grad and (
                    get_quadrant(obj[0]['pt_in'] , bulls_eye_coord )== get_quadrant(centers[idx] , bulls_eye_coord )
                ) : 
                    cntr_dict =dict({'contour' :contours[idx]})
                    
                    new_ob = new_ob + (cntr_dict, )
                    
            scoremap_with_contour.append(new_ob)  
    
    
    new_ob = (scoremap[0][0:2]), (scoremap[-1][0:2]), scoremap[0][2]
    for idx, g in enumerate(gradients[2:]):
      
        if float('-inf')<g < scoremap[0][1]['grad_avg'] or \
            scoremap[-1][0]['grad_avg'] < g < float('inf') :
           
            cntr_dict =dict({'contour' :contours[idx+2]})
           
            new_ob = new_ob + (cntr_dict, )
            
    scoremap_with_contour.append(new_ob)     
    return scoremap_with_contour

import math 
def calculate_gradient(x1, y1, x2, y2):
    if x2 - x1 == 0:
        return float('inf')  # Return infinity for vertical lines
    else:
        return (y2 - y1) / (x2 - x1)

# Function to calculate gradient of each point to the center
def calculate_gradients(points, center):
    gradients = []
    xc, yc = center
    for point in points:
        x, y = point
        gradient = calculate_gradient(x, y, xc, yc)
        gradients.append(gradient)
    return gradients
def get_ring_mask(image, white_contours , black_contours , inner_be, outer_be, Cx, Cy):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Draw filled white polygons on the mask using the contours
    for contour in white_contours:
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
    for contour in black_contours:
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Invert the mask to have the areas inside contours black
    mask_inv = cv2.bitwise_not(mask)
    ringimg = image.copy()
    color_mask = [0, 0, 0]
    white_pixels = np.all(inner_be == [255, 255, 255], axis=-1)
    ringimg = np.where(white_pixels[..., np.newaxis], color_mask, ringimg)
    white_pixels = np.all(outer_be == [255, 255, 255], axis=-1)
    ringimg = np.where(white_pixels[..., np.newaxis], color_mask, ringimg)
    height, width, _ = image.shape
    center_x, center_y = width // 2, height // 2
    h,w,c =image.shape
    rect_start_row = Cy-40
    rect_end_row = Cy+ 40
    rect_start_col = Cx-40
    rect_end_col = Cx+40
    roi = np.zeros_like(ringimg[rect_start_row:rect_end_row, rect_start_col:rect_end_col])
    denoised_roi = cv2.medianBlur(roi.astype(np.uint8), 9)
    ringimg[rect_start_row:rect_end_row, rect_start_col:rect_end_col] = denoised_roi
    plt.imshow(ringimg)
    plt.show()
    ringimg[mask == 255] = [0, 0, 0]
    blurred = cv2.medianBlur(ringimg.astype(np.uint8), 3)


  
    polar_image = cv2.linearPolar(blurred, (550,250), 600, cv2.WARP_FILL_OUTLIERS)

    polar_image = polar_image.astype(np.uint8)
    plt.imshow(polar_image)
    plt.show()
    #get pixels ehre
    gray = cv2.cvtColor(polar_image, cv2.COLOR_BGR2GRAY)
    all_c1 = np.zeros_like(gray)
    all_c2 = np.zeros_like(gray)
    for row in range(h): 
        non_black_positions = np.where(gray[row] > 0)[0]
        data = non_black_positions
        sorted_data = np.sort(data)
        
        # Find the largest gap between consecutive elements
        max_gap = 0
        max_gap_index = 0
        
        for i in range(1, len(sorted_data)):
            gap = sorted_data[i] - sorted_data[i - 1]
            if gap > max_gap:
                max_gap = gap
                max_gap_index = i
        
        # Split the data at the position of the largest gap
        cluster_1 = sorted_data[:max_gap_index]
        cluster_2 = sorted_data[max_gap_index:]
        for col in cluster_1:
            all_c1[row, col] =255
        for col in cluster_2:
            all_c2[row, col] =255
    inner_ring_img = cv2.linearPolar(all_c1, (550,
                                                   250),
                                      600, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    outer_ring_img = cv2.linearPolar(all_c2, (550,
                                                   250),
                                      600, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
 
    return inner_ring_img , outer_ring_img
def get_pixel_map(scoremap_with_contour , image_height, image_widht, inner_be, outer_be):
    pixel_map  = np.zeros((image_height,image_widht))
    for item in scoremap_with_contour:
        score = item[2]
        for i in range(3 , len(item)):
            cmap =item[i]['contour']
            cv2.drawContours(pixel_map , [cmap] , -1 , score , -1)
    inner_gray_image = cv2.cvtColor(inner_be, cv2.COLOR_BGR2GRAY)
    outer_gray_image = cv2.cvtColor(outer_be, cv2.COLOR_BGR2GRAY)
    
    for r in range(image_height):
        for c in range(image_widht):
            if inner_gray_image[r,c] != 0 : 
                pixel_map[r,c] = 50
            if outer_gray_image[r,c] != 0 :
                pixel_map[r,c] = 25    
    
    return pixel_map

def get_multiplier_map(inner_cnt, outer_cnt, image_height, image_widht):
    multiplier_map = np.ones((image_height, image_widht))
    # for c in inner_cnt : 
    #     cv2.drawContours(multiplier_map , [c] , -1, 3 , -1)
    # for c in outer_cnt : 
    #     cv2.drawContours(multiplier_map , [c] , -1, 2 , -1)
    multiplier_map[inner_cnt ==255] =3
    multiplier_map[outer_cnt ==255] =2  
    return multiplier_map

def get_bearings(points , cx, cy ):
        # Adjusted coordinates with respect to the origin at (cX, cY)
    adjusted_points = [(cx-x, cy-y) for x, y in points]

    # Unzip the adjusted points
    adjusted_X, adjusted_Y = zip(*adjusted_points)
    #print(adjusted_X , adjusted_Y)
    results= []
    coords = []
    for x, y in  zip(adjusted_X, adjusted_Y):
        print( f' base angle for {x},{y} : {round(math.degrees(math.atan2(x,y)), 2)}')
        print(x,y)
        if x>=0 and y >=0: 
           
            angle= math.atan2(x,y)
            result_degrees = math.degrees(angle)
        elif x>0 and y <0:
            angle =  math.atan2(abs(y),x)
            result_degrees = math.degrees(angle) +90
        elif x<0 and y< 0:
            angle =  math.atan2(abs(y),abs(x))
            result_degrees = -90  - math.degrees(angle)
        elif x<0 and y>=0 :
            angle =  math.atan2(x, y)
            result_degrees = math.degrees(angle) 

        results.append(result_degrees)
        coords.append((x,y))

    return results , coords


def sorts_contour_with_bearings(black_contours, black_bearings, white_contours, white_bearings, black_coords, white_coords):
    black_lists = [(black_bearings[i], black_contours[i] , black_coords[i]) for i in range(len(black_bearings))]
    white_lists = [(white_bearings[i], white_contours[i], white_coords[i]) for i in range(len(white_bearings))]
    sorted_combined = sorted(black_lists, key=lambda x: x[0])

    # Extract the sorted elements into separate lists
    black_bearings_sorted = [x[0] for x in sorted_combined]
    black_contours_sorted = [x[1] for x in sorted_combined]
    black_coords_sorted = [x[2] for x in sorted_combined]
    sorted_combined = sorted(white_lists, key=lambda x: x[0])

    # Extract the sorted elements into separate lists
    white_bearings_sorted = [x[0] for x in sorted_combined]
    white_contours_sorted = [x[1] for x in sorted_combined]
    white_coords_sorted = [x[2] for x in sorted_combined]
    return black_contours_sorted, black_bearings_sorted , white_contours_sorted , white_bearings_sorted , black_coords_sorted , white_coords_sorted
def save_scoremap_with_contour(image, scoremap_with_contours, config , output_file_path ):
    out_im = np.copy(image)
    for obj in scoremap_with_contours:
        score, angle, contour, coord = obj
        if score in  config['WHITE_SECTORS']: 
            cv2.fillPoly(out_im, [contour], color=(0,255,255))
        else:
            cv2.fillPoly(out_im, [contour], color=(0,255,140))

        
        M = cv2.moments(contour)
        
        # Calculate contour center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0  # Avoid division by zero   
            
        cv2.putText(out_im,str(score), (cX , cY ),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        cv2.putText(out_im,str(round(angle,2)), (cX+20  , cY ),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        cv2.putText(out_im,f'{coord[0]} , {coord[1]}', (cX+20  , cY-10 ),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    plt.imsave(output_file_path , out_im)
    # plt.imshow(out_im)
    # plt.show()



def get_score_map(image, config, i):
    im = image.copy()
    min_area = 400
    threshold = 50
    brightness_level = get_brightness_level(image)
    if brightness_level == BRIGHTNESS_LEVEL.HIGH:
        threshold = 85
        min_area = 300
    elif brightness_level == BRIGHTNESS_LEVEL.MEDIUM:
        threshold = 65

    color_corrected_image = make_black(image, threshold)

    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye(image)
    cX, cY = get_center_coordinates(bullseye_image)

    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye_by_coordinates(color_corrected_image, cX, cY)
    cX, cY = get_center_coordinates(bullseye_image)

    output_img_red, output_img_green = get_red_green_masks(color_corrected_image)
    output_img_red = remove_small_contours(output_img_red, min_area)
    red_green_image = cv2.add(output_img_red, output_img_green)
    rings_image = crop_region(red_green_image, cY)
    notext_image = get_outer_border(rings_image, color_corrected_image, (cX, cY), threshold)
    inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt = get_inner_outer_rings(notext_image)
    detected_regions = cv2.add(cv2.add(bullseye_image, outer_ring), inner_ring)
    playing_ground = ROI_mask(outer_ring, outer_ring_cnt, -1, [cX, cY])
    oi_mask = ROI_mask(outer_ring, outer_ring_cnt, -2, [cX, cY])
    io_mask = ROI_mask(inner_ring, inner_ring_cnt, -1, [cX, cY])

    roi = get_ROI_image(image, oi_mask, io_mask)
    sectors = get_sector_gradients(roi, [cX, cY], inner_ring_cnt, outer_ring_cnt, config["THRESH_SAME_RING"], image, i)
    median_center = [cX, cY] #calculate_center_as_median_intersection_point(sectors, [cX, cY])
    cX, cY = median_center

    sectors_sorted = sorted(sectors, key=lambda x: x["grad_avg"])

    score_map = map_pts(sectors_sorted, config)
    
    black_bg_image , white_bg_image = get_dartboard_roi(im)

    output_img_red, output_img_green = get_red_green_masks(white_bg_image)
    output_img_red = remove_small_contours(output_img_red, min_area)
    red_green_image = cv2.add(output_img_red, output_img_green)
    
    

    inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt = get_inner_outer_rings(red_green_image)
    white_roi_im= subtract_roi(white_bg_image , inner_ring, outer_ring , inner_be ,outer_be ,is_white=True)
    gray_image = preproces_image(white_roi_im)
    
    black_bg_image, black_contours = get_black_contour(gray_image , black_bg_image)

    black_roi_im = subtract_roi(black_bg_image,inner_ring, outer_ring , inner_be ,outer_be ,is_white=False )
    gray_image = preproces_image(black_roi_im)
    blurred = cv2.medianBlur(gray_image, 9)
    white_contours = get_white_contour(blurred)
  


    black_centers = get_contours_centers(black_contours)
    black_bearings = get_bearings(black_centers , cX , cY)
   

    white_centers = get_contours_centers(white_contours)
    white_bearings = get_bearings(white_centers , cX ,cY)
   
    
    black_contours , black_bearings, white_contours, white_bearings= sorts_contour_with_bearings(black_contours , black_bearings, white_contours, white_bearings)
    scoremap_with_contour = [] 
    for contour, score, angle in zip(black_contours, config['BLACK_SECTORS'] , black_bearings ):
        obj  = (score, angle, contour )
        scoremap_with_contour.append(obj)
    for contour, score, angle in zip(white_contours, config['WHITE_SECTORS'] , white_bearings ):
        obj  = (score, angle, contour )
        scoremap_with_contour.append(obj)
    
    save_scoremap_with_contour(im , scoremap_with_contour, config)

    return bullseye_image, inner_ring, outer_ring, inner_be, outer_be, playing_ground, roi, score_map, sectors, output_img_red, output_img_green, cX, cY , pixel_map_all , multiplier_map


params = {
    0: {
        "THRESH_SAME_LINE": 0.10,
        "POINTS_POS": [20, 5, 12, 9, 14, 11, 8, 16, 7, 19],
        "POINTS_NEG": [1, 18, 4, 13, 6, 10, 15, 2, 17, 3],
        "RELEVANCE_POINTS": [20, 5, 12, 1, 18],
        "THRESH_SAME_RING": 70,
         "BLACK_SECTORS": [20,20,12,12,14,14,8,8,7,7,3,3,2,2,10,10,13,13,18,18],
         "WHITE_SECTORS": [5,5,9,9,11,11,16,16,19,19,17,17,15,15,6,6,4,4,1,1]
    },

    1: {
        "THRESH_SAME_LINE": 0.20,
        "POINTS_POS": [11, 8, 16, 7, 19, 3, 17, 2, 15, 10],
        "POINTS_NEG": [14, 9, 12, 5, 20, 1, 18, 4, 13],
        "RELEVANCE_POINTS": [11, 8, 16, 14, 9],
        "THRESH_SAME_RING": 70,
         "BLACK_SECTORS": [8,8,7,7,3,3,2,2,10,10,13,13,18,18,20,20,12,12,14,14],
         
    "WHITE_SECTORS": [11,11,16,16,19,19,17,17,15,15,6,6,4,4,1,1,5,5,9,9]
    },

    2: {
        "THRESH_SAME_LINE": 0.10,
        "POINTS_POS": [3, 17, 2, 15, 10, 6, 13, 4, 18, 1],
        "POINTS_NEG": [19, 7, 16, 8, 11, 14, 9, 12, 5, 20],
        "RELEVANCE_POINTS": [3, 17, 2, 19, 7],
        "THRESH_SAME_RING": 70,
         "BLACK_SECTORS": [3,3,2,2,10,10,13,13,18,18,20,20,12,12,14,14,8,8,7,7],
    "WHITE_SECTORS": [17,17,15,15,6,6,4,4,1,1,5,5,9,9,11,11,16,16,19,19]
    },

    3: {
        "THRESH_SAME_LINE": 0.19,
        "POINTS_POS": [6, 13, 4, 18, 1, 20, 5, 12, 9, 14],
        "POINTS_NEG": [10, 15, 2, 17, 3, 19, 7, 16, 8, 11],
        "RELEVANCE_POINTS": [6, 13, 4, 10, 15],
        "THRESH_SAME_RING": 70,
         "BLACK_SECTORS": [13,13,18,118,20,20,12,12,14,14,8,8,7,7,3,3,2,2,10,10],
    "WHITE_SECTORS":[6,6, 4,4,1,1,5,5,9,9,11,11,16,16,19,19,17,17,15,15]
    },

    "BLACK_RED": [20, 12, 14, 8, 7, 3, 2, 10, 13, 18],
    "WHITE_GREEN": [5, 9, 11, 16, 19, 17, 15, 6, 4, 1],
    
    "BLACK_SECTORS": [20, 18, 13, 10, 2, 3, 7, 8, 14, 12],
    "WHITE_SECTORS": [19, 16, 11, 9, 5, 1, 4, 6, 15, 17]
}

board_map = [
    {
        "IMAGE": None,
        "MULT_3": None,
        "MULT_2": None,
        "BULLSEYE": None,
        "SCORE_MAP": None,
        "CENTER": None,
        "INNER_BE": None,
        "PLAYING_GROUND": None,
        "ROI": None
    },
    {
        "IMAGE": None,
        "MULT_3": None,
        "MULT_2": None,
        "BULLSEYE": None,
        "SCORE_MAP": None,
        "CENTER": None,
        "INNER_BE": None,
        "PLAYING_GROUND": None,
        "ROI": None
    },
    {
        "IMAGE": None,
        "MULT_3": None,
        "MULT_2": None,
        "BULLSEYE": None,
        "SCORE_MAP": None,
        "CENTER": None,
        "INNER_BE": None,
        "PLAYING_GROUND": None,
        "ROI": None
    },
    {
        "IMAGE": None,
        "MULT_3": None,
        "MULT_2": None,
        "BULLSEYE": None,
        "SCORE_MAP": None,
        "CENTER": None,
        "INNER_BE": None,
        "PLAYING_GROUND": None,
        "ROI": None
    }
]

import re


def getDartCoordinates(string):
    for line in string.splitlines():
        if line.startswith('dart:'):
            # print(line)
            left_x = int(re.search('left_x: +[0-9]+', line).group(0)[7:])
            top_y = int(re.search('top_y: +[0-9]+', line).group(0)[6:])
            width = int(re.search('width: +[0-9]+', line).group(0)[6:])
            height = int(re.search('height: +[0-9]+', line).group(0)[7:])

            pt1 = (left_x, top_y + height)
            pt2 = (left_x + width, top_y + height)

            return int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2)

        # confirming map for 4 picture


def show_map(images_all_ol_wd, board_map):
    for it in [0, 1, 2, 3]:
        image_name = images_all_ol_wd[it]
        #     print(image_name)
        image = cv2.imread(image_name)
        img = np.zeros_like(image)

        ox = board_map[it]["CENTER"][0]
        oy = board_map[it]["CENTER"][1]

        #     for k in [0,1,2,3,4,5,6,7,8,9]:
        #     print(len(board_map[it]["SCORE_MAP"])-2)
        for k in range(0, len(board_map[it]["SCORE_MAP"]) - 1):
            # for k in [1]:
            m = board_map[it]["SCORE_MAP"][k][1]
            #         print(it,k,m)
            c = oy - (1.0 * m) * ox
            x1_ = ox
            y1_ = oy

            y2_1 = 767
            x2_1 = (int)((y2_1 - (1.0 * c)) / (1.0 * m))

            x2_2 = 1023
            y2_2 = (int)(1.0 * m * x2_2 + 1.0 * c)

            x2_3 = 0
            y2_3 = (int)(1.0 * c)

            #     print(x2_1,y2_1)
            #     print(x2_2,y2_2)
            #     print(x2_3,y2_3)

            if (x2_1 < 1024) & (x2_1 > 0):
                x2_ = x2_1
                y2_ = y2_1
            #         print(x2_1,y2_1)

            elif (y2_2 < 768) & (y2_2 > 0):
                x2_ = x2_2
                y2_ = y2_2
            #         print(x2_2,y2_2)

            elif (y2_3 < 768) & (y2_3 > 0):
                x2_ = x2_3
                y2_ = y2_3
            #         print(x2_3,y2_3)

            # y2_=(int)(1.0*m)*x2_
            #     print(x1_,y1_,x2_,y2_)
            x1 = ox
            y1 = oy
            x2 = x2_
            y2 = y2_
            a = (1.0 * (y2 - y1)) / (1.0 * (x2 - x1))
            b = -a * x1 + y1
            y1_f = (int)(0)
            y2_f = (int)(np.shape(img)[1])
            x1_f = (int)((y1_f - b) / a)
            x2_f = (int)((y2_f - b) / a)

            cv2.line(img, (x1_f, y1_f), (x2_f, y2_f), (100 + (k * 10), 100, 100 + (k * 10)), 3)

        img_tmp = img.copy()
        board_map_tmp = board_map[it].copy()
        img_2 = cv2.bitwise_or(img_tmp, board_map_tmp["MULT_3"])
        img_2 = cv2.bitwise_or(img_2, board_map_tmp["MULT_2"])
        img_2 = cv2.bitwise_or(img_2, board_map_tmp["BULLSEYE"])
        img_2 = cv2.bitwise_and(img_2, board_map_tmp["PLAYING_GROUND"])
        img_2 = cv2.bitwise_or(img_2, image)
        # img_2=cv2.bitwise_and(img_2),board_map_tmp["PLAYING_GROUND"]) 
        ImageUtils.render_image(img_2)
