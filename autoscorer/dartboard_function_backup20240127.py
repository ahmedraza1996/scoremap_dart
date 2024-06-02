import json
import cv2
import matplotlib.pyplot as plt
import numpy as np


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
    radius = 50
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

    lower_green = np.array([50, 134, 50])
    upper_green = np.array([84, 255, 255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

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

    return output_image, inner_bullseye, outer_bullseye


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

    # lower_green = np.array([50, 134, 50])
    # upper_green = np.array([84, 255, 255])
    # mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    lower_green = np.array([50, 134, 50])  # np.array([50,134,50])
    upper_green = np.array([84, 255, 255])  # np.array([84,255,255])
    mask_green_1 = cv2.inRange(img_hsv, lower_green, upper_green)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask_green_2 = cv2.inRange(img_hsv, lower_green, upper_green)

    mask_green = mask_green_1 + mask_green_2

    # set my output img to zero everywhere except the green mask
    output_img_green = img.copy()
    output_img_green[np.where(mask_green == 0)] = 0

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

def get_outer_border(rings_image, image, center):
    mask = np.zeros_like(image)

    cv2.circle(mask, center, 360, (255, 255, 255), -1)
    cv2.rectangle(mask, (0, 0), (image.shape[1], center[1]), (0, 0, 0), -1)

    masked_image = cv2.bitwise_and(image, mask)

    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    ret, th = cv2.threshold(gray, 80, 255, 0)

    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = max(contours, key=cv2.contourArea)

    contour_mask = np.zeros_like(image)

    cv2.drawContours(contour_mask, [biggest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    cv2.drawContours(masked_image, [biggest_contour], -1, (0, 255, 0), 2)

    playing_area = cv2.bitwise_and(image.copy(), contour_mask)

    return cv2.bitwise_and(playing_area, rings_image)

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
    edges = cv2.Canny(roi, 150, 250)
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)

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
            grad1 = (pt_in[1] - center[1]) / (pt_in[0] - center[0])
            grad2 = (pt_out[1] - center[1]) / (pt_out[0] - center[0])
            grad_avg = (grad1 + grad2) / 2.0

            x1, y1 = pt_in
            x2, y2 = pt_out

            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            sectors.append((pt_in, pt_out, angle, grad_avg))

    median_center = center #calculate_center_as_median_intersection_point(sectors, center)

    angle_threshold = 4
    filtered_sectors = []

    for sector in sectors:
        pt_in, pt_out, angle, _ = sector

        is_unique = True
        for unique_sector in filtered_sectors:
            _, _, unique_angle, _ = unique_sector

            if abs(angle - unique_angle) < angle_threshold:
                is_unique = False
                break

        if is_unique:
            grad1 = (pt_in[1] - median_center[1]) / (pt_in[0] - median_center[0])
            grad2 = (pt_out[1] - median_center[1]) / (pt_out[0] - median_center[0])
            grad_avg = (grad1 + grad2) / 2.0

            x1, y1 = pt_in
            x2, y2 = pt_out

            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            filtered_sectors.append((pt_in, pt_out, angle, grad_avg))

            cv2.circle(image, pt_in, 4, (255, 0, 0), -1)
            cv2.circle(image, pt_out, 4, (0, 0, 255), -1)
            cv2.line(image, pt_in, pt_out, (0, 255, 0), 1)  # Draw the line in green with thickness 2



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


    file_path = f"/app/tmp/filtered_sectors_{i}.json"
    with open(file_path, 'w') as json_file:
        json.dump(filtered_sectors, json_file)

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


def map_pts(final_grads, config):
    #if not final_grads:  # Check if final_grads is empty
     #   return []  # Return an empty list or handle the error as needed

    points_pos = config["POINTS_POS"]  # [20,5,12,9,14,11,8,16,7,19]
    points_neg = config["POINTS_NEG"]  # [1,18,4,13,6,10,15,2,17,3]
    final_grads = np.array(final_grads)
    grad_pos = final_grads[final_grads > 0]
    grad_neg = final_grads[final_grads <= 0]
    grad_range = [(-np.inf, final_grads[0], points_pos[0]),
                  (final_grads[-1], np.inf, points_pos[0])]  # The 20 point sector
    for i in range(1, len(grad_neg)):
        grad_range.append((grad_neg[i - 1], grad_neg[i], points_neg[i - 1]))
    ind = 1
    for i in range(len(grad_pos) - 1, 0, -1):
        grad_range.append((grad_pos[i - 1], grad_pos[i], points_pos[ind]))
        ind += 1

    return sorted(grad_range, key=lambda x: (x[0], -x[1]))


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


def get_score_map(image, config, i):
    bullseye_image, inner_be, outer_be = get_inner_outer_bullseye(image)

    color_corrected_image = make_black(image, 85)

    cX, cY = get_center_coordinates(bullseye_image)

    output_img_red, output_img_green = get_red_green_masks(color_corrected_image)
    red_green_image = cv2.add(output_img_red, output_img_green)
    rings_image = crop_region(red_green_image, cY)
    notext_image = get_outer_border(rings_image, color_corrected_image, (cX, cY))
    inner_ring, outer_ring, inner_ring_cnt, outer_ring_cnt = get_inner_outer_rings(notext_image)
    detected_regions = cv2.add(cv2.add(bullseye_image, outer_ring), inner_ring)
    playing_ground = ROI_mask(outer_ring, outer_ring_cnt, -1, [cX, cY])
    oi_mask = ROI_mask(outer_ring, outer_ring_cnt, -2, [cX, cY])
    io_mask = ROI_mask(inner_ring, inner_ring_cnt, -1, [cX, cY])

    roi = get_ROI_image(image, oi_mask, io_mask)
    sectors = get_sector_gradients(roi, [cX, cY], inner_ring_cnt, outer_ring_cnt, config["THRESH_SAME_RING"], image, i)
    median_center = [cX, cY] #calculate_center_as_median_intersection_point(sectors, [cX, cY])
    cX, cY = median_center

    grads = [item[3] for item in sectors]

    grads_sorted = sorted(grads)

    score_map = map_pts(grads_sorted, config)

    if i == 2:
        cv2.imwrite('/app/tmp/images/image.jpg', image)
        cv2.imwrite('/app/tmp/images/inner_be.jpg', inner_be)
        cv2.imwrite('/app/tmp/images/outer_be.jpg', outer_be)

        cv2.imwrite('/app/tmp/images/inner_ring.jpg', inner_ring)
        cv2.imwrite('/app/tmp/images/outer_ring.jpg', outer_ring)

        cv2.imwrite('/app/tmp/images/bullseye_image.jpg', bullseye_image)

        cv2.imwrite('/app/tmp/images/output_img_red.jpg', output_img_red)
        cv2.imwrite('/app/tmp/images/output_img_green.jpg', output_img_green)
        cv2.imwrite('/app/tmp/images/red_green_image.jpg', red_green_image)
        cv2.imwrite('/app/tmp/images/rings_image.jpg', rings_image)
        cv2.imwrite('/app/tmp/images/notext_image.jpg', notext_image)
        cv2.imwrite('/app/tmp/images/detected_regions.jpg', detected_regions)
        cv2.imwrite('/app/tmp/images/playing_ground.jpg', playing_ground)
        cv2.imwrite('/app/tmp/images/oi_mask.jpg', oi_mask)
        cv2.imwrite('/app/tmp/images/io_mask.jpg', io_mask)
        cv2.imwrite('/app/tmp/images/roi.jpg', roi)

    file_path = f"/app/tmp/score_map_{i}.json"
    with open(file_path, 'w') as json_file:
        json.dump(score_map, json_file)

    return bullseye_image, inner_ring, outer_ring, inner_be, outer_be, playing_ground, score_map, cX, cY


params = {
    0: {
        "THRESH_SAME_LINE": 0.10,
        "POINTS_POS": [20, 5, 12, 9, 14, 11, 8, 16, 7, 19],
        "POINTS_NEG": [1, 18, 4, 13, 6, 10, 15, 2, 17, 3],
        "THRESH_SAME_RING": 80,
    },

    1: {
        "THRESH_SAME_LINE": 0.20,
        "POINTS_POS": [11, 8, 16, 7, 19, 3, 17, 2, 15, 10],
        "POINTS_NEG": [14, 9, 12, 5, 20, 1, 18, 4, 13],
        "THRESH_SAME_RING": 80,
    },

    2: {
        "THRESH_SAME_LINE": 0.10,
        "POINTS_POS": [3, 17, 2, 15, 10, 6, 13, 4, 18, 1],
        "POINTS_NEG": [19, 7, 16, 8, 11, 14, 9, 12, 5, 20],
        "THRESH_SAME_RING": 80,
    },

    3: {
        "THRESH_SAME_LINE": 0.19,
        "POINTS_POS": [6, 13, 4, 18, 1, 20, 5, 12, 9, 14],
        "POINTS_NEG": [10, 15, 2, 17, 3, 19, 7, 16, 8, 11],
        "THRESH_SAME_RING": 80,
    }
}

board_map = [
    {
        "MULT_3": None,
        "MULT_2": None,
        "BULLSEYE": None,
        "SCORE_MAP": None,
        "CENTER": None,
        "INNER_BE": None,
        "PLAYING_GROUND": None,
    },
    {
        "MULT_3": None,
        "MULT_2": None,
        "BULLSEYE": None,
        "SCORE_MAP": None,
        "CENTER": None,
        "INNER_BE": None,
        "PLAYING_GROUND": None,
    },
    {
        "MULT_3": None,
        "MULT_2": None,
        "BULLSEYE": None,
        "SCORE_MAP": None,
        "CENTER": None,
        "INNER_BE": None,
        "PLAYING_GROUND": None,
    },
    {
        "MULT_3": None,
        "MULT_2": None,
        "BULLSEYE": None,
        "SCORE_MAP": None,
        "CENTER": None,
        "INNER_BE": None,
        "PLAYING_GROUND": None,
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
