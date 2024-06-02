import logging
import threading
import time

import cv2
import numpy as np

#from .autoscorer import ImageUtils, BoardCalculatorHelper, slope, params

from integrations.darknet_integration import darknet


class Darknet:
    def __init__(self):
      cfg_file = "object-detection/yolov3.cfg"
      names_file = "object-detection/obj.data"
      weights_file = "object-detection/backup/yolov3_last.weights"

      # First thing we do is load the neural network.
      self.network, self.class_names, self.colours, self.metadata = darknet.load_network(cfg_file, names_file, weights_file)
      self.width = darknet.network_width(self.network)
      self.height = darknet.network_height(self.network)

      self.prediction_threshold = 0.25


    def process(self, image_name):
        image_bgr = cv2.imread(image_name)
        darknet_image = darknet.convert_cv2_image2darknet_image(image_bgr)
        resized_image = darknet.resize_image(darknet_image, self.width, self.height)

        detections = darknet.detect_image(self.network, self.class_names, resized_image,
                                          darknet.ImageDimension(width=darknet_image.w, height=darknet_image.h),
                                          thresh=self.prediction_threshold)

        darknet.free_image(darknet_image)
        darknet.free_image(resized_image)

        return detections
    
class DartsNoAnalyzeThread(threading.Thread):
    #pass image path, boardmap object, and cam no.
    def __init__(self, image, board_map, no_image):
        super().__init__()
        start_time = time.time()
        #initializeyolo v8 here
        self.darknet_process = Darknet()
        logging.info(f"Required time for get Darknet instance: {(time.time() - start_time):.2f} s")

        self.image = image
        self.no_image = no_image
        self.board_map = board_map
        self.result = None

    def run(self):
      start_time = time.time()
      tip, a, b, c, d, e, f, g, h, j = self.tip_detector(self.image)
      logging.info(f"Required time for TIP DETECTOR: {(time.time() - start_time):.2f} s")

      translation_x, translation_y = ImageUtils.compare_images(self.image, self.board_map["IMAGE"])
      translation_x, translation_y = round(translation_x), round(translation_y)
      translation = (translation_x, translation_y)

      if abs(translation_x) > 10 or abs(translation_y) > 10:
          self.result ="Board has moved too much" #Result.fail("The board has been moved too much compared to the calibration")
          return

      state_data = {
          'Cam': self.no_image,
          'tips': tip,
          'dx1': a,
          'dy1': b,
          'c1': c,
          'dx2': d,
          'dy2': e,
          'c2': f,
          'dx3': g,
          'dy3': h,
          'c3': j
      }
      data = self.__normalize_points_in_result__(translation, state_data)
      self.result = self.__calculate_result_for_all_darts__(data)

    def tip_detector(self, image_name):
      # use yolo model hereee
      detections = self.darknet_process.process(image_name)
      #DarknetManager.release_instance(self.darknet_process)

      var = self.parse_data(detections)

      if var is None:
          no_darts = 0
      else:
          no_darts = len(var)
      a = 0
      b = 0
      c = 0
      d = 0
      e = 0
      f = 0
      g = 0
      h = 0
      i = 0
      if no_darts == 1:
          a = var[0][0]
          b = var[0][1]
          c = var[0][2]
      if no_darts == 2:
          a = var[0][0]
          b = var[0][1]
          c = var[0][2]
          d = var[1][0]
          e = var[1][1]
          f = var[1][2]
      if no_darts == 3:
          a = var[0][0]
          b = var[0][1]
          c = var[0][2]
          d = var[1][0]
          e = var[1][1]
          f = var[1][2]
          g = var[2][0]
          h = var[2][1]
          i = var[2][2]
      return no_darts, a, b, c, d, e, f, g, h, i


    def parse_data(self, detections):
        matrix = []

        for detection in detections:
            percentage = float(detection[1])

            left_x = int(detection[2][0])
            top_y = int(detection[2][1])
            width = int(detection[2][2])
            height = int(detection[2][3])


            matrix.append([round(left_x + width / 2), round(top_y + height), percentage])

        return matrix


    def __calculate_result_for_all_darts__(self, data):
        results = None

        if data["tips"] == 0:
            results = [
                self.__create_result_dict__(1),
                self.__create_result_dict__(2),
                self.__create_result_dict__(3),
            ]

        elif data["tips"] == 1:
            result_1 = self.__calculate_result__(1, data["dx1"], data["dy1"], data["c1"])
            results = [
                result_1,
                self.__create_result_dict__(2),
                self.__create_result_dict__(3),
            ]

        elif data["tips"] == 2:
            result_1 = self.__calculate_result__(1, data["dx1"], data["dy1"], data["c1"])
            result_2 = self.__calculate_result__(2, data["dx2"], data["dy2"], data["c2"])

            results = [
                result_1,
                result_2,
                self.__create_result_dict__(3),
            ]

        elif data["tips"] >= 3:
            result_1 = self.__calculate_result__(1, data["dx1"], data["dy1"], data["c1"])
            result_2 = self.__calculate_result__(2, data["dx2"], data["dy2"], data["c2"])
            result_3 = self.__calculate_result__(3, data["dx3"], data["dy3"], data["c3"])

            results = [
                result_1,
                result_2,
                result_3
            ]

        return results


    def __normalize_points_in_result__(self, translation, result):
        translation_x, translation_y = translation
        if translation_x == 0 and translation_y == 0:
            return result

        dx1 = result["dx1"]
        dy1 = result["dy1"]
        if dx1 != 0 and dy1 != 0:
            result["dx1"], result["dy1"] = dx1 + translation_x, dy1 + translation_y
            # result.dx1[0], result.dy1[0] = self.__refine_tip_coordinates_to_lowest_point__(self.board_map[idx]["IMAGE"], image_path, dx1, dy1)

        dx2 = result["dx2"]
        dy2 = result["dy2"]
        if dx2 != 0 and dy2 != 0:
            result["dx2"], result["dy2"] = dx2 + translation_x, dy2 + translation_y
            # result.dx2[0], result.dy2[0] = self.__refine_tip_coordinates_to_lowest_point__(self.board_map[idx]["IMAGE"],
            #                                                                                image_path, dx2, dy2)

        dx3 = result["dx3"]
        dy3 = result["dy3"]
        if dx3 != 0 and dy3 != 0:
            result["dx3"], result["dy3"] = dx3 + translation_x, dy3 + translation_y
            # result.dx3[0], result.dy3[0] = self.__refine_tip_coordinates_to_lowest_point__(self.board_map[idx]["IMAGE"],
            #                                                                                image_path, dx3, dy3)

        return result

    def __calculate_score__(self, position):
        point_image = BoardCalculatorHelper.create_point_image(self.image, position)
        dX, dY = position
        board_map_tmp = self.board_map.copy()
        board_map_tmp["PLAYING_GROUND"] = cv2.bitwise_or(board_map_tmp["PLAYING_GROUND"], board_map_tmp["BULLSEYE"])

        corrected_image_2 = np.zeros_like(point_image.copy())
        cv2.line(corrected_image_2, (dX, dY), (dX, dY - 2), (255, 255, 255), 1)

        multiplier = 1
        in_playing_region = np.count_nonzero(cv2.bitwise_and(point_image.copy(), board_map_tmp["PLAYING_GROUND"]))
        if in_playing_region == 0:
            if not (ImageUtils.color_of_pixel_normalized(board_map_tmp["IMAGE"], board_map_tmp["RED_MASK"],
                                                       board_map_tmp["GREEN_MASK"], dX,
                                                       dY) == "white" and np.count_nonzero(
                cv2.bitwise_and(corrected_image_2.copy(), board_map_tmp["PLAYING_GROUND"]))):
                if dY > board_map_tmp["CENTER"][1]:
                    return 0, -1, 1

                return 0, 0, 1

            multiplier = 2

        corrected_image_3 = np.zeros_like(point_image.copy())
        cv2.line(corrected_image_3, (dX, dY), (dX, dY - 3), (255, 255, 255), 1)
        corrected_image_1 = np.zeros_like(point_image.copy())
        cv2.circle(corrected_image_1, (dX, dY - 1), 0, (255, 255, 255), -1)

        overlap_inner_bullseye = np.count_nonzero(cv2.bitwise_and(point_image.copy(), board_map_tmp["INNER_BE"]))
        if overlap_inner_bullseye > 0:
            return 100, 50, 1
        overlap_inner_bullseye = np.count_nonzero(cv2.bitwise_and(corrected_image_2, board_map_tmp["INNER_BE"]))
        if overlap_inner_bullseye > 0:
            return 50, 50, 1

        overlap_outer_bullseye = np.count_nonzero(cv2.bitwise_and(point_image.copy(), board_map_tmp["OUTER_BE"]))
        if overlap_outer_bullseye > 0:
            return 100, 25, 1
        overlap_outer_bullseye = np.count_nonzero(cv2.bitwise_and(corrected_image_2, board_map_tmp["OUTER_BE"]))
        if overlap_outer_bullseye > 0:
            return 50, 25, 1

        confidence_boost = None
        zone_3x = BoardCalculatorHelper.detect_3x_multiplication_zone(board_map_tmp, dX, dY)
        if zone_3x["is_detected"]:
            multiplier = 3
            confidence_boost = zone_3x["confidence_boost"]

        elif BoardCalculatorHelper.is_in_2x_multiplication_zone(board_map_tmp, dX, dY):
            multiplier = 2

        cv2.line(point_image, tuple(board_map_tmp["CENTER"]), (dX, dY), (255, 255, 255), 1)
        # ImageUtils.render_image(img)
        if dX == board_map_tmp["CENTER"][0]:
            return 100, multiplier * params[self.no_image]["POINTS_POS"][0], multiplier

        slope_current = slope(board_map_tmp["CENTER"], [dX, dY])

        # if point is left from center it needs to have negative slope - e.g. throw-5
        # or if points is right from center it needs to have positive slope
        if (
                (slope_current > 0 and dX < board_map_tmp["CENTER"][0]) or
                (slope_current < 0 and dX > board_map_tmp["CENTER"][0])
        ):
            slope_current = -1 * slope_current

        for score in board_map_tmp["SCORE_MAP"]:
            lower_sector = score[0]
            upper_sector = score[1]
            score_value = score[2]

            if slope_current >= lower_sector["grad_avg"] and slope_current <= upper_sector["grad_avg"]:
                point_sector = None
                slope_range_threshold = 0.1 if abs(upper_sector["grad_avg"]) - abs(
                    lower_sector["grad_avg"]) >= 0.25 else 0.05

                if BoardCalculatorHelper.is_slope_between_sector_grads(slope_current, lower_sector, slope_range_threshold):
                    point_sector = lower_sector
                if point_sector is None and BoardCalculatorHelper.is_slope_between_sector_grads(slope_current, upper_sector,
                                                                                   slope_range_threshold):
                    point_sector = upper_sector

                if point_sector is not None:
                    alternative_score = [tup for tup in board_map_tmp["SCORE_MAP"] if (
                            tup[0]['grad_avg'] == point_sector["grad_avg"] or tup[1]['grad_avg'] == point_sector[
                        "grad_avg"]) and tup[2] != score_value]

                    color = ImageUtils.color_of_pixel_normalized(board_map_tmp["IMAGE"], board_map_tmp["RED_MASK"],
                                                               board_map_tmp["GREEN_MASK"], dX, dY)

                    if len(alternative_score) == 1 and (
                            (color in ["red", "black"] and score_value not in params["BLACK_RED"]) or
                            (color in ["green"] and score_value not in params["WHITE_GREEN"])):
                        score_value = alternative_score[0][2]

                    elif color == "white":
                        colors = ImageUtils.above_below_pixel_with_nonwhite_color(board_map_tmp["IMAGE"],
                                                                                board_map_tmp["RED_MASK"],
                                                                                board_map_tmp["GREEN_MASK"], dX, dY, 5)

                        if (
                                len(alternative_score) == 1 and len(colors) == 2 and
                                BoardCalculatorHelper.check_if_list_have_combination_of_two_colors(
                                    [colors[0]["color"], colors[1]["color"]], ["red", "black"]
                                ) and
                                alternative_score[0][2] in params["BLACK_RED"]
                        ):
                            score_value = alternative_score[0][2]
                        elif (
                                len(alternative_score) == 1 and len(colors) == 2 and
                                BoardCalculatorHelper.check_if_list_have_combination_of_two_colors(
                                    [colors[0]["color"], colors[1]["color"]], ["green", "white"]
                                ) and
                                alternative_score[0][2] in params["WHITE_GREEN"]
                        ):
                            cX = board_map_tmp["CENTER"][0]
                            if dX < cX:
                                colors = ImageUtils.above_below_pixel_with_nonwhite_color(board_map_tmp["IMAGE"],
                                                                                        board_map_tmp["RED_MASK"],
                                                                                        board_map_tmp["GREEN_MASK"],
                                                                                        dX - 2, dY, 5)
                            else:
                                colors = ImageUtils.above_below_pixel_with_nonwhite_color(board_map_tmp["IMAGE"],
                                                                                        board_map_tmp["RED_MASK"],
                                                                                        board_map_tmp["GREEN_MASK"],
                                                                                        dX + 2, dY, 5)

                            if not BoardCalculatorHelper.check_if_list_have_combination_of_two_colors(
                                    [colors[0]["color"], colors[1]["color"]], ["red", "green"]
                            ):
                                score_value = alternative_score[0][2]
                        elif (
                                len(alternative_score) == 1 and len(colors) == 2 and
                                BoardCalculatorHelper.check_if_list_have_combination_of_two_colors(
                                    [colors[0]["color"], colors[1]["color"]], ["black", "white"]
                                ) and
                                alternative_score[0][2] in params["WHITE_GREEN"]
                        ):
                            cX = board_map_tmp["CENTER"][0]
                            above_pixel = colors[0]
                            if (
                                    (dX < cX and above_pixel["color"] == "white" and alternative_score[0][2] in params[
                                        "WHITE_GREEN"]) or
                                    (dX > cX and above_pixel["color"] == "black" and alternative_score[0][2] in params[
                                        "BLACK_RED"])
                            ):
                                score_value = alternative_score[0][2]
                        else:
                            cX = board_map_tmp["CENTER"][0]
                            if dX < cX:
                                colors = ImageUtils.above_below_pixel_with_nonwhite_color(board_map_tmp["IMAGE"],
                                                                                        board_map_tmp["RED_MASK"],
                                                                                        board_map_tmp["GREEN_MASK"],
                                                                                        dX + 2, dY, 5)
                            else:
                                colors = ImageUtils.above_below_pixel_with_nonwhite_color(board_map_tmp["IMAGE"],
                                                                                        board_map_tmp["RED_MASK"],
                                                                                        board_map_tmp["GREEN_MASK"],
                                                                                        dX - 2, dY, 5)

                            if (
                                    len(alternative_score) == 1 and len(colors) == 2 and
                                    BoardCalculatorHelper.check_if_list_have_combination_of_two_colors(
                                        [colors[0]["color"], colors[1]["color"]], ["red", "black"]
                                    ) and
                                    alternative_score[0][2] in params["BLACK_RED"]
                            ):
                                score_value = alternative_score[0][2]
                                multiplier = 3
                                if BoardCalculatorHelper.is_circle_in_zone(board_map_tmp["MULT_2"], dX, dY, 10):
                                    multiplier = 2

                if confidence_boost is not None:
                    return confidence_boost, multiplier * score_value, multiplier

                upper_limit_pixel_x, upper_limit_pixel_y = ImageUtils.find_lowest_white_pixel(board_map_tmp["OUTER_BE"])
                upper_limit_pixel = (upper_limit_pixel_x, upper_limit_pixel_y + 5)

                if dY < upper_limit_pixel[1]:
                    return 25, multiplier * score_value, multiplier

                elif score_value in params[self.no_image]["RELEVANCE_POINTS"]:
                    return 100, multiplier * score_value, multiplier

                return 50, multiplier * score_value, multiplier

        return 0, 0, 1

    def __calculate_result__(self, darts, dx=0, dy=0, c=0.0):
        boost_confidence, score, multiplier = self.__calculate_score__((dx, dy))
        if boost_confidence is not None:
            c += boost_confidence
        return self.__create_result_dict__(darts, dx, dy, c, score, multiplier)


    def __create_result_dict__(self, darts, dx=0, dy=0, c=0.0, score=0, multiplier=1):
        return {
            "Cam": self.no_image,
            "darts": darts,
            "dx": dx,
            "dy": dy,
            "c": c,
            "score": score,
            "multiplier": multiplier,
            "base_score": (score/multiplier)
        }

