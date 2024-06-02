import logging
import pickle
import json

import cv2

#from .darts_no_analyze_thread import DartsNoAnalyzeThread
from .dartboard_function import board_map
from .dartboard_function import get_score_map
from .dartboard_function import params


class AutoScorer:

    def __init__(self, board_id, board_images):
        self.board_id = board_id
        self.board_images = board_images
        self.board_map = board_map.copy()

    def set_board_images(self, board_images):
        self.board_images = board_images

    def load_board_map(self, board_map):
        self.board_map = board_map

        logging.info('Board map loaded')

    def calculate_board_map_and_cache(self):
        # logging.info('\nCreating Board map\n')
        for i in [0, 1, 2, 3]:
            #             print(images_all_ol_wd[i])
            # logging.info('\n' + i.__str__())
            try:
                print(self.board_images[i])
                image = cv2.imread(self.board_images[i])
                self.board_map[i]["IMAGE"] = image.copy()
                bullseye, mult_3, mult_2, inner_be, outer_be, pg, roi, score_map, sectors, output_img_red, output_img_green, cX, cY , pixel_map, multiplier_map = get_score_map(
                    image, params[i], i)

                if (not self.__valid_score_map__(score_map)) or not (self.__valid_sectors__(sectors)):
                    print(f"Camera {i}")
                    raise CalibrationException(i)

                # ImageUtils.render_image(inner_be)
                # ImageUtils.render_image(outer_be)
                self.board_map[i]["MULT_3"] = mult_3
                self.board_map[i]["MULT_2"] = mult_2
                self.board_map[i]["BULLSEYE"] = bullseye
                self.board_map[i]["SCORE_MAP"] = score_map
                self.board_map[i]["CENTER"] = [cX, cY]
                self.board_map[i]["INNER_BE"] = inner_be
                self.board_map[i]["OUTER_BE"] = outer_be
                self.board_map[i]["PLAYING_GROUND"] = pg
                self.board_map[i]["ROI"] = roi
                self.board_map[i]["RED_MASK"] = output_img_red
                self.board_map[i]["GREEN_MASK"] = output_img_green
                self.board_map[i]["SECTORS"] = sectors
                self.board_map[i]["PIXEL_MAP"] = pixel_map
                self.board_map[i]["MULTIPLIER_MAP"] = multiplier_map

                parts = self.board_images[i].split('/')
                parts.pop()
                calibration_id = parts[-1]
                calibration_folder_path = '/'.join(parts)
                self.board_map[i]["CALIBRATION_ID"] = calibration_id
                self.board_map[i]["CALIBRATION_FOLDER_PATH"] = calibration_folder_path
            except Exception:
                raise CalibrationException(i)

        # for image_path in self.board_images:
        #     os.remove(image_path)
        print(self.board_map)
        boardMapJsonFile = f"output/boardmap_json/{self.board_id}.json"
        f = open(boardMapJsonFile, 'wb')
        f.write(pickle.dumps(self.board_map))
        f.close()
        # with open(boardMapJsonFile, "w") as json_file:
        #     json.dump(self.board_map, json_file, indent=4)
        logging.info('Board created')

    def analyze(self, scorer_id, images_darts):
        results = []
        print('Running Detector on Cameras')
        threads = []
        for i in [0, 1, 2, 3]:
            image = images_darts[i]
            thread = DartsNoAnalyzeThread(image, self.board_map[i], i)
            threads.append(thread)
            thread.start()

        for idx, thread in enumerate(threads):
            thread.join()
            if not thread.result.success:
                raise AnalyzeException(idx, thread.result.error)

            results.extend(thread.result.value)

        results.insert(0, {"id": scorer_id})

        return results

    def __valid_score_map__(self, score_map):
        if len(score_map) < 10 or len(score_map) > 12:
            print("invalid scoremap")
            return False

        return True

    def __valid_sectors__(self, sectors):
        sectors_sorted_by_angle = sorted(sectors, key=lambda x: x["angle"])

        left_side_sectors = 0
        right_side_sectors = 0
        for sector in sectors_sorted_by_angle:
            if sector["angle"] < 90:
                left_side_sectors += 1
            else:
                right_side_sectors += 1

        i = 1
        first_angle_difference_area = abs(sectors_sorted_by_angle[0]["angle"] - sectors_sorted_by_angle[1]["angle"])
        if first_angle_difference_area > 40:
            print(f"first attempt invalid sector {first_angle_difference_area} sectors: {sectors_sorted_by_angle}")
            return False
        while i < (len(sectors_sorted_by_angle) - 1):
            second_angle_difference_area = abs(
                sectors_sorted_by_angle[i]["angle"] - sectors_sorted_by_angle[i + 1]["angle"]
            )

            if second_angle_difference_area > 40:
                print(f"invalid sector {second_angle_difference_area} sectors: {sectors_sorted_by_angle}")
                return False

            threshold = 5
            if i == 1:
                if right_side_sectors == 6:
                    threshold = 8
                else:
                    threshold = 6

            if i == (len(sectors_sorted_by_angle) - 2):
                if left_side_sectors == 6:
                    threshold = 8
                else:
                    threshold = 6

            if first_angle_difference_area > (second_angle_difference_area + threshold) and sectors_sorted_by_angle[i][
                "angle"] < 90:
                print(
                    f"invalid sector right side first_angle_diff: {first_angle_difference_area} second_angle_diff: {second_angle_difference_area} sector: {i} sectors: {sectors_sorted_by_angle}")
                return False
            if first_angle_difference_area < (second_angle_difference_area - threshold) and sectors_sorted_by_angle[i][
                "angle"] > 90:
                print(
                    f"invalid sector left side first_angle_diff: {first_angle_difference_area} second_angle_diff: {second_angle_difference_area} sector: {i} sectors: {sectors_sorted_by_angle}")
                return False

            first_angle_difference_area = second_angle_difference_area
            i += 1

        return True



class CalibrationException(Exception):
    def __init__(self, camera):
        super().__init__()
        self.camera = camera

class AnalyzeException(Exception):
    def __init__(self, camera, message):
        super().__init__()
        self.camera = camera
        self.message = message
