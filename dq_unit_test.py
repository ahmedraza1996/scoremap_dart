import unittest
import os

import cv2
from dotenv import load_dotenv
from datetime import datetime
import json

from integrations.google_integration import GoogleSheetsService

import requests
from autoscorer.autoscorer import AutoScorer


EDGE_NUMBERS = {
    0: [13, 6, 14, 11],
    1: [5, 20, 19, 3],
    2: [8, 11, 10, 6],
    3: [17, 3, 1, 20]
}



class DartResultsHelper:

    @staticmethod
    def count_darts_detected(json_data):
        cam0 = 0
        cam1 = 0
        cam2 = 0
        cam3 = 0

        # Iterate through the "result" array in the JSON object
        for item in json_data:
            if "Cam" in item:
                cam_num = item["Cam"]
                if item["dx"] != 0 or item["dy"] != 0:
                    if cam_num == 0:
                        cam0 += 1
                    elif cam_num == 1:
                        cam1 += 1
                    elif cam_num == 2:
                        cam2 += 1
                    elif cam_num == 3:
                        cam3 += 1

        return cam0, cam1, cam2, cam3

    @staticmethod
    def get_scores(json_data):
        dart1 = {"c": 0, "score": 0}
        dart2 = {"c": 0, "score": 0}
        dart3 = {"c": 0, "score": 0}

        for obj in json_data:
            if "c" in obj:
                if obj["darts"] == 1 and obj["score"] != 0:
                    if dart1["c"] < obj["c"]:
                        dart1 = obj
                if obj["darts"] == 2 and obj["score"] != 0:
                    if dart2["c"] < obj["c"]:
                        dart2 = obj
                if obj["darts"] == 3 and obj["score"] != 0:
                    if dart3["c"] < obj["c"]:
                        dart3 = obj

        return dart1["score"], dart2["score"], dart3["score"]



class AnalyzeTestSet:
    def test_analyze_test_set(self):
        load_dotenv()

        IMAGES_SERVER_URL = "https://autoscoring-image-viewer.dartsorakel.com"
        CALIBRATION_IMAGE_SERVER_ROOT = f"{IMAGES_SERVER_URL}/board_calibration"
        DARTS_IMAGE_SERVER_ROOT = f"{IMAGES_SERVER_URL}/auto_scorer"

        GOOGLE_JSON_KEYFILE_LOCATION = "/app/google_client_secret.json"
        SPREADSHEET_ID = "1kkOoQAsDHgnK0bdWh2wbQQqjezYEsXStZ4TsxsnlPXo"
        BASE_SET_SHEET_NAME = "Base set with three darts"
        NEW_SHEET_TIMESTAMP_NAME = f"THREE_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        spreadsheet_service = GoogleSheetsService(json_keyfile_name=GOOGLE_JSON_KEYFILE_LOCATION)

        spreadsheet_service.copy_sheet(SPREADSHEET_ID, BASE_SET_SHEET_NAME, NEW_SHEET_TIMESTAMP_NAME)

        sheet_data = spreadsheet_service.read_sheet(SPREADSHEET_ID, NEW_SHEET_TIMESTAMP_NAME)

        values = sheet_data.get("values", [])

        if not values:
            print("No data found.")

        headers = values[0]
        working_set = values[1:][:1254]
        updated_working_set = []
        row_num = 1

        for row in working_set:
            board_id = row[headers.index("Board ID")]
            calibration_id = row[headers.index("Calibration ID")]
            throw_id = row[headers.index("Throw ID")]
            correct_value = row[headers.index("Correct Value")]
            print(f"=============================     [{row_num}] Throw {throw_id}     =============================")

            row_num += 1
            print(f">>>{row}<<<")
            previous_state = row[headers.index("Previous State")] if 0 <= headers.index("Previous State") < len(row) else ""

            calibration_folder_path = f"/app/tmp/board_calibration/{calibration_id}"
            dart_images_folder_path = f"/app/tmp/auto_scorer/{throw_id}"

            # Check and create folders if they don't exist
            self.create_folder_if_not_exists(calibration_folder_path)
            self.create_folder_if_not_exists(dart_images_folder_path)

            calibration_image_paths = []
            dart_images = []

            for i in range(4):
                calibration_image_file_name = f"{calibration_folder_path}/{i}.jpg"
                dart_image_file_name = f"{dart_images_folder_path}/{i}.jpg"
                calibration_image_paths.append(calibration_image_file_name)
                dart_images.append(cv2.imread(dart_image_file_name))

                # Download images if they don't exist
                if not os.path.exists(calibration_image_file_name):
                    calibration_image_url = f"{CALIBRATION_IMAGE_SERVER_ROOT}/{calibration_id}/{i}.jpg"
                    self.download_image(calibration_image_url, calibration_image_file_name)

                if not os.path.exists(dart_image_file_name):
                    dart_image_url = f"{DARTS_IMAGE_SERVER_ROOT}/{throw_id}/{i}.jpg"
                    self.download_image(dart_image_url, dart_image_file_name)

            auto_scorer = AutoScorer(board_id, calibration_image_paths)
            auto_scorer.calculate_board_map_and_cache()
            if previous_state == "":
                result = auto_scorer.analyze(throw_id, dart_images)
                updated_working_set.append(self.prepare_row_data(board_id, throw_id, calibration_id, result, correct_value))
            else:
                result = auto_scorer.analyze(throw_id, dart_images)
                results = AnalyzeTestSet.find_all_result_from_parents(headers, updated_working_set, row[headers.index("Previous State")], [result])

                dart1_score = results[0]
                dart2_score = results[1] if len(results) >= 2 else None
                dart3_score = results[2] if len(results) == 3 else None

                detected_value = self.get_autoscoring_result(dart1_score, dart2_score, dart3_score)
                updated_working_set.append(self.prepare_row_data(board_id, throw_id, calibration_id, result, correct_value, previous_state, detected_value))

        spreadsheet_service = GoogleSheetsService(json_keyfile_name=GOOGLE_JSON_KEYFILE_LOCATION)
        spreadsheet_service.update(SPREADSHEET_ID, updated_working_set, "A2")


    @staticmethod
    def find_all_result_from_parents(headers, working_set, throw_id, results):
        raw_output_index = headers.index("Raw output")

        for row in working_set:
            if row[headers.index("Throw ID")] == throw_id:
                results.insert(0, json.loads(row[raw_output_index]))
                if row[headers.index("Previous State")] == "":
                    return results

                return AnalyzeTestSet.find_all_result_from_parents(headers, working_set, row[headers.index("Previous State")], results)

        return results


    @staticmethod
    def prepare_row_data(
            board_id,
            throw_id,
            calibration_id,
            result,
            correct_value,
            previous_state = None,
            detected_value = None
    ):
        cam0_detected_darts, cam1_detected_darts, cam2_detected_darts, cam3_detected_darts = DartResultsHelper.count_darts_detected(result)
        throw_folder_url = f"https://autoscoring-image-viewer.dartsorakel.com/auto_scorer/{throw_id}"
        board_folder_url = f"https://autoscoring-image-viewer.dartsorakel.com/board_calibration/{calibration_id}"
        if detected_value == None:
            dart1_score, dart2_score, dart3_score = DartResultsHelper.get_scores(result)
            detected_value = dart1_score if dart1_score != -1 else 0

        is_correct = "Yes" if int(detected_value) == int(correct_value) else "No"


        return [
            board_id,
            throw_id,
            calibration_id,
            cam0_detected_darts,
            cam1_detected_darts,
            cam2_detected_darts,
            cam3_detected_darts,
            detected_value,
            correct_value,
            is_correct,
            previous_state if previous_state is not None else "",
            json.dumps(result),
            f"{throw_folder_url}/0.jpg",
            f"{throw_folder_url}/1.jpg",
            f"{throw_folder_url}/2.jpg",
            f"{throw_folder_url}/3.jpg",
            f"{board_folder_url}/0.jpg",
            f"{board_folder_url}/1.jpg",
            f"{board_folder_url}/2.jpg",
            f"{board_folder_url}/3.jpg",
        ]

    def download_image(self, url, save_path):
        print(f"Downloading image from {url}...")
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Host": "autoscoring-image-viewer.dartsorakel.com",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        }
        session = requests.Session()
        response = session.get(url, headers=headers, allow_redirects=True, stream=True, timeout=30)

        if response.status_code == 200:
            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=128):
                    file.write(chunk)
            print(f"Downloaded image: {save_path}")
        else:
            print(f"Failed to download image from {url}. Status code: {response.status_code}")

    def create_folder_if_not_exists(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def get_autoscoring_result(self, dart1_payload, dart2_payload, dart3_payload):
        darts_score = {"1": [], "2": [], "3": []}

        if dart1_payload is not None:
            darts_score["1"].extend([
                el
                for el in dart1_payload
                if el.get("Cam") is not None and el.get("dx") != 0 and el.get("dy") != 0
            ])

        if dart2_payload is not None:
            filtered_indexes = []
            filtered_result = []
            score2_objects = [
                el
                for el in dart2_payload
                if el.get("Cam") is not None and el.get("dx") != 0 and el.get("dy") != 0
            ]
            for detected_result in darts_score["1"]:
                tolerance = 6

                detected_values = []
                detected_indexes = []
                not_detected = True
                for i, result in enumerate(score2_objects):
                    if result["Cam"] == detected_result["Cam"] and self.coordinates_match(result, detected_result,
                                                                                     tolerance):
                        not_detected = False

                        detected_values.append(result)
                        if i not in detected_indexes:
                            detected_indexes.append(i)

                if len(detected_values) > 1:
                    for i, result in enumerate(score2_objects):
                        if result["Cam"] == detected_result["Cam"] and self.coordinates_match(result, detected_result):
                            not_detected = False

                            filtered_result.append(result)
                            if i not in filtered_indexes:
                                filtered_indexes.append(i)

                elif len(detected_values) == 1:
                    not_detected = False
                    filtered_result.append(detected_values[0])
                    if detected_indexes[0] not in filtered_indexes:
                        filtered_indexes.append(detected_indexes[0])

                if not_detected:
                    filtered_result.append(detected_result)

            for index in sorted(filtered_indexes, reverse=True):
                score2_objects.pop(index)

            darts_score["1"].extend(filtered_result)
            darts_score["2"].extend(score2_objects)

        if dart3_payload is not None:
            score3_objects = [
                el
                for el in dart3_payload
                if el.get("Cam") is not None and el.get("dx") != 0 and el.get("dy") != 0
            ]
            cam_detected = DartResultsHelper.count_darts_detected(dart3_payload)

            filtered_indexes = []
            filtered_result = []
            for detected_result in darts_score["1"]:
                not_detected = True
                # if cam_detected[detected_result["Cam"]] == 3:
                tolerance = 6

                detected_values = []
                detected_indexes = []
                for i, result in enumerate(score3_objects):
                    if result["Cam"] == detected_result["Cam"] and self.coordinates_match(result, detected_result,
                                                                                     tolerance):
                        detected_values.append(result)
                        if i not in detected_indexes:
                            detected_indexes.append(i)

                detect = False
                if len(detected_values) > 1:
                    for i, found_value in enumerate(detected_values):
                        if found_value["score"] == detected_result["score"]:
                            filtered_result.append(found_value)
                            if detected_indexes[i] not in filtered_indexes:
                                filtered_indexes.append(detected_indexes[i])
                            detect = True
                            not_detected = False
                            break

                    if not detect:
                        for i, found_value in enumerate(detected_values):
                            if abs(found_value["dx"] - detected_result["dx"]) <= 4:
                                filtered_result.append(found_value)
                                if detected_indexes[i] not in filtered_indexes:
                                    filtered_indexes.append(detected_indexes[i])
                                detect = True
                                not_detected = False
                                break

                elif len(detected_values) == 1:
                    filtered_result.append(detected_values[0])
                    if detected_indexes[0] not in filtered_indexes:
                        filtered_indexes.append(detected_indexes[0])
                    not_detected = False
                # else:
                #     for i, result in enumerate(score3_objects):
                #         if result["Cam"] == detected_result["Cam"] and self.coordinates_match(result, detected_result):
                #             filtered_result.append(result)
                #             if i not in filtered_indexes:
                #                 filtered_indexes.append(i)
                #             not_detected = False

                if not_detected:
                    filtered_result.append(detected_result)

            darts_score["1"].extend(filtered_result)
            for index in sorted(filtered_indexes, reverse=True):
                score3_objects.pop(index)

            filtered_indexes = []
            filtered_result = []
            for detected_result in darts_score["2"]:
                not_detected = True
                # if cam_detected[detected_result["Cam"]] == 3:
                tolerance = 6

                detected_values = []
                detected_indexes = []
                for i, result in enumerate(score3_objects):
                    if result["Cam"] == detected_result["Cam"] and self.coordinates_match(result, detected_result,
                                                                                          tolerance):
                        detected_values.append(result)
                        if i not in detected_indexes:
                            detected_indexes.append(i)

                detect = False
                if len(detected_values) > 1:
                    for i, found_value in enumerate(detected_values):
                        if found_value["score"] == detected_result["score"]:
                            filtered_result.append(found_value)
                            if detected_indexes[i] not in filtered_indexes:
                                filtered_indexes.append(detected_indexes[i])
                            detect = True
                            not_detected = False
                            break

                    if not detect:
                        for i, found_value in enumerate(detected_values):
                            if abs(found_value["dx"] - detected_result["dx"]) <= 4:
                                filtered_result.append(found_value)
                                if detected_indexes[i] not in filtered_indexes:
                                    filtered_indexes.append(detected_indexes[i])
                                detect = True
                                not_detected = False
                                break

                elif len(detected_values) == 1:
                    filtered_result.append(detected_values[0])
                    if detected_indexes[0] not in filtered_indexes:
                        filtered_indexes.append(detected_indexes[0])
                    not_detected = False
                # else:
                #     for i, result in enumerate(score3_objects):
                #         if result["Cam"] == detected_result["Cam"] and self.coordinates_match(result, detected_result):
                #             filtered_result.append(result)
                #             if i not in filtered_indexes:
                #                 filtered_indexes.append(i)
                #             not_detected = False

                if not_detected:
                    filtered_result.append(detected_result)

            for index in sorted(filtered_indexes, reverse=True):
                score3_objects.pop(index)

            darts_score["2"].extend(filtered_result)

            darts_score["3"].extend(score3_objects)

        dart1_score, dart2_score, dart3_score = self.calculate_scores(darts_score)
        return (dart1_score + dart2_score + dart3_score)

    def coordinates_match(self, obj1, obj2, tolerance = 3):
        return (
                abs(obj1["dx"] - obj2["dx"]) <= tolerance
                and abs(obj1["dy"] - obj2["dy"]) <= tolerance
        )

    def get_key_with_max_value(self, score_dict):
        if not score_dict:
            return 0

    def get_score_from_dicts(self, score_dicts):
        if not score_dicts:
            return 0
        score_counter = {}
        # Iterating through dart1_scores
        for dart_score in score_dicts:
            score = dart_score["score"]
            if score in score_counter:
                score_counter[score] += 1
            else:
                score_counter[score] = 1
        max_score = max(score_counter.values())
        most_common_scores = [score for score, count in score_counter.items() if count == max_score]
        if len(most_common_scores) == 1:
            return most_common_scores[0] if most_common_scores[0] != -1 else 0
        score = self.get_score_with_max_confidence_level(score_dicts)

        return score if score != -1 else 0

    def get_score_with_max_confidence_level(self, score_dicts):
        if not score_dicts:
            return 0
        max_c_values = score_dicts[0]["c"]
        score = score_dicts[0]["score"]
        for i in range(1, len(score_dicts)):
            if max_c_values < score_dicts[i]["c"]:
                max_c_values = score_dicts[i]["c"]
                score = score_dicts[i]["score"]

        return score

    def calculate_scores(self, darts_score):
        dart1_scores = [dart_score for dart_score in darts_score["1"] if dart_score["score"] != 0]
        dart2_scores = [dart_score for dart_score in darts_score["2"] if dart_score["score"] != 0]
        dart3_scores = [dart_score for dart_score in darts_score["3"] if dart_score["score"] != 0]
        score3 = self.get_score_with_max_confidence_level(dart3_scores)
        if score3 == -1:
            score3 = 0

        return self.get_score_from_dicts(dart1_scores), self.get_score_from_dicts(dart2_scores), score3

if __name__ == "main":
    AnalyzeTestSet.test_analyze_test_set()
