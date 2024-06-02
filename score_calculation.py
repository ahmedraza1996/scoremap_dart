import threading
import requests
import os
import cv2
import json 

from autoscorer.autoscorer import AutoScorer

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def find_all_result_from_parents( working_set, throw_id, results):

    #  1 - throw id, 11 = previous state,  12  =  raw output of yolo
    for row in working_set:
        if row[1] == throw_id:
            results.insert(0, json.loads(row[12]))
            if row[11] == "":
                return results

            return find_all_result_from_parents( working_set, row[11], results)

    return results




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
                if result["Cam"] == detected_result["Cam"] and  coordinates_match(result, detected_result,
                                                                                    tolerance):
                    not_detected = False

                    detected_values.append(result)
                    if i not in detected_indexes:
                        detected_indexes.append(i)

            if len(detected_values) > 1:
                for i, result in enumerate(score2_objects):
                    if result["Cam"] == detected_result["Cam"] and  coordinates_match(result, detected_result):
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
                if result["Cam"] == detected_result["Cam"] and  coordinates_match(result, detected_result,
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
            #         if result["Cam"] == detected_result["Cam"] and  coordinates_match(result, detected_result):
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
                if result["Cam"] == detected_result["Cam"] and  coordinates_match(result, detected_result,
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
            #         if result["Cam"] == detected_result["Cam"] and  coordinates_match(result, detected_result):
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

    dart1_score, dart2_score, dart3_score =  calculate_scores(darts_score)
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
    score =  get_score_with_max_confidence_level(score_dicts)

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
    score3 =  get_score_with_max_confidence_level(dart3_scores)
    if score3 == -1:
        score3 = 0

    return  get_score_from_dicts(dart1_scores),  get_score_from_dicts(dart2_scores), score3

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
    #change url here
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

if __name__ =='__main__':

    exp_dict = {'throw_ids1','board_ids1' , 'previous_state_ids1' ,
                'throw_ids2','board_ids2' , 'previous_state_ids2' ,
                'throw_ids3','board_ids3' , 'previous_state_ids3'}
    
    
    updated_working_set = []
    
    # each row  board id throw id, previous state
    board_id = '120_20240322_151923_718'
    board_images_paths =['D:/dart/images_new/b_120_20240322_151923_718t_120_20240322_152134_228/board/0.jpg',
                         'D:/dart/images_new/b_120_20240322_151923_718t_120_20240322_152134_228/board/1.jpg',
                         'D:/dart/images_new/b_120_20240322_151923_718t_120_20240322_152134_228/board/2.jpg',
                         'D:/dart/images_new/b_120_20240322_151923_718t_120_20240322_152134_228/board/3.jpg']

    dart_images_paths =['D:/dart/images_new/b_120_20240322_151923_718t_120_20240322_152134_228/board/0.jpg',
                        'D:/dart/images_new/b_120_20240322_151923_718t_120_20240322_152134_228/board/1.jpg',
                        'D:/dart/images_new/b_120_20240322_151923_718t_120_20240322_152134_228/board/2.jpg',
                        'D:/dart/images_new/b_120_20240322_151923_718t_120_20240322_152134_228/board/3.jpg']

    
    
    auto_scorer = AutoScorer(board_id, board_images_paths)
    previous_state= ''
    throw_id = ''
    auto_scorer.calculate_board_map_and_cache()
    result = auto_scorer.analyze(throw_id, dart_images_paths)
    

    if previous_state =='':
        cam0_detected_darts, cam1_detected_darts, cam2_detected_darts, cam3_detected_darts = DartResultsHelper.count_darts_detected(result)
        dart1_score, dart2_score, dart3_score = DartResultsHelper.get_scores(result)
        detected_value = dart1_score if dart1_score != -1 else 0

        #for unit test add row preparation
        updated_working_set.append(prepare_row_data(board_id, throw_id, board_id, result, 0))

    else: 
        
        
        results = find_all_result_from_parents(updated_working_set, previous_state , [result] )
        dart1_score = results[0]
        dart2_score = results[1] if len(results) >= 2 else None
        dart3_score = results[2] if len(results) == 3 else None
        detected_value = get_autoscoring_result(dart1_score, dart2_score, dart3_score)
        updated_working_set.append(prepare_row_data(board_id, throw_id, board_id, result, 0, previous_state, detected_value))
