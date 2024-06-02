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
