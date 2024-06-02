def __calculate_result_for_all_darts__(self, data , boardmap):
        results = None

        if data["tips"] == 0:
            results = [
                self.__create_result_dict__(1),
                self.__create_result_dict__(2),
                self.__create_result_dict__(3),
            ]

        elif data["tips"] == 1:
            result_1 = self.__calculate_result__(1, data["dx1"], data["dy1"], data["c1"] , boardmap['PIXEL_MAP'] , boardmap['MULTIPLIER_MAP'], boardmap['CENTER'])
            results = [
                result_1,
                self.__create_result_dict__(2),
                self.__create_result_dict__(3),
            ]

        elif data["tips"] == 2:
            result_1 = self.__calculate_result__(1, data["dx1"], data["dy1"], data["c1"],   boardmap['PIXEL_MAP'] , boardmap['MULTIPLIER_MAP'], boardmap['CENTER'])
            result_2 = self.__calculate_result__(2, data["dx2"], data["dy2"], data["c2"] , boardmap['PIXEL_MAP'] , boardmap['MULTIPLIER_MAP'], boardmap['CENTER'])

            results = [
                result_1,
                result_2,
                self.__create_result_dict__(3),
            ]

        elif data["tips"] == 3:
            result_1 = self.__calculate_result__(1, data["dx1"], data["dy1"], data["c1"] , boardmap['PIXEL_MAP'] , boardmap['MULTIPLIER_MAP'], boardmap['CENTER'])
            result_2 = self.__calculate_result__(2, data["dx2"], data["dy2"], data["c2"] , boardmap['PIXEL_MAP'] , boardmap['MULTIPLIER_MAP'], boardmap['CENTER'])
            result_3 = self.__calculate_result__(3, data["dx3"], data["dy3"], data["c3"] , boardmap['PIXEL_MAP'] , boardmap['MULTIPLIER_MAP'], boardmap['CENTER'])

            results = [
                result_1,
                result_2,
                result_3
            ]

        return results
