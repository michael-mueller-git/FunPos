from lib.ffmpegstream import FFmpegStream
import cv2
import yaml
import copy
import numpy as np
from queue import Queue
from pynput.keyboard import Key, Listener


def read_yaml_config(config_file: str) -> dict:
    """ Parse a yaml config file

    Args:
        config_file (str): path to config file to parse

    Returns:
        dict: the configuration dictionary
    """
    with open(config_file) as f:
        return yaml.load(f, Loader = yaml.FullLoader)

PROJECTION = read_yaml_config("./lib/projection.yaml")

class VrProjection:

    def __init__(self, video_path, start_frame = 0):
        self.keypress_queue = Queue(maxsize=32)
        self.window_name = "vr selector"
        self.video_path = video_path
        self.start_frame = start_frame

    def draw_text(self, img: np.ndarray, txt: str, y :int = 50, color :tuple = (0,0,255)) -> np.ndarray:
        """ Draw text to an image/frame

        Args:
            img (np.ndarray): opencv image
            txt (str): the text to plot on the image
            y (int): y position
            colot (tuple): BGR Color tuple

        Returns:
            np.ndarray: opencv image with text
        """
        annotated_img = img.copy()
        cv2.putText(annotated_img, str(txt), (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return annotated_img

    def get_vr_projection_config(self, image :np.ndarray) -> dict:
            """ Get the projection ROI config form user input

            Args:
                image (np.ndarray): opencv vr 180 or 360 image

            Returns:
                dict: projection config
            """
            config = copy.deepcopy(PROJECTION['vr_he_180_sbs'])

            ui_texte = []
            if "keys" in config.keys():
                for param in config['keys'].keys():
                    if param in config['parameter'].keys() and all(x in config["keys"][param].keys() for x in ["increase", "decrease"]):
                        ui_texte.append("Use '{}', '{}' to increase/decrease {}".format(
                            config["keys"][param]["increase"],
                            config["keys"][param]["decrease"],
                            param)
                        )

            self.clear_keypress_queue()
            parameter_changed, selected = True, False
            while not selected:
                if parameter_changed:
                    parameter_changed = False
                    preview = FFmpegStream.get_projection(image, config)

                    preview = self.draw_text(preview, "Press 'space' to use current selected region of interest)",
                            y = 50, color = (255, 0, 0))
                    preview = self.draw_text(preview, "Press '0' (NULL) to reset view)",
                            y = 75, color = (255, 0, 0))
                    for line, txt in enumerate(ui_texte):
                        preview = self.draw_text(preview, txt, y = 100 + (line * 25), color = (0, 255, 0))

                cv2.imshow(self.window_name, preview)

                while self.keypress_queue.qsize() > 0:
                    pressed_key = '{0}'.format(self.keypress_queue.get())
                    if pressed_key == "Key.space":
                        selected = True
                        break

                    if pressed_key == "'0'":
                        config = copy.deepcopy(PROJECTION['vr_he_180_sbs'])
                        parameter_changed = True
                        break

                    if "keys" not in config.keys():
                        break

                    for param in config['keys'].keys():
                        if param in config['parameter'].keys() and all(x in config["keys"][param].keys() for x in ["increase", "decrease"]):
                            if pressed_key == "'" + config["keys"][param]["increase"] + "'":
                                config['parameter'][param] += 5
                                parameter_changed = True
                                break
                            elif pressed_key == "'" + config["keys"][param]["decrease"] + "'":
                                config['parameter'][param] -= 5
                                parameter_changed = True
                                break

                if cv2.waitKey(1) in [ord(' ')]: break

            bbox = cv2.selectROI(self.window_name, self.draw_text(FFmpegStream.get_projection(image, config), "Selected area"), False)
            config['zoom'] = bbox
            config['resize'] = (128, 128)
            cap = cv2.VideoCapture(self.video_path)
            config['fps'] = float(cap.get(cv2.CAP_PROP_FPS))
            config['length'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            try: cv2.destroyWindow(self.window_name)
            except: pass
            return config

    def clear_keypress_queue(self) -> None:
        while self.keypress_queue.qsize() > 0:
            self.keypress_queue.get()


    def was_key_pressed(self, key: str) -> bool:
        if key is None or len(key) == 0: return False
        while self.keypress_queue.qsize() > 0:
            if '{0}'.format(self.keypress_queue.get()) == "'"+key[0]+"'": return True
        return False

    def on_key_press(self, key: Key) -> None:
        if not self.keypress_queue.full():
            self.keypress_queue.put(key)

    def get_parameter(self) -> dict:
        with Listener(on_press=self.on_key_press) as _:
            first_frame = FFmpegStream.get_frame(self.video_path, self.start_frame)
            return self.get_vr_projection_config(first_frame)
