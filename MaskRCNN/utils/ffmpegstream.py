""" FFmpeg Video Stream
Borrowed from https://github.com/michael-mueller-git/Python-Funscript-Editor
"""

import cv2
import os
import logging
import time
import platform

from dataclasses import dataclass
from threading import Thread
from queue import Queue

import subprocess as sp
import numpy as np

from utils.watchdog import Watchdog

@dataclass
class VideoInfo:
    """ Video Info Dataclass """
    fps :float
    height :int
    width :int
    length :int


class FFmpegStream:
    """ FFmpeg Stream with frame buffer

    Args:
        video_path (str): path to video file
        config (dict): conversion parameter
        skip_frames (int): skip given number of frames between preview frames
        start_frame (int): start frame number
        queue_size (int): size of frame buffer
        watchdog_timeout (int): watchdog timeout in seconds
        log_queue_overrun (bool): log queue overruns
    """

    def __init__(self,
            video_path :str,
            config :dict,
            skip_frames :int = 0,
            start_frame :int = 0,
            queue_size :int = 256,
            watchdog_timeout :int = 4,
            log_queue_overrun :bool = False):

        self.video_path = video_path
        self.config = config
        self.skip_frames = skip_frames
        self.start_frame = start_frame
        self.queue_size = queue_size
        self.log_queue_overrun = log_queue_overrun

        self.stopped = False
        self.timeout = False
        self.current_frame = 0
        self.sleep_time = 0.001

        self.video_info = self.get_video_info(video_path)
        self.frame_buffer = Queue(maxsize=queue_size)

        self.watchdog = Watchdog(watchdog_timeout, self.watchdog_timeout)
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()


    def __del__(self):
        self.watchdog.stop()


    logger = logging.getLogger(__name__)


    def watchdog_timeout(self):
        """ Watchdog timeout for ffmpeg stream """
        if not self.stopped:
            self.logger.error("FFmpegStream Timeout")
            self.timeout = True
            self.stopped = True
        try: self.pipe.terminate()
        except: pass
        try: self.pipe.stdout.close()
        except: pass
        try: self.pipe.stderr.close()
        except: pass
        self.watchdog.stop()


    @staticmethod
    def get_video_info(
            video_path: str) -> VideoInfo:
        """ Get VideoInfo

        Args:
            video_path (str): path to video

        Returns:
            VideoInfo: video infos
        """
        cap = cv2.VideoCapture(video_path)
        video_info = VideoInfo(
                fps = float(cap.get(cv2.CAP_PROP_FPS)),
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            )
        cap.release()
        return video_info


    @staticmethod
    def get_frame(
            video_path :str,
            frame_number :int) -> np.ndarray:
        """ Get Video frame

        Args:
            video_path (str): path to video
            frame_number (int): frame number to extract from video

        Returns:
            np.ndarray: opencv image
        """
        cap = cv2.VideoCapture(video_path)
        if frame_number > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        _, frame = cap.read()
        cap.release()
        return frame


    @staticmethod
    def get_ffmpeg_command() -> str:
        """ Get FFmpeg binary command string

        Returns:
            str: FFmpeg command
        """
        ffmpeg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg")
        if platform.system() == "Windows":
            if os.path.exists(ffmpeg + '.exe'):
                return ffmpeg + '.exe'
        elif os.path.exists(ffmpeg):
            return ffmpeg

        # use ffmpeg in $PATH
        return "ffmpeg"


    @staticmethod
    def get_projection(
            frame :np.ndarray,
            config: dict) -> np.ndarray:
        """ Get projection of frame

        Args:
            frame (np.ndarray): opencv image
            config (dict): conversion parameter

        Returns:
            np.ndarray: projected opencv image
        """
        dimension = '{}x{}'.format(frame.shape[1], frame.shape[0])

        video_filter = config['video_filter']
        for k, v in config['parameter'].items():
            video_filter = video_filter.replace('${' + k + '}', str(v))

        command = [
                FFmpegStream.get_ffmpeg_command(),
                '-hide_banner',
                '-loglevel', 'warning',
                '-y',
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', dimension,
                '-pix_fmt', 'bgr24',
                '-i', '-',
                '-f', 'image2pipe',
                '-pix_fmt', 'bgr24',
                '-vsync', '0',
                '-vcodec', 'rawvideo',
                '-an',
                '-sn',
                '-vf', video_filter,
                '-'
            ]

        pipe = sp.Popen(
                command,
                stdin = sp.PIPE,
                stdout = sp.PIPE,
                bufsize = 3 * config['parameter']['width'] * config['parameter']['height']
            )

        pipe.stdin.write(frame.tobytes())
        projection = np.frombuffer(
                pipe.stdout.read(config['parameter']['width'] * config['parameter']['height'] * 3),
                dtype='uint8'
            ).reshape(
                    (config['parameter']['height'], config['parameter']['width'], 3)
                )

        pipe.terminate()
        try: pipe.stdin.close()
        except: pass
        try: pipe.stdout.close()
        except: pass

        return projection


    @staticmethod
    def frame_to_timestamp(frame_number: int, fps: float) -> str:
        """Get timestamp H:M:S.XXX for given frame number

        Args:
            frame_number (int): frame number
            fps (float): frames per seconds

        Returns:
            str: position in video as timestamp with H:M:S.XXX
        """
        return FFmpegStream.millisec_to_timestamp(
                FFmpegStream.frame_to_millisec(frame_number, fps)
            )


    @staticmethod
    def frame_to_millisec(frame_number: int, fps: float) -> int:
        """Get timestamp in milliseconds for given frame number

        Args:
            frame_number (int): frame number
            fps (float): frames per seconds

        Returns:
            int: timestamp in milliseconds
        """
        if frame_number <= 0: return 0
        return int(round(float(frame_number)*float(1000)/fps))


    @staticmethod
    def millisec_to_timestamp(millis :int)->str:
        """ Convert milliseconds to timestamp H:M:S.XXX

        Args:
            millis (int): position in video in milliseconds

        Returns:
            str: position in video as timestamp with H:M:S.XXX
        """
        millis = int(millis)
        seconds = int((millis / 1000) % 60)
        minutes = int((millis / (1000 * 60)) % 60)
        hours = int((millis / (1000 * 60 * 60)) % 24)
        millis = int(millis % 1000)

        return str(hours).zfill(2) \
                + ':' + str(minutes).zfill(2) \
                + ':' + str(seconds).zfill(2) \
                + '.' + str(millis).zfill(3)


    def stop(self) -> None:
        """ Stop FFmpeg video stream """
        self.stopped = True


    def read(self) -> np.ndarray:
        """ Get next projected frame from stream

        Returns:
            np.ndarray: opencv image data
        """
        while self.frame_buffer.qsize() == 0 and not self.stopped:
            time.sleep(self.sleep_time)
        return self.frame_buffer.get() if self.frame_buffer.qsize() > 0 else None


    def isOpen(self) -> bool:
        """ Check if FFmpeg video stream is open or a frame is still available in the buffer

        Returns:
            bool: True if video stream is open or a frame is still available in the buffer else False
        """
        return self.more() or not self.stopped


    def isTimeout(self) -> bool:
        """ Check if FFmpeg video stream has an timeout

        Returns:
            bool: True if ffmpeg video stream has an timeout else False
        """
        return self.timeout


    def more(self) -> bool:
        """ Check if frames in the frame bufer are available

        Returns:
            bool: True if a frame is available else False
        """
        tries = 0
        while self.frame_buffer.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(self.sleep_time)
            tries += 1

        return self.frame_buffer.qsize() > 0


    def run(self) -> None:
        """ Function to read transformed frames from ffmpeg video stream into a queue """

        try:
            video_filter = self.config['video_filter']
            for k, v in self.config['parameter'].items():
                video_filter = video_filter.replace('${' + k + '}', str(v))

            seek = FFmpegStream.frame_to_timestamp(self.start_frame, self.video_info.fps)

            command = [
                    FFmpegStream.get_ffmpeg_command(),
                    '-hide_banner',
                    '-loglevel', 'warning',
                    '-ss', str(seek),
                    '-hwaccel', 'auto',
                    '-i', self.video_path,
                    '-f', 'image2pipe',
                    '-pix_fmt', 'bgr24',
                    '-vsync', '0',
                    '-vcodec', 'rawvideo',
                    '-an',
                    '-sn',
                    '-vf', video_filter,
                    '-'
                ]

            self.watchdog.start()
            self.logger.info("FFmpeg Stream Watchdog started")
            self.logger.info("Open FFmpeg Stream")
            self.pipe = sp.Popen(
                    command,
                    stdout = sp.PIPE,
                    stderr = sp.PIPE,
                    bufsize= 3 * self.config['parameter']['height'] * self.config['parameter']['width']
                )

            while not self.stopped:
                self.watchdog.trigger()
                data = self.pipe.stdout.read(self.config['parameter']['width'] * self.config['parameter']['height'] * 3)
                if not data:
                    break

                frame = np.frombuffer(data, dtype='uint8').reshape(
                        (self.config['parameter']['height'], self.config['parameter']['width'], 3)
                    )
                if frame is None:
                    break

                self.current_frame += 1

                # NOTE: Use != 1 to ensure that the first difference is equal to the folowing (reqired for the interpolation)
                if self.skip_frames > 0 and self.current_frame % (self.skip_frames + 1) != 1:
                    continue

                wait_counter = 0
                while self.frame_buffer.full() and not self.stopped:
                    self.watchdog.trigger()
                    time.sleep(self.sleep_time)
                    wait_counter += 1
                    if self.current_frame - (self.skip_frames + 1)*self.queue_size > 3 and wait_counter == 2500:
                        if self.log_queue_overrun:
                            self.logger.error("FFmpeg Frame Buffer overrun!!!")

                if 'zoom' in self.config.keys():
                    frame = frame[
                        int(self.config['zoom'][1]):int(self.config['zoom'][1]+self.config['zoom'][3]),
                        int(self.config['zoom'][0]):int(self.config['zoom'][0]+self.config['zoom'][2])
                    ]

                if 'resize' in self.config.keys():
                    frame = cv2.resize(frame, self.config['resize'])

                self.frame_buffer.put(frame)

            self.stopped = True
            self.logger.info('Close FFmpeg Stream')
            self.watchdog.stop()
            self.logger.info("FFmpeg Stream Watchdog stoped")
            self.pipe.terminate()
            try: self.pipe.stdout.close()
            except: pass
            try: self.pipe.stderr.close()
            except: pass
        except Exception as ex:
            self.stopped = True
            self.logger.critical("FFmpegStream crashed due to a fatal error", exc_info=ex)

