"""The camera, always running, so a trigger never waits for it to start and never gets an old photo.

A webcam keeps a small queue of frames. Read one "on demand" after a pause and
you can get a frame taken BEFORE the kit arrived. So a background thread reads
continuously and keeps only the newest frame with its timestamp. On a trigger,
fresh_frame(after=trigger time) waits for the first frame taken after the trigger
(at 15 fps, at most ~70 ms).

Camera unplugged: reads fail or stop, fresh_frame raises CameraLost after
`timeout_s` (the station then faults and the PLC holds the kit), and the thread
keeps trying to reopen it.
"""

import threading
import time

from src.capture.camera import load_camera_config, open_camera, station_camera_config


class CameraLost(RuntimeError):
    pass


class CameraFeed:
    def __init__(self, cfg=None, reopen_s=2.0):
        self.cfg = cfg or load_camera_config()
        self.reopen_s = reopen_s
        self.frame, self.frame_t = None, 0.0
        self.error = "starting"
        self.new_frame = threading.Condition()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)

    def _run(self):
        while not self.stop.is_set():
            cap = None
            try:
                cap = open_camera(station_camera_config(self.cfg))
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise CameraLost("camera stopped sending frames")
                    with self.new_frame:
                        self.error = None
                        self.frame, self.frame_t = frame, time.monotonic()
                        self.new_frame.notify_all()
            except Exception as e:
                with self.new_frame:
                    self.error = str(e)
                    self.frame, self.frame_t = None, 0.0
                    self.new_frame.notify_all()
            finally:
                if cap is not None:
                    cap.release()
            self.stop.wait(self.reopen_s)

    def latest(self):
        """(frame, age in s) of the newest frame, for a live view. frame is None while disconnected."""
        with self.new_frame:
            return self.frame, time.monotonic() - self.frame_t

    def fresh_frame(self, after=None, timeout_s=1.0):
        """The first frame taken after `after` (default: now). Raises CameraLost if none arrives in time."""
        after = time.monotonic() if after is None else after
        deadline = after + timeout_s
        with self.new_frame:
            while self.frame_t <= after:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise CameraLost(self.error or f"no new frame within {timeout_s} s")
                self.new_frame.wait(left)
            return self.frame
