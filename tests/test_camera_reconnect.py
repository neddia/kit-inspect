from types import SimpleNamespace
from unittest.mock import Mock

from src.capture.camera import find_usb_camera
from src.station import camera_feed


def test_usb_capture_node_can_change_number(tmp_path):
    root = tmp_path / "class"
    dev = tmp_path / "usb" / "port"
    interface = dev / "interface"
    interface.mkdir(parents=True)
    (dev / "idVendor").write_text("046d")
    (dev / "idProduct").write_text("081d")
    for number, index in [(3, 1), (4, 0)]:
        node = root / f"video{number}"
        node.mkdir(parents=True)
        (node / "index").write_text(str(index))
        (node / "device").symlink_to(interface)
    assert find_usb_camera("046d:081d", root) == "/dev/video4"
    assert find_usb_camera("046d:1234", root) is None


def test_feed_reopens_after_a_read_exception(monkeypatch):
    feed = camera_feed.CameraFeed({"device": "/dev/test"}, reopen_s=0)
    def recovered():
        feed.stop.set()
        return True, "fresh frame"
    broken = SimpleNamespace(read=Mock(side_effect=OSError("unplugged")), release=Mock())
    working = SimpleNamespace(read=recovered, release=Mock())
    monkeypatch.setattr(camera_feed, "open_camera", Mock(side_effect=[broken, working]))
    feed._run()
    broken.release.assert_called_once()
    working.release.assert_called_once()
    assert feed.frame == "fresh frame" and feed.error is None
