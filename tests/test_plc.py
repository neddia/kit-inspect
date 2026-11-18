"""Handshake tests against a real simulated PLC (cpppo) on a spare port. No camera, no model.

Each test runs the vision side in a thread with a fake inspection, and plays the
PLC with src/plc/line.py. Run: uv run pytest tests/test_plc.py
"""

import copy
import threading
import time

import pytest

from src.inspection.decision import Decision
from src.plc.line import Line
from src.plc.tags import Tags, load_config, start_sim
from src.station.handshake import VisionSide

PORT = 44901
BASE, CODES = load_config()
CFG = copy.deepcopy(BASE) | {"timeout_s": 1.0, "heartbeat_timeout_s": 1.0, "poll_ms": 5, "heartbeat_ms": 100}


@pytest.fixture(scope="module")
def sim():
    proc = start_sim(CFG, port=PORT)
    yield
    proc.terminate()
    proc.wait()


@pytest.fixture
def plc(sim):
    with Tags(CFG, port=PORT) as t:
        yield Line(t, CFG)


class Station:
    """Run a VisionSide in a background thread for one test."""

    def __init__(self, inspect, cls=VisionSide, health=None):
        self.seen = []
        self.side = cls(inspect, lambda after: "frame", cfg=CFG, codes=CODES, port=PORT, on_result=self.seen.append,
                        health=health)
        self.thread = threading.Thread(target=self.side.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        time.sleep(0.3)                  # let it connect and start the heartbeat
        return self

    def __exit__(self, *exc):
        self.side.stop.set()
        self.thread.join(timeout=5)


def answers(result):
    return lambda frame, kit_id: {"decision": Decision(result, [f"{result} for {kit_id}"])}


def test_pass_releases_and_fail_holds(plc):
    with Station(answers("PASS")):
        out = plc.cycle(1)
    assert out["released"] and out["result"] == "PASS"
    assert plc.tags.read("release_permit") is True
    with Station(answers("FAIL")):
        out = plc.cycle(1)
    assert not out["released"] and out["result"] == "FAIL"
    assert plc.tags.read("release_permit") is False


def test_review_holds(plc):
    with Station(answers("REVIEW")):
        assert not plc.cycle(1)["released"]


def test_kit_code_is_translated_for_the_inspection(plc):
    with Station(answers("FAIL")) as s:
        plc.cycle(2)
        plc.cycle(9)
        plc.cycle(0)
    assert [r["kit_id"] for r in s.seen] == ["KIT-B", "code 9", ""]


def test_no_vision_station_times_out_and_holds(plc):
    out = plc.cycle(1)
    assert not out["released"] and out["why"].startswith("timeout")


def test_crash_sets_fault_and_holds(plc):
    def broken(frame, kit_id):
        raise RuntimeError("camera unplugged")
    with Station(broken) as s:
        out = plc.cycle(1)
    assert not out["released"] and out["why"].startswith("timeout")
    assert plc.tags.read("fault") is True
    assert "camera unplugged" in s.seen[0]["error"]


def test_station_recovers_after_a_fault(plc):
    calls = []

    def flaky(frame, kit_id):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("camera unplugged")
        return answers("PASS")(frame, kit_id)
    with Station(flaky):
        first = plc.cycle(1)
        later = [plc.cycle(1) for _ in range(3)]
    assert not first["released"]
    assert later[-1]["released"] and plc.tags.read("fault") is False


def test_lost_heartbeat_holds_even_a_pass(plc):
    class NoHeartbeat(VisionSide):
        def heartbeat(self):
            pass
    with Station(answers("PASS"), cls=NoHeartbeat):
        time.sleep(CFG["heartbeat_timeout_s"] + 0.2)
        out = plc.cycle(1)
    assert out["result"] == "PASS" and not out["released"] and "heartbeat" in out["why"]


def test_late_answer_is_never_used_for_the_next_kit(plc):
    calls = []

    def slow_then_fast(frame, kit_id):
        calls.append(1)
        if len(calls) == 1:
            time.sleep(CFG["timeout_s"] + 0.5)    # answers after the PLC gave up
        return answers("PASS")(frame, kit_id)
    with Station(slow_then_fast):
        first = plc.cycle(1)
        time.sleep(0.7)                           # the late PASS for cycle 1 lands now
        second = plc.cycle(1)
    assert not first["released"] and first["why"].startswith("timeout")
    assert second["released"] and second["cycle_id"] == first["cycle_id"] + 1


def test_camera_lost_while_idle_faults_at_once_and_recovers_by_itself(plc):
    camera = {"ok": True}
    health = lambda: None if camera["ok"] else "camera lost: unplugged"
    with Station(answers("PASS"), health=health) as s:
        camera["ok"] = False                                    # unplugged while no kit is waiting
        time.sleep(0.1)
        assert plc.tags.read("fault") is True                   # the PLC knows before the next kit arrives
        assert s.seen[-1]["error"] == "camera lost: unplugged" and s.seen[-1]["idle"]
        beat = plc.tags.read("heartbeat")
        time.sleep(0.3)
        assert plc.tags.read("heartbeat") == beat               # heartbeat stopped
        camera["ok"] = True                                     # plugged back in
        time.sleep(0.3)                                         # heartbeat back before the next kit
        assert plc.tags.read("fault") is False and s.seen[-1].get("recovered")
        out = plc.cycle(1)
    assert out["released"], out["why"]                          # the FIRST kit after recovery is inspected normally
