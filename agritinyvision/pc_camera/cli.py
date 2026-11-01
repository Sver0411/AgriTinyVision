import argparse
import hashlib
import json
import time

from .camera import Camera
from .image_preprocess import load_image_bgr, prepare_bgr
from .metrics import RunRecorder, generate_report
from .serial_client import SerialClient


def parser():
    p = argparse.ArgumentParser(description="PC camera → ESP32-S3 real device inference")
    p.add_argument("--port", help="ESP32 COM port, for example COM3")
    p.add_argument("--baudrate", type=int, default=921600)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0)
    p.add_argument("--mirror", action="store_true", help="mirror RGB input (recorded in experiment config)")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--output", default="runs/hardware")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--host", default="127.0.0.1", help="Web server bind address")
    p.add_argument("--http-port", type=int, default=8765, help="Web interface port")
    p.add_argument("--no-browser", action="store_true", help="serve Web UI without opening a browser")
    p.add_argument(
        "--frames", type=int, default=1, help="headless frame count; 0 with --duration for timed runs"
    )
    p.add_argument("--duration", type=float, default=0, help="headless minimum duration in seconds")
    p.add_argument("--image", help="send the same image to ESP32 for a reproducible deployment comparison")
    p.add_argument("--list-ports", action="store_true")
    p.add_argument("--check-camera", action="store_true")
    p.add_argument("--report", metavar="RUN_DIR", help="regenerate plots and Markdown from an experiment")
    return p


def headless(args):
    camera = Camera(args.camera)
    client = SerialClient(args.port, args.baudrate, args.timeout)
    recorder = None
    try:
        device = client.connect()
        recorder = RunRecorder(args.output, vars(args), device)
        image = load_image_bgr(args.image) if args.image else None
        if image is None:
            camera.connect()
        start = last = time.perf_counter()
        count = 0
        while count < args.frames or time.perf_counter() - start < args.duration:
            begin = time.perf_counter()
            frame = image if image is not None else camera.read()
            rgb = prepare_bgr(frame, args.rotation, args.mirror)
            r = client.infer(rgb.tobytes())
            r["rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
            end = time.perf_counter()
            r["actual_fps"] = 1 / (end - last)
            last = end
            count += 1
            recorder.record(r, end - start)
            print(json.dumps(r, allow_nan=False))
            time.sleep(max(0, 1 / args.fps - (time.perf_counter() - begin)))
    except (Exception, KeyboardInterrupt) as exc:
        if recorder:
            recorder.error(str(exc) or "Interrupted", client.sequence)
        raise
    finally:
        client.close()
        camera.disconnect()
        if recorder:
            recorder.close()
            print(f"Experiment: {recorder.path}")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.report:
        print(json.dumps(generate_report(args.report), indent=2))
        return
    if args.list_ports:
        from serial.tools.list_ports import comports

        for port in comports():
            print(f"{port.device}: {port.description} · {port.hwid}")
        return
    if args.check_camera:
        camera = Camera(args.camera)
        try:
            camera.connect()
            print(f"Camera {args.camera}: {camera.read().shape}")
        finally:
            camera.disconnect()
        return
    if args.headless and not args.port:
        p.error("--port COMx is required")
    if not 0 < args.fps <= 30 or args.timeout <= 0 or args.frames < 0 or args.duration < 0:
        p.error("fps must be (0,30], timeout > 0, frames/duration >= 0")
    if args.headless:
        if args.frames == 0 and args.duration == 0:
            p.error("headless requires frames > 0 or duration > 0")
        headless(args)
    else:
        from .web import serve

        serve(args)


if __name__ == "__main__":
    main()
