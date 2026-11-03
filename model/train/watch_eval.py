"""Wait for the next evaluation report using filesystem events, not polling."""

import argparse
import ctypes
import json
import os
import select
import struct
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--after", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=21600)
    args = parser.parse_args()
    directory = args.directory.resolve()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1 failed")
    mask = 0x8 | 0x80 | 0x100
    target_watch = None

    def watch(path):
        descriptor = libc.inotify_add_watch(fd, os.fsencode(path), mask)
        if descriptor < 0:
            raise OSError(ctypes.get_errno(), f"Cannot watch {path}")
        return descriptor

    def next_report():
        path = directory / "evaluations.jsonl"
        if path.is_file():
            for line in path.read_text().splitlines():
                record = json.loads(line)
                if record["step"] > args.after:
                    print(json.dumps(record, indent=2), flush=True)
                    return True
        return False

    try:
        parent_watch = watch(directory.parent)
        if directory.is_dir():
            target_watch = watch(directory)
        if next_report():
            return 0
        deadline = time.monotonic() + args.timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], max(0, remaining))[0]:
                print(
                    "No new checkpoint evaluation before the notification deadline.",
                    flush=True,
                )
                return 2
            events = os.read(fd, 65536)
            offset = 0
            while offset < len(events):
                descriptor, event_mask, _, length = struct.unpack_from(
                    "iIII", events, offset
                )
                name = os.fsdecode(
                    events[offset + 16 : offset + 16 + length].split(b"\0", 1)[0]
                )
                offset += 16 + length
                if (
                    descriptor == parent_watch
                    and name == directory.name
                    and target_watch is None
                    and directory.is_dir()
                ):
                    target_watch = watch(directory)
                    if next_report():
                        return 0
                if (
                    descriptor == target_watch
                    and name == "evaluations.jsonl"
                    and event_mask & (0x8 | 0x80)
                ):
                    if next_report():
                        return 0
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
