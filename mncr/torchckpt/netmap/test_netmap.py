#!/usr/bin/env python3
"""Prove the bind rewrite, in a child that preloads the library."""

import os
import subprocess
import sys

PROBE = r"""
import socket, sys
def try_bind(host):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, 0))
        return s.getsockname()[0]
    except OSError as exc:
        return f"error:{exc.errno}"
    finally:
        s.close()
# 192.0.2.0/24 is TEST-NET-1: never assigned to a host, so this bind fails
# without the shim and must be rewritten with it.
print("foreign", try_bind("192.0.2.7"))
print("loopback", try_bind("127.0.0.1"))
print("wildcard", try_bind("0.0.0.0"))
"""


def run(preload):
    env = dict(os.environ)
    if preload:
        env["LD_PRELOAD"] = os.path.abspath(preload)
    proc = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"probe failed: {proc.stderr}")
    return dict(line.split(" ", 1) for line in proc.stdout.strip().splitlines())


def main(lib):
    if not sys.platform.startswith("linux"):
        print("netmap test: Linux only, skipped")
        return 0
    without = run(None)
    assert without["foreign"].startswith("error:"), without
    with_shim = run(lib)
    assert not with_shim["foreign"].startswith("error:"), (
        f"bind to a foreign address was not rewritten: {with_shim}"
    )
    assert with_shim["foreign"] not in ("192.0.2.7", "0.0.0.0", "127.0.0.1"), with_shim
    assert with_shim["loopback"] == "127.0.0.1", with_shim
    assert with_shim["wildcard"] == "0.0.0.0", with_shim
    print(f"netmap: ok (foreign bind rewritten to {with_shim['foreign']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "./libmncr_netmap.so"))
