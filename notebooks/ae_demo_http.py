"""HTTP helpers for the AE demo notebook (demo_ae.ipynb).

`serve` starts a threaded HTTP server on 127.0.0.1 that answers single
Range requests (and HEAD) in one of two modes:

- local: files from a directory, optionally with an added delay before
  each response and a per-response bandwidth cap, to stand in for a
  remote host whose latency and throughput were measured;
- upstream: each request is forwarded to `upstream` + path over a
  persistent connection per server thread, following redirects (Hugging
  Face answers with one to its CDN), so a client such as GDAL or
  zarr-python can be pointed at the server and its requests to the real
  host counted and timed. `headers`, for example a token, go with each
  request to the upstream host and not to a host it redirects to.

Either way the server records each GET it answers as an event dict
(`kind` "get", `label`, `t0`, `t1` in time.perf_counter seconds,
`bytes`), the format of georvq.inference.records.Trace.

`probe` measures latency (time to first byte of small range requests)
and single-stream throughput (one large range request) to a URL.
"""
from __future__ import annotations

import http.client
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Server(ThreadingHTTPServer):
    # the default backlog of 5 drops connections when a client opens more
    # at once, and the client's retry then adds about a second
    request_queue_size = 256

_RANGE = re.compile(r"^bytes=(\d*)-(\d*)$")
_CHUNK = 64 * 1024
# some hosts refuse the default Python user agent
USER_AGENT = "georvq-demo"


class Recorder:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()

    def add(self, label, t0, t1, nbytes):
        with self.lock:
            self.events.append({"kind": "get", "label": label, "t0": t0,
                                "t1": t1, "bytes": int(nbytes)})

    def clear(self):
        with self.lock:
            self.events = []


def _span(rng, size):
    """(start, end) inclusive of a single Range header, or None."""
    m = _RANGE.match((rng or "").strip())
    if not m or (m.group(1) == "" and m.group(2) == ""):
        return None
    a, b = m.group(1), m.group(2)
    if a == "":
        return max(0, size - int(b)), size - 1
    return int(a), min(size - 1, int(b)) if b else size - 1


def _handler(root, upstream, latency, bandwidth, rec, extra=None):
    up = urllib.parse.urlsplit(upstream) if upstream else None
    extra = dict(extra or {})
    local = threading.local()

    def connection():
        c = getattr(local, "conn", None)
        if c is None:
            cls = (http.client.HTTPSConnection if up.scheme == "https"
                   else http.client.HTTPConnection)
            c = local.conn = cls(up.netloc, timeout=120)
        return c

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def do_HEAD(self):
            self._answer(body=False)

        def do_GET(self):
            self._answer(body=True)

        def _answer(self, body):
            t0 = time.perf_counter()
            rng = self.headers.get("Range")
            n = (self._forward if up else self._local)(body, rng)
            if body:
                rec.add(rng or "whole", t0, time.perf_counter(), n)

        def _forward(self, body, rng):
            path = up.path.rstrip("/") + self.path
            headers = {"User-Agent": USER_AGENT, **extra}
            if rng:
                headers["Range"] = rng
            for attempt in (0, 1):
                try:
                    c = connection()
                    c.request("GET" if body else "HEAD", path, headers=headers)
                    resp = c.getresponse()
                    break
                except (http.client.HTTPException, OSError):
                    local.conn = None
                    if attempt:
                        raise
            if resp.status in (301, 302, 303, 307, 308):
                resp.read()
                where = urllib.parse.urljoin(f"{up.scheme}://{up.netloc}{path}",
                                             resp.getheader("Location"))
                req = urllib.request.Request(
                    where, method="GET" if body else "HEAD",
                    headers={k: v for k, v in headers.items() if k in
                             ("User-Agent", "Range")})
                try:
                    resp = urllib.request.urlopen(req, timeout=120)
                except urllib.error.HTTPError as e:
                    resp = e
            self.send_response(resp.status)
            for k in ("Content-Length", "Content-Range", "Content-Type",
                      "ETag", "Last-Modified", "Accept-Ranges"):
                if resp.headers.get(k):
                    self.send_header(k, resp.headers.get(k))
            self.end_headers()
            n = 0
            while body:
                b = resp.read(_CHUNK)
                if not b:
                    break
                self.wfile.write(b)
                n += len(b)
            if not body:
                resp.read()
            return n

        def _local(self, body, rng):
            path = os.path.join(root, self.path.split("?")[0].lstrip("/"))
            if not os.path.isfile(path):
                self.send_error(404)
                return 0
            st = os.stat(path)
            size = st.st_size
            span = _span(rng, size) if rng else (0, size - 1)
            if span is None or span[0] > span[1]:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return 0
            a, b = span
            if latency:
                time.sleep(latency)
            self.send_response(206 if rng else 200)
            if rng:
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.send_header("Content-Length", str(b - a + 1))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", f'"{size:x}-{st.st_mtime_ns:x}"')
            self.end_headers()
            if not body:
                return 0
            t0, sent = time.perf_counter(), 0
            with open(path, "rb") as f:
                f.seek(a)
                while sent < b - a + 1:
                    chunk = f.read(min(_CHUNK, b - a + 1 - sent))
                    self.wfile.write(chunk)
                    sent += len(chunk)
                    if bandwidth:
                        wait = sent / bandwidth - (time.perf_counter() - t0)
                        if wait > 0:
                            time.sleep(wait)
            return sent

    return H


def serve(root=None, upstream=None, latency=0.0, bandwidth=None, headers=None):
    """Start a server; returns (base_url, recorder, server). Pass `root`
    (a directory) or `upstream` (a URL prefix). `latency` in seconds,
    `bandwidth` in bytes per second per response; both apply to local
    files only. `headers` go to the upstream host. Stop it with
    server.shutdown()."""
    if (root is None) == (upstream is None):
        raise ValueError("give root or upstream")
    rec = Recorder()
    srv = _Server(("127.0.0.1", 0),
                  _handler(root, upstream, latency, bandwidth, rec, headers))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", rec, srv


def _get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}",
                                               "User-Agent": USER_AGENT})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        first = resp.read(1)
        t1 = time.perf_counter()
        n = len(first) + len(resp.read())
    return t1 - t0, time.perf_counter() - t0, n


def probe(url, small=16 * 1024, large=16 * 1024 * 1024, repeats=5,
          offset=1 << 26):
    """Latency (median time to first byte of `repeats` small range
    requests) and throughput (bytes per second of one `large` range
    request after its first byte) to `url`."""
    ttfb = sorted(_get(url, offset + i * small, offset + (i + 1) * small - 1)[0]
                  for i in range(repeats))
    first, total, n = _get(url, offset, offset + large - 1)
    return {"latency_s": ttfb[len(ttfb) // 2], "ttfb_all_s": ttfb,
            "throughput_Bps": n / max(total - first, 1e-9),
            "large_bytes": n, "large_s": total}
