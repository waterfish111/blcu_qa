"""录一段演示 GIF：用 Edge 的 DevTools 协议开无头浏览器，录屏播报每一帧。

为什么不用 ffmpeg / OBS：
  - 机器上没有 ffmpeg，装它要走代理；
  - 本来就要"能自动化重复"——脚本录制比手动录屏更可靠，改一版重录一条命令即可。

流程：Page.startScreencast 拿到 JPEG 帧 → 用 CDP 驱动真实交互（逐字输入、点击、滚动、点角标）
→ Pillow 合成 GIF。产物：docs/screenshots/demo.gif
"""

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import websocket
from PIL import Image

EDGE = os.environ.get("BLCU_EDGE") or r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9223
URL = os.environ.get("BLCU_URL", "http://127.0.0.1:8000/")
OUT = str(Path(__file__).resolve().parents[1] / "docs" / "screenshots" / "demo.gif")
QUESTION = "2026 级数字经济专业准予毕业总学分是多少？"
COHORT = "2026级"

frames = []          # [(t, jpeg_bytes)]
TYPING = [0.0, 0.0]    # 打字动画的时间窗，这段不压缩
lock = threading.Lock()
stop = threading.Event()


class CDP:
    def __init__(self, ws_url):
        self.ws = websocket.create_connection(ws_url, timeout=60,
                                              max_size=32 * 1024 * 1024)
        self.next_id = 1
        self.pending = {}
        self.responses = {}
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        while not stop.is_set():
            try:
                msg = json.loads(self.ws.recv())
            except Exception:
                return
            if msg.get("method") == "Page.screencastFrame":
                p = msg["params"]
                with lock:
                    frames.append((time.time(), base64.b64decode(p["data"])))
                self.send("Page.screencastFrameAck", {"sessionId": p["sessionId"]})
            elif "id" in msg:
                self.responses[msg["id"]] = msg

    def send(self, method, params=None):
        with lock:
            mid = self.next_id
            self.next_id += 1
        self.ws.send(json.dumps({"id": mid, "method": method,
                                 "params": params or {}}))
        return mid

    def call(self, method, params=None, timeout=60):
        mid = self.send(method, params)
        end = time.time() + timeout
        while time.time() < end:
            if mid in self.responses:
                return self.responses.pop(mid)
            time.sleep(0.01)
        raise TimeoutError(method)

    def js(self, expr, await_promise=False):
        r = self.call("Runtime.evaluate", {"expression": expr, "returnByValue": True,
                                           "awaitPromise": await_promise})
        return r.get("result", {}).get("result", {}).get("value")

    def wait_js(self, expr, timeout=40, label=""):
        end = time.time() + timeout
        while time.time() < end:
            if self.js(expr):
                return True
            time.sleep(0.15)
        print("[warn] 等待超时:", label or expr)
        return False


def type_text(cdp, text, per_char=0.06):
    cdp.js("document.querySelector('#question').focus()")
    for ch in text:
        cdp.js("(() => { const el = document.querySelector('#question');"
               " el.value += %s; el.dispatchEvent(new Event('input', "
               "{bubbles:true})); })()" % json.dumps(ch))
        time.sleep(per_char)


def start_edge(profile):
    proc = subprocess.Popen([
        EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--remote-debugging-port=%d" % PORT,
        "--user-data-dir=%s" % profile,
        "--window-size=1180,1500", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/json/list" % PORT,
                                        timeout=2) as f:
                targets = json.load(f)
            pages = [t for t in targets if t.get("type") == "page"]
            if pages:
                return proc, pages[0]["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.4)
    raise RuntimeError("Edge 没起来")


def main():
    profile = os.path.join(os.environ.get("TEMP", "."), "edge-gif-profile")
    proc, ws_url = start_edge(profile)
    cdp = CDP(ws_url)
    try:
        cdp.call("Page.enable")
        cdp.call("Network.enable")
        cdp.call("Network.setCacheDisabled", {"cacheDisabled": True})
        cdp.call("Emulation.setDeviceMetricsOverride", {
            "width": 1180, "height": 620, "deviceScaleFactor": 1, "mobile": False})
        cdp.call("Page.navigate", {"url": URL})
        cdp.wait_js("document.readyState === 'complete'", 30, "页面加载")
        time.sleep(1.6)

        # 从这一刻开始录
        cdp.call("Page.startScreencast", {"format": "jpeg", "quality": 72,
                                          "maxWidth": 1180, "maxHeight": 620,
                                          "everyNthFrame": 1})
        print("开始录制…")
        time.sleep(0.9)

        TYPING[0] = time.time()
        type_text(cdp, QUESTION, 0.075)
        TYPING[1] = time.time() + 0.4
        time.sleep(0.35)

        cdp.js("document.querySelector('#cohort').value = %s;"
               " document.querySelector('#cohort').dispatchEvent("
               "new Event('change',{bubbles:true}))" % json.dumps(COHORT))
        time.sleep(0.45)
        cdp.js("document.querySelector('#submit').click()")
        print("已提交，等答案…")

        cdp.wait_js("!!document.querySelector('.answer, .abstain')", 60, "答案出现")
        time.sleep(1.2)

        # 滚到答案区
        cdp.js("window.scrollTo({top: document.querySelector('#answer-panel')"
               ".offsetTop - 30, behavior: 'smooth'})")
        time.sleep(1.5)
        # 再滚到出处卡
        cdp.js("window.scrollTo({top: document.querySelector('#sources')"
               ".offsetTop - 30, behavior: 'smooth'})")
        time.sleep(1.5)
        # 点一下 [1] 号角标，卡片高亮、跳回第一张
        cdp.js("(() => { const c = document.querySelector('.cite'); if (c) c.click(); })()")
        time.sleep(1.4)
        print("录制结束，帧数:", len(frames))
    finally:
        stop.set()
        try:
            cdp.ws.close()
        except Exception:
            pass
        proc.terminate()

    if not frames:
        sys.exit("没抓到帧")

    # ---- 抽帧：太像的丢掉，太密的拉稀 ----
    kept = []
    last_t = 0.0
    last_sig = None
    for t, data in frames:
        if t - last_t < 0.09:
            continue
        sig = data[:600]
        if sig == last_sig:
            continue
        kept.append((t, data))
        last_t, last_sig = t, sig
    print("抽帧后:", len(kept))

    imgs, durs = [], []
    for i, (t, data) in enumerate(kept):
        im = Image.open(io.BytesIO(data)).convert("RGB")
        imgs.append(im)
        dt = (kept[i + 1][0] - t) if i + 1 < len(kept) else 0.5
        durs.append(int(max(70, min(600, dt * 1000))))

    # ---- 压掉"什么都没变"的长段：等答案那 6 秒没人愿意看 ----
    def thumb(im):
        return im.convert("L").resize((64, 64))

    from PIL import ImageChops, ImageStat
    groups = []          # [{frames: [...], dur: ms}]
    for i, im in enumerate(imgs):
        t = kept[i][0]
        if groups:
            prev_i = groups[-1]["frames"][-1]
            diff = ImageStat.Stat(ImageChops.difference(thumb(imgs[prev_i]),
                                                       thumb(im))).mean[0]
        else:
            diff = 999.0
        typing = TYPING[0] <= t <= TYPING[1]
        if groups and diff < 3.0 and not typing:
            # 画面没变：并进上一组，但只给它加时间、不加帧（等答案那 6 秒压成 0.9 秒）
            groups[-1]["dur"] = min(900, groups[-1]["dur"] + durs[i])
        else:
            # 打字那一段保留逐帧，别让输入框"瞬间填满"
            groups.append({"frames": [i], "dur": min(durs[i], 140) if typing else durs[i]})

    picked = [imgs[g["frames"][0]] for g in groups]
    pdurs = [int(max(80, min(900, g["dur"]))) for g in groups]
    pdurs[-1] = max(pdurs[-1], 900)      # 最后一帧多停一会儿
    print("压缩静态段后:", len(picked), "帧")

    width = 860
    scaled = [im.resize((width, int(im.height * width / im.width)),
                        Image.LANCZOS) for im in picked]
    pal = [im.quantize(colors=128, method=Image.MEDIANCUT) for im in scaled]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=pdurs,
                loop=0, optimize=True, disposal=2)
    print("GIF:", OUT, round(os.path.getsize(OUT) / 1024 / 1024, 2), "MB",
          "frames", len(pal), "总时长", round(sum(pdurs) / 1000, 1), "s")


if __name__ == "__main__":
    main()


