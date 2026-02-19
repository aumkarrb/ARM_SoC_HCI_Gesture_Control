import socket, subprocess, threading, time, os, json, tempfile
from typing import Optional, List

MPV_SOCKET   = "/tmp/mpvsocket"
VOLUME_STEP  = 5
SEEK_SECONDS = 10

class MPVSocketController:
    def __init__(self, video_path=None, playlist=None,
                 auto_launch=True, socket_path=MPV_SOCKET):
        self._socket_path = socket_path
        self._mpv_proc    = None
        self._sock        = None
        self._lock        = threading.Lock()
        self._connected   = False
        self._playlist_file = None

        if os.path.exists(socket_path):
            os.remove(socket_path)

        videos = playlist if playlist else ([video_path] if video_path else [])

        if auto_launch and videos:
            self._launch_mpv(videos)

        self._connect(timeout=10.0)

    def _launch_mpv(self, videos):
        if len(videos) > 1:
            tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            for v in videos:
                tmp.write(v + "\n")
            tmp.close()
            self._playlist_file = tmp.name
            cmd = ["mpv", f"--input-ipc-server={self._socket_path}",
                   f"--playlist={self._playlist_file}",
                   "--loop-playlist", "--no-terminal", "--really-quiet"]
        else:
            cmd = ["mpv", f"--input-ipc-server={self._socket_path}",
                   "--loop", "--no-terminal", "--really-quiet", videos[0]]

        print(f"[MPV-IPC] Launching: {' '.join(cmd)}")
        self._mpv_proc = subprocess.Popen(cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
        print(f"[MPV-IPC] mpv started (PID {self._mpv_proc.pid})")

    def _connect(self, timeout=10.0):
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self._socket_path)
                s.settimeout(2.0)
                self._sock = s
                self._connected = True
                print(f"[MPV-IPC] Connected at {self._socket_path}")
                return True
            except (FileNotFoundError, ConnectionRefusedError) as e:
                last_err = e
                time.sleep(0.2)
        print(f"[MPV-IPC] Could not connect: {last_err}")
        return False

    def _reconnect(self):
        self._connected = False
        if self._sock:
            try: self._sock.close()
            except: pass
            self._sock = None
        return self._connect(timeout=5.0)

    def _send(self, command):
        msg = json.dumps({"command": command}) + "\n"
        with self._lock:
            for attempt in range(2):
                try:
                    if not self._connected or not self._sock:
                        if not self._reconnect(): return False
                    self._sock.sendall(msg.encode())
                    return True
                except (BrokenPipeError, OSError):
                    if attempt == 0: self._reconnect()
                    else: return False
        return False

    def play_pause(self):   self._send(["cycle", "pause"]);          return "play_pause"
    def stop(self):         self._send(["stop"]);                    return "stop"
    def volume_up(self):    self._send(["add", "volume", VOLUME_STEP]); return "volume_up"
    def volume_down(self):  self._send(["add", "volume", -VOLUME_STEP]); return "volume_down"
    def seek_forward(self): self._send(["seek", SEEK_SECONDS, "relative"]); return "seek_forward"
    def seek_back(self):    self._send(["seek", -SEEK_SECONDS, "relative"]); return "seek_back"
    def next_track(self):   self._send(["playlist-next", "force"]);  return "next_track"
    def prev_track(self):   self._send(["playlist-prev", "force"]);  return "prev_track"

    def dispatch(self, gesture):
        actions = {
            "fist":            self.play_pause,
            "palm":            self.volume_up,
            "stop":            self.stop,
            "two_up":          self.next_track,
            "two_up_inverted": self.prev_track,
            "ok":              self.seek_forward,
        }
        fn = actions.get(gesture)
        return fn() if fn else f"unknown: {gesture}"

    def __del__(self):
        for attr, action in [('_sock', 'close'), ('_mpv_proc', 'terminate')]:
            obj = getattr(self, attr, None)
            if obj:
                try: getattr(obj, action)()
                except: pass
        pf = getattr(self, '_playlist_file', None)
        if pf and os.path.exists(pf):
            os.remove(pf)
