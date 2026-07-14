import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from web_controller import WebPokerController


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "web" / "static"
controller = WebPokerController()


class PokerWebHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self._send_json(controller.public_state())
            return
        if parsed.path in {"/", "/index.html"}:
            self._send_file(STATIC_DIR / "index.html")
            return
        if parsed.path.startswith("/static/"):
            relative_path = parsed.path.removeprefix("/static/")
            self._send_file(STATIC_DIR / relative_path)
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            data = self._read_json()
            if parsed.path == "/api/start":
                state = controller.start(
                    data.get("player_types", []),
                    db_filename=data.get("db", "LearningAgent_Shared_db.json"),
                    log_file=data.get("log", "web_state_log.txt"),
                    starting_chips=int(data.get("starting_chips", 1000)),
                    ante=int(data.get("ante", 1)),
                    replay_dir=data.get("replay_dir", "replays"),
                    game_mode=data.get("game_mode", "cash"),
                )
                self._send_json(state)
                return
            if parsed.path in {"/api/next_round", "/api/next_hand"}:
                state = controller.start_next_round()
                self._send_json(state)
                return
            if parsed.path == "/api/discard":
                state = controller.submit_discard(
                    data["player"],
                    int(data["discard_index"]),
                    int(data["reveal_index"]),
                )
                self._send_json(state)
                return
            if parsed.path == "/api/action":
                state = controller.submit_action(data["player"], data["action"])
                self._send_json(state)
                return
            self._send_json({"error": "Not found"}, status=404)
        except Exception as exc:
            self._send_json({"error": str(exc), "state": controller.public_state()}, status=400)

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw_body = self.rfile.read(length).decode("utf-8")
        return json.loads(raw_body)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        raw_payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw_payload)))
        self.end_headers()
        self.wfile.write(raw_payload)

    def _send_file(self, path: Path) -> None:
        try:
            resolved_path = path.resolve()
            resolved_path.relative_to(STATIC_DIR.resolve())
        except ValueError:
            self._send_json({"error": "Invalid static path"}, status=403)
            return
        if not resolved_path.exists() or not resolved_path.is_file():
            self._send_json({"error": "Not found"}, status=404)
            return

        content = resolved_path.read_bytes()
        content_type = mimetypes.guess_type(str(resolved_path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local web GUI for 7-stud poker.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), PokerWebHandler)
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
