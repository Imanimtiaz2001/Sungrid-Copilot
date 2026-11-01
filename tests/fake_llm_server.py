"""A tiny local HTTP server speaking just enough of the OpenAI and Anthropic APIs to test
our adapters end-to-end through the real SDKs (no network, no keys)."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLLMServer:
    def __init__(self):
        self.requests: list[dict] = []
        self.text_reply = "Bills are due within 21 days [1]."
        self.reject_legacy = False
        self.tool_args = {"intent": "eligibility_check", "program": "rooftop", "household_zip": "94103"}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "body": body, "headers": dict(self.headers)})
                if self.path.endswith("/chat/completions") and outer.reject_legacy and "max_tokens" in body:
                    err = json.dumps(
                        {"error": {"message": "Unsupported parameter: 'max_tokens'", "type": "invalid_request_error"}}
                    ).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return
                if self.path.endswith("/chat/completions"):
                    resp = outer._openai(body)
                elif self.path.endswith("/messages"):
                    resp = outer._anthropic(body)
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def _openai(self, body):
        msg = {"role": "assistant", "content": self.text_reply}
        finish = "stop"
        if body.get("tools"):
            name = body["tools"][0]["function"]["name"]
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(self.tool_args)},
                    }
                ],
            }
            finish = "tool_calls"
        return {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def _anthropic(self, body):
        if body.get("tools"):
            content = [{"type": "tool_use", "id": "toolu_1", "name": body["tools"][0]["name"], "input": self.tool_args}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": self.text_reply}]
            stop = "end_turn"
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

    def close(self):
        self.httpd.shutdown()
