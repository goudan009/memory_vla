#!/usr/bin/env python3
"""Serve only the AI frontend prototype, without importing any execution managers."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ASSETS = {
    '/': ('ai_workbench.html', 'text/html'),
    '/ai': ('ai_workbench.html', 'text/html'),
    '/ai/': ('ai_workbench.html', 'text/html'),
    '/ai_workbench.html': ('ai_workbench.html', 'text/html'),
    '/ai_workbench.css': ('ai_workbench.css', 'text/css'),
    '/ai_workbench.js': ('ai_workbench.js', 'application/javascript'),
    '/ai/architecture': ('ai_architecture.html', 'text/html'),
    '/ai/architecture/': ('ai_architecture.html', 'text/html'),
    '/ai_architecture.html': ('ai_architecture.html', 'text/html'),
    '/ai_architecture.css': ('ai_architecture.css', 'text/css'),
    '/ai_architecture.js': ('ai_architecture.js', 'application/javascript'),
}


class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        asset = ASSETS.get(urlparse(self.path).path)
        if asset is None:
            self.send_error(404)
            return
        name, content_type = asset
        body = Path(__file__).with_name(name).read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8772)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), PreviewHandler)
    print(f'AI 工作流前端预览：http://{args.host}:{server.server_port}/ai', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
