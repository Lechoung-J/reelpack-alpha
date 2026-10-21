"""RapidOCR child entry point: deny network before importing optional packages."""
import runpy
from pathlib import Path
import socket
import sys


def blocked(*args, **kwargs):
    raise RuntimeError('RapidOCR network access and automatic downloads are disabled')


if __name__ == '__main__':
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    worker = str(Path(__file__).with_name('optional_backend_worker.py'))
    sys.argv[0] = worker
    runpy.run_path(worker, run_name='__main__')
