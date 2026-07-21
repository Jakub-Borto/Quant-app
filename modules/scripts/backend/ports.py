"""Free-port allocation for Streamlit script instances.

The port is pre-allocated here (instead of letting Streamlit pick one) so the
instance's URL and label are known before the server prints anything. The
bind-then-close race is closed well enough by `exclude` — the ports already
handed to instances that haven't bound yet; if Streamlit still loses the race
it exits with "Port N is already in use", which surfaces as a crashed
instance with the message in its retained log.
"""

import socket


def find_free_port(exclude: set[int] = frozenset()) -> int:
    """An OS-assigned free TCP port on 127.0.0.1, not in `exclude`."""
    for _ in range(20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in exclude:
            return port
    raise RuntimeError("Could not find a free port for the Streamlit server")
