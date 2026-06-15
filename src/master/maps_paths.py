"""Fixed map storage paths — independent of process cwd."""

import os

TAROS_WS_DIR = "/home/robot/workspaces/taros_autonomy_ws"
TAROS_MAPS_DIR = os.path.join(TAROS_WS_DIR, "maps")


def maps_dir() -> str:
    return os.environ.get("TAROS_MAPS_DIR", TAROS_MAPS_DIR)


def map_path(name: str, *parts: str) -> str:
    return os.path.join(maps_dir(), name, *parts)
