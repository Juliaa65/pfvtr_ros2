#!/usr/bin/env python3
"""Print a YAML config file's top-level keys as `KEY=value` shell assignments.

Usage: source <(python3 yaml_to_env.py mapping.cfg)

Booleans print as true/false (so they drop straight into a ROS2 action-goal
YAML string); everything else is shell-quoted.
"""
import shlex
import sys

import yaml


def main() -> None:
    with open(sys.argv[1]) as f:
        data = yaml.safe_load(f) or {}

    for key, value in data.items():
        if isinstance(value, bool):
            shell_value = "true" if value else "false"
        else:
            shell_value = str(value)
        print(f"{key.upper()}={shlex.quote(shell_value)}")


if __name__ == "__main__":
    main()
