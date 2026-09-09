"""Allows ``python -m pharos``."""

import sys

from pharos import cli

if __name__ == "__main__":
  sys.exit(cli.main())
