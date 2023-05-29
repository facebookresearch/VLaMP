# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.


import os
from pathlib import Path
from typing import Literal

DEBUG = bool(os.environ.get("TSL_DEBUG", False))
ROOT_DIR = Path(os.environ.get("ROOT_DIR", "."))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT_DIR / ".data"))

ObservationType = Literal["pre", "post"]
