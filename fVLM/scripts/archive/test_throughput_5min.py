#!/usr/bin/env python3
"""Quick 5-minute test of the precompute pipeline to verify throughput."""

import os
import sys

# Modify the config for testing
os.environ["PRECOMPUTE_TEST_MODE"] = "1"

# Import and modify the main script's config
import importlib.util
spec = importlib.util.spec_from_file_location("precompute", "fVLM/scripts/precompute_6h_max.py")
module = importlib.util.module_from_spec(spec)

# Patch config before loading
import precompute_6h_max_test

