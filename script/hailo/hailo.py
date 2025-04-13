# -*- coding: UTF-8 -*-

import hailo_platform

devices = hailo_platform.scan_devices()
print("Detected Hailo devices:", devices)

