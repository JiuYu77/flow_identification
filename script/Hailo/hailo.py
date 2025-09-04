# -*- coding: UTF-8 -*-

# import hailo_platform

devices = hailo_platform.scan_devices()
print("Detected Hailo devices:", devices)

import hailo
print("支持的硬件架构:", hailo.SUPPORTED_HW_ARCHS)
