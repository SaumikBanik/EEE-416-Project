#pragma once

// Same network arrangement as data collection:
// 1) Turn on the laptop's 2.4 GHz Windows hotspot.
// 2) Use ipconfig to find the hotspot adapter IPv4 address.
// 3) Fill these three values before uploading to ESP32.

#define LIVE_WIFI_SSID       "LAPTOP-8OR98973 8605"
#define LIVE_WIFI_PASSWORD   "4=U237h2"
#define LIVE_SERVER_IP       "192.168.137.1"
#define LIVE_SERVER_PORT     5000
