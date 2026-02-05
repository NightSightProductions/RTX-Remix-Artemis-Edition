#pragma once

// RTX MegaGeo verbose logging control
// Set to 1 to enable all info logging for debugging, 0 to disable for performance
#define RTXMG_VERBOSE_LOGGING 0

#if RTXMG_VERBOSE_LOGGING
#define RTXMG_LOG(msg) dxvk::Logger::info(msg)
#else
#define RTXMG_LOG(msg) ((void)0)
#endif
