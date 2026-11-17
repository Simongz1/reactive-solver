#pragma once
/* #undef EXODUS_THREADSAFE */

#ifndef SEACAS_DEPRECATED
#  if (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#    define SEACAS_DEPRECATED  __attribute__((__deprecated__))
#  else
#    define SEACAS_DEPRECATED
#  endif
#endif

#ifndef SEACAS_DEPRECATED_MSG
#  if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 5))
#    define SEACAS_DEPRECATED_MSG(MSG)  __attribute__((__deprecated__ (#MSG) ))
#  elif (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#    define SEACAS_DEPRECATED_MSG(MSG)  __attribute__((__deprecated__))
#  else
#    define SEACAS_DEPRECATED_MSG(MSG)
#  endif
#endif


