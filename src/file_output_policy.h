#pragma once
#include <cstdlib>
#include <cstring>
// Production builds cannot write SDK diagnostic/recording files, even when old
// sdk.ini or environment settings request them. Lab builds must opt in twice:
// -DHERMES_ENABLE_FILE_OUTPUT=1 and HERMES_FILE_OUTPUT=1 before construction.
#ifndef HERMES_ENABLE_FILE_OUTPUT
#define HERMES_ENABLE_FILE_OUTPUT 0
#endif
namespace hermes_io {
inline bool enabled() {
#if HERMES_ENABLE_FILE_OUTPUT
 const char* value=std::getenv("HERMES_FILE_OUTPUT");
 return value&&std::strcmp(value,"1")==0;
#else
 return false;
#endif
}
}
