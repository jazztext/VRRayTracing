// Logger.cpp

#include "Logger.h"
#include <stdarg.h>
#include <time.h>
#include <iostream>

#ifndef PROJECT_NAME
// This macro should be defined in CMakeLists.txt
#define PROJECT_NAME "RiftSkeleton"
#endif

/// Default constructor: called the first time Instance() is called.
/// Open the output file.
Logger::Logger()
{
    m_stream.open(PROJECT_NAME "-log.txt");
    Write(PROJECT_NAME "log");
}

/// Flush and close the output file.
Logger::~Logger()
{
    m_stream.close();
}

/// Write a message to the log's output stream.
///@param format The string to write to the log.
void Logger::Write(const char* format, ...)
{
    const unsigned int bufSz = 1024;
    char buffer[bufSz];

    va_list args;
    va_start(args, format);
    vsnprintf(buffer, bufSz, format, args);
    va_end(args);

    // Prefix entry by timestamp
    char timestamp[128];
    struct tm* tm;
    time_t t = time(0);
    tm = gmtime(&t);
    strftime(timestamp, sizeof(timestamp), "%Y %b %d %H:%M:%S - ", tm);

    m_stream << timestamp << buffer << std::endl;
    std::cout << buffer << std::endl;
}
