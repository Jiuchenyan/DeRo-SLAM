#pragma once

#include <iostream>
#include <string>
#include <cstdarg>
#include <utility>

// Color codes for terminal output
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */


template<typename T>
inline auto to_cstr(T&& t) noexcept -> decltype(std::forward<T>(t)) {
    return std::forward<T>(t);
}

inline const char* to_cstr(const std::string& s) noexcept {
    return s.c_str();
}

inline const char* to_cstr(std::string& s) noexcept {
    return s.c_str();
}

inline const char* to_cstr(std::string&& s) noexcept {
    return s.c_str();
}

static inline std::string format_internal(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    
    int size = vsnprintf(nullptr, 0, fmt, args);
    va_end(args);
    
    if (size <= 0) return "";
    
    std::string buffer(size, '\0');
    
    va_start(args, fmt);
    vsnprintf(&buffer[0], size + 1, fmt, args);
    va_end(args);
    
    return buffer;
}


template<typename... Args>
inline void mprt(const char* color, const char* fmt, Args&&... args) {
    std::string buffer = format_internal(fmt, to_cstr(std::forward<Args>(args))...);
    std::cout << color << buffer << RESET;
    std::cout.flush();
}

template<typename... Args>
inline void mprt(const char* fmt, Args&&... args) {
    std::string buffer = format_internal(fmt, to_cstr(std::forward<Args>(args))...);
    std::cout << buffer;
    std::cout.flush();
}

template<typename... Args>
inline void mprtln(const char* color, const char* fmt, Args&&... args) {
    std::string buffer = format_internal(fmt, to_cstr(std::forward<Args>(args))...);
    std::cout << color << buffer << RESET << std::endl;
}

template<typename... Args>
inline void mprtln(const char* fmt, Args&&... args) {
    std::string buffer = format_internal(fmt, to_cstr(std::forward<Args>(args))...);
    std::cout << buffer << std::endl;
}