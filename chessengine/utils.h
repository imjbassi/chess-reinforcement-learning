#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>

namespace ChessEngine {

// String helpers shared across the engine.
std::string trim(const std::string& str);
std::vector<std::string> split(const std::string& str, char delimiter);
std::string toLower(const std::string& str);
std::string toUpper(const std::string& str);
bool startsWith(const std::string& str, const std::string& prefix);
bool endsWith(const std::string& str, const std::string& suffix);

} // namespace ChessEngine

#endif
