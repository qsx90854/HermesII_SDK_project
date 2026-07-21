// ini_read_test.cpp
//
// Isolation test: reads an .ini file and prints exactly what was read --
// NOTHING ELSE. No HermesII_sdk.h, no SDK calls, no NPU, no Init(). Purpose:
// find out whether reading/parsing a physical file (possibly over NFS) is
// itself unreliable on the edge board, independent of any SDK/NPU code path.
//
// Usage:
//   ./ini_read_test [path=sdk_gray_count_falls.ini]
//
// Prints, in order:
//   1. Raw file size in bytes.
//   2. The raw file content, byte for byte, as text.
//   3. A hex+ASCII dump of the raw bytes (so invisible corruption, CRLF,
//      truncation, or stray bytes are impossible to miss).
//   4. Every [section].key = value pair the INI parser extracted.
//   5. A focused deep-dive on Bed.Points: the raw string value, its exact
//      byte length, a hex dump of just that value, and the parsed x,y pairs
//      -- this is the exact value that showed up corrupted when read by
//      sdk_gray_count_falls.cpp on the edge board.
//
// If THIS tool prints a clean, correct Bed.Points value on the board, the
// corruption is happening somewhere after this point (SDK/NPU init).
// If THIS tool already prints a garbled value, the problem is in file
// reading itself (NFS, libc, or the ARM toolchain) -- not the SDK.

#include <cctype>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

// Byte-for-byte the same INI loader used by the other examples
// (examples/sdk_gray_test.cpp, sdk_gray_count_falls.cpp, event_replay_verify.cpp)
// so this test exercises the identical code path.
class SimpleConfig {
    std::map<std::string, std::string> settings;
public:
    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        std::string line, section;
        while (std::getline(f, line)) {
            size_t first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || line[first] == ';' || line[first] == '#') continue;
            std::string trimmed = line.substr(first);
            if (trimmed[0] == '[') {
                size_t end = trimmed.find(']');
                if (end != std::string::npos && end > 1) section = trimmed.substr(1, end - 1);
            } else {
                size_t eq = trimmed.find('=');
                if (eq != std::string::npos) {
                    std::string key = trimmed.substr(0, eq);
                    std::string val = trimmed.substr(eq + 1);
                    size_t k_last = key.find_last_not_of(" \t\r\n");
                    if (k_last != std::string::npos) key.erase(k_last + 1);
                    size_t v_first = val.find_first_not_of(" \t\r\n");
                    if (v_first != std::string::npos) val.erase(0, v_first);
                    size_t v_last = val.find_last_not_of(" \t\r\n");
                    if (v_last != std::string::npos) val.erase(v_last + 1);
                    settings[section.empty() ? key : (section + "." + key)] = val;
                }
            }
        }
        return true;
    }
    const std::map<std::string, std::string>& all() const { return settings; }
    std::string getStr(const std::string& key, const std::string& def) const {
        auto it = settings.find(key);
        return it != settings.end() ? it->second : def;
    }
};

// Byte-for-byte the same bed-point parser used by sdk_gray_count_falls.cpp.
std::vector<std::pair<int, int>> ParseBedPoints(const std::string& s) {
    std::vector<int> nums;
    std::string cur;
    for (char c : s + ",") {
        if (c == ',') { if (!cur.empty()) { nums.push_back(std::atoi(cur.c_str())); cur.clear(); } }
        else cur += c;
    }
    std::vector<std::pair<int, int>> pts;
    for (size_t i = 0; i + 1 < nums.size(); i += 2) pts.push_back({nums[i], nums[i + 1]});
    return pts;
}

void HexDump(const std::string& data) {
    const unsigned char* p = (const unsigned char*)data.data();
    size_t n = data.size();
    for (size_t off = 0; off < n; off += 16) {
        printf("  %06zx: ", off);
        size_t line_len = (n - off < 16) ? (n - off) : 16;
        for (size_t i = 0; i < 16; ++i) {
            if (i < line_len) printf("%02x ", p[off + i]);
            else printf("   ");
            if (i == 7) printf(" ");
        }
        printf(" |");
        for (size_t i = 0; i < line_len; ++i) {
            unsigned char c = p[off + i];
            putchar(std::isprint(c) ? c : '.');
        }
        printf("|\n");
    }
}

int main(int argc, char** argv) {
    std::string path = (argc >= 2) ? argv[1] : "sdk_gray_count_falls.ini";

    std::cout << "=========================================\n";
    std::cout << "ini_read_test -- pure file I/O, NO SDK, NO NPU\n";
    std::cout << "=========================================\n";
    std::cout << "File: " << path << "\n\n";

    // --- 1 & 2 & 3: raw bytes ---
    std::ifstream raw(path, std::ios::binary);
    if (!raw.is_open()) {
        std::cerr << "Error: cannot open " << path << std::endl;
        return 1;
    }
    std::ostringstream ss;
    ss << raw.rdbuf();
    std::string content = ss.str();
    raw.close();

    std::cout << "--- 1. Raw size ---\n";
    std::cout << content.size() << " bytes\n\n";

    std::cout << "--- 2. Raw content (as text) ---\n";
    std::cout << content << "\n";
    if (!content.empty() && content.back() != '\n') std::cout << "\n";

    std::cout << "--- 3. Hex dump (full file) ---\n";
    HexDump(content);
    std::cout << "\n";

    // --- 4: parsed key/value pairs ---
    SimpleConfig cfg;
    bool loaded = cfg.load(path);
    std::cout << "--- 4. Parsed key/value pairs (load() returned "
              << (loaded ? "true" : "false") << ") ---\n";
    for (const auto& kv : cfg.all()) {
        std::cout << "  " << kv.first << " = \"" << kv.second << "\"  (" << kv.second.size() << " bytes)\n";
    }
    std::cout << "\n";

    // --- 5: focused Bed.Points deep-dive ---
    std::cout << "--- 5. Bed.Points deep-dive ---\n";
    std::string bed_str = cfg.getStr("Bed.Points", "");
    std::cout << "Raw value: \"" << bed_str << "\"\n";
    std::cout << "Length: " << bed_str.size() << " bytes\n";
    std::cout << "Hex:\n";
    HexDump(bed_str);
    std::vector<std::pair<int, int>> pts = ParseBedPoints(bed_str);
    std::cout << "Parsed " << pts.size() << " point(s):";
    for (const auto& p : pts) std::cout << " (" << p.first << "," << p.second << ")";
    std::cout << "\n";
    if (pts.size() == 4) {
        std::cout << "RESULT: OK -- 4 points parsed correctly.\n";
    } else {
        std::cout << "RESULT: BAD -- expected 4 points, got " << pts.size()
                  << ". File reading/parsing itself is unreliable here,\n"
                     "         independent of any SDK/NPU code -- this tool never touches either.\n";
    }
    std::cout << "=========================================\n";

    return (pts.size() == 4) ? 0 : 2;
}
