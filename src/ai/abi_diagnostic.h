#pragma once
#include <cstdint>
// C diagnostic boundary: no STL objects until layout checks pass.
struct HermesAbiSnapshot {
    uint32_t pointer_size, string_size, string_align, image_config_size;
    uint32_t image_path_offset, motion_config_size, fall_config_size;
    uint32_t ofstream_size, ostringstream_size, cxx11_abi, glibcxx_date, cplusplus;
};
struct HermesIoProbe {
    uint32_t initial_state, text_state, int_state, uint64_state, double_state;
    uint32_t flush_state, close_state;
    int error_number, exception, num_put_available;
};
extern "C" {
void hermes_abi_snapshot(HermesAbiSnapshot* out);
// kind: 0=ofstream, 1=classic locale ofstream, 2=ostringstream -> fwrite, 3=stdio.
int hermes_abi_io_probe(const char* path, int kind, HermesIoProbe* out);
int hermes_abi_string_probe(void* text, void* config);
}
