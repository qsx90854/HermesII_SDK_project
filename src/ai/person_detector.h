#pragma once
#include <cstdint>
#include <string>
#include <vector>
namespace hermes_person {
// Backend-neutral contract; see experiments/yolo_bg/LINUX_AI_PORTING.md.
// Box coordinates are float pixel edges in the submitted image, not model/letterbox
// coordinates or normalized values. x2/y2 may equal w/h. Controller restores any
// excluded top-row offset exactly once. Classes: 0=normal person, 1=fallen person;
// both count as person evidence (neither directly emits a fall event).
struct Box { float x1=0,y1=0,x2=0,y2=0,confidence=0; int class_id=0; };
struct Config { std::string model_path, runtime_path; float confidence=.25f,nms_iou=.45f; int threads=4; };
// Echo request_id and the source frame timestamp in milliseconds unchanged.
// Use the same monotonic stream time as SDK Step(), NOT inference completion time.
// success=true with no persons is a valid negative; success=false is unknown/error.
// inference_ms measures backend execution; total_ms includes preprocessing/postprocessing.
struct Result { uint64_t request_id=0,source_timestamp_ms=0; bool success=false; std::vector<Box> persons; double inference_ms=0,total_ms=0; std::string error; };
class Detector {
public:
 virtual ~Detector()=default;
// Input: packed uint8 grayscale, w*h bytes, row stride=w; no padded pitch API.
// Nonblocking admission: false means busy/stopped, not a negative person result.
// Windows copies the input before returning true; replacement backends must own/copy
// accepted data so the caller can reuse its frame buffer immediately.
// One outstanding request, including a completed result not yet collected.
 virtual bool TrySubmit(const uint8_t* gray,int w,int h,uint64_t stamp,uint64_t id)=0;
// Nonblocking poll: false means no completed result. true transfers one Result,
// including failures, and releases the outstanding slot. Do not fabricate negatives.
 virtual bool TryGetResult(Result& result)=0;
// Stop admission and safely finish/join in-flight work before releasing resources.
// Windows may discard queued work on shutdown; never depend on a final result.
// The controller recreates the detector after a stream reset when needed.
 virtual void Shutdown()=0;
};
}
