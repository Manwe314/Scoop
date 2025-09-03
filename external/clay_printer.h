// clay_printer.h
#pragma once
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdint>
#include <algorithm> // std::clamp
#include "clay.h"

// Helper: indent
static inline void indent(std::ostream& os, int n) {
    for (int i = 0; i < n; ++i) os.put(' ');
}

// Helper: stringify color (both 0-255 and normalized)
static inline void printColor(std::ostream& os, const Clay_Color& c) {
    int R = std::clamp((int)std::lround(c.r), 0, 255);
    int G = std::clamp((int)std::lround(c.g), 0, 255);
    int B = std::clamp((int)std::lround(c.b), 0, 255);
    int A = std::clamp((int)std::lround(c.a), 0, 255);
    os << "rgba(" << R << "," << G << "," << B << "," << A << ")"
       << " [" << std::fixed << std::setprecision(3)
       << (c.r/255.f) << "," << (c.g/255.f) << "," << (c.b/255.f) << "," << (c.a/255.f) << "]";
}

// Helper: bbox
static inline void printBBox(std::ostream& os, const Clay_BoundingBox& bb) {
    os << "{x:" << bb.x << ", y:" << bb.y << ", w:" << bb.width << ", h:" << bb.height << "}";
}

// Helper: corner radius
static inline void printCornerRadius(std::ostream& os, const Clay_CornerRadius& cr) {
    os << "{tl:" << cr.topLeft
       << ", tr:" << cr.topRight
       << ", br:" << cr.bottomRight
       << ", bl:" << cr.bottomLeft << "}";
}

// Helper: border widths
static inline void printBorderWidth(std::ostream& os, const Clay_BorderWidth& bw) {
    os << "{l:" << bw.left << ", r:" << bw.right << ", t:" << bw.top << ", b:" << bw.bottom
       << ", betweenChildren:" << bw.betweenChildren << "}";
}

// Helper: string from Clay_StringSlice (not null-terminated!)
static inline std::string toString(const Clay_StringSlice& s) {
    return std::string(s.chars, s.length);
}

// Map command type to string (covers current public types)
static inline const char* cmdTypeToStr(uint8_t t) {
    switch (t) {
        case CLAY_RENDER_COMMAND_TYPE_NONE:          return "NONE";
        case CLAY_RENDER_COMMAND_TYPE_RECTANGLE:     return "RECTANGLE";
        case CLAY_RENDER_COMMAND_TYPE_TEXT:          return "TEXT";
        case CLAY_RENDER_COMMAND_TYPE_BORDER:        return "BORDER";
        case CLAY_RENDER_COMMAND_TYPE_SCISSOR_START: return "SCISSOR_START";
        case CLAY_RENDER_COMMAND_TYPE_SCISSOR_END:   return "SCISSOR_END";
        case CLAY_RENDER_COMMAND_TYPE_IMAGE:         return "IMAGE";
        case CLAY_RENDER_COMMAND_TYPE_CUSTOM:        return "CUSTOM";
        default:                                     return "UNKNOWN";
    }
}

// Print a single command
static inline void PrintRenderCommand(const Clay_RenderCommand& rc, std::ostream& os, int pad = 2) {
    indent(os, pad);   os << "- id: " << rc.id << "  z:" << rc.zIndex
                          << "  type:" << cmdTypeToStr(rc.commandType) << "\n";
    indent(os, pad+2); os << "bbox: "; printBBox(os, rc.boundingBox); os << "\n";

    // Render-data per type
    switch (rc.commandType) {
        case CLAY_RENDER_COMMAND_TYPE_RECTANGLE: {
            const Clay_RectangleRenderData& cfg = rc.renderData.rectangle;
            indent(os, pad+2); os << "rectangle.background: "; printColor(os, cfg.backgroundColor); os << "\n";
            indent(os, pad+2); os << "rectangle.cornerRadius: "; printCornerRadius(os, cfg.cornerRadius); os << "\n";
        } break;

        case CLAY_RENDER_COMMAND_TYPE_TEXT: {
            const Clay_TextRenderData& cfg = rc.renderData.text;
            indent(os, pad+2); os << "text: \"" << toString(cfg.stringContents) << "\"\n";
            indent(os, pad+2); os << "text.color: "; printColor(os, cfg.textColor); os << "\n";
            indent(os, pad+2); os << "text.fontId:" << cfg.fontId
                                  << " size:" << cfg.fontSize
                                  << " letterSpacing:" << cfg.letterSpacing
                                  << " lineHeight:" << cfg.lineHeight << "\n";
        } break;

        case CLAY_RENDER_COMMAND_TYPE_BORDER: {
            const Clay_BorderRenderData& cfg = rc.renderData.border;
            indent(os, pad+2); os << "border.color: "; printColor(os, cfg.color); os << "\n";
            indent(os, pad+2); os << "border.cornerRadius: "; printCornerRadius(os, cfg.cornerRadius); os << "\n";
            indent(os, pad+2); os << "border.widths: "; printBorderWidth(os, cfg.width); os << "\n";
        } break;

        case CLAY_RENDER_COMMAND_TYPE_IMAGE: {
            const Clay_ImageRenderData& cfg = rc.renderData.image;
            indent(os, pad+2); os << "image.bg: "; printColor(os, cfg.backgroundColor); os << "\n";
            indent(os, pad+2); os << "image.cornerRadius: "; printCornerRadius(os, cfg.cornerRadius); os << "\n";
            indent(os, pad+2); os << "image.data:" << cfg.imageData << "\n";
        } break;

        case CLAY_RENDER_COMMAND_TYPE_SCISSOR_START:
        case CLAY_RENDER_COMMAND_TYPE_SCISSOR_END: {
            // Clip render data is an alias of scroll/clip flags {horizontal, vertical}
            const Clay_ClipRenderData& cfg = rc.renderData.clip;
            indent(os, pad+2); os << "clip.horizontal:" << (cfg.horizontal ? "true" : "false")
                                  << " clip.vertical:" << (cfg.vertical ? "true" : "false") << "\n";
        } break;

        case CLAY_RENDER_COMMAND_TYPE_CUSTOM: {
            const Clay_CustomRenderData& cfg = rc.renderData.custom;
            indent(os, pad+2); os << "custom.userData:" << cfg.customData << "\n";
        } break;

        default:
            // nothing more to print
            break;
    }

    if (rc.userData) {
        indent(os, pad+2); os << "userData:" << rc.userData << "\n";
    }
}

// Print the whole array
static inline void PrintRenderCommandArray(const Clay_RenderCommandArray& arr,
                                           std::ostream& os = std::cout) {
    os << "RenderCommandArray length=" << arr.length
       << " capacity=" << arr.capacity << "\n";
    for (int i = 0; i < arr.length; ++i) {
        const Clay_RenderCommand& rc = arr.internalArray[i]; // as shown in the README
        PrintRenderCommand(rc, os, /*pad=*/2);
    }
}
