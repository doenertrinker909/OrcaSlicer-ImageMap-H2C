#ifndef WipeTower_
#define WipeTower_

#include <cmath>
#include <string>
#include <sstream>
#include <utility>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <vector>

#include "libslic3r/Point.hpp"
#include "libslic3r/Polygon.hpp"
#include "ColorSolver.hpp"
#include "libslic3r/TextureMapping.hpp"
#include "libslic3r/Polyline.hpp"
#include "libslic3r/TriangleMesh.hpp"
#include <unordered_set>
#include "libslic3r/MultiNozzleUtils.hpp"
namespace Slic3r
{

class WipeTowerWriter;
class PrintConfig;
enum GCodeFlavor : unsigned char;

struct PrimeTowerTextureRenderSettings
{
    enum ColorMode : int {
        Auto = 0,
        GenericSolver,
        CMY,
        CMYK,
        CMYW,
        RGB,
        RGBK,
        RGBW,
        BW,
        CMYKW,
        RGBKW
    };

    bool enabled = false;
    float angle_offset_deg = 0.f;
    bool preserve_aspect_ratio = false;
    int color_mode = Auto;
    bool generic_fallback_for_missing_channels = false;
    bool compact_offset_mode = true;
    bool settings_zone_enabled = false;
    int texture_mapping_mode = int(TextureMappingZone::TextureMappingFilamentBlending);
    int texture_filament_color_mode = int(TextureMappingZone::FilamentColorAny);
    int generic_solver_lookup_mode = TextureMappingZone::DefaultGenericSolverLookupMode;
    int generic_solver_mode = TextureMappingZone::DefaultGenericSolverMode;
    int generic_solver_mix_model = TextureMappingZone::DefaultGenericSolverMixModel;
    float filament_overhang_contrast_pct = TextureMappingZone::DefaultFilamentOverhangContrastPct;
    float tone_gamma = 1.f;
    float global_strength = 1.f;
    float max_line_width = 0.95f;
    float min_line_width = 0.32f;
    std::vector<uint8_t> image_rgba;
    unsigned int image_width = 0;
    unsigned int image_height = 0;
    std::vector<uint8_t> image_rgba_back;
    unsigned int image_width_back = 0;
    unsigned int image_height_back = 0;
    std::vector<std::string> filament_colours;
    std::vector<size_t> tool_indices;
    std::vector<unsigned int> component_ids;
    std::vector<float> filament_strengths_pct;
    std::vector<float> filament_minimum_offsets_pct;
    float z_min = 0.f;
    float z_max = 0.f;

    bool valid() const
    {
        return enabled && (image_valid(false) || image_valid(true));
    }

    float sample_tool_visibility(size_t tool, float u, float v, float wrap_width_mm = 0.f, float height_mm = 0.f) const
    {
        const float raw_visibility = sample_tool_visibility_raw(tool, u, v, nullptr, wrap_width_mm, height_mm);
        if (!compact_offset_mode)
            return adjusted_tool_visibility(tool, raw_visibility);

        float max_visibility = std::clamp(raw_visibility, 0.f, 1.f);
        if (!tool_indices.empty()) {
            for (const size_t candidate_tool : tool_indices) {
                if (candidate_tool != tool) {
                    const float candidate_visibility = sample_tool_visibility_raw(candidate_tool, u, v, nullptr, wrap_width_mm, height_mm);
                    max_visibility = std::max(max_visibility, std::clamp(candidate_visibility, 0.f, 1.f));
                }
            }
        } else {
            for (size_t candidate_tool = 0; candidate_tool < filament_colours.size(); ++candidate_tool) {
                if (candidate_tool != tool) {
                    const float candidate_visibility = sample_tool_visibility_raw(candidate_tool, u, v, nullptr, wrap_width_mm, height_mm);
                    max_visibility = std::max(max_visibility, std::clamp(candidate_visibility, 0.f, 1.f));
                }
            }
        }
        return max_visibility > 1e-6f ? adjusted_tool_visibility(tool, raw_visibility / max_visibility) :
                                        adjusted_tool_visibility(tool, raw_visibility);
    }

    float sample_tool_visibility(size_t tool,
                                 float u,
                                 float v,
                                 const std::vector<size_t> &normalization_tools,
                                 float wrap_width_mm = 0.f,
                                 float height_mm = 0.f) const
    {
        const float raw_visibility = sample_tool_visibility_raw(tool, u, v, &normalization_tools, wrap_width_mm, height_mm);
        if (!compact_offset_mode || normalization_tools.empty())
            return adjusted_tool_visibility(tool, raw_visibility);

        float max_visibility = std::clamp(raw_visibility, 0.f, 1.f);
        for (const size_t normalization_tool : normalization_tools) {
            const float normalization_visibility =
                sample_tool_visibility_raw(normalization_tool, u, v, &normalization_tools, wrap_width_mm, height_mm);
            max_visibility = std::max(max_visibility, std::clamp(normalization_visibility, 0.f, 1.f));
        }

        return max_visibility > 1e-6f ? adjusted_tool_visibility(tool, raw_visibility / max_visibility) :
                                        adjusted_tool_visibility(tool, raw_visibility);
    }

    std::vector<size_t> component_tools_for_layer_sequence() const
    {
        std::vector<size_t> tools;
        if (!valid())
            return tools;

        if (settings_zone_enabled && !component_ids.empty()) {
            tools.reserve(component_ids.size());
            for (const unsigned int physical_id : component_ids) {
                if (physical_id >= 1 && physical_id <= filament_colours.size()) {
                    const size_t tool = size_t(physical_id - 1);
                    if (std::find(tools.begin(), tools.end(), tool) == tools.end())
                        tools.emplace_back(tool);
                }
            }
            if (!tools.empty())
                return tools;
        }

        if (color_mode == GenericSolver || color_mode == Auto)
            return generic_solver_tool_candidates();

        const std::vector<std::array<float, 3>> ideals = color_mode_ideals(color_mode);
        if (ideals.empty())
            return tools;

        std::vector<size_t> candidates;
        if (!tool_indices.empty()) {
            for (const size_t tool : tool_indices)
                if (tool < filament_colours.size() && std::find(candidates.begin(), candidates.end(), tool) == candidates.end())
                    candidates.emplace_back(tool);
        } else {
            candidates.reserve(filament_colours.size());
            for (size_t tool = 0; tool < filament_colours.size(); ++tool)
                candidates.emplace_back(tool);
        }
        if (candidates.empty())
            return tools;

        std::vector<char> used(filament_colours.size(), 0);
        tools.reserve(ideals.size());
        for (const std::array<float, 3> &ideal : ideals) {
            size_t best_tool = size_t(-1);
            float best_distance = std::numeric_limits<float>::max();
            for (const size_t candidate : candidates) {
                if (candidate >= used.size() || used[candidate])
                    continue;
                const float distance = color_distance2(tool_color(candidate), ideal);
                if (distance < best_distance) {
                    best_distance = distance;
                    best_tool = candidate;
                }
            }
            if (best_tool != size_t(-1)) {
                used[best_tool] = 1;
                tools.emplace_back(best_tool);
            }
        }
        return tools;
    }

    size_t tool_for_layer_sequence(int layer_index) const
    {
        const std::vector<size_t> tools = component_tools_for_layer_sequence();
        if (tools.empty())
            return size_t(-1);
        const int tool_count = int(tools.size());
        const int index = ((layer_index % tool_count) + tool_count) % tool_count;
        return tools[size_t(index)];
    }

private:
    mutable ColorSolverCandidateCache m_color_solver_candidate_cache;

    static std::vector<std::array<float, 3>> color_mode_ideals(int mode)
    {
        switch (mode) {
        case CMY:
            return {std::array<float, 3>{0.f, 1.f, 1.f}, std::array<float, 3>{1.f, 0.f, 1.f}, std::array<float, 3>{1.f, 1.f, 0.f}};
        case CMYK:
            return {std::array<float, 3>{0.f, 1.f, 1.f},
                    std::array<float, 3>{1.f, 0.f, 1.f},
                    std::array<float, 3>{1.f, 1.f, 0.f},
                    std::array<float, 3>{0.f, 0.f, 0.f}};
        case CMYW:
            return {std::array<float, 3>{0.f, 1.f, 1.f},
                    std::array<float, 3>{1.f, 0.f, 1.f},
                    std::array<float, 3>{1.f, 1.f, 0.f},
                    std::array<float, 3>{1.f, 1.f, 1.f}};
        case RGB:
            return {std::array<float, 3>{1.f, 0.f, 0.f}, std::array<float, 3>{0.f, 1.f, 0.f}, std::array<float, 3>{0.f, 0.f, 1.f}};
        case RGBK:
            return {std::array<float, 3>{1.f, 0.f, 0.f},
                    std::array<float, 3>{0.f, 1.f, 0.f},
                    std::array<float, 3>{0.f, 0.f, 1.f},
                    std::array<float, 3>{0.f, 0.f, 0.f}};
        case RGBW:
            return {std::array<float, 3>{1.f, 0.f, 0.f},
                    std::array<float, 3>{0.f, 1.f, 0.f},
                    std::array<float, 3>{0.f, 0.f, 1.f},
                    std::array<float, 3>{1.f, 1.f, 1.f}};
        case BW:
            return {std::array<float, 3>{0.f, 0.f, 0.f}, std::array<float, 3>{1.f, 1.f, 1.f}};
        case CMYKW:
            return {std::array<float, 3>{0.f, 1.f, 1.f},
                    std::array<float, 3>{1.f, 0.f, 1.f},
                    std::array<float, 3>{1.f, 1.f, 0.f},
                    std::array<float, 3>{0.f, 0.f, 0.f},
                    std::array<float, 3>{1.f, 1.f, 1.f}};
        case RGBKW:
            return {std::array<float, 3>{1.f, 0.f, 0.f},
                    std::array<float, 3>{0.f, 1.f, 0.f},
                    std::array<float, 3>{0.f, 0.f, 1.f},
                    std::array<float, 3>{0.f, 0.f, 0.f},
                    std::array<float, 3>{1.f, 1.f, 1.f}};
        default: return {};
        }
    }

    float sample_tool_visibility_raw(size_t                     tool,
                                     float                      u,
                                     float                      v,
                                     const std::vector<size_t> *solver_tools,
                                     float                      wrap_width_mm = 0.f,
                                     float                      height_mm = 0.f) const
    {
        if (!valid())
            return 1.f;
        u -= std::floor(u);
        v = std::clamp(v, 0.f, 1.f);
        const bool front_valid = image_valid(false);
        const bool back_valid = image_valid(true);
        bool use_back = false;
        if (front_valid && back_valid) {
            use_back = u >= 0.5f;
            u = use_back ? (u - 0.5f) * 2.f : u * 2.f;
        } else {
            use_back = back_valid;
        }
        const float side_width_mm = front_valid && back_valid ? 0.5f * wrap_width_mm : wrap_width_mm;
        apply_preserved_aspect_ratio(u, v, use_back, side_width_mm, height_mm);
        if (settings_zone_enabled)
            return sample_settings_zone_tool_visibility(tool, u, v, use_back, solver_tools);
        return sample_image_tool_visibility(tool, u, v, use_back, solver_tools);
    }

    void apply_preserved_aspect_ratio(float &u, float &v, bool back, float side_width_mm, float height_mm) const
    {
        if (!preserve_aspect_ratio || side_width_mm <= 1e-6f || height_mm <= 1e-6f)
            return;
        const unsigned int width = back ? image_width_back : image_width;
        const unsigned int height = back ? image_height_back : image_height;
        if (width == 0 || height == 0)
            return;

        const float image_aspect = float(width) / float(height);
        const float target_aspect = side_width_mm / height_mm;
        if (!std::isfinite(image_aspect) || !std::isfinite(target_aspect) || image_aspect <= 1e-6f || target_aspect <= 1e-6f)
            return;

        if (target_aspect > image_aspect + 1e-6f) {
            const float visible_height = std::clamp(image_aspect / target_aspect, 0.f, 1.f);
            v = std::clamp(v * visible_height, 0.f, 1.f);
        } else if (image_aspect > target_aspect + 1e-6f) {
            const float visible_width = std::clamp(target_aspect / image_aspect, 0.f, 1.f);
            u = std::clamp(0.5f * (1.f - visible_width) + u * visible_width, 0.f, 1.f);
        }
    }

    bool image_valid(bool back) const
    {
        return back ?
            image_width_back > 0 && image_height_back > 0 &&
                image_rgba_back.size() >= size_t(image_width_back) * size_t(image_height_back) * 4 :
            image_width > 0 && image_height > 0 &&
                image_rgba.size() >= size_t(image_width) * size_t(image_height) * 4;
    }

    std::array<float, 4> sample_image_rgba(float u, float v, bool back) const
    {
        const std::vector<uint8_t> &rgba = back ? image_rgba_back : image_rgba;
        const unsigned int width = back ? image_width_back : image_width;
        const unsigned int height = back ? image_height_back : image_height;
        if (width == 0 || height == 0 || rgba.size() < size_t(width) * size_t(height) * 4)
            return {1.f, 1.f, 1.f, 1.f};

        const float x = std::clamp(u, 0.f, 1.f) * float(width > 1 ? width - 1 : 0);
        const float y = std::clamp(1.f - v, 0.f, 1.f) * float(height > 1 ? height - 1 : 0);
        const unsigned int x0 = std::min<unsigned int>(width - 1, unsigned(std::floor(x)));
        const unsigned int y0 = std::min<unsigned int>(height - 1, unsigned(std::floor(y)));
        const unsigned int x1 = std::min<unsigned int>(width - 1, x0 + 1);
        const unsigned int y1 = std::min<unsigned int>(height - 1, y0 + 1);
        const float tx = x - float(x0);
        const float ty = y - float(y0);
        auto sample_channel = [&rgba, width](unsigned int sx, unsigned int sy, size_t channel) {
            const size_t idx = (size_t(sy) * size_t(width) + size_t(sx)) * 4 + channel;
            return float(rgba[idx]) / 255.f;
        };

        std::array<float, 4> out{};
        for (size_t channel = 0; channel < out.size(); ++channel) {
            const float c00 = sample_channel(x0, y0, channel);
            const float c10 = sample_channel(x1, y0, channel);
            const float c01 = sample_channel(x0, y1, channel);
            const float c11 = sample_channel(x1, y1, channel);
            const float cx0 = c00 + (c10 - c00) * tx;
            const float cx1 = c01 + (c11 - c01) * tx;
            out[channel] = std::clamp(cx0 + (cx1 - cx0) * ty, 0.f, 1.f);
        }
        return out;
    }

    std::array<float, 3> sample_image_rgb(float u, float v, bool back) const
    {
        const std::array<float, 4> rgba = sample_image_rgba(u, v, back);
        return {rgba[0], rgba[1], rgba[2]};
    }

    float sample_image_tool_visibility(size_t tool, float u, float v, bool back, const std::vector<size_t> *solver_tools = nullptr) const
    {
        const std::array<float, 3> rgb = sample_image_rgb(u, v, back);
        if (color_mode == GenericSolver || color_mode == Auto)
            return generic_solver_tool_visibility(tool, rgb[0], rgb[1], rgb[2], solver_tools);
        return fixed_mode_visibility(tool, rgb[0], rgb[1], rgb[2]);
    }

    static std::array<float, 3> parse_color(const std::string &hex)
    {
        auto hex_byte = [](char hi, char lo) {
            auto nibble = [](char c) {
                if (c >= '0' && c <= '9') return int(c - '0');
                if (c >= 'a' && c <= 'f') return int(c - 'a') + 10;
                if (c >= 'A' && c <= 'F') return int(c - 'A') + 10;
                return 0;
            };
            return float(nibble(hi) * 16 + nibble(lo)) / 255.f;
        };
        if (hex.size() >= 7 && hex[0] == '#')
            return {hex_byte(hex[1], hex[2]), hex_byte(hex[3], hex[4]), hex_byte(hex[5], hex[6])};
        return {1.f, 1.f, 1.f};
    }

    static float color_distance2(const std::array<float, 3> &a, const std::array<float, 3> &b)
    {
        const float dr = a[0] - b[0];
        const float dg = a[1] - b[1];
        const float db = a[2] - b[2];
        return dr * dr + dg * dg + db * db;
    }

    static float print_visibility_strength(float value)
    {
        return std::clamp(std::pow(std::max(0.f, value), 0.85f), 0.f, 1.f);
    }

    static float safe_div(float numerator, float denominator)
    {
        return denominator <= 1e-6f ? 0.f : std::clamp(numerator / denominator, 0.f, 1.f);
    }

    std::array<float, 3> tool_color(size_t tool) const
    {
        return tool < filament_colours.size() ? parse_color(filament_colours[tool]) : std::array<float, 3>{1.f, 1.f, 1.f};
    }

    std::vector<float> generic_solver_weights(const std::vector<std::array<float, 3>> &component_colors,
                                              const std::array<float, 3>              &target) const
    {
        if (component_colors.empty())
            return {};
        const ColorSolverCandidateSet &candidates =
            color_solver_candidates(m_color_solver_candidate_cache,
                                    component_colors,
                                    color_solver_mix_model_from_index(generic_solver_mix_model));
        return solve_color_solver_weights_for_target(candidates,
                                                     target,
                                                     color_solver_lookup_mode_from_index(generic_solver_lookup_mode),
                                                     color_solver_mode_from_index(TextureMappingZone::effective_generic_solver_mode(generic_solver_mode)));
    }

    std::vector<size_t> generic_solver_tool_candidates(const std::vector<size_t> *preferred_tools = nullptr) const
    {
        std::vector<size_t> tools;
        if (preferred_tools != nullptr && !preferred_tools->empty()) {
            tools.reserve(preferred_tools->size());
            for (const size_t tool : *preferred_tools)
                if (tool < filament_colours.size() && std::find(tools.begin(), tools.end(), tool) == tools.end())
                    tools.emplace_back(tool);
        } else if (!tool_indices.empty()) {
            tools.reserve(tool_indices.size());
            for (const size_t tool : tool_indices) {
                if (tool < filament_colours.size() && std::find(tools.begin(), tools.end(), tool) == tools.end())
                    tools.emplace_back(tool);
            }
        } else {
            tools.reserve(filament_colours.size());
            for (size_t tool = 0; tool < filament_colours.size(); ++tool)
                tools.emplace_back(tool);
        }
        return tools;
    }

    float generic_solver_tool_visibility(size_t tool, float r, float g, float b, const std::vector<size_t> *solver_tools = nullptr) const
    {
        const std::vector<size_t> tools = generic_solver_tool_candidates(solver_tools);
        std::vector<std::array<float, 3>> component_colors;
        component_colors.reserve(tools.size());
        for (const size_t candidate_tool : tools)
            component_colors.emplace_back(tool_color(candidate_tool));

        const std::vector<float> weights = generic_solver_weights(component_colors, {r, g, b});
        if (weights.size() == tools.size()) {
            for (size_t idx = 0; idx < tools.size(); ++idx)
                if (tools[idx] == tool)
                    return std::clamp(weights[idx], 0.f, 1.f);
        }
        return generic_visibility(tool, r, g, b);
    }

    float generic_visibility(size_t tool, float r, float g, float b) const
    {
        const std::array<float, 3> target{r, g, b};
        const float distance = std::sqrt(color_distance2(tool_color(tool), target));
        return std::clamp(1.f - distance / std::sqrt(3.f), 0.f, 1.f);
    }

    static float apply_tone_gamma(float channel, float gamma)
    {
        const float safe_channel = std::clamp(channel, 0.f, 1.f);
        const float safe_gamma = (!std::isfinite(gamma) || gamma <= 0.f) ? 1.f : std::clamp(gamma, 0.5f, 3.f);
        return std::abs(safe_gamma - 1.f) <= 1e-5f ? safe_channel : std::clamp(std::pow(safe_channel, 1.f / safe_gamma), 0.f, 1.f);
    }

    std::vector<float> generic_component_weights(float r, float g, float b) const
    {
        std::vector<std::array<float, 3>> component_colors;
        component_colors.reserve(component_ids.size());
        for (size_t idx = 0; idx < component_ids.size(); ++idx) {
            const unsigned int id = component_ids[idx];
            if (id == 0 || id > filament_colours.size())
                return generic_component_weights_fallback(r, g, b);
            component_colors.emplace_back(tool_color(size_t(id - 1)));
        }
        std::vector<float> weights = generic_solver_weights(component_colors, {r, g, b});
        if (weights.size() == component_ids.size())
            return weights;
        return generic_component_weights_fallback(r, g, b);
    }

    std::vector<float> generic_component_weights_fallback(float r, float g, float b) const
    {
        std::vector<float> weights(component_ids.size(), 0.f);
        for (size_t idx = 0; idx < component_ids.size(); ++idx) {
            const unsigned int id = component_ids[idx];
            weights[idx] = id > 0 ? generic_visibility(size_t(id - 1), r, g, b) : 0.f;
        }
        return weights;
    }

    static std::vector<float> fixed_mode_weights(int mode, size_t component_count, float r, float g, float b)
    {
        r = std::clamp(r, 0.f, 1.f);
        g = std::clamp(g, 0.f, 1.f);
        b = std::clamp(b, 0.f, 1.f);
        const float whiteness = std::min({r, g, b});
        const float darkness = 1.f - std::max({r, g, b});
        switch (mode) {
        case int(TextureMappingZone::FilamentColorRGB):
            return component_count == 3 ?
                std::vector<float>{print_visibility_strength(r), print_visibility_strength(g), print_visibility_strength(b)} :
                std::vector<float>{};
        case int(TextureMappingZone::FilamentColorCMY):
            return component_count == 3 ?
                std::vector<float>{print_visibility_strength(1.f - r),
                                   print_visibility_strength(1.f - g),
                                   print_visibility_strength(1.f - b)} :
                std::vector<float>{};
        case int(TextureMappingZone::FilamentColorBW): {
            if (component_count != 2)
                return {};
            const float gray = std::clamp(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.f, 1.f);
            return {print_visibility_strength(gray >= 0.5f ? 2.f * (1.f - gray) : 1.f),
                    print_visibility_strength(gray <= 0.5f ? 2.f * gray : 1.f)};
        }
        default:
            break;
        }
        if (component_count == 5) {
            const float chroma = std::max(0.f, 1.f - darkness - whiteness);
            if (mode == int(TextureMappingZone::FilamentColorCMYKW)) {
                const float c = chroma <= 1e-6f ? 0.f : 1.f - safe_div(r - whiteness, chroma);
                const float m = chroma <= 1e-6f ? 0.f : 1.f - safe_div(g - whiteness, chroma);
                const float y = chroma <= 1e-6f ? 0.f : 1.f - safe_div(b - whiteness, chroma);
                return {print_visibility_strength(c),
                        print_visibility_strength(m),
                        print_visibility_strength(y),
                        print_visibility_strength(darkness),
                        print_visibility_strength(whiteness)};
            }
            if (mode == int(TextureMappingZone::FilamentColorRGBKW)) {
                return {print_visibility_strength(safe_div(r - whiteness, chroma)),
                        print_visibility_strength(safe_div(g - whiteness, chroma)),
                        print_visibility_strength(safe_div(b - whiteness, chroma)),
                        print_visibility_strength(darkness),
                        print_visibility_strength(whiteness)};
            }
        }
        if (component_count != 4)
            return {};
        if (mode == int(TextureMappingZone::FilamentColorCMYK)) {
            const float k = std::clamp(darkness, 0.f, 1.f);
            const float inv = 1.f - k;
            return {print_visibility_strength(safe_div(1.f - r - k, inv)),
                    print_visibility_strength(safe_div(1.f - g - k, inv)),
                    print_visibility_strength(safe_div(1.f - b - k, inv)),
                    print_visibility_strength(k)};
        }
        if (mode == int(TextureMappingZone::FilamentColorCMYW)) {
            const float inv = 1.f - whiteness;
            const float r_no_w = safe_div(r - whiteness, inv);
            const float g_no_w = safe_div(g - whiteness, inv);
            const float b_no_w = safe_div(b - whiteness, inv);
            return {print_visibility_strength(std::clamp((1.f - r_no_w) * inv, 0.f, 1.f)),
                    print_visibility_strength(std::clamp((1.f - g_no_w) * inv, 0.f, 1.f)),
                    print_visibility_strength(std::clamp((1.f - b_no_w) * inv, 0.f, 1.f)),
                    std::clamp(std::pow(whiteness, 1.35f), 0.f, 1.f)};
        }
        if (mode == int(TextureMappingZone::FilamentColorRGBK)) {
            const float k = std::clamp(darkness, 0.f, 1.f);
            const float inv = 1.f - k;
            return {print_visibility_strength(safe_div(r - k, inv)),
                    print_visibility_strength(safe_div(g - k, inv)),
                    print_visibility_strength(safe_div(b - k, inv)),
                    print_visibility_strength(k)};
        }
        if (mode == int(TextureMappingZone::FilamentColorRGBW)) {
            const float inv = 1.f - whiteness;
            return {print_visibility_strength(safe_div(r - whiteness, inv)),
                    print_visibility_strength(safe_div(g - whiteness, inv)),
                    print_visibility_strength(safe_div(b - whiteness, inv)),
                    print_visibility_strength(whiteness)};
        }
        return {};
    }

    static void apply_filament_overhang_contrast(std::vector<float> &weights, float contrast_factor, size_t mapped_count)
    {
        const size_t count = std::min(mapped_count, weights.size());
        if (count == 0)
            return;
        float mean = 0.f;
        for (size_t idx = 0; idx < count; ++idx)
            mean += std::clamp(weights[idx], 0.f, 1.f);
        mean /= float(count);
        for (size_t idx = 0; idx < count; ++idx)
            weights[idx] = std::clamp(mean + (std::clamp(weights[idx], 0.f, 1.f) - mean) * contrast_factor, 0.f, 1.f);
    }

    float adjusted_visibility_factor(unsigned int physical_id, float value) const
    {
        const size_t idx = physical_id > 0 ? size_t(physical_id - 1) : size_t(-1);
        const float strength = idx < filament_strengths_pct.size() && std::isfinite(filament_strengths_pct[idx]) ?
            std::clamp(filament_strengths_pct[idx] / 100.f, 0.f, 1.f) :
            1.f;
        const float minimum = idx < filament_minimum_offsets_pct.size() && std::isfinite(filament_minimum_offsets_pct[idx]) ?
            std::clamp(filament_minimum_offsets_pct[idx] / 100.f, 0.f, 1.f) :
            0.f;
        return std::clamp(minimum + std::clamp(value, 0.f, 1.f) * strength * (1.f - minimum), 0.f, 1.f);
    }

    float adjusted_tool_visibility(size_t tool, float value) const
    {
        const unsigned int physical_id = unsigned(tool + 1);
        return settings_zone_enabled && std::find(component_ids.begin(), component_ids.end(), physical_id) != component_ids.end() ?
                   adjusted_visibility_factor(physical_id, value) :
                   std::clamp(value, 0.f, 1.f);
    }

    float sample_settings_zone_tool_visibility(size_t                     tool,
                                               float                      u,
                                               float                      v,
                                               bool                       back,
                                               const std::vector<size_t> *solver_tools = nullptr) const
    {
        const unsigned int physical_id = unsigned(tool + 1);
        const auto component_it = std::find(component_ids.begin(), component_ids.end(), physical_id);
        if (component_it == component_ids.end())
            return sample_image_tool_visibility(tool, u, v, back, solver_tools);
        const size_t component_idx = size_t(component_it - component_ids.begin());

        std::array<float, 3> rgb = sample_image_rgb(u, v, back);
        rgb[0] = apply_tone_gamma(rgb[0], tone_gamma);
        rgb[1] = apply_tone_gamma(rgb[1], tone_gamma);
        rgb[2] = apply_tone_gamma(rgb[2], tone_gamma);

        std::vector<float> weights(component_ids.size(), 0.f);
        size_t mapped_count = component_ids.size();
        if (texture_mapping_mode == int(TextureMappingZone::TextureMappingRawValues)) {
            const float channels[3] = {rgb[0], rgb[1], rgb[2]};
            mapped_count = std::min(component_ids.size(), size_t(3));
            for (size_t idx = 0; idx < mapped_count; ++idx)
                weights[idx] = std::clamp(channels[idx], 0.f, 1.f);
        } else {
            weights = fixed_mode_weights(texture_filament_color_mode, component_ids.size(), rgb[0], rgb[1], rgb[2]);
            if (weights.size() != component_ids.size())
                weights = generic_component_weights(rgb[0], rgb[1], rgb[2]);
        }
        if (weights.size() != component_ids.size() || component_idx >= weights.size())
            return sample_image_tool_visibility(tool, u, v, back, solver_tools);

        apply_filament_overhang_contrast(weights, std::clamp(filament_overhang_contrast_pct, 25.f, 300.f) / 100.f, mapped_count);
        return std::clamp(weights[component_idx], 0.f, 1.f);
    }

    float fixed_mode_visibility(size_t tool, float r, float g, float b) const
    {
        std::vector<std::array<float, 3>> ideals;
        std::vector<float> weights;
        r = std::clamp(r, 0.f, 1.f);
        g = std::clamp(g, 0.f, 1.f);
        b = std::clamp(b, 0.f, 1.f);
        const float whiteness = std::min({r, g, b});
        const float darkness = 1.f - std::max({r, g, b});
        switch (color_mode) {
        case CMY:
            ideals = {{{0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f}}};
            weights = {print_visibility_strength(1.f - r),
                       print_visibility_strength(1.f - g),
                       print_visibility_strength(1.f - b)};
            break;
        case CMYK: {
            const float k = std::clamp(darkness, 0.f, 1.f);
            const float inv = 1.f - k;
            ideals = {{{0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f}, {0.f, 0.f, 0.f}}};
            weights = {print_visibility_strength(safe_div(1.f - r - k, inv)),
                       print_visibility_strength(safe_div(1.f - g - k, inv)),
                       print_visibility_strength(safe_div(1.f - b - k, inv)),
                       print_visibility_strength(k)};
            break;
        }
        case CMYW: {
            const float inv = 1.f - whiteness;
            const float r_no_w = safe_div(r - whiteness, inv);
            const float g_no_w = safe_div(g - whiteness, inv);
            const float b_no_w = safe_div(b - whiteness, inv);
            ideals = {{{0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f}, {1.f, 1.f, 1.f}}};
            weights = {print_visibility_strength(std::clamp((1.f - r_no_w) * inv, 0.f, 1.f)),
                       print_visibility_strength(std::clamp((1.f - g_no_w) * inv, 0.f, 1.f)),
                       print_visibility_strength(std::clamp((1.f - b_no_w) * inv, 0.f, 1.f)),
                       std::clamp(std::pow(whiteness, 1.35f), 0.f, 1.f)};
            break;
        }
        case RGB:
            ideals = {{{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}}};
            weights = {print_visibility_strength(r),
                       print_visibility_strength(g),
                       print_visibility_strength(b)};
            break;
        case RGBK: {
            const float k = std::clamp(darkness, 0.f, 1.f);
            const float inv = 1.f - k;
            ideals = {{{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, 0.f}}};
            weights = {print_visibility_strength(safe_div(r - k, inv)),
                       print_visibility_strength(safe_div(g - k, inv)),
                       print_visibility_strength(safe_div(b - k, inv)),
                       print_visibility_strength(k)};
            break;
        }
        case RGBW: {
            const float inv = 1.f - whiteness;
            ideals = {{{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 1.f, 1.f}}};
            weights = {print_visibility_strength(safe_div(r - whiteness, inv)),
                       print_visibility_strength(safe_div(g - whiteness, inv)),
                       print_visibility_strength(safe_div(b - whiteness, inv)),
                       print_visibility_strength(whiteness)};
            break;
        }
        case BW: {
            const float gray = std::clamp(0.2126f * r + 0.7152f * g + 0.0722f * b, 0.f, 1.f);
            ideals = {{{0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}}};
            weights = {print_visibility_strength(gray >= 0.5f ? 2.f * (1.f - gray) : 1.f),
                       print_visibility_strength(gray <= 0.5f ? 2.f * gray : 1.f)};
            break;
        }
        case CMYKW: {
            const float chroma = std::max(0.f, 1.f - darkness - whiteness);
            const float c = chroma <= 1e-6f ? 0.f : 1.f - safe_div(r - whiteness, chroma);
            const float m = chroma <= 1e-6f ? 0.f : 1.f - safe_div(g - whiteness, chroma);
            const float y = chroma <= 1e-6f ? 0.f : 1.f - safe_div(b - whiteness, chroma);
            ideals = {{{0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f}, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}}};
            weights = {print_visibility_strength(c),
                       print_visibility_strength(m),
                       print_visibility_strength(y),
                       print_visibility_strength(darkness),
                       print_visibility_strength(whiteness)};
            break;
        }
        case RGBKW: {
            const float chroma = std::max(0.f, 1.f - darkness - whiteness);
            ideals = {{{1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {0.f, 0.f, 0.f}, {1.f, 1.f, 1.f}}};
            weights = {print_visibility_strength(safe_div(r - whiteness, chroma)),
                       print_visibility_strength(safe_div(g - whiteness, chroma)),
                       print_visibility_strength(safe_div(b - whiteness, chroma)),
                       print_visibility_strength(darkness),
                       print_visibility_strength(whiteness)};
            break;
        }
        default:
            ideals = {{{0.f, 1.f, 1.f}, {1.f, 0.f, 1.f}, {1.f, 1.f, 0.f}, {0.f, 0.f, 0.f}}};
            weights = {print_visibility_strength(1.f - r),
                       print_visibility_strength(1.f - g),
                       print_visibility_strength(1.f - b),
                       print_visibility_strength(darkness)};
            break;
        }
        const std::array<float, 3> actual = tool_color(tool);
        size_t best = 0;
        float best_distance = std::numeric_limits<float>::max();
        for (size_t i = 0; i < ideals.size(); ++i) {
            const float distance = color_distance2(actual, ideals[i]);
            if (distance < best_distance) {
                best_distance = distance;
                best = i;
            }
        }
        if (generic_fallback_for_missing_channels && best_distance > 0.35f)
            return generic_solver_tool_visibility(tool, r, g, b);
        return best < weights.size() ? std::clamp(weights[best], 0.f, 1.f) : generic_solver_tool_visibility(tool, r, g, b);
    }
};

// Cuts the tower wall polygon open at each skip point (a toolchange's entry position)
// so the entry travel can pass through instead of crossing the printed wall. Defined in
// WipeTower.cpp, shared by WipeTower and WipeTower2.
Polylines construct_gap_for_skip_points(
    const Polygon& polygon, const std::vector<Vec2f>& skip_points, float wt_width, float gap_length, Polygon& insert_skip_polygon);

// Klipper acts on commands the instant it parses them, and its G4 reads only P (milliseconds),
// so the zero-second and seconds-valued dwells every other flavor uses neither synchronize nor
// pause there. Both defined in WipeTower.cpp, shared by WipeTower and WipeTower2.
const char* flush_planner_queue_command(GCodeFlavor flavor); // finish queued moves, e.g. around M104/M109
std::string wait_command(GCodeFlavor flavor, float seconds);  // pause for `seconds`

class WipeTower
{
public:
    friend class WipeTowerWriter;
    using ProgressCallback = std::function<void(size_t, size_t)>;
    static const std::string never_skip_tag() { return "_GCODE_WIPE_TOWER_NEVER_SKIP_TAG"; }

	// WipeTower height to minimum depth map
	static const std::map<float, float> min_depth_per_height;
    static float get_limit_depth_by_height(float max_height);
    static float get_auto_brim_by_height(float max_height);
    // Both generators lay the brim in whole loops one line spacing apart, so the printed width
    // differs from the configured one. WipeTower reports it with half a spacing of line width
    // added, WipeTower2 reports the loops alone; an estimate has to round like the generator
    // whose G-code it stands in for.
    static float estimate_brim_real_width(float brim_width, float nozzle_diameter, float first_layer_height, bool type2);
    // Depth a Type1 tower reserves once nothing but wrapping detection asks for one.
    static float get_wrapping_detection_depth();
    // Line width of the nozzle-change purge lines at this nozzle diameter.
    static float nozzle_change_perimeter_width(float nozzle_diameter);
    static TriangleMesh                 its_make_rib_tower(float width, float depth, float height, float rib_length, float rib_width, bool fillet_wall);
    static TriangleMesh                 its_make_rib_brim(const Polygon& brim, float layer_height);
    static Polygon                      rib_section(float width, float depth, float rib_length, float rib_width, bool fillet_wall);
    // One filament's share of a Type1 tower layer, as plan_tower_new() reserves it.
    struct PurgeEstimate
    {
        float prime_volume           = 0.f;   // mm3 wiped after changing to this filament
        int   category               = 0;     // filament_adhesiveness_category; one purge block per category
        float filament_change_length = 0.f;   // mm of filament rammed when it leaves its nozzle; 0 when no nozzle change is planned
        float filament_diameter      = 1.75f;
    };
    // Depth of the Type1 purge stack at the given width (also the rectangle-wall depth): each
    // purge is whole lines at the block infill gap, one block per adhesiveness category sized by
    // its worst layer, stacked behind one perimeter width.
    static float estimate_tower_blocks_depth(const std::vector<PurgeEstimate> &purges, float width, float layer_height, float nozzle_diameter, float extra_spacing);
    // Side of the square bounding a rib-wall tower's first layer, brim excluded: the body plus the
    // rib bulge, with the ribs extended to the height-based minimum as both generators do.
    static float rib_footprint_side(float width, float depth, float rib_width, float extra_rib_length, float max_height);
    // Type1 rib tower: plan_tower_new() squares the tower from the depth at the configured width,
    // then re-plans the depth at the squared width.
    static float estimate_rib_tower_bbox_side(const std::vector<PurgeEstimate> &purges, float width, float layer_height, float nozzle_diameter, float extra_spacing, float rib_width, float extra_rib_length, float max_height);
    // Translation that brings a footprint inside the printable outline, padded by offset. The prime
    // tower is validated against the real outline (see layered_print_cleareance_valid), so clamping
    // against the bounding box alone would leave it off a delta or hexagonal bed. box and polygons
    // must share one scaled coordinate frame; the translation comes back in millimeters.
    static Vec2f                        move_box_inside_polygon(const BoundingBox &box, const Polygons &polygons, coord_t offset = 0);
    static Vec2f                        move_box_inside_box(const BoundingBox &box1, const BoundingBox &box2, int offset = 0);
    static Polygon                      rounding_polygon(Polygon &polygon, double rounding = 2., double angle_tol = 30. / 180. * PI);
    struct Extrusion
    {
		Extrusion(const Vec2f &pos, float width, unsigned int tool) : pos(pos), width(width), tool(tool) {}
		// End position of this extrusion.
		Vec2f				pos;
		// Width of a squished extrusion, corrected for the roundings of the squished extrusions.
		// This is left zero if it is a travel move.
		float 			width;
		// Current extruder index.
		unsigned int    tool;
	};

	struct NozzleChangeResult
    {
        std::string gcode;

        Vec2f start_pos;  // rotated
        Vec2f end_pos;

		Vec2f origin_start_pos;  // not rotated

        std::vector<Vec2f> wipe_path;
        bool is_extruder_change{true};
    };

	struct ToolChangeResult
	{
		// Print heigh of this tool change.
		float					print_z;
		float 					layer_height;
		// G-code section to be directly included into the output G-code.
		std::string				gcode;
		// For path preview.
		std::vector<Extrusion> 	extrusions;
		// Initial position, at which the wipe tower starts its action.
		// At this position the extruder is loaded and there is no Z-hop applied.
		Vec2f						start_pos;
		// Last point, at which the normal G-code generator of Slic3r shall continue.
		// At this position the extruder is loaded and there is no Z-hop applied.
		Vec2f						end_pos;
		// Time elapsed over this tool change.
		// This is useful not only for the print time estimation, but also for the control of layer cooling.
		float  				    elapsed_time;

        // Is this a priming extrusion? (If so, the wipe tower rotation & translation will not be applied later)
        bool                    priming;

		bool                    is_tool_change{false};
		Vec2f                   tool_change_start_pos;

        // Pass a polyline so that normal G-code generator can do a wipe for us.
        // The wipe cannot be done by the wipe tower because it has to pass back
        // a loaded extruder, so it would have to either do a wipe with no retraction
        // (leading to https://github.com/prusa3d/PrusaSlicer/issues/2834) or do
        // an extra retraction-unretraction pair.
        std::vector<Vec2f> wipe_path;

		// BBS
        float purge_volume = 0.f;

        // Initial tool
        int initial_tool;

        // New tool
        int new_tool;

        // BBS: in bbl filament_change_gcode, toolhead will be moved to the wipe tower automatically.
        // But if finish_layer_tcr is before tool_change_tcr, we have to travel to the wipe tower before
        // executing the gcode finish_layer_tcr.
        bool is_finish_first = false;

        bool               is_contact = false;
        NozzleChangeResult nozzle_change_result;

        // Orca: folded into a later, thicker layer, so the emitter drops it. Set by the tower, so
        // the two cannot disagree about which layers print.
        bool               combined_away = false;

		// Sum the total length of the extrusion.
		float total_extrusion_length_in_plane() {
			float e_length = 0.f;
			for (size_t i = 1; i < this->extrusions.size(); ++ i) {
				const Extrusion &e = this->extrusions[i];
				if (e.width > 0) {
					Vec2f v = e.pos - (&e - 1)->pos;
					e_length += v.norm();
				}
			}
			return e_length;
		}
		// Orca: set by WipeTower2 (non-BBL tower) to force a travel to the tower even when the
		// previous position is unknown; read by WipeTowerIntegration::append_tcr2 (GCode.cpp).
		bool force_travel = false;
	};

    struct box_coordinates
    {
        box_coordinates(float left, float bottom, float width, float height) :
            ld(left        , bottom         ),
            lu(left        , bottom + height),
            rd(left + width, bottom         ),
            ru(left + width, bottom + height) {}
        box_coordinates(const Vec2f &pos, float width, float height) : box_coordinates(pos(0), pos(1), width, height) {}
        void translate(const Vec2f &shift) {
            ld += shift; lu += shift;
            rd += shift; ru += shift;
        }
        void translate(const float dx, const float dy) { translate(Vec2f(dx, dy)); }
        void expand(const float offset) {
            ld += Vec2f(- offset, - offset);
            lu += Vec2f(- offset,   offset);
            rd += Vec2f(  offset, - offset);
            ru += Vec2f(  offset,   offset);
        }
        void expand(const float offset_x, const float offset_y) {
            ld += Vec2f(- offset_x, - offset_y);
            lu += Vec2f(- offset_x,   offset_y);
            rd += Vec2f(  offset_x, - offset_y);
            ru += Vec2f(  offset_x,   offset_y);
        }
        Vec2f ld;  // left down
        Vec2f lu;	// left upper
        Vec2f rd;	// right lower
        Vec2f ru;  // right upper
    };

    // Construct ToolChangeResult from current state of WipeTower and WipeTowerWriter.
    // WipeTowerWriter is moved from !
    ToolChangeResult construct_tcr(WipeTowerWriter& writer,
                                   bool priming,
                                   size_t old_tool,
                                   bool is_finish,
		                           bool is_tool_change, float purge_volume, bool is_contact) const;

    ToolChangeResult construct_block_tcr(WipeTowerWriter& writer,
                                   bool priming,
                                   size_t filament_id,
                                   bool is_finish, float purge_volume) const;


	// x			-- x coordinates of wipe tower in mm ( left bottom corner )
	// y			-- y coordinates of wipe tower in mm ( left bottom corner )
	// width		-- width of wipe tower in mm ( default 60 mm - leave as it is )
	// wipe_area	-- space available for one toolchange in mm
	// BBS: add partplate logic
	WipeTower(const PrintConfig& config, int plate_idx, Vec3d plate_origin, size_t initial_tool, const float wipe_tower_height, const std::vector<unsigned int>& slice_used_filaments);


	// Set the extruder properties.
    void set_extruder(size_t idx, const PrintConfig& config);

    void set_shared_print_bed(const Polygons &bed) { m_shared_print_bed = bed; }
    // Orca: has_filament_switcher is not a static PrintConfig member here, so it is pushed in from
    // Print via a setter rather than read in the ctor. Device-set only.
    void set_has_filament_switcher(bool v) { m_has_filament_switcher = v; }
	// Appends into internal structure m_plan containing info about the future wipe tower
	// to be used before building begins. The entries must be added ordered in z.
	void plan_toolchange(float        z_par,
                         float        layer_height_par,
                         unsigned int old_tool,
                         unsigned int new_tool,
                         float        wipe_volume_ec = 0.f,
                         float        wipe_volume_nc = 0.f,
                         float        purge_volume = 0.f,
                         bool         texture_mapping_single_component_layer = false,
                         const std::vector<unsigned int> &texture_mapping_layer_tools = {},
                         int          texture_mapping_wall_tool = -1);

	// Iterates through prepared m_plan, generates ToolChangeResults and appends them to "result"
	void generate(std::vector<std::vector<ToolChangeResult>> &result, ProgressCallback progress_callback = {});
    void set_prime_tower_texture(const PrimeTowerTextureRenderSettings &settings) { m_prime_tower_texture = settings; }

	WipeTower::ToolChangeResult only_generate_out_wall(bool is_new_mode = false);
    Polygon generate_support_wall(WipeTowerWriter &writer, const box_coordinates &wt_box, double feedrate, bool first_layer);
    Polygon generate_support_wall_new(WipeTowerWriter &writer, const box_coordinates &wt_box, double feedrate, bool first_layer,bool rib_wall, bool extrude_perimeter, bool skip_points);

    Polygon generate_rib_polygon(const box_coordinates &wt_box);
    float get_depth() const { return m_wipe_tower_depth; }
    float get_brim_width() const { return m_wipe_tower_brim_width_real; }
    BoundingBoxf get_bbx() const {
        if (m_outer_wall.empty()) return BoundingBoxf({Vec2d(0,0)});
        BoundingBox  box = get_extents(m_outer_wall.begin()->second);
        BoundingBoxf res = BoundingBoxf(unscale(box.min), unscale(box.max));
        return res;
    }
    std::map<float, Polylines> get_outer_wall() const
    {
        return m_outer_wall;
    }
    float get_height() const { return m_wipe_tower_height; }
    float get_layer_height() const { return m_layer_height; }
    float get_rib_length() const { return m_rib_length; }
    float get_rib_width() const { return m_rib_width; }

	void set_last_layer_extruder_fill(bool extruder_fill) {
        if (!m_plan.empty()) {
			m_plan.back().extruder_fill = extruder_fill;
		}
	}


	// Switch to a next layer.
	void set_layer(
		// Print height of this layer.
		float print_z,
		// Layer height, used to calculate extrusion the rate.
		float layer_height,
		// Maximum number of tool changes on this layer or the layers below.
		size_t max_tool_changes,
		// Is this the first layer of the print? In that case print the brim first.
		bool is_first_layer,
		// Is this the last layer of the waste tower?
		bool is_last_layer)
	{
		m_z_pos 				= print_z;
		m_layer_height			= layer_height;
		m_depth_traversed  = 0.f;
        m_current_layer_finished = false;
		//m_current_shape = (! is_first_layer && m_current_shape == SHAPE_NORMAL) ? SHAPE_REVERSED : SHAPE_NORMAL;
		m_current_shape = SHAPE_NORMAL;
		if (is_first_layer) {
            m_num_layer_changes = 0;
            m_num_tool_changes 	= 0;
        } else
            ++ m_num_layer_changes;

		// Calculate extrusion flow from desired line width, nozzle diameter, filament diameter and layer_height:
		m_extrusion_flow = extrusion_flow(layer_height);
        // Advance m_layer_info iterator, making sure we got it right
		while (!m_plan.empty() && m_layer_info->z < print_z - WT_EPSILON && m_layer_info+1 != m_plan.end())
			++m_layer_info;
	}

	// Return the wipe tower position.
	const Vec2f& 		 position() const { return m_wipe_tower_pos; }
	// Return the wipe tower width.
	float     		 width()    const { return m_wipe_tower_width; }
	// The wipe tower is finished, there should be no more tool changes or wipe tower prints.
	bool 	  		 finished() const { return m_max_color_changes == 0; }

	// Returns gcode to prime the nozzles at the front edge of the print bed.
	std::vector<ToolChangeResult> prime(
		// print_z of the first layer.
		float 						initial_layer_print_height,
		// Extruder indices, in the order to be primed. The last extruder will later print the wipe tower brim, print brim and the object.
		const std::vector<unsigned int> &tools,
		// If true, the last priming are will be the same as the other priming areas, and the rest of the wipe will be performed inside the wipe tower.
		// If false, the last priming are will be large enough to wipe the last extruder sufficiently.
		bool 						last_wipe_inside_wipe_tower);

	// Returns gcode for a toolchange and a final print head position.
	// On the first layer, extrude a brim around the future wipe tower first.
	// BBS
    ToolChangeResult tool_change(size_t new_tool, bool extrude_perimeter = false, bool first_toolchange_to_nonsoluble = false);

	NozzleChangeResult nozzle_change(int old_filament_id, int new_filament_id);

	// Fill the unfilled space with a sparse infill.
	// Call this method only if layer_finished() is false.
    ToolChangeResult finish_layer(bool extruder_perimeter = true, bool extruder_fill = true);

	// Calculates extrusion flow needed to produce required line width for given layer height
    float extrusion_flow(float layer_height = -1.f) const // negative layer_height - return current m_extrusion_flow
    {
        if (layer_height < 0) return m_extrusion_flow;
        return layer_height * (m_perimeter_width - layer_height * (1.f - float(M_PI) / 4.f)) / filament_area();
    }
    float nozzle_change_extrusion_flow(float layer_height = -1.f) const // negative layer_height - return current m_extrusion_flow
    {
        if (layer_height < 0)
            return m_extrusion_flow;
        return layer_height * (m_nozzle_change_perimeter_width - layer_height * (1.f - float(M_PI) / 4.f)) / filament_area();
    }

	bool get_floating_area(float& start_pos_y, float& end_pos_y) const;
	bool need_thick_bridge_flow(float pos_y) const;
    float get_extrusion_flow() const { return m_extrusion_flow; }

	// Is the current layer finished?
	bool 			 layer_finished() const {
        return m_current_layer_finished;
	}

    std::vector<float> get_used_filament() const { return m_used_filament_length; }
    int get_number_of_toolchanges() const { return m_num_tool_changes; }

	void set_has_tpu_filament(bool has_tpu) { m_has_tpu_filament = has_tpu; }

    bool has_tpu_filament() const { return m_has_tpu_filament; }
    struct FilamentParameters {
        std::string 	    material = "PLA";
        int                 category;
        bool                is_soluble = false;
        // BBS
        bool                is_support = false;
        int  			    nozzle_temperature = 0;
        int  			    nozzle_temperature_initial_layer = 0;
        // BBS: remove useless config
        //float               loading_speed = 0.f;
        //float               loading_speed_start = 0.f;
        //float               unloading_speed = 0.f;
        //float               unloading_speed_start = 0.f;
        //float               delay = 0.f ;
        //int                 cooling_moves = 0;
        //float               cooling_initial_speed = 0.f;
        //float               cooling_final_speed = 0.f;
        float               ramming_line_width_multiplicator = 1.f;
        float               ramming_step_multiplicator = 1.f;
        float               max_e_speed = std::numeric_limits<float>::max();
        std::vector<float>  ramming_speed;
        float               nozzle_diameter;
        float               filament_area;
        float               retract_length;
        float               retract_speed;
        float               wipe_dist;
        std::pair<float,float>  max_e_ramming_speed;//[0]extruder change [1]nozzle change
        std::pair<float, float> ramming_travel_time; // Travel time after ramming
        std::pair<std::vector<float>,std::vector<float>>  precool_t;//Pre-cooling time, set to 0 to ensure the ramming speed is controlled solely by ramming volumetric speed.
        std::pair<std::vector<float>, std::vector<float>> precool_t_first_layer;
        std::pair<int,int>    precool_target_temp;
        float filament_cooling_before_tower = 0.f;
        float flat_iron_area;
        float filament_tower_interface_print_temp;
        float filament_tower_interface_pre_extrusion_dist = 0;
        float filament_tower_interface_pre_extrusion_length = 0;
        float filament_petg_pre_extrusion_offset_dist = 0;
        // Tallest layer this filament's nozzle can lay down; caps the sparse layer combination.
        float max_layer_height = 0.f;
    };


    void set_used_filament_ids(const std::vector<int> &used_filament_ids) { m_used_filament_ids = used_filament_ids; };
    void set_filament_categories(const std::vector<int> & filament_categories) { m_filament_categories = filament_categories;};
    void set_nozzle_group_result(const MultiNozzleUtils::LayeredNozzleGroupResult &multi_nozzle_group_result) { m_multi_nozzle_group_result = &multi_nozzle_group_result; };
    std::vector<int> m_used_filament_ids;
    std::vector<int> m_filament_categories;
    const MultiNozzleUtils::LayeredNozzleGroupResult *m_multi_nozzle_group_result{nullptr};

    enum class WipeTowerLayerType : unsigned char { Normal, Contact, Solid, Contact_UP};// Contact layer should be solid and reduce feed

	struct WipeTowerBlock
    {
        int              block_id{0};
        int              filament_adhesiveness_category{0};
        std::vector<float>      layer_depths;
        //std::vector<bool>       solid_infill;
        std::vector<float>      finish_depth{0}; // the start pos of finish frame for every layer
        std::vector<WipeTowerLayerType> layers_type;     // type of the layer, normal, Contact or Solid
        float            depth{0};
        float            start_depth{0};
        float            cur_depth{0};
        int              last_filament_change_id{-1};
        int              last_nozzle_change_id{-1};
	};

	struct BlockDepthInfo
    {
        int category{-1};
        float depth{0};
        float nozzle_change_depth{0};
	};

	std::vector<std::vector<BlockDepthInfo>> m_all_layers_depth;
	std::vector<WipeTowerBlock> m_wipe_tower_blocks;
    int                  m_last_block_id;
    WipeTowerBlock*      m_cur_block{nullptr};

	// help function
    WipeTowerBlock* get_block_by_category(int filament_adhesiveness_category, bool create);
    void add_depth_to_block(int filament_id, int filament_adhesiveness_category, float depth, bool is_nozzle_change = false);
	int get_filament_category(int filament_id);
	void reset_block_status();
    int get_wall_filament_for_all_layer();
	// for generate new wipe tower
    void generate_new(std::vector<std::vector<WipeTower::ToolChangeResult>> &result, ProgressCallback progress_callback = {});

	void plan_tower_new();
	void generate_wipe_tower_blocks(bool add_solid_flag);
    void update_all_layer_depth(float wipe_tower_depth);
    void set_nozzle_last_layer_id();
    void set_first_layer_flow_ratio(const float flow_ratio);
    // Orca: default/initial-layer/travel acceleration are object-scope options here (PrintConfig
    // members in BBS), so Print pushes the resolved per-variant columns in via this setter.
    void set_accelerations(const std::vector<double> &normal, const std::vector<double> &first_layer_normal,
                           const std::vector<double> &travel, const std::vector<double> &first_layer_travel);
    void calc_block_infill_gap();
    ToolChangeResult   tool_change_new(size_t new_tool, bool solid_change = false, bool solid_nozzlechange=false);
    NozzleChangeResult ramming(int old_filament_id, int new_filament_id, bool solid_change = false, bool extruder_change = true); // extruder_chang means nozzle_change
    ToolChangeResult   finish_layer_new(bool extrude_perimeter = true, bool extrude_fill = true, bool extrude_fill_wall = true);
    ToolChangeResult   finish_block(const WipeTowerBlock &block, int filament_id, bool extrude_fill = true);
    ToolChangeResult   finish_block_solid(const WipeTowerBlock &block, int filament_id, bool extrude_fill = true, WipeTowerLayerType layer_type = WipeTowerLayerType::Normal);
    void toolchange_wipe_new(WipeTowerWriter &writer, const box_coordinates &cleaning_box, float wipe_length,bool solid_toolchange=false);
    Vec2f              get_rib_offset() const { return m_rib_offset; }
    bool               is_need_ramming(int filament_id_1, int filament_id_2, int layer_id) const;
    bool               is_same_extruder(int filament_id_1, int filament_id_2, int layer_id) const;
    bool               is_same_nozzle(int filament_id_1, int filament_id_2, int layer_id) const;
    int                get_nozzle_id(int filament_id, int layer_id) const;
    int                get_extruder_id(int filament_id, int layer_id) const;

private:
	enum wipe_shape // A fill-in direction
	{
		SHAPE_NORMAL = 1,
		SHAPE_REVERSED = -1
	};

    const float Width_To_Nozzle_Ratio = 1.25f; // desired line width (oval) in multiples of nozzle diameter - may not be actually neccessary to adjust
    const float WT_EPSILON            = 1e-3f;
    float filament_area() const {
        return m_filpar[0].filament_area; // all extruders are assumed to have the same filament diameter at this point
    }

    int    m_slice_used_filaments      = 0;
    int    m_wrapping_detection_layers = 0;
    bool   m_enable_wrapping_detection = false;
	bool   m_enable_timelapse_print = false;
	bool   m_semm               = true; // Are we using a single extruder multimaterial printer?
    Vec2f  m_wipe_tower_pos; 			// Left front corner of the wipe tower in mm.
	float  m_wipe_tower_width; 			// Width of the wipe tower.
	float  m_wipe_tower_depth 	= 0.f; 	// Depth of the wipe tower
	// BBS
	float  m_wipe_tower_height = 0.f;
    float  m_wipe_tower_brim_width      = 0.f; 	// Width of brim (mm) from config
    float  m_wipe_tower_brim_width_real = 0.f; 	// Width of brim (mm) after generation
	float  m_wipe_tower_rotation_angle = 0.f; // Wipe tower rotation angle in degrees (with respect to x axis)
    float  m_internal_rotation  = 0.f;
	float  m_y_shift			= 0.f;  // y shift passed to writer
	float  m_z_pos 				= 0.f;  // Current Z position.
	float  m_layer_height 		= 0.f; 	// Current layer height.
	size_t m_max_color_changes 	= 0; 	// Maximum number of color changes per layer.
    int    m_old_temperature    = -1;   // To keep track of what was the last temp that we set (so we don't issue the command when not neccessary)
    float  m_travel_speed       = 0.f;
    float  m_first_layer_speed  = 0.f;
    size_t m_first_layer_idx    = size_t(-1);
    Vec2f            m_origin;
    std::vector<int>    m_last_layer_id;
    std::pair<std::vector<double>,std::vector<double>> m_filaments_change_length;//[0]extruder change [1]nozzle change
    size_t       m_cur_layer_id;
    NozzleChangeResult m_nozzle_change_result;
    bool               m_has_tpu_filament{false};
    bool               m_is_multi_extruder{false};
    bool               m_use_gap_wall{false};
    bool               m_use_rib_wall{false};
    float              m_rib_length=0.f;
    float              m_rib_width=0.f;
    float              m_extra_rib_length=0.f;
    bool               m_used_fillet{false};
    Vec2f              m_rib_offset{Vec2f(0.f, 0.f)};
    bool               m_tower_framework{false};
    bool               m_need_reverse_travel{false};
    bool               m_enable_tower_interface_features{false};
	// G-code generator parameters.
    // BBS: remove useless config
    //float           m_cooling_tube_retraction   = 0.f;
    //float           m_cooling_tube_length       = 0.f;
    //float           m_parking_pos_retraction    = 0.f;
    //float           m_extra_loading_move        = 0.f;
    float           m_bridging                  = 0.f;
    bool            m_sparse_layers_skipped     = false;
    bool            m_sparse_layers_combined    = false;
    // BBS: remove useless config
    //bool            m_set_extruder_trimpot      = false;
    bool            m_adhesion                  = true;
    GCodeFlavor     m_gcode_flavor;
    bool                      m_is_multiple_nozzle = false;
    std::vector<unsigned int> m_normal_accels;
    std::vector<unsigned int> m_first_layer_normal_accels;
    std::vector<unsigned int> m_travel_accels;
    std::vector<unsigned int> m_first_layer_travel_accels;
    unsigned int              m_max_accels;
    bool                      m_accel_to_decel_enable;
    float                     m_accel_to_decel_factor;
    bool                      m_enable_arc_fitting = true;
    std::vector<double>       m_hotend_heating_rate;
    std::vector<double>       m_hotend_cooling_rate;
    Polygons                  m_shared_print_bed;

    // Bed properties
    enum {
        RectangularBed,
        CircularBed,
        CustomBed
    } m_bed_shape;
    float m_bed_width; // width of the bed bounding box
    Vec2f m_bed_bottom_left; // bottom-left corner coordinates (for rectangular beds)

    float m_first_layer_flow_ratio;
	float m_perimeter_width = 0.4f * Width_To_Nozzle_Ratio; // Width of an extrusion line, also a perimeter spacing for 100% infill.
    float m_nozzle_change_perimeter_width = 0.4f * Width_To_Nozzle_Ratio;
	float m_extrusion_flow = 0.038f; //0.029f;// Extrusion flow is derived from m_perimeter_width, layer height and filament diameter.
    std::unordered_map<int, std::pair<float,float>> m_block_infill_gap_width; // categories to infill_gap: toolchange gap, nozzlechange gap
	// Extruder specific parameters.
    std::vector<FilamentParameters> m_filpar;


	// State of the wipe tower generator.
	unsigned int m_num_layer_changes = 0; // Layer change counter for the output statistics.
	unsigned int m_num_tool_changes  = 0; // Tool change change counter for the output statistics.
	///unsigned int 	m_idx_tool_change_in_layer = 0; // Layer change counter in this layer. Counting up to m_max_color_changes.
	bool m_print_brim = true;
	// A fill-in direction (positive Y, negative Y) alternates with each layer.
	wipe_shape   	m_current_shape = SHAPE_NORMAL;
    size_t 	m_current_tool  = 0;
	// BBS
    //const std::vector<std::vector<float>> wipe_volumes;

	float           m_depth_traversed = 0.f; // Current y position at the wipe tower.
    bool            m_current_layer_finished = false;
	bool 			m_left_to_right   = true;
	float			m_extra_spacing   = 1.f;
	float           m_tpu_fixed_spacing = 2;
    float           m_max_speed = 5400.f;  // the maximum printing speed on the prime tower.
    std::vector<std::vector<Vec2f>> m_wall_skip_points;
    std::map<float,Polylines> m_outer_wall;
    std::vector<double>        m_printable_height;
    bool is_first_layer() const { return size_t(m_layer_info - m_plan.begin()) == m_first_layer_idx; }
    bool                       is_valid_last_layer(int tool, int layer_id, double layer_z) const;
    bool                       m_flat_ironing=false;
    bool                       m_contact_ironing = false;
    bool                       m_has_filament_switcher = false;
    float                      m_contact_speed   = 20 * 60.f;
    std::vector<int>           m_physical_extruder_map;
	// Calculates length of extrusion line to extrude given volume
	float volume_to_length(float volume, float line_width, float layer_height) const {
		return std::max(0.f, volume / (layer_height * (line_width - layer_height * (1.f - float(M_PI) / 4.f))));
	}
    // Calculates volume of extrusion line
    float length_to_volume(float length,float line_width, float layer_height) const
    {
        return std::max(0.f, length * (layer_height * (line_width - layer_height * (1.f - float(M_PI) / 4.f))));
    }
	// Calculates depth for all layers and propagates them downwards
	void plan_tower();
	// Whether the layer reaches the G-code, and so whether its extrusions count as filament used.
	bool layer_is_printed(bool toolchanges_on_layer) const;

	// Goes through m_plan and recalculates depths and width of the WT to make it exactly square - experimental
	void make_wipe_tower_square();

	Vec2f get_next_pos(const WipeTower::box_coordinates &cleaning_box, float wipe_length, bool solid_toolchange);

    // Goes through m_plan, calculates border and finish_layer extrusions and subtracts them from last wipe
    void save_on_last_wipe();

	bool is_tpu_filament(int filament_id) const;
	bool is_petg_filament(int filament_id) const;
    bool is_need_reverse_travel(int filament, bool extruder_change) const;
	// BBS
	box_coordinates align_perimeter(const box_coordinates& perimeter_box);

    void set_for_wipe_tower_writer(WipeTowerWriter &writer);

    // to store information about tool changes for a given layer
	struct WipeTowerInfo{
		struct ToolChange {
            size_t old_tool;
            size_t new_tool;
			float required_depth;
            float ramming_depth;
            float first_wipe_line;
            float wipe_volume;
			float wipe_length;
            float nozzle_change_depth{0};
            float nozzle_change_length{0};
			// BBS
			float purge_volume;
            ToolChange(size_t old, size_t newtool, float depth=0.f, float ramming_depth=0.f, float fwl=0.f, float wv=0.f, float wl = 0, float pv = 0)
				: old_tool{ old }, new_tool{ newtool }, required_depth{ depth }, ramming_depth{ ramming_depth }, first_wipe_line{ fwl }, wipe_volume{ wv }, wipe_length{ wl }, purge_volume{ pv } {}
		};
		float z;		// z position of the layer
		float height;	// layer height
		float depth;	// depth of the layer based on all layers above
        float extra_spacing;
        bool  extruder_fill{true};
        bool  texture_mapping_single_component_layer{false};
        std::vector<size_t> texture_mapping_layer_tools;
        size_t texture_mapping_wall_tool{size_t(-1)};
		// Folded into a later, thicker layer, so this one prints nothing at all.
		bool  combined_away{false};
		float toolchanges_depth() const { float sum = 0.f; for (const auto &a : tool_changes) sum += a.required_depth; return sum; }

		std::vector<ToolChange> tool_changes;

		WipeTowerInfo(float z_par, float layer_height_par)
			: z{z_par}, height{layer_height_par}, depth{0}, extra_spacing{1.f} {}
	};

	std::vector<WipeTowerInfo> m_plan; 	// Stores information about all layers and toolchanges for the future wipe tower (filled by plan_toolchange(...))
	std::vector<WipeTowerInfo>::iterator m_layer_info = m_plan.end();

    // Stores information about used filament length per extruder:
    std::vector<float> m_used_filament_length;
    PrimeTowerTextureRenderSettings m_prime_tower_texture;

    // BBS: consider both soluable and support properties
    // Return index of first toolchange that switches to non-soluble extruder
    // ot -1 if there is no such toolchange.
    int first_toolchange_to_nonsoluble_nonsupport(
            const std::vector<WipeTowerInfo::ToolChange>& tool_changes) const;
    WipeTowerInfo::ToolChange set_toolchange(int old_tool, int new_tool, float layer_height, float wipe_volume, float purge_volume,int layer_id);
	void toolchange_Unload(
		WipeTowerWriter &writer,
		const box_coordinates  &cleaning_box,
		const std::string&	 	current_material,
		const int 				new_temperature);

	void toolchange_Change(
		WipeTowerWriter &writer,
        const size_t		new_tool,
		const std::string& 		new_material);

	void toolchange_Load(
		WipeTowerWriter &writer,
		const box_coordinates  &cleaning_box);

	void toolchange_Wipe(
		WipeTowerWriter &writer,
		const box_coordinates  &cleaning_box,
		float wipe_volume);
    void get_wall_skip_points(const WipeTowerInfo &layer,int layer_id);
    void get_all_wall_skip_points();
    ToolChangeResult merge_tcr(ToolChangeResult &first, ToolChangeResult &second);
    float            get_block_gap_width(int tool, bool is_nozzlechangle = false);
};


// Compaction rule for wipe_tower_no_sparse_layers. Shared by the G-code emitter and by the
// clearance validator so that both agree on where the compacted tower actually sits; a drift
// between the two would either let a real nozzle collision through or reject a safe plate.

// Whether sparse layers are really skipped, i.e. whether the tower is compacted at all. Smooth
// timelapse and wrapping detection put a tower on every layer, so no layer is ever dropped and the
// tower keeps following the object even though the option is on. Tower planning, G-code emission and
// the clearance validator all ask this single question, so none of them can compact on its own.
bool wipe_tower_sparse_layers_skipped(const PrintConfig &config);

// A planned layer prints no tower at all when its only toolchange keeps the same filament.
bool wipe_tower_layer_is_sparse(const std::vector<WipeTower::ToolChangeResult> &layer_tool_changes);

// Print z the compacted tower reaches on every planned layer. Sparse layers carry over the
// previous value, so the tower falls one layer height behind the object for each of them. base_z is
// the z the tower starts from, which Orca offsets by z_offset.
std::vector<float> compute_compacted_wipe_tower_z(const std::vector<std::vector<WipeTower::ToolChangeResult>> &tool_changes,
                                                  float base_z = 0.f);


// Combination rule for wipe_tower_sparse_layers_combination. Nothing is compacted - the tower keeps
// following the object - but a run of consecutive toolchange-free layers prints as one thicker layer,
// the way infill combination merges sparse infill. Shared so that neither tower generator nor the
// G-code emitter can combine on its own.

// Whether sparse layers are really combined. Skipping them outright is the stronger answer to the
// same problem and wins over this; smooth timelapse and wrapping detection need a tower on every
// layer, so they rule it out too.
bool wipe_tower_sparse_layers_combined(const PrintConfig &config);

// A planned layer folded into a later, thicker one prints nothing at all.
bool wipe_tower_layer_is_combined_away(const std::vector<WipeTower::ToolChangeResult> &layer_tool_changes);

// Folds runs of sparse layers into one. layer_height is raised in place on the layer that prints a
// run - always its last, so the merged extrusion lands on top of what it covers - and the returned
// mask marks the layers that now print nothing. A run stops growing once one more layer would pass
// max_layer_height of the nozzle that prints it. first_layer_idx and below never combine: the
// tower's first layer carries the brim.
std::vector<char> combine_sparse_wipe_tower_layers(std::vector<float>       &layer_height,
                                                   const std::vector<char>  &layer_is_sparse,
                                                   const std::vector<float> &max_layer_height,
                                                   size_t                    first_layer_idx);

// Applies the rule above to a planned tower. Either generator's plan fits: both carry height,
// tool_changes and combined_away per layer, and index their filament parameters by tool.
template<class PlanLayers, class FilamentParams>
void combine_sparse_wipe_tower_plan(PlanLayers &plan, const FilamentParams &filpar, size_t first_layer_idx, size_t initial_tool)
{
    const size_t       n = plan.size();
    std::vector<float> heights(n);
    std::vector<char>  sparse(n);
    std::vector<float> caps(n);

    // A layer with no toolchange prints with the filament the layer below left loaded.
    size_t tool = initial_tool;
    for (const auto &layer : plan)
        if (! layer.tool_changes.empty()) {
            tool = layer.tool_changes.front().old_tool;
            break;
        }
    for (size_t i = 0; i < n; ++i) {
        heights[i] = plan[i].height;
        sparse[i]  = plan[i].tool_changes.empty() ? 1 : 0;
        caps[i]    = tool < filpar.size() ? filpar[tool].max_layer_height : 0.f;
        if (! plan[i].tool_changes.empty())
            tool = plan[i].tool_changes.back().new_tool;
    }

    const std::vector<char> combined_away = combine_sparse_wipe_tower_layers(heights, sparse, caps, first_layer_idx);
    for (size_t i = 0; i < n; ++i) {
        plan[i].height        = heights[i];
        plan[i].combined_away = combined_away[i] != 0;
    }
}


} // namespace Slic3r

#endif // WipeTowerPrusaMM_hpp_
