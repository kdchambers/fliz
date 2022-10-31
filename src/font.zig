// SPDX-License-Identifier: GPL-3.0
// Copyright (c) 2022 Keith Chambers
// This program is free software: you can redistribute it and/or modify it under the terms
// of the GNU General Public License as published by the Free Software Foundation, version 3.

const std = @import("std");
const log = std.log;
const Allocator = std.mem.Allocator;
const toNative = std.mem.toNative;
const bigToNative = std.mem.bigToNative;
const eql = std.mem.eql;
const assert = std.debug.assert;

const geometry = @import("geometry.zig");
const graphics = @import("graphics.zig");
const Scale2D = geometry.Scale2D;
const Shift2D = geometry.Shift2D;

const util = struct {
    pub fn print(text: []const u8) void {
        var stdout = std.io.getStdErr().writer();
        _ = stdout.write(text) catch return;
    }
};

const print = util.print;

inline fn floatCompare(first: f64, second: f64) bool {
    const float_accuracy_threshold: f64 = 0.00001;
    if (first < (second + float_accuracy_threshold) and first > (second - float_accuracy_threshold))
        return true;
    return false;
}

// TODO: Rename to reverseLerp + make generic perhaps
inline fn calcPercentage(from: f64, to: f64, value: f64) f64 {
    // TODO: remove
    if (!(value <= to)) {
        std.log.err("value {d} to {d}", .{ value, to });
    }
    std.debug.assert(value >= from);
    std.debug.assert(value <= to);
    const diff = to - from;
    const relative = value - from;
    const result = relative / diff;
    std.debug.assert(result >= 0.0);
    std.debug.assert(result <= 1.0);
    return result;
}

test "calculatePercentage" {
    try std.testing.expect(calcPercentage(2.0, 4.0, 3.0) == 0.5);
    try std.testing.expect(calcPercentage(2.0, 4.0, 2.0) == 0.0);
    try std.testing.expect(calcPercentage(2.0, 244.0, 244.0) == 1.0);
    try std.testing.expect(calcPercentage(1.0, 11.0, 3.5) == 0.25);
}

/// Converts array of Vertex into array of Outline (Our own format)
/// Applies Y flip and scaling
fn createOutlines(allocator: Allocator, vertices: []Vertex, height: f64, scale: f32) ![]Outline {
    // TODO:
    std.debug.assert(@intToEnum(VMove, vertices[0].kind) == .move);

    var outline_segment_lengths = [1]u32{0} ** 32;
    const outline_count: u32 = blk: {
        var count: u32 = 0;
        for (vertices[1..]) |vertex| {
            if (@intToEnum(VMove, vertex.kind) == .move) {
                count += 1;
                continue;
            }
            outline_segment_lengths[count] += 1;
        }
        break :blk count + 1;
    };

    var outlines = try allocator.alloc(Outline, outline_count);
    {
        var i: u32 = 0;
        while (i < outline_count) : (i += 1) {
            outlines[i].segments = try allocator.alloc(OutlineSegment, outline_segment_lengths[i]);
        }
    }

    {
        var vertex_index: u32 = 1;
        var outline_index: u32 = 0;
        var outline_segment_index: u32 = 0;
        while (vertex_index < vertices.len) {
            switch (@intToEnum(VMove, vertices[vertex_index].kind)) {
                .move => {
                    vertex_index += 1;
                    outline_index += 1;
                    outline_segment_index = 0;
                },
                .line => {
                    const from = vertices[vertex_index - 1];
                    const to = vertices[vertex_index];
                    outlines[outline_index].segments[outline_segment_index] = OutlineSegment{
                        .from = Point(f64){
                            .x = @intToFloat(f64, from.x) * scale,
                            .y = height - (@intToFloat(f64, from.y) * scale),
                        },
                        .to = Point(f64){
                            .x = @intToFloat(f64, to.x) * scale,
                            .y = height - (@intToFloat(f64, to.y) * scale),
                        },
                    };
                    vertex_index += 1;
                    outline_segment_index += 1;
                },
                .curve => {
                    const from = vertices[vertex_index - 1];
                    const to = vertices[vertex_index];
                    outlines[outline_index].segments[outline_segment_index] = OutlineSegment{
                        .from = Point(f64){
                            .x = @intToFloat(f64, from.x) * scale,
                            .y = height - (@intToFloat(f64, from.y) * scale),
                        },
                        .to = Point(f64){
                            .x = @intToFloat(f64, to.x) * scale,
                            .y = height - (@intToFloat(f64, to.y) * scale),
                        },
                        .control_opt = Point(f64){
                            .x = @intToFloat(f64, to.control1_x) * scale,
                            .y = height - (@intToFloat(f64, to.control1_y) * scale),
                        },
                    };
                    vertex_index += 1;
                    outline_segment_index += 1;
                },
                // TODO:
                else => unreachable,
            }
        }
    }
    return outlines;
}

fn rasterize(allocator: Allocator, dimensions: Dimensions2D(u32), vertices: []Vertex, scale: f32) !Bitmap {
    printVertices(vertices, scale);
    const outlines = try createOutlines(allocator, vertices, @intToFloat(f64, dimensions.height), scale);
    defer allocator.free(outlines);
    printOutlines(outlines);

    const bitmap_pixel_count = @intCast(usize, dimensions.width) * dimensions.height;
    var bitmap = Bitmap{
        .width = @intCast(u32, dimensions.width),
        .height = @intCast(u32, dimensions.height),
        .pixels = try allocator.alloc(graphics.RGBA(f32), bitmap_pixel_count),
    };
    const null_pixel = graphics.RGBA(f32){ .r = 0.0, .g = 0.0, .b = 0.0, .a = 0.0 };
    std.mem.set(graphics.RGBA(f32), bitmap.pixels, null_pixel);

    var scanline_lower: usize = 1;
    var intersections_upper = try calculateHorizontalLineIntersections(0, outlines);
    while (scanline_lower < bitmap.height) : (scanline_lower += 1) {
        // while (scanline_lower < 12) : (scanline_lower += 1) {
        const scanline_upper = scanline_lower - 1;
        const base_index = scanline_upper * dimensions.width;
        var intersections_lower = try calculateHorizontalLineIntersections(@intToFloat(f64, scanline_lower), outlines);
        if (intersections_lower.len > 0 or intersections_upper.len > 0) {
            const uppers = intersections_upper.buffer[0..intersections_upper.len];
            const lowers = intersections_lower.buffer[0..intersections_lower.len];
            const connected_intersections = try combineIntersectionLists(uppers, lowers, @intToFloat(f64, scanline_upper), outlines);
            const samples_per_pixel = 3;
            for (connected_intersections.buffer[0..connected_intersections.len]) |intersect_pair| {
                const invert_coverage = intersect_pair.flags.invert_coverage;
                const upper_opt = intersect_pair.upper;
                const lower_opt = intersect_pair.lower;
                if (upper_opt != null and lower_opt != null) {
                    //
                    // Ideal situation, we have two points on upper and lower scanline (4 in total)
                    // This forms a quadralateral in the range y (0.0 - 1.0) and x (0.0 - dimensions.width)
                    //
                    const upper = upper_opt.?;
                    const lower = lower_opt.?;
                    var fill_start: i32 = std.math.maxInt(i32);
                    var fill_end: i32 = 0;
                    {
                        //
                        // Start Anti-aliasing
                        //
                        const start_x = @minimum(upper.start.x_intersect, lower.start.x_intersect);
                        const end_x = @maximum(upper.start.x_intersect, lower.start.x_intersect);
                        const ends_upper = if (end_x == upper.start.x_intersect) true else false;
                        const pixel_start = @floatToInt(usize, @floor(start_x));
                        const pixel_end = @floatToInt(usize, @floor(end_x));
                        const is_vertical = (@floor(start_x) == @floor(end_x));
                        if (is_vertical) {
                            //
                            // The upper and lower parts of the initial intersection lie on the same pixel.
                            // Coverage of pixel is the horizonal average between both points and there are
                            // no more pixels that need anti-aliasing calculated
                            //
                            const c = 255 - @floatToInt(u8, @divTrunc((@mod(start_x, 1.0) + @mod(end_x, 1.0)) * 255.0, 2.0));
                            std.debug.assert(c <= 255);
                            std.debug.assert(c >= 0);
                            bitmap.pixels[pixel_start + base_index] = graphics.RGBA(f32).fromInt(u8, c, c, c, 255);
                        } else {
                            const start_top = if (start_x == upper.start.x_intersect) true else false;
                            var i = pixel_start;
                            var entry_point = Point(f64){ .x = start_x - @floor(start_x), .y = 0.0 };

                            const pixel_range = pixel_end - pixel_start;
                            if (pixel_range > 1) {
                                std.log.info("Pixel range: {d}", .{pixel_range});
                            }

                            while (i <= pixel_end) : (i += 1) {
                                const last_point = Point(f64){ .x = end_x - entry_point.x, .y = if (ends_upper) 1.0 else 0.0 };
                                const exit_point = interpolateBoundryPoint(entry_point, last_point);
                                std.debug.assert(exit_point.x >= 0.0);
                                std.debug.assert(exit_point.x <= 1.0);
                                std.debug.assert(exit_point.y >= 0.0);
                                std.debug.assert(exit_point.y <= 1.0);
                                const fill_anchor_point = Point(f64){ .x = 1.0, .y = if (start_top) 0.0 else 1.0 };
                                const c = @floatToInt(u8, @floor(255.0 * triangleArea(entry_point, exit_point, fill_anchor_point)));
                                std.debug.assert(c <= 255);
                                std.debug.assert(c >= 0);
                                bitmap.pixels[i + base_index] = graphics.RGBA(f32).fromInt(u8, c, c, c, 255);
                                entry_point = Point(f64){ .x = 0.0, .y = exit_point.y };
                            }
                        }
                        fill_start = @floatToInt(i32, @floor(end_x)) + 1;
                    }
                    {
                        //
                        // End Anti-aliasing
                        //
                        const start_x = @minimum(upper.end.x_intersect, lower.end.x_intersect);
                        const end_x = @maximum(upper.end.x_intersect, lower.end.x_intersect);
                        const ends_upper = if (end_x == upper.end.x_intersect) true else false;
                        const pixel_start = @floatToInt(usize, @floor(start_x));
                        const pixel_end = @floatToInt(usize, @floor(end_x));
                        const is_vertical = (@floor(start_x) == @floor(end_x));
                        if (is_vertical) {
                            const c = @floatToInt(u8, @divTrunc((@mod(start_x, 1.0) + @mod(end_x, 1.0)) * 255.0, 2.0));
                            std.debug.assert(c <= 255);
                            std.debug.assert(c >= 0);
                            bitmap.pixels[pixel_start + base_index] = graphics.RGBA(f32).fromInt(u8, c, c, c, 255);
                        } else {
                            const start_top = if (start_x == upper.end.x_intersect) true else false;
                            var i = pixel_start;
                            var entry_point = Point(f64){ .x = start_x - @floor(start_x), .y = 0.0 };
                            while (i <= pixel_end) : (i += 1) {
                                const last_point = Point(f64){ .x = end_x - entry_point.x, .y = if (ends_upper) 1.0 else 0.0 };
                                const exit_point = interpolateBoundryPoint(entry_point, last_point);
                                std.debug.assert(exit_point.x >= 0.0);
                                std.debug.assert(exit_point.x <= 1.0);
                                std.debug.assert(exit_point.y >= 0.0);
                                std.debug.assert(exit_point.y <= 1.0);
                                const fill_anchor_point = Point(f64){ .x = 1.0, .y = if (start_top) 0.0 else 1.0 };
                                const c = @floatToInt(u8, @floor(255.0 * triangleArea(entry_point, exit_point, fill_anchor_point)));
                                std.debug.assert(c <= 255);
                                std.debug.assert(c >= 0);
                                bitmap.pixels[i + base_index] = graphics.RGBA(f32).fromInt(u8, c, c, c, 255);
                                entry_point = Point(f64){ .x = 0.0, .y = exit_point.y };
                            }
                        }
                        fill_end = @floatToInt(i32, @floor(start_x)) - 1;
                    }
                    //
                    // Inner fill
                    //
                    var i: usize = @intCast(usize, fill_start);
                    while (i <= @intCast(usize, fill_end)) : (i += 1) {
                        bitmap.pixels[i + base_index] = graphics.RGBA(f32).fromInt(u8, 255, 255, 255, 255);
                    }
                } else {
                    //
                    // We only have a upper or lower scanline
                    //

                    //
                    // Idea: You might also be able to set the invert flag based on the direction of t
                    //       Taking advantage of winding order
                    //
                    print("Rasterizing with only one scanline\n");
                    std.debug.assert(lower_opt == null or upper_opt == null);

                    const is_upper = (lower_opt == null);
                    const pair = if (is_upper) upper_opt.? else lower_opt.?;
                    const outline_index = pair.start.outline_index;
                    std.log.info("Outline index {d} segment count {d}", .{ outline_index, outlines[outline_index].segments.len });
                    std.debug.assert(outline_index == pair.end.outline_index);

                    const outline = outlines[outline_index];
                    const sample_t_max = @intToFloat(f64, outlines[outline_index].segments.len);

                    std.log.info("Fill: (t {d}, x {d}) -> (t {d}, x {d})", .{
                        pair.start.t,
                        pair.start.x_intersect,
                        pair.end.t,
                        pair.end.x_intersect,
                    });

                    const pixel_start = @floatToInt(usize, @floor(pair.start.x_intersect));
                    const pixel_end = @floatToInt(usize, @floor(pair.end.x_intersect));
                    std.debug.assert(pixel_start <= pixel_end);
                    const pixel_count: usize = pixel_end - pixel_start;

                    var pixel_x = pixel_start;
                    if (pixel_count == 0) {
                        // TODO
                        const c = 255.0;
                        bitmap.pixels[pixel_x + base_index] = graphics.RGBA(f32).fromInt(u8, c, c, c, 255);
                        continue;
                    }

                    var fill_anchor_point = Point(f64){
                        .x = 1.0,
                        .y = if (is_upper) 1.0 else 0.0,
                    };

                    const samples_to_take: usize = pixel_count * samples_per_pixel;
                    const sample_t_start = pair.start.t;
                    const sample_t_end = pair.end.t;

                    //
                    // These need to be calculated based on whether which direction (forward / reverse)
                    // is most suitable (I.e Closest)
                    //
                    var sample_t_length: f64 = undefined;
                    var sample_t_increment: f64 = undefined;

                    if (sample_t_start < sample_t_end) {
                        const forward = sample_t_end - sample_t_start;
                        const backward = sample_t_start + (sample_t_max - sample_t_end);
                        if (forward < backward) {
                            sample_t_length = forward;
                            sample_t_increment = forward / @intToFloat(f64, samples_to_take);
                        } else {
                            sample_t_length = backward;
                            sample_t_increment = -backward / @intToFloat(f64, samples_to_take);
                        }
                    } else {
                        const forward = sample_t_end + (sample_t_max - sample_t_start);
                        const backward = sample_t_start - sample_t_end;
                        if (forward < backward) {
                            sample_t_length = forward;
                            sample_t_increment = forward / @intToFloat(f64, samples_to_take);
                        } else {
                            sample_t_length = backward;
                            sample_t_increment = -backward / @intToFloat(f64, samples_to_take);
                        }
                    }
                    std.log.info("Sample range: {d}->{d} {d} increment {d} count {d}", .{
                        sample_t_start,
                        sample_t_end,
                        sample_t_length,
                        sample_t_increment,
                        samples_to_take,
                    });
                    std.debug.assert(sample_t_length <= (sample_t_max / 2.0));

                    {
                        var end_sample_abs = @mod(sample_t_start + (sample_t_increment * @intToFloat(f64, samples_to_take)), sample_t_max);
                        if (end_sample_abs < 0.0) {
                            end_sample_abs += sample_t_max;
                        }
                        std.debug.assert(floatCompare(end_sample_abs, sample_t_end));
                    }

                    var previous_sampled_point = Point(f64){
                        .x = pair.start.x_intersect - @intToFloat(f64, pixel_start),
                        .y = if (is_upper) 1.0 else 0.0,
                    };
                    const base_y = @intToFloat(f64, scanline_upper);
                    var sample_index: usize = 1;
                    var current_sampled_point: Point(f64) = undefined;
                    var sample_t: f64 = undefined;
                    var coverage: f64 = 0.0;

                    while (sample_index < samples_to_take) : (sample_index += 1) {
                        current_sampled_point = blk: {
                            sample_t = blk2: {
                                const val = @mod(sample_t_start + (sample_t_increment * @intToFloat(f64, sample_index)), sample_t_max);
                                break :blk2 if (val >= 0.0) val else val + sample_t_max;
                            };
                            const absolute_sampled_point = outline.samplePoint(sample_t);
                            break :blk Point(f64){
                                .x = absolute_sampled_point.x - @intToFloat(f64, pixel_x),
                                .y = absolute_sampled_point.y - base_y,
                            };
                        };
                        std.log.info("Sample #{d}: t {d} current ({d}, {d})", .{
                            sample_index,
                            sample_t,
                            current_sampled_point.x,
                            current_sampled_point.y,
                        });
                        std.debug.assert(current_sampled_point.y >= 0.0);
                        std.debug.assert(current_sampled_point.y <= 1.0);

                        if (current_sampled_point.x >= 1.0) {
                            // We've sampled into the neigbouring right pixel.
                            // Interpolate a pixel on the rightside and then set the pixel value.
                            std.log.info("Point crosses righthand pixel border. Interpolating endpoint", .{});
                            std.debug.assert(current_sampled_point.x > previous_sampled_point.x);
                            const interpolated_point = interpolateBoundryPoint(previous_sampled_point, current_sampled_point);
                            coverage += triangleArea(interpolated_point, previous_sampled_point, fill_anchor_point);
                            std.debug.assert(coverage >= 0.0);
                            std.debug.assert(coverage <= 1.0);
                            if (invert_coverage) {
                                coverage = 1.0 - coverage;
                            }
                            const c = @floatToInt(u8, coverage * 255.0);
                            bitmap.pixels[pixel_x + base_index] = graphics.RGBA(f32).fromInt(u8, c, c, c, 255);

                            //
                            // Adjust for next pixel
                            //
                            previous_sampled_point = .{ .x = 0.0, .y = interpolated_point.y };
                            current_sampled_point.x -= 1.0;
                            std.debug.assert(current_sampled_point.x >= 0.0);
                            std.debug.assert(current_sampled_point.x <= 1.0);
                            std.log.info("Calculating coverage for pixel {d}", .{pixel_x});
                            fill_anchor_point.x = 0.0;
                            pixel_x += 1;
                            //
                            // Calculate first coverage for next pixel
                            //
                            coverage = triangleArea(current_sampled_point, previous_sampled_point, fill_anchor_point);
                            previous_sampled_point = current_sampled_point;
                        } else {
                            coverage += triangleArea(current_sampled_point, previous_sampled_point, fill_anchor_point);
                            std.debug.assert(coverage >= 0.0);
                            std.debug.assert(coverage <= 1.0);
                            previous_sampled_point = current_sampled_point;
                        }
                    } // end while
                    //
                    // Rasterize last pixel
                    //
                    std.log.info("Rasterizing last pixel", .{});
                    const interpolated_point = blk: {
                        if (@floatToInt(usize, @floor(current_sampled_point.x)) == pixel_end) {
                            std.log.info("Can use sampled point", .{});
                            break :blk current_sampled_point;
                        }
                        // We actually need to interpolate a point between current_sampled_point and last point
                        const end_point = Point(f64){
                            .x = pair.end.x_intersect - @intToFloat(f64, pixel_end),
                            .y = if (is_upper) 1.0 else 0.0,
                        };
                        break :blk interpolateBoundryPoint(current_sampled_point, end_point);
                    };
                    coverage += triangleArea(interpolated_point, previous_sampled_point, fill_anchor_point);
                    if (invert_coverage) {
                        coverage = 1.0 - coverage;
                    }
                    const c = @floatToInt(u8, coverage * 255.0);
                    bitmap.pixels[pixel_x + base_index] = graphics.RGBA(f32).fromInt(u8, c, c, c, 255);
                } // end if
            }
        } else {
            std.log.info("Skipping rasterization", .{});
        }
        intersections_upper = intersections_lower;
    }

    for (outlines) |*outline| {
        allocator.free(outline.segments);
    }

    return bitmap;
}

const Outline = struct {
    segments: []OutlineSegment,

    pub fn samplePoint(self: @This(), t: f64) Point(f64) {
        const t_floored: f64 = @floor(t);
        const segment_index = @floatToInt(usize, t_floored);
        std.debug.assert(segment_index < self.segments.len);
        return self.segments[segment_index].sample(t - t_floored);
    }

    // When a fill region is ended, the direction around the outline_segment will be negative
    pub fn sampleAtDistance(self: @This(), ideal: f64, threshold: f64, base_point: SampledPoint) SampledPoint {
        const t_floored: f64 = @floor(base_point.t);
        const outline_segment_index = @floatToInt(usize, t_floored);
        var new_base = SampledPoint{
            .p = base_point.p,
            .t = base_point.t - t_floored,
            .t_increment = base_point.t_increment,
        };
        if (self.outline_segment[outline_segment_index].sampleAtDistance(ideal, threshold, new_base)) |sample| {
            return sample;
        }
        const next_outline_segment_index = (outline_segment_index + 1) % self.outline_segment.len;
        new_base.t = 0.0;
        // TODO: Is is possible though that distance will be large enough to skip a contour,
        //       or even loop around entirely
        if (self.outline_segment[next_outline_segment_index].sampleAtDistance(ideal, threshold, new_base)) |sample| {
            return sample;
        }
        // This function cannot fail (Barring a bug in the code), as an outline is a closed loop
        unreachable;
    }
};

const SampledPoint = struct {
    p: Point(f64),
    t: f64,
    t_increment: f64,
};

const YIntersection = struct {
    outline_index: u32,
    x_intersect: f64,
    t: f64, // t value (sample) of outline

    pub fn print(self: @This(), comptime indent_level: comptime_int, comptime newline: bool, index: usize) void {
        const printf = std.debug.print;
        const indent = "  " ** indent_level;
        printf("{s}{d:.2}. t {d:.4}  x_intersect {d:.4} outline_index: {d}", .{
            indent,
            index,
            self.t,
            self.x_intersect,
            self.outline_index,
        });
        if (newline) {
            std.debug.print("\n", .{});
        }
    }
};

const YIntersectionPair = struct {
    start: YIntersection,
    end: YIntersection,

    pub fn print(self: @This(), comptime indent_level: comptime_int, comptime newline: bool) void {
        const indent = "  " ** indent_level;
        std.debug.print("{s}Start ", .{indent});
        self.start.print();
        std.debug.print("\n", .{});
        std.debug.print("{s}End ", .{indent});
        self.end.print();
        if (newline) {
            std.debug.print("\n", .{});
        }
    }
};

const YIntersectionPairList = struct {
    const capacity = 32;
    buffer: [capacity]YIntersectionPair,
    len: u64,

    pub fn print(self: @This()) void {
        std.debug.print("Pair list\n", .{});
        for (self.buffer[0..self.len]) |pair| {
            pair.print(1, true);
        }
    }

    pub fn add(self: *@This(), intersection: YIntersectionPair) !void {
        if (self.len == capacity) {
            return error.BufferFull;
        }
        self.buffer[self.len] = intersection;
        self.len += 1;
    }

    pub fn toSlice(self: @This()) []const YIntersectionPair {
        return self.buffer[0..self.len];
    }
};

const YIntersectionList = struct {
    const capacity = 32;
    buffer: [capacity]YIntersection,
    len: u64,

    pub fn print(self: @This()) void {
        std.debug.print("Intersection list\n", .{});
        for (self.buffer[0..self.len]) |pair| {
            pair.print(1, true);
        }
    }

    pub fn add(self: *@This(), intersection: YIntersection) !void {
        std.debug.assert(intersection.x_intersect >= 0.0);
        if (self.len == capacity) {
            return error.BufferFull;
        }
        self.buffer[self.len] = intersection;
        self.len += 1;
    }

    pub fn toSlice(self: @This()) []const YIntersection {
        return self.buffer[0..self.len];
    }
};

const IntersectionConnection = struct {
    const Flags = packed struct {
        invert_coverage: bool = false,
    };

    lower: ?YIntersectionPair,
    upper: ?YIntersectionPair,
    flags: Flags = .{},

    pub fn print(self: @This(), index: usize, comptime indent_level: comptime_int) void {
        const indent = "  " ** indent_level;
        const printf = std.debug.print;
        if (self.upper) |upper| {
            printf("{s}{d:.2}. upper t {d} ({d}) ==> t {d} ({d})\n", .{
                indent,
                index,
                upper.start.t,
                upper.start.x_intersect,
                upper.end.t,
                upper.end.x_intersect,
            });
        } else {
            printf("{s}{d:.2}. upper: null\n", .{ indent, index });
        }
        if (self.lower) |lower| {
            printf("{s}{d:.2}. lower t {d} ({d}) ==> t {d} ({d})\n", .{
                indent,
                index,
                lower.start.t,
                lower.start.x_intersect,
                lower.end.t,
                lower.end.x_intersect,
            });
        } else {
            printf("{s}{d:.2}. lower: null\n", .{ indent, index });
        }
    }
};

const IntersectionConnectionList = struct {
    const capacity = 32;

    buffer: [capacity]IntersectionConnection,
    len: u32,

    pub fn print(self: @This()) void {
        std.debug.print("** Intersection Connection List **\n", .{});
        for (self.buffer[0..self.len]) |connection, i| {
            connection.print(i, 1);
        }
    }

    pub fn toSlice(self: @This()) []IntersectionConnection {
        return self.buffer[0..self.len];
    }

    pub fn add(self: *@This(), connection: IntersectionConnection) !void {
        if (self.len == capacity) {
            return error.BufferFull;
        }
        self.buffer[self.len] = connection;
        self.len += 1;
    }
};

inline fn minTDistance(a: f64, b: f64, max: f64) f64 {
    const positive = (a <= b);
    const dist_forward = if (positive) b - a else b + (max - a);
    const dist_reverse = if (positive) max - dist_forward else a - b;
    const dist_min = @minimum(dist_forward, dist_reverse);
    return dist_min;
}

test "minTDistance" {
    const expect = std.testing.expect;
    try expect(minTDistance(0.2, 0.4, 1.0) == 0.2);
    try expect(minTDistance(1.0, 1.0, 1.0) == 0.0);
    try expect(minTDistance(0.6, 0.2, 0.7) == 0.3);
    try expect(minTDistance(0.65, 0.25, 0.75) == 0.35);
}

inline fn minTMiddle(a: f64, b: f64, max: f64) f64 {
    // std.debug.print("minTMiddle: a {d}, b {d}, max {d}\n", .{ a, b, max });
    const positive = (a <= b);
    const dist_forward = if (positive) b - a else b + (max - a);
    const dist_reverse = if (positive) max - dist_forward else a - b;
    if (dist_forward < dist_reverse) {
        return @mod(a + (dist_forward / 2.0), max);
    }
    var middle = @mod(b + (dist_reverse / 2.0), max);
    const result = if (middle >= 0.0) middle else middle + max;
    // std.debug.print("result: {d}\n", .{result});
    std.debug.assert(result >= 0.0);
    std.debug.assert(result <= max);
    return result;
}

test "minTMiddle" {
    const expect = std.testing.expect;
    try expect(minTMiddle(0.2, 0.5, 1.0) == 0.35);
    try expect(minTMiddle(0.5, 0.4, 1.0) == 0.45);
    try expect(minTMiddle(0.8, 0.2, 1.0) == 0.0);
    try expect(minTMiddle(16.0, 2.0, 20.0) == 19.0);
}

const IntersectionList = struct {
    const capacity = 64;

    upper_index_start: u32,
    lower_index_start: u32,
    upper_count: u32,
    lower_count: u32,
    increment: i32,
    buffer: [capacity]YIntersection,

    fn print(self: @This()) void {
        {
            util.print("Upper:\n");
            var i: usize = 0;
            while (i < self.upper_count) : (i += 1) {
                const index = i + self.upper_index_start;
                self.buffer[index].print(1, true, i);
            }
        }
        {
            util.print("Lower:\n");
            var i: usize = 0;
            while (i < self.lower_count) : (i += 1) {
                const index = i + self.lower_index_start;
                self.buffer[index].print(1, true, i);
            }
        }
    }

    fn makeFromSeperateScanlines(uppers: []const YIntersection, lowers: []const YIntersection) IntersectionList {
        std.debug.assert(uppers.len + lowers.len <= IntersectionList.capacity);
        var result = IntersectionList{
            .upper_index_start = 0,
            .lower_index_start = @intCast(u32, uppers.len),
            .upper_count = @intCast(u32, uppers.len),
            .lower_count = @intCast(u32, lowers.len),
            .increment = 1,
            .buffer = undefined,
        };

        var i: usize = 0;
        for (uppers) |upper| {
            result.buffer[i] = upper;
            i += 1;
        }
        for (lowers) |lower| {
            result.buffer[i] = lower;
            i += 1;
        }
        const total_len = uppers.len + lowers.len;
        std.debug.assert(result.length() == total_len);
        std.debug.assert(result.toSlice().len == total_len);
        return result;
    }

    inline fn toSlice(self: @This()) []const YIntersection {
        const start_index = if (self.upper_index_start < self.lower_index_start) self.upper_index_start else self.lower_index_start;
        return self.buffer[start_index .. start_index + (self.upper_count + self.lower_count)];
    }

    inline fn length(self: @This()) usize {
        return self.upper_count + self.lower_count;
    }

    inline fn at(self: @This(), index: usize) YIntersection {
        return self.toSlice()[index];
    }

    inline fn isUpper(self: @This(), index: usize) bool {
        const upper_start: u32 = self.upper_index_start;
        const upper_end: i64 = upper_start + @intCast(i64, self.upper_count) - 1;
        std.log.info("isUpper: start {d}, end {d}, index {d}", .{ upper_start, upper_end, index });
        return (upper_start <= upper_end and index >= upper_start and index <= upper_end);
    }

    // Two points are 't_connected', if there doesn't exist a closer t value
    // going in the same direction (Forward or reverse)
    inline fn isTConnected(self: @This(), base_index: usize, candidate_index: usize, max_t: f64) bool {
        const slice = self.toSlice();

        const base_outline_index = slice[base_index].outline_index;
        std.debug.assert(base_outline_index == slice[candidate_index].outline_index);

        const base_t = slice[base_index].t;
        const candidate_t = slice[candidate_index].t;
        if (base_t == candidate_t) return true;

        const dist_forward = @mod(candidate_t + (max_t - base_t), max_t);
        std.debug.assert(dist_forward >= 0.0);
        std.debug.assert(dist_forward < max_t);

        const dist_reverse = @mod(@fabs(base_t + (max_t - candidate_t)), max_t);
        std.debug.assert(dist_reverse >= 0.0);
        std.debug.assert(dist_reverse < max_t);

        const is_forward = if (dist_forward < dist_reverse) true else false;
        if (is_forward) {
            for (slice) |other, other_i| {
                if (other.t == base_t or other.t == candidate_t) continue;
                if (other_i == candidate_index or other_i == base_index) continue;
                if (other.outline_index != base_outline_index) continue;
                const dist_other = @mod(other.t + (max_t - base_t), max_t);
                if (dist_other < dist_forward) {
                    // std.debug.print("Better match (forward) [{d}] t {d}: [{d}] t {d}\n", .{
                    //     base_index, base_t,
                    //     other_i,    other.t,
                    // });
                    return false;
                }
            }
            return true;
        }
        for (slice) |other, other_i| {
            if (other.t == base_t or other.t == candidate_t) continue;
            if (other_i == candidate_index or other_i == base_index) continue;
            if (other.outline_index != base_outline_index) continue;
            const dist_other = @mod(@fabs(base_t + (max_t - other.t)), max_t);
            if (dist_other < dist_reverse) {
                // std.debug.print("Better match (reverse) [{d}] t {d}: [{d}] t {d}\n", .{
                //     base_index, base_t,
                //     other_i,    other.t,
                // });
                return false;
            }
        }
        return true;
    }

    fn swapScanlines(self: *@This()) !void {
        self.lower_index_start = self.upper_index_start;
        self.lower_count = self.upper_count;
        self.upper_count = 0;
        self.upper_index_start = blk: {
            const lower_index_end = self.lower_index_start + self.lower_count;
            const space_forward = self.capacity - lower_index_end;
            const space_behind = self.lower_index_start;
            if (space_forward > space_behind) {
                self.increment = 1;
                break :blk lower_index_end;
            }
            self.increment = 1;
            break :blk self.lower_index_start - 1;
        };
    }

    inline fn add(self: *@This(), intersection: YIntersection) !void {
        self.buffer[self.upper_index_start + self.increment] = intersection;
        self.upper_count += 1;
    }
};

test "isTConnected" {
    const expect = std.testing.expect;
    const max_t = 20.0;
    // 1, 5, 0, 4, 3, 2
    const example_outline = [_]f64{ 6.5, 0.3, 19.3, 15.6, 8.7, 3.4 };
    try expect(isTConnected(example_outline[0..], 5, 0, max_t) == true);
    try expect(isTConnected(example_outline[0..], 2, 3, max_t) == true);
    try expect(isTConnected(example_outline[0..], 1, 2, max_t) == true);
    try expect(isTConnected(example_outline[0..], 5, 1, max_t) == true);
    try expect(isTConnected(example_outline[0..], 4, 0, max_t) == true);
    try expect(isTConnected(example_outline[0..], 4, 3, max_t) == true);
    try expect(isTConnected(example_outline[0..], 2, 3, max_t) == true);
    try expect(isTConnected(example_outline[0..], 2, 1, max_t) == true);
    try expect(isTConnected(example_outline[0..], 1, 0, max_t) == false);
    try expect(isTConnected(example_outline[0..], 5, 4, max_t) == false);
    try expect(isTConnected(example_outline[0..], 5, 3, max_t) == false);
    try expect(isTConnected(example_outline[0..], 4, 2, max_t) == false);
    try expect(isTConnected(example_outline[0..], 1, 3, max_t) == false);
}

inline fn isTConnected(other_slice: []const f64, base_index: usize, candidate_index: usize, max_t: f64) bool {
    const base_t = other_slice[base_index];
    const candidate_t = other_slice[candidate_index];
    if (base_t == candidate_t) return true;

    const dist_forward = @mod(candidate_t + (max_t - base_t), max_t);
    std.debug.assert(dist_forward >= 0.0);
    std.debug.assert(dist_forward < max_t);

    const dist_reverse = @mod(@fabs(base_t + (max_t - candidate_t)), max_t);
    std.debug.assert(dist_reverse >= 0.0);
    std.debug.assert(dist_reverse < max_t);

    const is_forward = if (dist_forward < dist_reverse) true else false;
    if (is_forward) {
        for (other_slice) |other, other_i| {
            if (other == base_t or other == candidate_t) continue;
            if (other_i == candidate_index or other_i == base_index) continue;
            const dist_other = @mod(other + (max_t - base_t), max_t);
            if (dist_other < dist_forward) {
                // std.debug.print("Better match (forward) [{d}] t {d}: [{d}] t {d}\n", .{
                //     base_index, base_t,
                //     other_i,    other,
                // });
                return false;
            }
        }
        return true;
    }
    for (other_slice) |other, other_i| {
        if (other == base_t or other == candidate_t) continue;
        if (other_i == candidate_index or other_i == base_index) continue;
        const dist_other = @mod(@fabs(base_t + (max_t - other)), max_t);
        if (dist_other < dist_reverse) {
            // std.debug.print("Better match (reverse) [{d}] t {d}: [{d}] t {d}\n", .{
            //     base_index, base_t,
            //     other_i,    other,
            // });
            return false;
        }
    }
    return true;
}

/// Takes a list of upper and lower intersections, and groups them into
/// 2 or 4 point intersections that makes it easy for the rasterizer
fn combineIntersectionLists(
    uppers: []const YIntersection,
    lowers: []const YIntersection,
    base_scanline: f64,
    outlines: []const Outline,
) !IntersectionConnectionList {
    //
    // Lines are connected if:
    // 1. Connected by T
    // 2. Middle t lies within scanline
    // 3. Part of the same outline
    //
    const intersections = IntersectionList.makeFromSeperateScanlines(uppers, lowers);
    print("Printing custom intersection buffer\n");
    intersections.print();

    const total_count: usize = uppers.len + lowers.len;
    std.debug.assert(intersections.length() == total_count);

    // TODO: Hard coded size
    //       Replace [2]usize with struct
    var pair_list: [10][2]usize = undefined;
    var pair_count: usize = 0;
    {
        var matched = [1]bool{false} ** 32;
        for (intersections.toSlice()) |intersection, intersection_i| {
            if (intersection_i == intersections.length() - 1) break;
            if (matched[intersection_i] == true) continue;
            const intersection_outline_index = intersection.outline_index;
            const intersection_outline = outlines[intersection_outline_index];
            const outline_max_t = @intToFloat(f64, intersection_outline.segments.len);
            var other_i: usize = intersection_i + 1;
            std.debug.print("Checking matches of {d}\n", .{intersection_i});
            var smallest_t_diff = std.math.floatMax(f64);
            var best_match_index: ?usize = null;
            while (other_i < total_count) : (other_i += 1) {
                if (matched[other_i] == true) continue;
                const other_intersection = intersections.at(other_i);
                if (intersection.t == other_intersection.t) continue;
                const other_intersection_outline_index = other_intersection.outline_index;
                if (other_intersection_outline_index != intersection_outline_index) continue;
                std.debug.print("  Comparing with {d}\n", .{other_i});
                const within_scanline = blk: {
                    const middle_t = minTMiddle(intersection.t, other_intersection.t, outline_max_t);
                    // TODO: Specialized implementation of samplePoint just for y value
                    const sample_point = intersection_outline.samplePoint(middle_t);
                    std.debug.print("  Sample point @t {d}: {d}, {d}\n", .{ middle_t, sample_point.x, sample_point.y });
                    const relative_y = sample_point.y - base_scanline;
                    std.debug.print("  relative_y: {d}\n", .{relative_y});
                    break :blk relative_y >= 0.0 and relative_y <= 1.0;
                };
                if (!within_scanline) {
                    continue;
                }
                const is_t_connected = intersections.isTConnected(intersection_i, other_i, outline_max_t);
                if (is_t_connected) {
                    // TODO: This doesn't take into account wrapping
                    const t_diff = @fabs(intersection.t - other_intersection.t);
                    if (t_diff < smallest_t_diff) {
                        smallest_t_diff = t_diff;
                        best_match_index = other_i;
                    }
                    std.log.info("Potential match with index {d}", .{other_i});
                } else {
                    std.log.info("Rejected because not t connected", .{});
                }
            }
            if (best_match_index) |match_index| {
                std.log.info("Matched with index {d}", .{match_index});
                const match_intersection = intersections.at(match_index);
                const swap = intersection.x_intersect > match_intersection.x_intersect;
                pair_list[pair_count][0] = if (swap) match_index else intersection_i;
                pair_list[pair_count][1] = if (swap) intersection_i else match_index;
                matched[match_index] = true;
                matched[intersection_i] = true;
                pair_count += 1;
            } else {
                std.debug.assert(false);
            }
        }
    }

    {
        var i: usize = 0;
        print("Connected pairs:\n");
        while (i < pair_count) : (i += 1) {
            const p = pair_list[i];
            const a = intersections.at(p[0]);
            const b = intersections.at(p[1]);
            std.debug.print("[{d}]: t {d} x {d} -> t {d} x {d}\n", .{
                i,
                a.t,
                a.x_intersect,
                b.t,
                b.x_intersect,
            });
        }
        print("\n");
    }

    const min_pair_count = @divTrunc(total_count, 2);
    std.debug.print("Min pair count: {d} actual count {d}\n", .{ min_pair_count, pair_count });

    std.debug.assert(pair_count >= min_pair_count);

    var connection_list = IntersectionConnectionList{ .buffer = undefined, .len = 0 };
    {
        var matched = [1]bool{false} ** 32;
        var i: usize = 0;
        while (i < pair_count) : (i += 1) {
            if (matched[i]) continue;
            const index_start = pair_list[i][0];
            const index_end = pair_list[i][1];
            std.debug.print("Matching pair {d} & {d}\n", .{ index_start, index_end });
            const start = intersections.at(index_start);
            const end = intersections.at(index_end);
            std.debug.print("Start x {d} end x {d}\n", .{ start.x_intersect, end.x_intersect });
            std.debug.assert(start.x_intersect <= end.x_intersect);
            const start_is_upper = intersections.isUpper(index_start);
            const end_is_upper = intersections.isUpper(index_end);
            if (start_is_upper == end_is_upper) {
                print("On same scanline. Adding as-is\n");
                const intersection_pair = YIntersectionPair{
                    .start = start,
                    .end = end,
                };
                if (start_is_upper) {
                    try connection_list.add(.{ .upper = intersection_pair, .lower = null });
                } else {
                    try connection_list.add(.{ .upper = null, .lower = intersection_pair });
                }
            } else {
                print("Different scanline, searching for closing match\n");
                //
                // Pair touches both scanlines
                // You need to find the match
                // Match criteria:
                //   1. Also across both scanlines
                //   2. Has the most leftmost point
                //
                var x: usize = i + 1;
                const ref_x_intersect: f64 = @maximum(start.x_intersect, end.x_intersect);
                var smallest_x: f64 = std.math.floatMax(f64);
                var smallest_index_opt: ?usize = null;
                while (x < pair_count) : (x += 1) {
                    const comp_index_start = pair_list[x][0];
                    const comp_index_end = pair_list[x][1];
                    const comp_start_is_upper = intersections.isUpper(comp_index_start);
                    const comp_end_is_upper = intersections.isUpper(comp_index_end);
                    if (comp_start_is_upper == comp_end_is_upper) {
                        std.log.info("Comp on same scanline", .{});
                        continue;
                    }

                    const comp_start = intersections.at(comp_index_start);
                    const comp_end = intersections.at(comp_index_end);
                    const comp_x = @minimum(comp_start.x_intersect, comp_end.x_intersect);
                    std.debug.print("Comp_x: {d} against {d}\n", .{ comp_x, ref_x_intersect });
                    if (comp_x >= ref_x_intersect and comp_x < smallest_x) {
                        smallest_x = comp_x;
                        smallest_index_opt = x;
                    }
                }
                if (smallest_index_opt) |smallest_index| {
                    const match_pair = pair_list[smallest_index];
                    const match_start = intersections.at(match_pair[0]);
                    const match_end = intersections.at(match_pair[1]);
                    const match_start_is_upper = intersections.isUpper(match_pair[0]);
                    std.debug.assert(match_start.x_intersect <= match_end.x_intersect);

                    const upper_start = if (start_is_upper) start else end;
                    const upper_end = if (match_start_is_upper) match_start else match_end;
                    std.debug.assert(upper_start.x_intersect <= upper_end.x_intersect);

                    const lower_start = if (start_is_upper) end else start;
                    const lower_end = if (match_start_is_upper) match_end else match_start;
                    std.debug.assert(lower_start.x_intersect <= lower_end.x_intersect);

                    const upper = YIntersectionPair{
                        .start = upper_start,
                        .end = upper_end,
                    };
                    const lower = YIntersectionPair{
                        .start = lower_start,
                        .end = lower_end,
                    };
                    try connection_list.add(.{ .upper = upper, .lower = lower });
                    matched[smallest_index] = true;
                } else {
                    std.debug.assert(false);
                    // return error.FailedToFindMatch;
                }
            }
        }
    }

    std.debug.print("Final connection list\n", .{});
    connection_list.print();

    return connection_list;
}

fn calculateHorizontalLineIntersections(scanline_y: f64, outlines: []Outline) !YIntersectionList {
    var intersection_list = YIntersectionList{ .len = 0, .buffer = undefined };
    const printf = std.debug.print;
    std.log.info("Calculating intersections for scanline {d}", .{scanline_y});
    for (outlines) |outline, outline_i| {
        for (outline.segments) |segment, segment_i| {
            const point_a = segment.from;
            const point_b = segment.to;
            const max_y = @maximum(point_a.y, point_b.y);
            const min_y = @minimum(point_a.y, point_b.y);
            if (segment.isCurve()) {
                // TODO: Improve bounding box calculation
                // https://iquilezles.org/articles/bezierbbox/
                const control_point = segment.control_opt.?;
                printf("  {d:.2} Vertex (curve) A({d:.2}, {d:.2}) --> B({d:.2}, {d:.2}) C ({d:.2}, {d:.2}) -- ", .{
                    segment_i,
                    point_a.x,
                    point_a.y,
                    point_b.x,
                    point_b.y,
                    control_point.x,
                    control_point.y,
                });
                const bezier = BezierQuadratic{ .a = point_a, .b = point_b, .control = control_point };
                const inflection_y = quadradicBezierInflectionPoint(bezier).y;
                const is_middle_higher = (inflection_y > max_y) and scanline_y > inflection_y;
                const is_middle_lower = (inflection_y < min_y) and scanline_y < inflection_y;
                if (is_middle_higher or is_middle_lower) {
                    printf("REJECT - Outsize Y range\n", .{});
                    {
                        // TODO: Remove paranoia check
                        const intersections = quadraticBezierPlaneIntersections(bezier, scanline_y);
                        std.debug.assert(intersections[0] == null);
                        std.debug.assert(intersections[1] == null);
                    }
                    continue;
                }
                const optional_intersection_points = quadraticBezierPlaneIntersections(bezier, scanline_y);
                if (optional_intersection_points[0]) |first_intersection| {
                    {
                        const intersection = YIntersection{
                            .outline_index = @intCast(u32, outline_i),
                            .x_intersect = first_intersection.x,
                            .t = @intToFloat(f64, segment_i) + first_intersection.t,
                        };
                        try intersection_list.add(intersection);
                    }
                    if (optional_intersection_points[1]) |second_intersection| {
                        const x_diff_threshold = 0.001;
                        if (@fabs(second_intersection.x - first_intersection.x) > x_diff_threshold) {
                            const t_second = @intToFloat(f64, segment_i) + second_intersection.t;
                            const intersection = YIntersection{
                                .outline_index = @intCast(u32, outline_i),
                                .x_intersect = second_intersection.x,
                                .t = t_second,
                            };
                            try intersection_list.add(intersection);
                            printf("Curve (1st) w/ t {d} && (2nd) w/ t {d}\n", .{ @intToFloat(f64, segment_i) + first_intersection.t, t_second });
                        } else {
                            std.log.warn("Collapsing two points in curve into one w/ t {d}", .{@intToFloat(f64, segment_i) + first_intersection.t});
                        }
                    } else {
                        printf("Curve (1st) w/ t {d}\n", .{@intToFloat(f64, segment_i) + first_intersection.t});
                    }
                } else if (optional_intersection_points[1]) |second_intersection| {
                    try intersection_list.add(.{
                        .outline_index = @intCast(u32, outline_i),
                        .x_intersect = second_intersection.x,
                        .t = @intToFloat(f64, segment_i) + second_intersection.t,
                    });
                    printf("Curve (2nd) w/ t {d}\n", .{@intToFloat(f64, segment_i) + second_intersection.t});
                } else {
                    printf("REJECT - Non intersecting\n", .{});
                    const i = quadraticBezierPlaneIntersections(bezier, scanline_y);
                    std.debug.assert(i[0] == null);
                    std.debug.assert(i[1] == null);
                }
                continue;
            }

            //
            // Outline segment is a line
            //
            printf("  {d:.2} Vertex (line) {d:^5.2} x {d:^5.2} --> {d:^5.2} x {d:^5.2} -- ", .{ segment_i, point_a.x, point_a.y, point_b.x, point_b.y });
            std.debug.assert(max_y >= min_y);
            if (scanline_y > max_y or scanline_y < min_y) {
                printf("REJECT - Outsize Y range\n", .{});
                continue;
            }
            if (point_a.y == point_b.y) {
                printf("REJECT - horizontal\n", .{});
                continue;
            }

            const interp_t = blk: {
                if (scanline_y == 0) {
                    if (point_a.y == 0.0) break :blk 0.0;
                    if (point_b.y == 0.0) break :blk 1.0;
                    unreachable;
                }
                // TODO: Add comment. Another varient of lerp func
                // a - (b - a) * t = p`
                // p - a = (b - a) * t
                // (p - a) / (b - a) = t
                break :blk (scanline_y - point_a.y) / (point_b.y - point_a.y);
            };
            // std.log.info("\nInterp_t {d} scanline {d} : a ({d}, {d}), b ({d}, {d})", .{
            //     interp_t,
            //     scanline_y,
            //     point_a.x,
            //     point_a.y,
            //     point_b.x,
            //     point_b.x,
            // });
            std.debug.assert(interp_t >= 0.0 and interp_t <= 1.0);
            const t = @intToFloat(f64, segment_i) + interp_t;
            if (point_a.x == point_b.x) {
                try intersection_list.add(.{
                    .outline_index = @intCast(u32, outline_i),
                    .x_intersect = point_a.x,
                    .t = t,
                });
                printf("Vertical line w/ t {d}\n", .{t});
            } else {
                const x_diff = point_b.x - point_a.x;
                const x_intersect = point_a.x + (x_diff * interp_t);
                try intersection_list.add(.{
                    .outline_index = @intCast(u32, outline_i),
                    .x_intersect = x_intersect,
                    .t = t,
                });
                printf("Line w/ t {d}\n", .{t});
            }
        }
    }

    std.log.info("{d} intersections found", .{intersection_list.len});

    // TODO:
    std.debug.assert(intersection_list.len % 2 == 0);

    for (intersection_list.toSlice()) |intersection| {
        std.debug.assert(intersection.outline_index >= 0);
        std.debug.assert(intersection.outline_index < outlines.len);
        const max_t = @intToFloat(f64, outlines[intersection.outline_index].segments.len);
        std.debug.assert(intersection.t >= 0.0);
        std.debug.assert(intersection.t < max_t);
    }

    // Sort by x_intersect ascending
    var step: usize = 1;
    while (step < intersection_list.len) : (step += 1) {
        const key = intersection_list.buffer[step];
        var x = @intCast(i64, step) - 1;
        while (x >= 0 and intersection_list.buffer[@intCast(usize, x)].x_intersect > key.x_intersect) : (x -= 1) {
            intersection_list.buffer[@intCast(usize, x) + 1] = intersection_list.buffer[@intCast(usize, x)];
        }
        intersection_list.buffer[@intCast(usize, x + 1)] = key;
    }

    for (intersection_list.buffer[0..intersection_list.len]) |intersection| {
        std.debug.assert(intersection.outline_index >= 0);
        std.debug.assert(intersection.outline_index < outlines.len);
        const max_t = @intToFloat(f64, outlines[intersection.outline_index].segments.len);
        std.debug.assert(intersection.t >= 0.0);
        std.debug.assert(intersection.t < max_t);
    }

    // TODO:
    if (intersection_list.len == 2) {
        const a = intersection_list.buffer[0];
        const b = intersection_list.buffer[1];
        if (a.t == b.t) {
            std.log.warn("Removing pair with same t", .{});
            intersection_list.len = 0;
        }
    }

    return intersection_list;
}

const OutlineSegment = struct {
    from: Point(f64),
    to: Point(f64),
    control_opt: ?Point(f64) = null,
    // bounding_box: BoundingBox(f64),

    pub inline fn isCurve(self: @This()) bool {
        return self.control_opt != null;
    }

    pub inline fn isLine(self: @This()) bool {
        return self.control_opt == null;
    }

    pub fn sample(self: @This(), t: f64) Point(f64) {
        std.debug.assert(t <= 1.0);
        std.debug.assert(t >= 0.0);
        if (self.control_opt) |control| {
            const bezier = BezierQuadratic{
                .a = self.from,
                .b = self.to,
                .control = control,
            };
            return quadraticBezierPoint(bezier, t);
        }
        return .{
            .x = self.from.x + (self.to.x - self.from.x) * t,
            .y = self.from.y + (self.to.y - self.from.y) * t,
        };
    }

    pub fn sampleAtDistance(self: @This(), ideal: f64, threshold: f64, base_point: SampledPoint) ?SampledPoint {
        if (self.control_opt) |_| {
            const distance_max: f64 = (ideal - threshold);
            const distance_min: f64 = (ideal - threshold);
            var t_increment: f64 = base_point.t_increment;
            var i: usize = 0;
            while (i < 20) : (i += 1) {
                const sampled_point = self.sample(base_point + t_increment);
                const distance = distanceBetweenPoints(sampled_point, base_point.p);
                if ((distance > distance_max) or (distance < distance_min)) {
                    t_increment = std.math.clamp(t_increment / (distance / ideal), 0.0, 1.0);
                    continue;
                }
                return SampledPoint{
                    .p = sampled_point,
                    .t = base_point + t_increment,
                    .t_increment = t_increment,
                };
            }
            return null;
        }
        const line_length = distanceBetweenPoints(self.to, self.from);
        const t_increment: f64 = ideal / line_length;
        const new_t: f64 = base_point.t + t_increment;
        if (new_t > 1.0) return null;
        return SampledPoint{
            .p = self.sample(new_t),
            .t = new_t,
            .t_increment = t_increment,
        };
    }
};

// TODO: Double check formula and write some tests
fn distanceBetweenPoints(point_a: Point(f64), point_b: Point(f64)) f64 {
    const pow = std.math.pow;
    const sqrt = std.math.sqrt;
    return sqrt(pow(point_b.y - point_a.y, 2) + pow(point_a.x - point_b.x, 2));
}

/// Given two points, one that lies inside a normalized boundry and one that lies outside
/// Interpolate a point between them that lies on the boundry of the imaginary 1x1 square
fn interpolateBoundryPoint(inside: Point(f64), outside: Point(f64)) Point(f64) {
    std.debug.assert(inside.x >= 0.0);
    std.debug.assert(inside.x <= 1.0);
    std.debug.assert(inside.y >= 0.0);
    std.debug.assert(inside.y <= 1.0);
    std.debug.assert(outside.x >= 1.0 or outside.x <= 0.0 or outside.y >= 1.0 or outside.y <= 0.0);

    if (outside.x == inside.x) {
        return Point(f64){
            .x = outside.x,
            .y = if (outside.y > inside.y) 1.0 else 0.0,
        };
    }

    if (outside.y == inside.y) {
        return Point(f64){
            .x = if (outside.x > inside.x) 1.0 else 0.0,
            .y = outside.y,
        };
    }

    const x_difference: f64 = outside.x - inside.x;
    const y_difference: f64 = outside.y - inside.y;
    const t: f64 = blk: {
        //
        // Based on lerp function `a - (b - a) * t = p`. Can be rewritten as follows:
        // `(-a + p) / (b - a) = t` where p is our desired value in the spectrum
        // 0.0 or 1.0 in our case, as they represent the left and right (Or top and bottom) sides of the pixel
        // We know whether we want 0.0 or 1.0 based on where the outside point lies in relation to the inside point
        //
        // Taking the x axis for example, if the outside point is to the right of our pixel bounds, we know that
        // we're looking for a p value of 1.0 as the line moves from left to right, otherwise it would be 0.0.
        //
        const side_x: f64 = if (inside.x > outside.x) 0.0 else 1.0;
        const side_y: f64 = if (inside.y > outside.y) 0.0 else 1.0;
        const t_x: f64 = (-inside.x + side_x) / (x_difference);
        const t_y: f64 = (-inside.y + side_y) / (y_difference);
        break :blk if (t_x > 1.0 or t_x < 0.0 or t_y < t_x) t_y else t_x;
    };

    std.debug.assert(t >= 0.0);
    std.debug.assert(t <= 1.0);

    return Point(f64){
        .x = inside.x + (x_difference * t),
        .y = inside.y + (y_difference * t),
    };
}

test "interpolateBoundryPoint" {
    {
        const in = Point(f64){
            .x = 0.5,
            .y = 0.5,
        };
        const out = Point(f64){
            .x = -2.0,
            .y = 0.5,
        };
        const result = interpolateBoundryPoint(in, out);
        try std.testing.expect(result.y == 0.5);
        try std.testing.expect(result.x == 0.0);
    }

    {
        const in = Point(f64){
            .x = 0.5,
            .y = 0.5,
        };
        const out = Point(f64){
            .x = 2.0,
            .y = 0.5,
        };
        const result = interpolateBoundryPoint(in, out);
        try std.testing.expect(result.y == 0.5);
        try std.testing.expect(result.x == 1.0);
    }

    {
        const in = Point(f64){
            .x = 0.5,
            .y = 0.5,
        };
        const out = Point(f64){
            .x = 0.5,
            .y = 2.0,
        };
        const result = interpolateBoundryPoint(in, out);
        try std.testing.expect(result.y == 1.0);
        try std.testing.expect(result.x == 0.5);
    }

    {
        const in = Point(f64){
            .x = 0.25,
            .y = 0.25,
        };
        const out = Point(f64){
            .x = 1.5,
            .y = 2.0,
        };
        const result = interpolateBoundryPoint(in, out);
        try std.testing.expect(result.y == 1.0);
        try std.testing.expect(result.x == 0.7857142857142857);
    }

    {
        const in = Point(f64){
            .x = 0.75,
            .y = 0.25,
        };
        const out = Point(f64){
            .x = -1.5,
            .y = 2.0,
        };
        const result = interpolateBoundryPoint(in, out);
        try std.testing.expect(result.y == 0.8333333333333333);
        try std.testing.expect(result.x == 0.0);
    }

    {
        const in = Point(f64){
            .x = 0.0,
            .y = 0.0,
        };
        const out = Point(f64){
            .x = -1.5,
            .y = -2.0,
        };
        const result = interpolateBoundryPoint(in, out);
        try std.testing.expect(result.y == 0.0);
        try std.testing.expect(result.x == 0.0);
    }

    {
        const in = Point(f64){
            .x = 1.0,
            .y = 1.0,
        };
        const out = Point(f64){
            .x = 2.5,
            .y = 1.0,
        };
        const result = interpolateBoundryPoint(in, out);
        try std.testing.expect(result.y == 1.0);
        try std.testing.expect(result.x == 1.0);
    }
}

fn quadraticBezierPlaneIntersections(bezier: BezierQuadratic, horizontal_axis: f64) [2]?CurveYIntersection {
    const a: f64 = bezier.a.y;
    const b: f64 = bezier.control.y;
    const c: f64 = bezier.b.y;

    //
    // Handle edge-case where control.y is exactly inbetween end points (Leading to NaN)
    // A control point in the middle can be ignored and a normal percent based calculation is used.
    //
    const term_a = a - (2 * b) + c;
    if (term_a == 0.0) {
        const min = @minimum(a, c);
        const max = @maximum(a, c);
        if (horizontal_axis < min or horizontal_axis > max) return .{ null, null };
        const dist = c - a;
        const t = (horizontal_axis - a) / dist;
        std.debug.assert(t >= 0.0 and t <= 1.0);
        return .{
            CurveYIntersection{ .t = t, .x = quadraticBezierPoint(bezier, t).x },
            null,
        };
    }

    const term_b = 2 * (b - a);
    const term_c = a - horizontal_axis;

    const sqrt_calculation = std.math.sqrt((term_b * term_b) - (4.0 * term_a * term_c));

    const first_intersection_t = ((-term_b) + sqrt_calculation) / (2.0 * term_a);
    const second_intersection_t = ((-term_b) - sqrt_calculation) / (2.0 * term_a);

    const is_first_valid = (first_intersection_t <= 1.0 and first_intersection_t >= 0.0);
    const is_second_valid = (second_intersection_t <= 1.0 and second_intersection_t >= 0.0);

    return .{
        if (is_first_valid) CurveYIntersection{ .t = first_intersection_t, .x = quadraticBezierPoint(bezier, first_intersection_t).x } else null,
        if (is_second_valid) CurveYIntersection{ .t = second_intersection_t, .x = quadraticBezierPoint(bezier, second_intersection_t).x } else null,
    };
}

test "quadraticBezierPlaneIntersection" {
    const expect = std.testing.expect;
    {
        const b = BezierQuadratic{
            .a = .{ .x = 16.882635839283466, .y = 0.0 },
            .b = .{
                .x = 23.494,
                .y = 1.208,
            },
            .control = .{ .x = 20.472, .y = 0.0 },
        };
        const s = quadraticBezierPlaneIntersections(b, 0.0);
        try expect(s[0].?.x == 16.882635839283466);
        try expect(s[0].?.t == 0.0);
        try expect(s[1].?.x == 16.882635839283466);
        try expect(s[1].?.t == 0.0);
    }
}

// x = (1 - t) * (1 - t) * p[0].x + 2 * (1 - t) * t * p[1].x + t * t * p[2].x;
// y = (1 - t) * (1 - t) * p[0].y + 2 * (1 - t) * t * p[1].y + t * t * p[2].y;
fn quadraticBezierPoint(bezier: BezierQuadratic, t: f64) Point(f64) {
    std.debug.assert(t >= 0.0);
    std.debug.assert(t <= 1.0);
    const one_minus_t: f64 = 1.0 - t;
    const t_squared: f64 = t * t;
    const p0 = bezier.a;
    const p1 = bezier.b;
    const control = bezier.control;
    return .{
        .x = @floatCast(f64, (one_minus_t * one_minus_t) * p0.x + (2 * one_minus_t * t * control.x + (t_squared * p1.x))),
        .y = @floatCast(f64, (one_minus_t * one_minus_t) * p0.y + (2 * one_minus_t * t * control.y + (t_squared * p1.y))),
    };
}

test "quadraticBezierPoint" {
    const expect = std.testing.expect;
    {
        const b = BezierQuadratic{
            .a = .{ .x = 16.882635839283466, .y = 0.0 },
            .b = .{
                .x = 23.494,
                .y = 1.208,
            },
            .control = .{ .x = 20.472, .y = 0.0 },
        };
        const s = quadraticBezierPoint(b, 0.0);
        try expect(s.x == 16.882635839283466);
        try expect(s.y == 0.0);
    }
}

fn quadrilateralArea(points: [4]Point(f64)) f64 {
    // ((x1y2 + x2y3 + x3y4 + x4y1) - (x2y1 + x3y2 + x4y3 + x1y4)) / 2
    const p1 = points[0];
    const p2 = points[1];
    const p3 = points[2];
    const p4 = points[3];
    return @fabs(((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) - (p2.x * p1.y + p3.x * p2.y + p4.x * p3.y + p1.x * p4.y)) / 2.0);
}

fn triangleArea(p1: Point(f64), p2: Point(f64), p3: Point(f64)) f64 {
    // |x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|
    return @fabs((p1.x * (p2.y - p3.y)) + (p2.x * (p3.y - p1.y)) + (p3.x * (p1.y - p2.y))) / 2.0;
}

inline fn horizontalPlaneIntersection(vertical_axis: f64, a: Point(f64), b: Point(f64)) f64 {
    const m = (a.y - b.y) / (a.x - b.x);
    const s = -1.0 * ((m * a.x) - a.y);
    return (vertical_axis - s) / m;
}

const BezierQuadratic = extern struct {
    a: Point(f64),
    b: Point(f64),
    control: Point(f64),
};

fn quadradicBezierInflectionPoint(bezier: BezierQuadratic) Point(f64) {
    const line_ab_constant = bezier.a.y;
    const line_ab_t = (bezier.control.y - bezier.a.y);
    const line_bc_constant = bezier.control.y;
    const line_bc_t = (bezier.b.y - bezier.control.y);
    const t_total = line_ab_t - line_bc_t;

    const constant_total = line_ab_constant + line_bc_constant;
    const t = constant_total / t_total;

    const ab_lerp_x = bezier.a.x + ((bezier.control.x - bezier.a.x) * t);
    const ab_lerp_y = bezier.a.y + ((bezier.control.y - bezier.a.y) * t);

    const bc_lerp_x = bezier.control.x + ((bezier.b.x - bezier.control.x) * t);
    const bc_lerp_y = bezier.control.y + ((bezier.b.y - bezier.control.y) * t);

    return .{
        .x = ab_lerp_x + ((bc_lerp_x - ab_lerp_x) * t),
        .y = ab_lerp_y + ((bc_lerp_y - ab_lerp_y) * t),
    };
}

const CurveYIntersection = struct {
    x: f64,
    t: f64,
};

pub fn getCodepointBitmap(allocator: Allocator, info: FontInfo, scale: Scale2D(f32), codepoint: i32) !Bitmap {
    const shift = Shift2D(f32){ .x = 0.0, .y = 0.0 };
    const offset = Offset2D(u32){ .x = 0, .y = 0 };
    return try getCodepointBitmapSubpixel(allocator, info, scale, shift, codepoint, offset);
}

const FWORD = i16;
const UFWORD = u16;

const TableHHEA = struct {
    const index = struct {
        const major_version = 0;
        const minor_version = 2;
        const ascender = 4;
        const descender = 6;
        const line_gap = 8;
        const advance_width_max = 10;
        const min_leftside_bearing = 12;
        const min_rightside_bearing = 14;
        const x_max_extent = 16;
        const caret_slope_rise = 18;
        const caret_slope_run = 20;
        const caret_offset = 22;
        const reserved_1 = 24;
        const reserved_2 = 26;
        const reserved_3 = 28;
        const reserved_4 = 30;
        const metric_data_format = 32;
        const number_of_hmetics = 34;
    };
};

pub fn getAscent(info: FontInfo) i16 {
    const offset = info.hhea.offset + TableHHEA.index.ascender;
    return bigToNative(i16, @intToPtr(*i16, @ptrToInt(info.data.ptr) + offset).*);
}

pub fn getDescent(info: FontInfo) i16 {
    const offset = info.hhea.offset + TableHHEA.index.descender;
    return bigToNative(i16, @intToPtr(*i16, @ptrToInt(info.data.ptr) + offset).*);
}

pub const FontInfo = struct {
    // zig fmt: off
    data: []u8,
    glyph_count: i32 = 0,
    loca: SectionRange = .{},
    head: SectionRange = .{},
    glyf: SectionRange = .{},
    hhea: SectionRange = .{},
    hmtx: SectionRange = .{},
    kern: SectionRange = .{},
    gpos: SectionRange = .{},
    svg: SectionRange = .{},
    maxp: SectionRange = .{},
    cff: Buffer = .{},
    index_map: i32 = 0, 
    index_to_loc_format: i32 = 0,
    cmap_encoding_table_offset: u32 = 0,
// zig fmt: on
};

const Buffer = struct {
    data: []u8 = undefined,
    cursor: u32 = 0,
    size: u32 = 0,
};

const FontType = enum { none, truetype_1, truetype_2, opentype_cff, opentype_1, apple };

fn fontType(font: []const u8) FontType {
    const TrueType1Tag: [4]u8 = .{ 49, 0, 0, 0 };
    const OpenTypeTag: [4]u8 = .{ 0, 1, 0, 0 };

    if (eql(u8, font, TrueType1Tag[0..])) return .truetype_1; // TrueType 1
    if (eql(u8, font, "typ1")) return .truetype_2; // TrueType with type 1 font -- we don't support this!
    if (eql(u8, font, "OTTO")) return .opentype_cff; // OpenType with CFF
    if (eql(u8, font, OpenTypeTag[0..])) return .opentype_1; // OpenType 1.0
    if (eql(u8, font, "true")) return .apple; // Apple specification for TrueType fonts

    return .none;
}

pub fn getFontOffsetForIndex(font_collection: []u8, index: i32) i32 {
    const font_type = fontType(font_collection);
    if (font_type == .none) {
        return if (index == 0) 0 else -1;
    }
    return -1;
}

pub fn printFontTags(data: []u8) void {
    const tables_count_addr: *u16 = @intToPtr(*u16, @ptrToInt(data.ptr) + 4);
    const tables_count = toNative(u16, tables_count_addr.*, .Big);
    const table_dir: u32 = 12;

    var i: u32 = 0;
    while (i < tables_count) : (i += 1) {
        const loc: u32 = table_dir + (16 * i);
        const tag: *[4]u8 = @intToPtr(*[4]u8, @ptrToInt(data.ptr) + loc);
        log.info("Tag: '{s}'", .{tag.*[0..]});
    }
}

const TableType = enum { cmap, loca, head, glyf, hhea, hmtx, kern, gpos, maxp };

const TableTypeList: [9]*const [4:0]u8 = .{
    "cmap",
    "loca",
    "head",
    "glyf",
    "hhea",
    "hmtx",
    "kern",
    "GPOS",
    "maxp",
};

pub fn Byte(comptime T: type) type {
    return T;
}

// Describe structure of TTF file
const offset_subtable_start: u32 = 0;
const offset_subtable_length: Byte(u32) = 12;

const table_directory_start: u32 = offset_subtable_length;
const table_directory_length: Byte(u32) = 16;

pub fn Dimensions2D(comptime T: type) type {
    return packed struct {
        width: T,
        height: T,
    };
}

fn BoundingBox(comptime T: type) type {
    return struct {
        x0: T,
        y0: T,
        x1: T,
        y1: T,
    };
}

// TODO: Wrap in a function that lets you select pixel type
const Bitmap = struct {
    width: u32,
    height: u32,
    pixels: []graphics.RGBA(f32),
};

const Vertex = packed struct {
    x: i16,
    y: i16,
    // Refer to control points for bezier curves
    control1_x: i16,
    control1_y: i16,
    // control2_x: i16,
    // control2_y: i16,
    kind: u8,
    is_active: u8 = 0,
};

fn printVertices(vertices: []Vertex, scale: f64) void {
    for (vertices) |vertex, i| {
        assert(vertex.kind <= @enumToInt(VMove.cubic));
        std.debug.print("{d:^2} : {} xy ({d:^5.2}, {d:^5.2}) cxcy ({d:^5.2},{d:^5.2})\n", .{
            i,
            @intToEnum(VMove, vertex.kind),
            @intToFloat(f64, vertex.x) * scale,
            @intToFloat(f64, vertex.y) * scale,
            @intToFloat(f64, vertex.control1_x) * scale,
            @intToFloat(f64, vertex.control1_y) * scale,
        });
    }
}

fn readBigEndian(comptime T: type, index: usize) T {
    return bigToNative(T, @intToPtr(*T, index).*);
}

fn getGlyfOffset(info: FontInfo, glyph_index: i32) !usize {
    assert(info.cff.size == 0);

    if (glyph_index >= info.glyph_count) return error.InvalidGlyphIndex;

    if (info.index_to_loc_format != 0 and info.index_to_loc_format != 1) return error.InvalidIndexToLocationFormat;
    const loca_start = @ptrToInt(info.data.ptr) + info.loca.offset;
    const glyf_offset = @intCast(usize, info.glyf.offset);

    var glyph_data_offset: usize = 0;
    var next_glyph_data_offset: usize = 0;

    // Use 16 or 32 bit offsets based on index_to_loc_format
    // https://docs.microsoft.com/en-us/typography/opentype/spec/head
    if (info.index_to_loc_format == 0) {
        // Location values are stored as half the actual value.
        // https://docs.microsoft.com/en-us/typography/opentype/spec/loca#short-version
        const loca_table_offset: usize = loca_start + (@intCast(usize, glyph_index) * 2);
        glyph_data_offset = glyf_offset + @intCast(u32, readBigEndian(u16, loca_table_offset + 0)) * 2;
        next_glyph_data_offset = glyf_offset + @intCast(u32, readBigEndian(u16, loca_table_offset + 2)) * 2;
    } else {
        glyph_data_offset = glyf_offset + readBigEndian(u32, loca_start + (@intCast(usize, glyph_index) * 4) + 0);
        next_glyph_data_offset = glyf_offset + readBigEndian(u32, loca_start + (@intCast(usize, glyph_index) * 4) + 4);
    }

    if (glyph_data_offset == next_glyph_data_offset) {
        // https://docs.microsoft.com/en-us/typography/opentype/spec/loca
        // If loca[n] == loca[n + 1], that means the glyph has no outline (E.g space character)
        return error.GlyphHasNoOutline;
    }

    return glyph_data_offset;
}

//
// https://docs.microsoft.com/en-us/typography/opentype/spec/glyf
//
const GlyphFlags = struct {
    const none: u8 = 0x00;
    const on_curve_point: u8 = 0x01;
    const x_short_vector: u8 = 0x02;
    const y_short_vector: u8 = 0x04;
    const repeat_flag: u8 = 0x08;
    const positive_x_short_vector: u8 = 0x10;
    const same_x: u8 = 0x10;
    const positive_y_short_vector: u8 = 0x20;
    const same_y: u8 = 0x20;
    const overlap_simple: u8 = 0x40;

    pub fn isFlagSet(value: u8, flag: u8) bool {
        return (value & flag) != 0;
    }
};

fn closeShape(vertices: []Vertex, vertices_count: u32, was_off: bool, start_off: bool, sx: i32, sy: i32, scx: i32, scy: i32, control1_x: i32, control1_y: i32) u32 {
    var vertices_count_local: u32 = vertices_count;

    if (start_off) {
        if (was_off) {
            setVertex(&vertices[vertices_count_local], .curve, @divFloor(control1_x + scx, 2), @divFloor(control1_y + scy, 2), control1_x, control1_y);
            vertices_count_local += 1;
        }
        setVertex(&vertices[vertices_count_local], .curve, sx, sy, scx, scy);
        vertices_count_local += 1;
    } else {
        if (was_off) {
            setVertex(&vertices[vertices_count_local], .curve, sx, sy, control1_x, control1_y);
            vertices_count_local += 1;
        } else {
            setVertex(&vertices[vertices_count_local], .line, sx, sy, 0, 0);
            vertices_count_local += 1;
        }
    }

    return vertices_count_local;
}

fn setVertex(vertex: *Vertex, kind: VMove, x: i32, y: i32, control1_x: i32, control1_y: i32) void {
    vertex.kind = @enumToInt(kind);
    vertex.x = @intCast(i16, x);
    vertex.y = @intCast(i16, y);
    vertex.control1_x = @intCast(i16, control1_x);
    vertex.control1_y = @intCast(i16, control1_y);
}

const VMove = enum(u8) {
    none,
    move = 1,
    line,
    curve,
    cubic,
};

fn isFlagSet(value: u8, bit_mask: u8) bool {
    return (value & bit_mask) != 0;
}

pub fn scaleForPixelHeight(info: FontInfo, height: f32) f32 {
    assert(info.hhea.offset != 0);
    const base_index: usize = @ptrToInt(info.data.ptr) + info.hhea.offset;
    const first = bigToNative(i16, @intToPtr(*i16, (base_index + 4)).*); //ascender
    const second = bigToNative(i16, @intToPtr(*i16, (base_index + 6)).*); // descender
    std.log.info("Ascender: {d}, Descender: {d}", .{ first, second });
    const fheight = @intToFloat(f32, first - second);
    return height / fheight;
}

const GlyhHeader = extern struct {
    // See: https://docs.microsoft.com/en-us/typography/opentype/spec/glyf
    //
    //  If the number of contours is greater than or equal to zero, this is a simple glyph.
    //  If negative, this is a composite glyph — the value -1 should be used for composite glyphs.
    contour_count: i16,
    x_minimum: i16,
    y_minimum: i16,
    x_maximum: i16,
    y_maximum: i16,
};

fn getGlyphShape(allocator: Allocator, info: FontInfo, glyph_index: i32) ![]Vertex {
    if (info.cff.size != 0) {
        return error.CffFound;
    }

    const data = info.data;

    var vertices: []Vertex = undefined;
    var vertices_count: u32 = 0;

    var min_x: i16 = undefined;
    var min_y: i16 = undefined;
    var max_x: i16 = undefined;
    var max_y: i16 = undefined;
    var glyph_dimensions: geometry.Dimensions2D(u32) = undefined;

    // Find the byte offset of the glyh table
    const glyph_offset = try getGlyfOffset(info, glyph_index);
    const glyph_offset_index: usize = @ptrToInt(data.ptr) + glyph_offset;

    if (glyph_offset < 0) {
        return error.InvalidGlypOffset;
    }

    // Beginning of the glyf table
    // See: https://docs.microsoft.com/en-us/typography/opentype/spec/glyf
    const contour_count_signed = readBigEndian(i16, glyph_offset_index);

    min_x = readBigEndian(i16, glyph_offset_index + 2);
    min_y = readBigEndian(i16, glyph_offset_index + 4);
    max_x = readBigEndian(i16, glyph_offset_index + 6);
    max_y = readBigEndian(i16, glyph_offset_index + 8);

    std.log.info("Glyph vertex range: min {d} x {d} max {d} x {d}", .{ min_x, min_y, max_x, max_y });
    std.log.info("Stripped dimensions {d} x {d}", .{ max_x - min_x, max_y - min_y });

    glyph_dimensions.width = @intCast(u32, max_x - min_x + 1);
    glyph_dimensions.height = @intCast(u32, max_y - min_y + 1);

    if (contour_count_signed > 0) {
        const contour_count: u32 = @intCast(u16, contour_count_signed);

        var j: i32 = 0;
        var m: u32 = 0;
        var n: u16 = 0;

        // Index of the next point that begins a new contour
        // This will correspond to value after end_points_of_contours
        var next_move: i32 = 0;

        var off: usize = 0;

        // end_points_of_contours is located directly after GlyphHeader in the glyf table
        const end_points_of_contours = @intToPtr([*]u16, glyph_offset_index + @sizeOf(GlyhHeader));
        const end_points_of_contours_size = @intCast(usize, contour_count * @sizeOf(u16));

        const simple_glyph_table_index = glyph_offset_index + @sizeOf(GlyhHeader);

        // Get the size of the instructions so we can skip past them
        const instructions_size_bytes = readBigEndian(i16, simple_glyph_table_index + end_points_of_contours_size);

        var glyph_flags: [*]u8 = @intToPtr([*]u8, glyph_offset_index + @sizeOf(GlyhHeader) + (@intCast(usize, contour_count) * 2) + 2 + @intCast(usize, instructions_size_bytes));

        // NOTE: The number of flags is determined by the last entry in the endPtsOfContours array
        n = 1 + readBigEndian(u16, @ptrToInt(end_points_of_contours) + (@intCast(usize, contour_count - 1) * 2));

        // What is m here?
        // Size of contours
        {
            // Allocate space for all the flags, and vertices
            m = n + (2 * contour_count);
            vertices = try allocator.alloc(Vertex, @intCast(usize, m) * @sizeOf(Vertex));

            assert((m - n) > 0);
            off = (2 * contour_count);
        }

        var flags: u8 = GlyphFlags.none;
        var flags_len: u32 = 0;
        {
            var i: usize = 0;
            var flag_count: u8 = 0;
            while (i < n) : (i += 1) {
                if (flag_count == 0) {
                    flags = glyph_flags[0];
                    glyph_flags = glyph_flags + 1;
                    if (isFlagSet(flags, GlyphFlags.repeat_flag)) {
                        // If `repeat_flag` is set, the next flag is the number of times to repeat
                        flag_count = glyph_flags[0];
                        glyph_flags = glyph_flags + 1;
                    }
                } else {
                    flag_count -= 1;
                }
                vertices[@intCast(usize, off) + @intCast(usize, i)].kind = flags;
                flags_len += 1;
            }
        }

        {
            var x: i16 = 0;
            var i: usize = 0;
            while (i < n) : (i += 1) {
                flags = vertices[@intCast(usize, off) + @intCast(usize, i)].kind;
                if (isFlagSet(flags, GlyphFlags.x_short_vector)) {
                    const dx: i16 = glyph_flags[0];
                    glyph_flags += 1;
                    x += if (isFlagSet(flags, GlyphFlags.positive_x_short_vector)) dx else -dx;
                } else {
                    if (!isFlagSet(flags, GlyphFlags.same_x)) {

                        // The current x-coordinate is a signed 16-bit delta vector
                        const abs_x = (@intCast(i16, glyph_flags[0]) << 8) + glyph_flags[1];

                        x += abs_x;
                        glyph_flags += 2;
                    }
                }
                // If: `!x_short_vector` and `same_x` then the same `x` value shall be appended
                vertices[off + i].x = x;
            }
        }

        {
            var y: i16 = 0;
            var i: usize = 0;
            while (i < n) : (i += 1) {
                flags = vertices[off + i].kind;
                if (isFlagSet(flags, GlyphFlags.y_short_vector)) {
                    const dy: i16 = glyph_flags[0];
                    glyph_flags += 1;
                    y += if (isFlagSet(flags, GlyphFlags.positive_y_short_vector)) dy else -dy;
                } else {
                    if (!isFlagSet(flags, GlyphFlags.same_y)) {
                        // The current y-coordinate is a signed 16-bit delta vector
                        const abs_y = (@intCast(i16, glyph_flags[0]) << 8) + glyph_flags[1];
                        y += abs_y;
                        glyph_flags += 2;
                    }
                }
                // If: `!y_short_vector` and `same_y` then the same `y` value shall be appended
                vertices[off + i].y = y;
            }
        }
        assert(vertices_count == 0);

        var i: usize = 0;
        var x: i16 = 0;
        var y: i16 = 0;

        var control1_x: i32 = 0;
        var control1_y: i32 = 0;
        var start_x: i32 = 0;
        var start_y: i32 = 0;

        var scx: i32 = 0; // start_control_point_x
        var scy: i32 = 0; // start_control_point_y

        var was_off: bool = false;
        var first_point_off_curve: bool = false;

        next_move = 0;
        while (i < n) : (i += 1) {
            const current_vertex = vertices[off + i];
            if (next_move == i) { // End of contour
                if (i != 0) {
                    vertices_count = closeShape(vertices, vertices_count, was_off, first_point_off_curve, start_x, start_y, scx, scy, control1_x, control1_y);
                }

                first_point_off_curve = ((current_vertex.kind & GlyphFlags.on_curve_point) == 0);
                if (first_point_off_curve) {
                    scx = current_vertex.x;
                    scy = current_vertex.y;
                    if (!isFlagSet(vertices[off + i + 1].kind, GlyphFlags.on_curve_point)) {
                        start_x = x + (vertices[off + i + 1].x >> 1);
                        start_y = y + (vertices[off + i + 1].y >> 1);
                    } else {
                        start_x = current_vertex.x + (vertices[off + i + 1].x);
                        start_y = current_vertex.y + (vertices[off + i + 1].y);
                        i += 1;
                    }
                } else {
                    start_x = current_vertex.x;
                    start_y = current_vertex.y;
                }
                setVertex(&vertices[vertices_count], .move, start_x, start_y, 0, 0);
                vertices_count += 1;
                was_off = false;
                next_move = 1 + readBigEndian(i16, @ptrToInt(end_points_of_contours) + (@intCast(usize, j) * 2));
                j += 1;
            } else {
                // Continue current contour
                if (0 == (current_vertex.kind & GlyphFlags.on_curve_point)) {
                    if (was_off) {
                        // Even though we've encountered 2 control points in a row, this is still a simple
                        // quadradic bezier (I.e 1 control point, 2 real points)
                        // We can calculate the real point that lies between them by taking the average
                        // of the two control points (It's omitted because it's redundant information)
                        // https://stackoverflow.com/questions/20733790/
                        const average_x = @divFloor(control1_x + current_vertex.x, 2);
                        const average_y = @divFloor(control1_y + current_vertex.y, 2);
                        setVertex(&vertices[vertices_count], .curve, average_x, average_y, control1_x, control1_y);
                        vertices_count += 1;
                    }
                    control1_x = current_vertex.x;
                    control1_y = current_vertex.y;
                    was_off = true;
                } else {
                    if (was_off) {
                        setVertex(&vertices[vertices_count], .curve, current_vertex.x, current_vertex.y, control1_x, control1_y);
                    } else {
                        setVertex(&vertices[vertices_count], .line, current_vertex.x, current_vertex.y, 0, 0);
                    }
                    vertices_count += 1;
                    was_off = false;
                }
            }
        }
        vertices_count = closeShape(vertices, vertices_count, was_off, first_point_off_curve, start_x, start_y, scx, scy, control1_x, control1_y);
    } else if (contour_count_signed < 0) {
        // Glyph is composite
        return error.InvalidContourCount;
    } else {
        unreachable;
    }

    // printVertices(vertices[0..vertices_count]);
    return allocator.shrink(vertices, vertices_count);
}

pub fn getRequiredDimensions(info: FontInfo, codepoint: i32, scale: Scale2D(f32)) !Dimensions2D(u32) {
    const glyph_index = @intCast(i32, findGlyphIndex(info, codepoint));
    const shift = Shift2D(f32){ .x = 0.0, .y = 0.0 };
    const bounding_box = try getGlyphBitmapBoxSubpixel(info, glyph_index, scale, shift);
    std.debug.assert(bounding_box.x1 >= bounding_box.x0);
    std.debug.assert(bounding_box.y1 >= bounding_box.y0);
    return Dimensions2D(u32){
        .width = @intCast(u32, bounding_box.x1 - bounding_box.x0),
        .height = @intCast(u32, bounding_box.y1 - bounding_box.y0),
    };
}

pub fn getVerticalOffset(info: FontInfo, codepoint: i32, scale: Scale2D(f32)) !i16 {
    const glyph_index: i32 = @intCast(i32, findGlyphIndex(info, codepoint));
    const shift = Shift2D(f32){ .x = 0.0, .y = 0.0 };
    const bounding_box = try getGlyphBitmapBoxSubpixel(info, glyph_index, scale, shift);
    return @intCast(i16, bounding_box.y1);
}

pub fn getCodepointBitmapBox(info: FontInfo, codepoint: i32, scale: Scale2D(f32)) !BoundingBox(i32) {
    const shift = Shift2D(f32){ .x = 0, .y = 0 };
    return try getCodepointBitmapBoxSubpixel(info, codepoint, scale, shift);
}

fn getCodepointBitmapBoxSubpixel(info: FontInfo, codepoint: i32, scale: Scale2D(f32), shift: Shift2D(f32)) !BoundingBox(i32) {
    const glyph_index = @intCast(i32, findGlyphIndex(info, codepoint));
    return try getGlyphBitmapBoxSubpixel(info, glyph_index, scale, shift);
}

fn getCodepointBitmapSubpixel(allocator: Allocator, info: FontInfo, scale: Scale2D(f32), shift: Shift2D(f32), codepoint: i32, offset: Offset2D(u32)) !Bitmap {
    const glyph_index: i32 = @intCast(i32, findGlyphIndex(info, codepoint));
    return try getGlyphBitmapSubpixel(allocator, info, scale, shift, glyph_index, offset);
}

pub fn getCodepointBoundingBoxScaled(info: FontInfo, codepoint: i32, scale: Scale2D(f32)) !BoundingBox(i32) {
    const glyph_index = @intCast(i32, findGlyphIndex(info, codepoint));
    return try getGlyphBoundingBoxScaled(info, glyph_index, scale);
}

// TODO:
fn getGlyphBoundingBox(info: FontInfo, glyph_index: i32) !BoundingBox(i32) {
    const bounding_box_opt: ?BoundingBox(i32) = getGlyphBox(info, glyph_index);
    if (bounding_box_opt) |bounding_box| {
        return bounding_box;
    }
    return error.GetBitmapBoxFailed;
}

fn getGlyphBoundingBoxScaled(info: FontInfo, glyph_index: i32, scale: Scale2D(f32)) !BoundingBox(i32) {
    const bounding_box_opt: ?BoundingBox(i32) = getGlyphBox(info, glyph_index);
    if (bounding_box_opt) |bounding_box| {
        return BoundingBox(i32){
            .x0 = @floatToInt(i32, @floor(@intToFloat(f64, bounding_box.x0) * scale.x)),
            .y0 = @floatToInt(i32, @floor(@intToFloat(f64, bounding_box.y0) * scale.y)),
            .x1 = @floatToInt(i32, @ceil(@intToFloat(f64, bounding_box.x1) * scale.x)),
            .y1 = @floatToInt(i32, @ceil(@intToFloat(f64, bounding_box.y1) * scale.y)),
        };
    }
    return error.GetBitmapBoxFailed;
}

fn getGlyphBitmapBoxSubpixel(info: FontInfo, glyph_index: i32, scale: Scale2D(f32), shift: Shift2D(f32)) !BoundingBox(i32) {
    const bounding_box_opt: ?BoundingBox(i32) = getGlyphBox(info, glyph_index);
    if (bounding_box_opt) |bounding_box| {
        return BoundingBox(i32){
            .x0 = @floatToInt(i32, @floor(@intToFloat(f32, bounding_box.x0) * scale.x + shift.x)),
            .y0 = @floatToInt(i32, @floor(@intToFloat(f32, -bounding_box.y1) * scale.y + shift.y)),
            .x1 = @floatToInt(i32, @ceil(@intToFloat(f32, bounding_box.x1) * scale.x + shift.x)),
            .y1 = @floatToInt(i32, @ceil(@intToFloat(f32, -bounding_box.y0) * scale.y + shift.y)),
        };
    }
    return error.GetBitmapBoxFailed;
}

fn getGlyphBox(info: FontInfo, glyph_index: i32) ?BoundingBox(i32) {
    assert(info.cff.size == 0);

    const g: usize = getGlyfOffset(info, glyph_index) catch |err| {
        log.warn("Error in getGlyfOffset {}", .{err});
        return null;
    };

    if (g == 0) {
        log.warn("Failed to get glyf offset", .{});
        return null;
    }

    const base_index: usize = @ptrToInt(info.data.ptr) + g;
    return BoundingBox(i32){
        .x0 = bigToNative(i16, @intToPtr(*i16, base_index + 2).*), // min_x
        .y0 = bigToNative(i16, @intToPtr(*i16, base_index + 4).*), // min_y
        .x1 = bigToNative(i16, @intToPtr(*i16, base_index + 6).*), // max_x
        .y1 = bigToNative(i16, @intToPtr(*i16, base_index + 8).*), // max_y
    };
}

fn Offset2D(comptime T: type) type {
    return extern struct {
        x: T,
        y: T,
    };
}

fn getGlyphBitmapSubpixel(allocator: Allocator, info: FontInfo, desired_scale: Scale2D(f32), shift: Shift2D(f32), glyph_index: i32, offset: Offset2D(u32)) !Bitmap {
    _ = shift;
    _ = offset;

    var scale = desired_scale;

    var bitmap: Bitmap = undefined;
    const vertices = try getGlyphShape(allocator, info, glyph_index);
    // TODO: Allocated inside of getGlyphShape
    defer allocator.free(vertices);

    if (scale.x == 0) {
        scale.x = scale.y;
    }

    if (scale.y == 0) {
        if (scale.x == 0) {
            return error.WhoKnows;
        }
        scale.y = scale.x;
    }

    std.debug.assert(scale.x == scale.y);

    std.log.info("Scale: {d}", .{scale.x});

    const bounding_box = try getGlyphBoundingBoxScaled(info, glyph_index, scale);

    const unscaled_bb = try getGlyphBoundingBox(info, glyph_index);
    std.log.info("Generated BB {d} {d} {d} {d}", .{ unscaled_bb.x0, unscaled_bb.y0, unscaled_bb.x1, unscaled_bb.y1 });

    // printVertices(vertices);

    // Scale, shift and invert y for vertices
    std.debug.assert(unscaled_bb.y1 >= unscaled_bb.y0);
    // const unscaled_y_range: i16 = @intCast(i16, unscaled_bb.y1 - unscaled_bb.y0);
    for (vertices) |*vertex| {
        vertex.x -= @intCast(i16, unscaled_bb.x0);
        vertex.y -= @intCast(i16, unscaled_bb.y0);
        // vertex.y = unscaled_y_range - vertex.y;
        if (@intToEnum(VMove, vertex.kind) == .curve) {
            vertex.control1_x -= @intCast(i16, unscaled_bb.x0);
            vertex.control1_y -= @intCast(i16, unscaled_bb.y0);
            // vertex.control1_y = unscaled_y_range - vertex.control1_y;
        }
    }
    const dimensions = Dimensions2D(u32){
        .width = @intCast(u32, bounding_box.x1 - bounding_box.x0),
        .height = @intCast(u32, bounding_box.y1 - bounding_box.y0),
    };

    bitmap.width = @intCast(u32, bounding_box.x1 - bounding_box.x0);
    bitmap.height = @intCast(u32, bounding_box.y1 - bounding_box.y0);

    if (bitmap.width != 0 and bitmap.height != 0) {
        bitmap = try rasterize(allocator, dimensions, vertices, scale.x);
    }

    return bitmap;
}

fn Point(comptime T: type) type {
    return extern struct {
        x: T,
        y: T,
    };
}

fn printPoints(points: []Point(f64)) void {
    for (points) |point, i| {
        print("  {d:2} xy ({d}, {d})\n", .{ i, point.x, point.y });
    }
    print("Done\n", .{});
}

fn calculateLineSegmentSlope(point_a: Point(f64), point_b: Point(f64)) f64 {
    return (point_a.y - point_b.y) / (point_a.x - point_b.x);
}

fn pixelFill(coverage: f64) graphics.RGBA(f32) {
    return .{
        .r = 1.0,
        .g = 1.0,
        .b = 1.0,
        .a = @floatCast(f32, coverage),
    };
}

fn pixelCurve(coverage: f64) graphics.RGBA(f32) {
    return .{
        .r = 1.0,
        .g = 0.0,
        .b = 0.0,
        .a = @floatCast(f32, coverage),
    };
}

fn pixelCurve2(coverage: f64) graphics.RGBA(f32) {
    return .{
        .r = 1.0,
        .g = 1.0,
        .b = 1.0,
        .a = @floatCast(f32, coverage),
    };
}

const Side = enum {
    left,
    right,
    top,
    bottom,
};

fn sideOfPixel(point: Point(f64)) Side {
    if (point.x == 0.0) return .left;
    if (point.x == 1.0) return .right;
    if (point.y == 0.0) return .bottom;
    if (point.y == 1.0) return .top;

    std.log.info("Point: {d}, {d}", .{ point.x, point.y });
    std.debug.assert(false);
    return .top;
}

fn clampTo(value: f64, edge: f64, threshold: f64) f64 {
    if (@fabs(value - edge) <= threshold) {
        return edge;
    }
    return value;
}

fn printOutlines(outlines: []Outline) void {
    for (outlines) |outline, outline_i| {
        const printf = std.debug.print;
        printf("Outline #{d}\n", .{outline_i + 1});
        for (outline.segments) |outline_segment, outline_segment_i| {
            const to = outline_segment.to;
            const from = outline_segment.from;
            printf("  {d:2} {d:.5}, {d:.5} -> {d:.3}, {d:.3}", .{ outline_segment_i, from.x, from.y, to.x, to.y });
            if (outline_segment.control_opt) |control_point| {
                printf(" w/ control {d:.3}, {d:.3}", .{ control_point.x, control_point.y });
            }
            printf("\n", .{});
        }
    }
}

const OffsetSubtable = struct {
    scaler_type: u32,
    tables_count: u16,
    search_range: u16,
    entry_selector: u16,
    range_shift: u16,

    pub fn fromBigEndianBytes(bytes: *align(4) [@sizeOf(OffsetSubtable)]u8) @This() {
        var result = @ptrCast(*OffsetSubtable, bytes).*;

        result.scaler_type = toNative(u32, result.scaler_type, .Big);
        result.tables_count = toNative(u16, result.tables_count, .Big);
        result.search_range = toNative(u16, result.search_range, .Big);
        result.entry_selector = toNative(u16, result.entry_selector, .Big);
        result.range_shift = toNative(u16, result.range_shift, .Big);

        return result;
    }
};

// NOTE: This should be packed
const TableDirectory = struct {
    tag: [4]u8,
    checksum: u32,
    offset: u32,
    length: u32,

    pub fn isChecksumValid(self: @This()) bool {
        assert(@sizeOf(@This()) == 16);

        var sum: u32 = 0;
        var iteractions_count: u32 = @sizeOf(@This()) / 4;

        var bytes = @ptrCast(*const u32, &self);
        while (iteractions_count > 0) : (iteractions_count -= 1) {
            _ = @addWithOverflow(u32, sum, bytes.*, &sum);
            bytes = @intToPtr(*const u32, @ptrToInt(bytes) + @sizeOf(u32));
        }

        const checksum = self.checksum;

        return (sum == checksum);
    }

    pub fn fromBigEndianBytes(bytes: *align(4) [@sizeOf(TableDirectory)]u8) ?TableDirectory {
        var result: TableDirectory = @ptrCast(*align(4) TableDirectory, bytes).*;

        // Disabled as not working
        // if (!result.isChecksumValid()) {
        // return null;
        // }

        result.length = toNative(u32, result.length, .Big);
        result.offset = toNative(u32, result.offset, .Big);

        return result;
    }
};

const Head = struct {
    version_major: i16,
    version_minor: i16,
    font_revision_major: i16,
    font_revision_minor: i16,
    checksum_adjustment: u32,
    magic_number: u32, // 0x5F0F3CF5
    flags: Flags,
    units_per_em: u16,
    created_timestamp: i64,
    modified_timestamp: i64,
    x_min: i16,
    y_min: i16,
    x_max: i16,
    y_max: i16,
    mac_style: MacStyle,
    lowest_rec_ppem: u16,
    font_direction_hint: i16,
    index_to_loc_format: i16,
    glyph_data_format: i16,

    const Flags = packed struct(u16) {
        y0_specifies_baseline: bool,
        left_blackbit_is_lsb: bool,
        scaled_point_size_differs: bool,
        use_integer_scaling: bool,
        reserved_microsoft: bool,
        layout_vertically: bool,
        reserved_0: bool,
        requires_layout_for_ling_rendering: bool,
        aat_font_with_metamorphosis_effects: bool,
        strong_right_to_left: bool,
        indic_style_effects: bool,
        reserved_adobe_0: bool,
        reserved_adobe_1: bool,
        reserved_adobe_2: bool,
        reserved_adobe_3: bool,
        simple_generic_symbols: bool,
    };

    const MacStyle = packed struct(u16) {
        bold: bool,
        italic: bool,
        underline: bool,
        outline: bool,
        shadow: bool,
        extended: bool,
        unused_bit_6: bool,
        unused_bit_7: bool,
        unused_bit_8: bool,
        unused_bit_9: bool,
        unused_bit_10: bool,
        unused_bit_11: bool,
        unused_bit_12: bool,
        unused_bit_13: bool,
        unused_bit_14: bool,
        unused_bit_15: bool,
    };
};

const cff_magic_number: u32 = 0x5F0F3CF5;

const PlatformID = enum(u8) { unicode = 0, max = 1, iso = 2, microsoft = 3 };

const CmapIndex = struct {
    version: u16,
    subtables_count: u16,
};

const CMAPPlatformID = enum(u16) {
    unicode = 0,
    macintosh = 1,
    reserved = 2,
    microsoft = 3,
};

const CMAPPlatformSpecificID = packed union {
    const Unicode = enum(u16) {
        version1_0,
        version1_1,
        iso_10646,
        unicode2_0_bmp_only,
        unicode2_0,
        unicode_variation_sequences,
        last_resort,
        other, // This value is allowed but shall be ignored
    };

    const Macintosh = enum(u16) {
        roman,
        japanese,
        traditional_chinese,
        // etc: https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6name.html
    };

    const Microsoft = enum(u16) {
        symbol,
        unicode_bmp_only,
        shift_jis,
        prc,
        big_five,
        johab,
        unicode_ucs_4,
    };

    unicode: Unicode,
    microsoft: Microsoft,
    macintosh: Macintosh,
};

const CMAPSubtable = extern struct {
    pub fn fromBigEndianBytes(bytes: []u8) ?CMAPSubtable {
        var table: CMAPSubtable = undefined;

        const platform_id_u16 = toNative(u16, @ptrCast(*u16, @alignCast(2, bytes.ptr)).*, .Big);

        if (platform_id_u16 > @enumToInt(CMAPPlatformID.microsoft)) {
            log.warn("Invalid platform ID '{d}' parsed from CMAP subtable", .{platform_id_u16});
            return null;
        }

        table.platform_id = @intToEnum(CMAPPlatformID, platform_id_u16);

        table.offset = toNative(u32, @ptrCast(*u32, @alignCast(4, &bytes.ptr[4])).*, .Big);

        const platform_specific_id_u16 = toNative(u16, @ptrCast(*u16, @alignCast(2, &bytes.ptr[2])).*, .Big);

        switch (table.platform_id) {
            .unicode => {
                if (platform_specific_id_u16 < @enumToInt(CMAPPlatformSpecificID.Unicode.last_resort)) {
                    table.platform_specific_id = .{ .unicode = @intToEnum(CMAPPlatformSpecificID.Unicode, platform_specific_id_u16) };
                } else {
                    table.platform_specific_id = .{ .unicode = .other };
                }
                log.info("Platform specific ID for '{}' => '{}'", .{ table.platform_id, table.platform_specific_id.unicode });
            },
            .microsoft => {
                unreachable;
            },
            .macintosh => {
                unreachable;
            },
            .reserved => {
                unreachable;
            },
        }

        return table;
    }

    platform_id: CMAPPlatformID,
    platform_specific_id: CMAPPlatformSpecificID,
    offset: u32,
};

const CMAPFormat2 = struct {
    format: u16,
    length: u16,
    language: u16,
};

const SectionRange = struct {
    offset: u32 = 0,
    length: u32 = 0,

    pub fn isNull(self: @This()) bool {
        return self.offset == 0;
    }
};

const DataSections = struct {
    dsig: SectionRange = .{},
    loca: SectionRange = .{},
    head: SectionRange = .{},
    glyf: SectionRange = .{},
    hhea: SectionRange = .{},
    hmtx: SectionRange = .{},
    hvar: SectionRange = .{},
    kern: SectionRange = .{},
    gpos: SectionRange = .{},
    svg: SectionRange = .{},
    maxp: SectionRange = .{},
    cmap: SectionRange = .{},
    name: SectionRange = .{},
};

pub fn initializeFont(allocator: Allocator, font_data: []u8) !FontInfo {
    _ = allocator;

    var data_sections = DataSections{};

    {
        var fixed_buffer_stream = std.io.FixedBufferStream([]const u8){ .buffer = font_data, .pos = 0 };
        var reader = fixed_buffer_stream.reader();

        const scaler_type = try reader.readIntBig(u32);
        const tables_count = try reader.readIntBig(u16);
        const search_range = try reader.readIntBig(u16);
        const entry_selector = try reader.readIntBig(u16);
        const range_shift = try reader.readIntBig(u16);

        _ = scaler_type;
        _ = search_range;
        _ = entry_selector;
        _ = range_shift;

        var i: usize = 0;
        while (i < tables_count) : (i += 1) {
            var tag_buffer: [4]u8 = undefined;
            var tag = tag_buffer[0..];
            _ = try reader.readAll(tag[0..]);
            const checksum = try reader.readIntBig(u32);
            // TODO: Use checksum
            _ = checksum;
            const offset = try reader.readIntBig(u32);
            const length = try reader.readIntBig(u32);

            std.debug.print("{d:2}.    {s}\n", .{ i + 1, tag });

            if (std.mem.eql(u8, "cmap", tag)) {
                data_sections.cmap.offset = offset;
                data_sections.cmap.length = length;
                continue;
            }

            if (std.mem.eql(u8, "DSIG", tag)) {
                data_sections.dsig.offset = offset;
                data_sections.dsig.length = length;
                continue;
            }

            if (std.mem.eql(u8, "loca", tag)) {
                data_sections.loca.offset = offset;
                data_sections.loca.length = length;
                continue;
            }

            if (std.mem.eql(u8, "head", tag)) {
                data_sections.head.offset = offset;
                data_sections.head.length = length;
                continue;
            }

            if (std.mem.eql(u8, "hvar", tag)) {
                data_sections.hvar.offset = offset;
                data_sections.hvar.length = length;
                continue;
            }

            if (std.mem.eql(u8, "glyf", tag)) {
                data_sections.glyf.offset = offset;
                data_sections.glyf.length = length;
                continue;
            }

            if (std.mem.eql(u8, "hhea", tag)) {
                data_sections.hhea.offset = offset;
                data_sections.hhea.length = length;
                continue;
            }

            if (std.mem.eql(u8, "hmtx", tag)) {
                data_sections.hmtx.offset = offset;
                data_sections.hmtx.length = length;
                continue;
            }

            if (std.mem.eql(u8, "kern", tag)) {
                data_sections.kern.offset = offset;
                data_sections.kern.length = length;
                continue;
            }

            if (std.mem.eql(u8, "GPOS", tag)) {
                data_sections.gpos.offset = offset;
                data_sections.gpos.length = length;
                continue;
            }

            if (std.mem.eql(u8, "maxp", tag)) {
                data_sections.maxp.offset = offset;
                data_sections.maxp.length = length;
                continue;
            }

            if (std.mem.eql(u8, "name", tag)) {
                data_sections.name.offset = offset;
                data_sections.name.length = length;
                continue;
            }
        }
    }

    var font_info = FontInfo{
        .data = font_data,
        .hhea = data_sections.hhea,
        .loca = data_sections.loca,
        .glyf = data_sections.glyf,
    };

    {
        std.debug.assert(!data_sections.maxp.isNull());

        var fixed_buffer_stream = std.io.FixedBufferStream([]const u8){ .buffer = font_data, .pos = data_sections.maxp.offset };
        var reader = fixed_buffer_stream.reader();
        const version_major = try reader.readIntBig(i16);
        const version_minor = try reader.readIntBig(i16);
        _ = version_major;
        _ = version_minor;
        font_info.glyph_count = try reader.readIntBig(u16);
        std.log.info("Glyphs found: {d}", .{font_info.glyph_count});
    }

    var head: Head = undefined;
    {
        var fixed_buffer_stream = std.io.FixedBufferStream([]const u8){ .buffer = font_data, .pos = data_sections.head.offset };
        var reader = fixed_buffer_stream.reader();

        head.version_major = try reader.readIntBig(i16);
        head.version_minor = try reader.readIntBig(i16);
        head.font_revision_major = try reader.readIntBig(i16);
        head.font_revision_minor = try reader.readIntBig(i16);
        head.checksum_adjustment = try reader.readIntBig(u32);
        head.magic_number = try reader.readIntBig(u32);

        if (head.magic_number != 0x5F0F3CF5) {
            std.log.warn("Magic number not set to 0x5F0F3CF5. File might be corrupt", .{});
        }

        head.flags = try reader.readStruct(Head.Flags);

        head.units_per_em = try reader.readIntBig(u16);
        head.created_timestamp = try reader.readIntBig(i64);
        head.modified_timestamp = try reader.readIntBig(i64);

        head.x_min = try reader.readIntBig(i16);
        head.y_min = try reader.readIntBig(i16);
        head.x_max = try reader.readIntBig(i16);
        head.y_max = try reader.readIntBig(i16);

        std.debug.assert(head.x_min <= head.x_max);
        std.debug.assert(head.y_min <= head.y_max);

        head.mac_style = try reader.readStruct(Head.MacStyle);

        head.lowest_rec_ppem = try reader.readIntBig(u16);

        head.font_direction_hint = try reader.readIntBig(i16);
        head.index_to_loc_format = try reader.readIntBig(i16);
        head.glyph_data_format = try reader.readIntBig(i16);

        font_info.index_to_loc_format = head.index_to_loc_format;

        std.debug.assert(font_info.index_to_loc_format == 0 or font_info.index_to_loc_format == 1);
    }

    font_info.cmap_encoding_table_offset = outer: {
        std.debug.assert(!data_sections.cmap.isNull());

        var fixed_buffer_stream = std.io.FixedBufferStream([]const u8){ .buffer = font_data, .pos = data_sections.cmap.offset };
        var reader = fixed_buffer_stream.reader();

        const version = try reader.readIntBig(u16);
        const subtable_count = try reader.readIntBig(u16);

        _ = version;

        var i: usize = 0;
        while (i < subtable_count) : (i += 1) {
            comptime {
                std.debug.assert(@sizeOf(CMAPPlatformID) == 2);
                std.debug.assert(@sizeOf(CMAPPlatformSpecificID) == 2);
            }
            const platform_id = try reader.readEnum(CMAPPlatformID, .Big);
            const platform_specific_id = blk: {
                switch (platform_id) {
                    .unicode => break :blk CMAPPlatformSpecificID{ .unicode = try reader.readEnum(CMAPPlatformSpecificID.Unicode, .Big) },
                    else => return error.InvalidSpecificPlatformID,
                }
            };
            _ = platform_specific_id;
            const offset = try reader.readIntBig(u32);
            std.log.info("Platform: {}", .{platform_id});
            if (platform_id == .unicode) break :outer data_sections.cmap.offset + offset;
        }
        return error.InvalidPlatform;
    };

    return font_info;
}

fn findGlyphIndex(font_info: FontInfo, unicode_codepoint: i32) u32 {
    const data = font_info.data;
    const encoding_offset = font_info.cmap_encoding_table_offset;

    if (unicode_codepoint > 0xffff) {
        log.info("Invalid codepoint", .{});
        return 0;
    }

    const base_index: usize = @ptrToInt(data.ptr) + encoding_offset;
    const format: u16 = bigToNative(u16, @intToPtr(*u16, base_index).*);

    // TODO:
    assert(format == 4);

    const segcount = toNative(u16, @intToPtr(*u16, base_index + 6).*, .Big) >> 1;
    var search_range = toNative(u16, @intToPtr(*u16, base_index + 8).*, .Big) >> 1;
    var entry_selector = toNative(u16, @intToPtr(*u16, base_index + 10).*, .Big);
    const range_shift = toNative(u16, @intToPtr(*u16, base_index + 12).*, .Big) >> 1;

    const end_count: u32 = encoding_offset + 14;
    var search: u32 = end_count;

    if (unicode_codepoint >= toNative(u16, @intToPtr(*u16, @ptrToInt(data.ptr) + search + (range_shift * 2)).*, .Big)) {
        search += range_shift * 2;
    }

    search -= 2;

    while (entry_selector != 0) {
        var end: u16 = undefined;
        search_range = search_range >> 1;

        end = toNative(u16, @intToPtr(*u16, @ptrToInt(data.ptr) + search + (search_range * 2)).*, .Big);

        if (unicode_codepoint > end) {
            search += search_range * 2;
        }
        entry_selector -= 1;
    }

    search += 2;

    {
        var offset: u16 = undefined;
        var start: u16 = undefined;
        const item: u32 = (search - end_count) >> 1;

        assert(unicode_codepoint <= toNative(u16, @intToPtr(*u16, @ptrToInt(data.ptr) + end_count + (item * 2)).*, .Big));
        start = toNative(u16, @intToPtr(*u16, @ptrToInt(data.ptr) + encoding_offset + 14 + (segcount * 2) + 2 + (2 * item)).*, .Big);

        if (unicode_codepoint < start) {
            // TODO: return error
            return 0;
        }

        offset = toNative(u16, @intToPtr(*u16, @ptrToInt(data.ptr) + encoding_offset + 14 + (segcount * 6) + 2 + (item * 2)).*, .Big);
        if (offset == 0) {
            const base = bigToNative(i16, @intToPtr(*i16, base_index + 14 + (segcount * 4) + 2 + (2 * item)).*);
            return @intCast(u32, unicode_codepoint + base);
        }

        const result_addr_index = @ptrToInt(data.ptr) + offset + @intCast(usize, unicode_codepoint - start) * 2 + encoding_offset + 14 + (segcount * 6) + 2 + (2 * item);

        const result_addr = @intToPtr(*u8, result_addr_index);
        const result_addr_aligned = @ptrCast(*u16, @alignCast(2, result_addr));

        return @intCast(u32, toNative(u16, result_addr_aligned.*, .Big));
    }
}

//
// Construction
//

fn getGlyphShape2(allocator: Allocator, info: FontInfo, glyph_index: i32) ![]Outline {
    if (info.cff.size != 0) {
        return error.CffFound;
    }

    // TODO:
    const data = info.data;

    var vertices: []Vertex = undefined;
    var vertices_count: u32 = 0;

    // Find the byte offset of the glyh table
    const glyph_offset = try getGlyfOffset(info, glyph_index);
    const glyph_offset_index: usize = @ptrToInt(data.ptr) + glyph_offset;

    if (glyph_offset < 0) {
        return error.InvalidGlypOffset;
    }

    // Beginning of the glyf table
    // See: https://docs.microsoft.com/en-us/typography/opentype/spec/glyf
    const contour_count_signed = readBigEndian(i16, glyph_offset_index);

    const min_x = readBigEndian(i16, glyph_offset_index + 2);
    const min_y = readBigEndian(i16, glyph_offset_index + 4);
    const max_x = readBigEndian(i16, glyph_offset_index + 6);
    const max_y = readBigEndian(i16, glyph_offset_index + 8);

    std.log.info("Glyph vertex range: min {d} x {d} max {d} x {d}", .{ min_x, min_y, max_x, max_y });
    std.log.info("Stripped dimensions {d} x {d}", .{ max_x - min_x, max_y - min_y });

    if (contour_count_signed > 0) {
        const contour_count: u32 = @intCast(u16, contour_count_signed);

        var j: i32 = 0;
        var m: u32 = 0;
        var n: u16 = 0;

        // Index of the next point that begins a new contour
        // This will correspond to value after end_points_of_contours
        var next_move: i32 = 0;

        var off: usize = 0;

        // end_points_of_contours is located directly after GlyphHeader in the glyf table
        const end_points_of_contours = @intToPtr([*]u16, glyph_offset_index + @sizeOf(GlyhHeader));
        const end_points_of_contours_size = @intCast(usize, contour_count * @sizeOf(u16));

        const simple_glyph_table_index = glyph_offset_index + @sizeOf(GlyhHeader);

        // Get the size of the instructions so we can skip past them
        const instructions_size_bytes = readBigEndian(i16, simple_glyph_table_index + end_points_of_contours_size);

        var glyph_flags: [*]u8 = @intToPtr([*]u8, glyph_offset_index + @sizeOf(GlyhHeader) + (@intCast(usize, contour_count) * 2) + 2 + @intCast(usize, instructions_size_bytes));

        // NOTE: The number of flags is determined by the last entry in the endPtsOfContours array
        n = 1 + readBigEndian(u16, @ptrToInt(end_points_of_contours) + (@intCast(usize, contour_count - 1) * 2));

        const x_coords = glyph_flags[n .. n * 2];
        const y_coords = glyph_flags[n * 2 .. n * 3];

        // What is m here?
        // Size of contours
        {
            // Allocate space for all the flags, and vertices
            m = n + (2 * contour_count);
            vertices = try allocator.alloc(Vertex, @intCast(usize, m));

            assert((m - n) > 0);
            off = (2 * contour_count);
        }

        {
            //
            // Construction
            //
            var i: usize = 0;
            var flag_repeat_count: i32 = 0;
            var x_index: u32 = 0;
            var y_index: u32 = 0;
            var flag_index: u32 = 0;
            // On curve points
            var x: i16 = 0;
            var y: i16 = 0;
            // Off curve (bezier control) points
            // var control_x: i16 = 0;
            // var control_y: i16 = 0;

            // var next_outline_end_index = end_points_of_contours[0];

            while (i < n) : (i += 1) {
                const flags = glyph_flags[flag_index];
                if (flag_repeat_count <= 0) {
                    if (isFlagSet(flags, GlyphFlags.repeat_flag)) {
                        flag_repeat_count = glyph_flags[i + 1];
                        i += 1;
                    }
                    flag_index += 1;
                }
                flag_repeat_count -= 1;

                //
                // Load x coordinates
                //
                if (isFlagSet(flags, GlyphFlags.x_short_vector)) {
                    const dx: i16 = x_coords[x_index];
                    x += if (isFlagSet(flags, GlyphFlags.positive_x_short_vector)) dx else -dx;
                    x_index += 1;
                } else {
                    if (!isFlagSet(flags, GlyphFlags.same_x)) {
                        // The current x-coordinate is a signed 16-bit delta vector
                        x += (@intCast(i16, x_coords[x_index]) << 8) + x_coords[x_index + 1];
                        x_index += 2;
                    } else {
                        // If `!x_short_vector` and `same_x` then the same `x` value shall be appended
                        // https://learn.microsoft.com/en-us/typography/opentype/spec/glyf
                        // See: X_IS_SAME_OR_POSITIVE_X_SHORT_VECTOR
                    }
                }

                //
                // Load y coordinates
                //
                if (isFlagSet(flags, GlyphFlags.y_short_vector)) {
                    const dy: i16 = y_coords[y_index];
                    y_index += 1;
                    y += if (isFlagSet(flags, GlyphFlags.positive_y_short_vector)) dy else -dy;
                } else {
                    if (!isFlagSet(flags, GlyphFlags.same_y)) {
                        // The current y-coordinate is a signed 16-bit delta vector
                        y += (@intCast(i16, y_coords[y_index]) << 8) + y_coords[y_index + 1];
                        y_index += 2;
                    } else {
                        // If `!y_short_vector` and `same_y` then the same `y` value shall be appended
                        // https://learn.microsoft.com/en-us/typography/opentype/spec/glyf
                        // See: Y_IS_SAME_OR_POSITIVE_Y_SHORT_VECTOR
                    }
                }
            }
        }

        var flags: u8 = GlyphFlags.none;
        {
            var i: usize = 0;
            var flag_count: u8 = 0;
            while (i < n) : (i += 1) {
                if (flag_count == 0) {
                    flags = glyph_flags[0];
                    glyph_flags = glyph_flags + 1;
                    if (isFlagSet(flags, GlyphFlags.repeat_flag)) {
                        // If `repeat_flag` is set, the next flag is the number of times to repeat
                        flag_count = glyph_flags[0];
                        glyph_flags = glyph_flags + 1;
                    }
                } else {
                    flag_count -= 1;
                }
                vertices[@intCast(usize, off) + @intCast(usize, i)].kind = flags;
            }
        }

        {
            var x: i16 = 0;
            var i: usize = 0;
            while (i < n) : (i += 1) {
                flags = vertices[@intCast(usize, off) + @intCast(usize, i)].kind;
                if (isFlagSet(flags, GlyphFlags.x_short_vector)) {
                    const dx: i16 = glyph_flags[0];
                    glyph_flags += 1;
                    x += if (isFlagSet(flags, GlyphFlags.positive_x_short_vector)) dx else -dx;
                } else {
                    if (!isFlagSet(flags, GlyphFlags.same_x)) {

                        // The current x-coordinate is a signed 16-bit delta vector
                        const abs_x = (@intCast(i16, glyph_flags[0]) << 8) + glyph_flags[1];

                        x += abs_x;
                        glyph_flags += 2;
                    }
                }
                // If: `!x_short_vector` and `same_x` then the same `x` value shall be appended
                vertices[off + i].x = x;
            }
        }

        {
            var y: i16 = 0;
            var i: usize = 0;
            while (i < n) : (i += 1) {
                flags = vertices[off + i].kind;
                if (isFlagSet(flags, GlyphFlags.y_short_vector)) {
                    const dy: i16 = glyph_flags[0];
                    glyph_flags += 1;
                    y += if (isFlagSet(flags, GlyphFlags.positive_y_short_vector)) dy else -dy;
                } else {
                    if (!isFlagSet(flags, GlyphFlags.same_y)) {
                        // The current y-coordinate is a signed 16-bit delta vector
                        const abs_y = (@intCast(i16, glyph_flags[0]) << 8) + glyph_flags[1];
                        y += abs_y;
                        glyph_flags += 2;
                    }
                }
                // If: `!y_short_vector` and `same_y` then the same `y` value shall be appended
                vertices[off + i].y = y;
            }
        }
        assert(vertices_count == 0);

        var i: usize = 0;
        var x: i16 = 0;
        var y: i16 = 0;

        var control1_x: i32 = 0;
        var control1_y: i32 = 0;
        var start_x: i32 = 0;
        var start_y: i32 = 0;

        var scx: i32 = 0; // start_control_point_x
        var scy: i32 = 0; // start_control_point_y

        var was_off: bool = false;
        var first_point_off_curve: bool = false;

        next_move = 0;
        while (i < n) : (i += 1) {
            const current_vertex = vertices[off + i];
            if (next_move == i) { // End of contour
                if (i != 0) {
                    vertices_count = closeShape(vertices, vertices_count, was_off, first_point_off_curve, start_x, start_y, scx, scy, control1_x, control1_y);
                }

                first_point_off_curve = ((current_vertex.kind & GlyphFlags.on_curve_point) == 0);
                if (first_point_off_curve) {
                    scx = current_vertex.x;
                    scy = current_vertex.y;
                    if (!isFlagSet(vertices[off + i + 1].kind, GlyphFlags.on_curve_point)) {
                        start_x = x + (vertices[off + i + 1].x >> 1);
                        start_y = y + (vertices[off + i + 1].y >> 1);
                    } else {
                        start_x = current_vertex.x + (vertices[off + i + 1].x);
                        start_y = current_vertex.y + (vertices[off + i + 1].y);
                        i += 1;
                    }
                } else {
                    start_x = current_vertex.x;
                    start_y = current_vertex.y;
                }
                setVertex(&vertices[vertices_count], .move, start_x, start_y, 0, 0);
                vertices_count += 1;
                was_off = false;
                next_move = 1 + readBigEndian(i16, @ptrToInt(end_points_of_contours) + (@intCast(usize, j) * 2));
                j += 1;
            } else {
                // Continue current contour
                if (0 == (current_vertex.kind & GlyphFlags.on_curve_point)) {
                    if (was_off) {
                        // Even though we've encountered 2 control points in a row, this is still a simple
                        // quadradic bezier (I.e 1 control point, 2 real points)
                        // We can calculate the real point that lies between them by taking the average
                        // of the two control points (It's omitted because it's redundant information)
                        // https://stackoverflow.com/questions/20733790/
                        const average_x = @divFloor(control1_x + current_vertex.x, 2);
                        const average_y = @divFloor(control1_y + current_vertex.y, 2);
                        setVertex(&vertices[vertices_count], .curve, average_x, average_y, control1_x, control1_y);
                        vertices_count += 1;
                    }
                    control1_x = current_vertex.x;
                    control1_y = current_vertex.y;
                    was_off = true;
                } else {
                    if (was_off) {
                        setVertex(&vertices[vertices_count], .curve, current_vertex.x, current_vertex.y, control1_x, control1_y);
                    } else {
                        setVertex(&vertices[vertices_count], .line, current_vertex.x, current_vertex.y, 0, 0);
                    }
                    vertices_count += 1;
                    was_off = false;
                }
            }
        }
        vertices_count = closeShape(vertices, vertices_count, was_off, first_point_off_curve, start_x, start_y, scx, scy, control1_x, control1_y);
    } else if (contour_count_signed < 0) {
        // Glyph is composite
        return error.InvalidContourCount;
    } else {
        unreachable;
    }

    return allocator.shrink(vertices, vertices_count);
}
