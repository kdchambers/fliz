// SPDX-License-Identifier: GPL-3.0
// Copyright (c) 2021 Keith Chambers
// This program is free software: you can redistribute it and/or modify it under the terms
// of the GNU General Public License as published by the Free Software Foundation, version 3.

const std = @import("std");
const log = std.log;
const fs = std.fs;
const Allocator = std.mem.Allocator;
const toNative = std.mem.toNative;
const bigToNative = std.mem.bigToNative;
const eql = std.mem.eql;
const assert = std.debug.assert;
const print = std.debug.print;

const geometry = @import("geometry.zig");
const Scale2D = geometry.Scale2D;
const Shift2D = geometry.Shift2D;

// TODO:
pub fn abs(value: f64) f64 {
    if (value < 0) {
        return value * -1;
    }
    return value;
}

pub fn getCodepointBitmap(allocator: Allocator, info: FontInfo, scale: Scale2D(f32), codepoint: i32) !Bitmap {
    const shift = Shift2D(f32){ .x = 0.0, .y = 0.0 };
    const offset = Offset2D(u32){ .x = 0, .y = 0 };
    return try getCodepointBitmapSubpixel(allocator, info, scale, shift, codepoint, offset);
}

pub fn getAscent(info: FontInfo) i16 {
    return bigToNative(i16, @intToPtr(*i16, @ptrToInt(info.data.ptr) + info.hhea.offset + 4).*);
}

pub fn getDescent(info: FontInfo) i16 {
    return bigToNative(i16, @intToPtr(*i16, @ptrToInt(info.data.ptr) + info.hhea.offset + 6).*);
}

const TableLookup = struct {
    offset: u32 = 0,
    length: u32 = 0,
};

pub const FontInfo = struct {
    // zig fmt: off
    userdata: *void,
    data: []u8,
    glyph_count: i32 = 0,
    loca: TableLookup,
    head: TableLookup,
    glyf: TableLookup,
    hhea: TableLookup,
    hmtx: TableLookup,
    kern: TableLookup,
    gpos: TableLookup,
    svg: TableLookup,
    maxp: TableLookup,
    index_map: i32 = 0, 
    index_to_loc_format: i32 = 0,
    cff: Buffer,
    char_strings: Buffer,
    gsubrs: Buffer,
    subrs: Buffer,
    font_dicts: Buffer,
    fd_select: Buffer,
    cmap_encoding_table_offset: u32 = 0,
// zig fmt: on
};

const Buffer = struct {
    data: []u8,
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

const BoundingBox = packed struct {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
};

// TODO: Wrap in a function that lets you select pixel type
const Bitmap = struct {
    width: u32,
    height: u32,
    pixels: []u8,
};

const coordinate_types: []u8 = undefined;
const coordinates = []Point(i16);
const control_points = []Point(i16);

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
        print("{d:^2} : {} xy ({d:^5.2}, {d:^5.2}) cxcy ({d:^5.2},{d:^5.2})\n", .{
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
    var g1: usize = 0;
    var g2: usize = 0;

    assert(info.cff.size == 0);

    if (glyph_index >= info.glyph_count) return error.InvalidGlyphIndex;

    if (info.index_to_loc_format >= 2) return error.InvalidIndexToLocationFormat;

    const base_index = @ptrToInt(info.data.ptr) + info.loca.offset;

    if (info.index_to_loc_format == 0) {
        assert(false);
        g1 = @intCast(usize, info.glyf.offset) + readBigEndian(u16, base_index + (@intCast(usize, glyph_index) * 2) + 0) * 2;
        g2 = @intCast(usize, info.glyf.offset) + readBigEndian(u16, base_index + (@intCast(usize, glyph_index) * 2) + 2) * 2;
    } else {
        g1 = @intCast(usize, info.glyf.offset) + readBigEndian(u32, base_index + (@intCast(usize, glyph_index) * 4) + 0);
        g2 = @intCast(usize, info.glyf.offset) + readBigEndian(u32, base_index + (@intCast(usize, glyph_index) * 4) + 4);
    }

    if (g1 == g2) {
        return error.GlyphIndicesMatch;
    }

    return g1;
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

const GlyhHeader = packed struct {
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

// See: https://docs.microsoft.com/en-us/typography/opentype/spec/glyf
//      Simple Glyph Table
//

// const SimpleGlyphTable = packed struct {
// // Array of point indices for the last point of each contour, in increasing numeric order.
// end_points_of_contours: [contour_count]u16,
// instruction_length: u16,
// instructions: [instruction_length]u8,
// flags: [*]u8,
// // Contour point x-coordinates.
// // Coordinate for the first point is relative to (0,0); others are relative to previous point.
// x_coordinates: [*]u8,
// // Contour point y-coordinates.
// // Coordinate for the first point is relative to (0,0); others are relative to previous point.
// y_coordinates: [*]u8,
// };

var min_x: i16 = undefined;
var min_y: i16 = undefined;
var max_x: i16 = undefined;
var max_y: i16 = undefined;
var glyph_dimensions: geometry.Dimensions2D(u32) = undefined;

fn getGlyphShape(allocator: Allocator, info: FontInfo, glyph_index: i32) ![]Vertex {
    if (info.cff.size != 0) {
        return error.CffFound;
    }

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

        // {
        //     var r: u32 = 0;
        //     while (r < contour_count) : (r += 1) {
        //         print("END PT: {d}\n", .{bigToNative(u16, end_points_of_contours[r])});
        //     }
        // }

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

        // {
        //     const start = @intCast(usize, off);
        //     for(vertices[start..start + flags_len]) |vertex| {
        //         const printf = std.debug.print;
        //         printf("on_curve: {d}\n", .{vertex.kind & GlyphFlags.on_curve_point});
        //     }
        // }

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

pub fn getCodepointBitmapBox(info: FontInfo, codepoint: i32, scale: Scale2D(f32)) !BoundingBox {
    const shift = Shift2D(f32){ .x = 0, .y = 0 };
    return try getCodepointBitmapBoxSubpixel(info, codepoint, scale, shift);
}

fn getCodepointBitmapBoxSubpixel(info: FontInfo, codepoint: i32, scale: Scale2D(f32), shift: Shift2D(f32)) !BoundingBox {
    const glyph_index = @intCast(i32, findGlyphIndex(info, codepoint));
    return try getGlyphBitmapBoxSubpixel(info, glyph_index, scale, shift);
}

fn getCodepointBitmapSubpixel(allocator: Allocator, info: FontInfo, scale: Scale2D(f32), shift: Shift2D(f32), codepoint: i32, offset: Offset2D(u32)) !Bitmap {
    const glyph_index: i32 = @intCast(i32, findGlyphIndex(info, codepoint));
    return try getGlyphBitmapSubpixel(allocator, info, scale, shift, glyph_index, offset);
}

pub fn getCodepointBoundingBoxScaled(info: FontInfo, codepoint: i32, scale: Scale2D(f32)) !BoundingBox {
    const glyph_index = @intCast(i32, findGlyphIndex(info, codepoint));
    return try getGlyphBoundingBoxScaled(info, glyph_index, scale);
}

fn getGlyphBoundingBox(info: FontInfo, glyph_index: i32) !BoundingBox {
    const bounding_box_opt: ?BoundingBox = getGlyphBox(info, glyph_index);
    if (bounding_box_opt) |bounding_box| {
        return bounding_box;
    }
    return error.GetBitmapBoxFailed;
}

fn getGlyphBoundingBoxScaled(info: FontInfo, glyph_index: i32, scale: Scale2D(f32)) !BoundingBox {
    const bounding_box_opt: ?BoundingBox = getGlyphBox(info, glyph_index);
    if (bounding_box_opt) |bounding_box| {
        return BoundingBox{
            .x0 = @floatToInt(i32, @floor(@intToFloat(f64, bounding_box.x0) * scale.x)),
            .y0 = @floatToInt(i32, @floor(@intToFloat(f64, bounding_box.y0) * scale.y)),
            .x1 = @floatToInt(i32, @ceil(@intToFloat(f64, bounding_box.x1) * scale.x)),
            .y1 = @floatToInt(i32, @ceil(@intToFloat(f64, bounding_box.y1) * scale.y)),
        };
    }
    return error.GetBitmapBoxFailed;
}

fn getGlyphBitmapBoxSubpixel(info: FontInfo, glyph_index: i32, scale: Scale2D(f32), shift: Shift2D(f32)) !BoundingBox {
    const bounding_box_opt: ?BoundingBox = getGlyphBox(info, glyph_index);
    if (bounding_box_opt) |bounding_box| {
        return BoundingBox{
            .x0 = @floatToInt(i32, @floor(@intToFloat(f32, bounding_box.x0) * scale.x + shift.x)),
            .y0 = @floatToInt(i32, @floor(@intToFloat(f32, -bounding_box.y1) * scale.y + shift.y)),
            .x1 = @floatToInt(i32, @ceil(@intToFloat(f32, bounding_box.x1) * scale.x + shift.x)),
            .y1 = @floatToInt(i32, @ceil(@intToFloat(f32, -bounding_box.y0) * scale.y + shift.y)),
        };
    }
    return error.GetBitmapBoxFailed;
}

fn getGlyphBox(info: FontInfo, glyph_index: i32) ?BoundingBox {
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
    return BoundingBox{
        .x0 = bigToNative(i16, @intToPtr(*i16, base_index + 2).*), // min_x
        .y0 = bigToNative(i16, @intToPtr(*i16, base_index + 4).*), // min_y
        .x1 = bigToNative(i16, @intToPtr(*i16, base_index + 6).*), // max_x
        .y1 = bigToNative(i16, @intToPtr(*i16, base_index + 8).*), // min_y
    };
}

fn Offset2D(comptime T: type) type {
    return packed struct {
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

    // printVertices(vertices);

    // std.log.info("Generated BB {d} {d} {d} {d}", .{bounding_box.x0, bounding_box.y0, bounding_box.x1, bounding_box.y1});

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
    return packed struct {
        x: T,
        y: T,

        // pub fn print(self: @This()) void {
        //     std.debug.print("{d}, {d}\n", .{self.x, self.y});
        // }
    };
}

fn printPoints(points: []Point(f32)) void {
    for (points) |point, i| {
        print("  {d:2} xy ({d}, {d})\n", .{ i, point.x, point.y });
    }
    print("Done\n", .{});
}

fn pointValue(a: Point(i64), b: Point(i64), p: Point(i64)) i64 {
    return ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x));
}

fn isAbove(a: Point(i64), b: Point(i64), p: Point(i64)) bool {
    return ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) >= 0;
}

inline fn max(comptime Type: type, a: Type, b: Type) Type {
    return if (a > b) a else b;
}

inline fn min(comptime Type: type, a: Type, b: Type) Type {
    return if (a > b) b else a;
}

inline fn horizontalPlaneIntersection(vertical_axis: f64, a: Point(f64), b: Point(f64)) f64 {
    const m = (a.y - b.y) / (a.x - b.x);
    const s = -1.0 * ((m * a.x) - a.y);
    return (vertical_axis - s) / m;
}

const BezierQuadratic = packed struct {
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

fn quadraticBezierPlaneIntersections3(bezier: BezierQuadratic, horizontal_axis: f64) [2]?CurveYIntersection {
    const a = bezier.a.y;
    const b = bezier.control.y;
    const c = bezier.b.y;

    const term_a = a - (2 * b) + c;
    const term_b = 2 * (b - a);
    const term_c = a - horizontal_axis;

    const sqrt_calculation = std.math.sqrt((term_b * term_b) - (4 * term_a * term_c));

    const first_intersection_t = ((-term_b) + sqrt_calculation) / (2 * term_a);
    const second_intersection_t = ((-term_b) - sqrt_calculation) / (2 * term_a);

    std.log.info("T values {d} {d}", .{ first_intersection_t, second_intersection_t });

    const is_first_valid = (first_intersection_t <= 1.0 and first_intersection_t >= 0.0);
    const is_second_valid = (second_intersection_t <= 1.0 and second_intersection_t >= 0.0);

    return .{
        if (is_first_valid) CurveYIntersection{ .t = first_intersection_t, .x = quadraticBezierPoint(bezier, first_intersection_t).x } else null,
        if (is_second_valid) CurveYIntersection{ .t = second_intersection_t, .x = quadraticBezierPoint(bezier, second_intersection_t).x } else null,
    };
}

fn quadraticBezierPlaneIntersection2(horizontal_axis: f64, bezier: BezierQuadratic) f64 {
    // (ABy + BCy) / 2 = horizontal_axis
    const ab_y_constant = bezier.a.y;
    const ab_y_t = (bezier.control.y - bezier.a.y);

    const bc_y_constant = bezier.control.y;
    const bc_y_t = (bezier.b.y - bezier.control.y);

    const t_total = ab_y_t - bc_y_t;
    const constant_total = ab_y_constant + bc_y_constant;

    const t = ((horizontal_axis * 2) - constant_total) / t_total;

    const ab_lerp_x = bezier.a.x + ((bezier.control.x - bezier.a.x) * t);
    const bc_lerp_x = bezier.control.x + ((bezier.b.x - bezier.control.x) * t);
    return ab_lerp_x + ((bc_lerp_x - ab_lerp_x) * t);
}

// t = -(sqrt(-2 b y_1 + y_0 (b - y_2) + b y_2 + y_1^2) - y_0 + y_1) / (y_0 - 2 y_1 + y_2) and y_0 + y_2!=2 y_1
fn quadraticBezierPlaneIntersection(vertical_axis: f64, bezier: BezierQuadratic) f64 {

    // ISSUE: This apparently works.. but, this equation needs to return up to two points of intersection

    const y0 = bezier.a.y;
    const y1 = bezier.control.y;
    const y2 = bezier.b.y;
    const b = vertical_axis;

    const bottom = y0 - (2 * y1) + y2;
    const top = -(std.math.sqrt((-2.0 * b * y1) + y0 * (b - y2) + (b * y2) + (y1 * y1)) - y0 + y1);
    const t = top / bottom;

    // const control_squared = bezier.control.y * bezier.control.y;
    // const a = bezier.a.y - (2.0 * bezier.control.y) + bezier.b.y;
    // const top = (bezier.a.y - bezier.control.y) - std.math.sqrt((vertical_axis * a) + control_squared - (bezier.a.y * bezier.b.y));
    // const t = top / a;

    std.log.info("T({d})", .{t});

    // std.log.info("T({d}) - {d}", .{t, quadraticBezierPoint(t, bezier).x});

    std.debug.assert(t <= 1.0);
    std.debug.assert(t >= 0.0);
    // std.debug.assert(quadraticBezierPoint(t, bezier).y == vertical_axis);

    return quadraticBezierPoint(t, bezier).x;
}

// C(t) = (1 - t)^2 * p_0 + 2t(1 - t) * p_1 + t ^ 2 * p_2

fn quadraticBezierPoint2(bezier: BezierQuadratic, t: f64) Point(f64) {
    const ab_lerp_x = bezier.a.x + ((bezier.control.x - bezier.a.x) * t);
    const ab_lerp_y = bezier.a.y + ((bezier.control.y - bezier.a.y) * t);
    const bc_lerp_x = bezier.control.x + ((bezier.b.x - bezier.control.x) * t);
    const bc_lerp_y = bezier.control.y + ((bezier.b.y - bezier.control.y) * t);
    return .{
        .x = ab_lerp_x + ((bc_lerp_x - ab_lerp_x) * t),
        .y = ab_lerp_y + ((bc_lerp_y - ab_lerp_y) * t),
    };
}

// x = (1 - t) * (1 - t) * p[0].x + 2 * (1 - t) * t * p[1].x + t * t * p[2].x;
// y = (1 - t) * (1 - t) * p[0].y + 2 * (1 - t) * t * p[1].y + t * t * p[2].y;
fn quadraticBezierPoint(bezier: BezierQuadratic, t: f64) Point(f64) {
    std.debug.assert(t >= 0.0);
    std.debug.assert(t <= 1.0);
    const one_minus_t: f128 = 1.0 - t;
    const t_squared: f128 = t * t;
    const p0 = bezier.a;
    const p1 = bezier.b;
    const control = bezier.control;
    return .{
        .x = @floatCast(f64, (one_minus_t * one_minus_t) * p0.x + (2 * one_minus_t * t * control.x + (t_squared * p1.x))),
        .y = @floatCast(f64, (one_minus_t * one_minus_t) * p0.y + (2 * one_minus_t * t * control.y + (t_squared * p1.y))),
    };
}

fn quadrilateralArea(points: [4]Point(f64)) f64 {
    // ((x1y2 + x2y3 + x3y4 + x4y1) - (x2y1 + x3y2 + x4y3 + x1y4)) / 2
    const p1 = points[0];
    const p2 = points[1];
    const p3 = points[2];
    const p4 = points[3];
    return abs(((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) - (p2.x * p1.y + p3.x * p2.y + p4.x * p3.y + p1.x * p4.y)) / 2.0);
}

const Direction = enum {
    horizonal,
    vertical,
    positive,
    negative,
};

const Intersection = struct {
    x_position: f64,
    outline_index: u32,
    t: ?f64 = null,

    pub fn isCurve(self: @This()) bool {
        return (self.t != null);
    }

    pub fn isLine(self: @This()) bool {
        return !self.isCurve();
    }
};

const IntersectionList = struct {
    buffer: []Intersection,
    count: u32 = 0,

    pub fn reset(self: *@This()) void {
        self.count = 0;
    }

    pub fn add(self: *@This(), intersection: Intersection, extra: []const u8) !void {
        std.debug.assert(intersection.x_position >= 0);
        if (self.count == (self.buffer.len - 1)) {
            return error.BufferFull;
        }
        self.buffer[self.count].x_position = intersection.x_position;
        self.buffer[self.count].outline_index = intersection.outline_index;
        self.buffer[self.count].t = intersection.t;

        if(self.buffer[self.count].t) |t| {
            std.debug.print("Intersection V {d} w/ t {d} ({s}) added at X {d:.4}\n", .{ self.buffer[self.count].outline_index, t, extra, self.buffer[self.count].x_position });
        } else {
            std.debug.print("Intersection V {d} ({s}) added at X {d:.4}\n", .{ self.buffer[self.count].outline_index, extra, self.buffer[self.count].x_position });
        }
        
        self.count += 1;
    }

    pub fn bufferSlice(self: @This()) []Intersection {
        return self.buffer[0..self.count];
    }
};

const Coverage = struct {
    entry: Point(f64), // This is always the y intersection
    exit: Point(f64), // This can be all other sides
    // This is an intermediate point if line connects within pixel bounds
    aux_opt: ?Point(f64), // A point with the pixel bounds

    pub fn calculateCoverage(self: @This()) f64 {
        std.debug.print("Entry {d}, {d}\n", .{ self.entry.x, self.entry.y });
        std.debug.print("Exit {d}, {d}\n", .{ self.exit.x, self.exit.y });
        std.debug.assert(self.entry.x >= 0.0);
        std.debug.assert(self.entry.x <= 1.0);
        std.debug.assert(self.exit.x >= 0.0);
        std.debug.assert(self.exit.x <= 1.0);
        if (self.aux_opt) |aux| {
            std.debug.print("Aux {d}, {d}\n", .{ aux.x, aux.y });
            std.debug.assert(self.entry.x >= 0.0);
            std.debug.assert(self.entry.x <= 1.0);
            const points = [4]Point(f64){
                self.entry,
                self.exit,
                aux,
                Point(f64){
                    .x = 1.0,
                    // If the slope is positive (Trending upwards) we put the additional point
                    // in the bottom right corner. If negative, in the top right
                    .y = if (self.exit.x > self.entry.x) 0.0 else 1.0,
                },
            };
            const result = quadrilateralArea(points);
            std.debug.assert(result >= 0.0 and result <= 1.0);
            return result;
        }

        const entry = self.entry;
        const exit = self.exit;

        if (entry.y == 0.0 and exit.x == 1.0) {
            std.log.info("Coverage A", .{});
            return 0.5 * abs(self.entry.x - self.exit.x) * self.exit.y;
        }
        if (entry.y == 0.0 and exit.y == 1.0) {
            std.log.info("Coverage B", .{});
            return (exit.x + entry.x) / 2.0;
        }
        if (entry.x == 0.0 and exit.y == 1.0) {
            std.log.info("Coverage C", .{});
            return 1.0 - (0.5 * abs(self.entry.x - self.exit.x) * self.exit.y);
        }
        // rectangle
        if (entry.x == 0.0 and exit.x == 1.0) {
            std.log.info("Coverage D", .{});
            // Get the average y and return that as a percentage
            // Area under the y is covered. Unit square
            return (exit.y + entry.y) / 2.0;
        }

        // Typical negative line
        if (entry.y == 0 and exit.x == 0.0) {
            std.log.info("Coverage E", .{});
            return 0.5 * abs(self.entry.x - self.exit.x) * self.exit.y;
        }

        if (entry.x == 1.0 and exit.y == 1.0) {
            std.log.info("Coverage F", .{});
            return 0.5 * abs(self.entry.x - self.exit.x) * self.exit.y;
        }

        std.log.warn("Unhandled coverage", .{});
        std.debug.assert(false);
        return 0.0;
    }
};

const LineSegment = struct {
    left: Point(f64),
    right: Point(f64),

    pub fn direction(self: @This()) Direction {
        const left = self.left;
        const right = self.right;
        if (left.y < right.y) {
            return .positive;
        }
        return .negative;
    }

    pub fn isHorizontal(self: @This()) bool {
        return (self.left.y == self.right.y);
    }

    pub fn isVertical(self: @This()) bool {
        return (self.left.x == self.right.x);
    }
};

fn calculateCoverage(entry: Point(f64), exit: Point(f64), is_positive: bool) f64 {
    const coverage = blk: {
        // const entry = entry_intersection;
        // const exit = exit_intersection;

        std.debug.assert(entry.x >= 0.0);
        std.debug.assert(entry.x <= 1.0);
        std.debug.assert(entry.y >= 0.0);
        std.debug.assert(entry.y <= 1.0);

        std.debug.assert(exit.x >= 0.0);
        std.debug.assert(exit.x <= 1.0);
        std.debug.assert(exit.y >= 0.0);
        std.debug.assert(exit.y <= 1.0);

        std.debug.print("  ENTRY ({d:.2}, {d:.2}) EXIT ({d:.2}, {d:.2})\n", .{
            entry.x,
            entry.y,
            exit.x,
            exit.y,
        });

        // Bottom -> Right
        if (entry.y == 0.0 and exit.x == 1.0) {
            std.debug.assert(is_positive);
            const value = 0.5 * abs(entry.x - exit.x) * exit.y;
            std.log.info("Coverage A {d:.2}", .{value});
            break :blk value;
        }

        // Bottom -> Top
        if (entry.y == 0.0 and exit.y == 1.0) {
            const base = ((exit.x + entry.x) / 2.0);
            const value = if (is_positive) 1.0 - base else 1.0 - base;
            std.log.info("Coverage B {d:.2}", .{value});
            break :blk value;
        }

        // Left -> Top
        if (entry.x == 0.0 and exit.y == 1.0) {
            std.debug.assert(is_positive);
            const value = 1.0 - (0.5 * abs(entry.x - exit.x) * exit.y);
            std.log.info("Coverage C {d:.2}", .{value});
            break :blk value;
        }

        // Left -> Right
        if (entry.x == 0.0 and exit.x == 1.0) {
            // std.debug.assert(false);
            const value = (exit.y + entry.y) / 2.0;
            std.log.info("Coverage D {d:.2}", .{value});
            break :blk value;
        }

        // Bottom -> Left
        if (entry.y == 0 and exit.x == 0.0) {
            std.debug.assert(!is_positive);
            const value = 1.0 - (0.5 * abs(entry.x - exit.x) * exit.y);
            std.log.info("Coverage E {d:.2}", .{value});
            break :blk value;
        }

        // Right -> Top
        if (entry.x == 1.0 and exit.y == 1.0) {
            std.debug.assert(!is_positive);
            const value = (0.5 * abs(entry.x - exit.x) * exit.y);
            std.log.info("Coverage F {d:.2}", .{value});
            break :blk value;
        }

        // Right -> Left
        if (entry.x == 1.0 and exit.x == 0.0) {
            const value = (exit.y + entry.y) / 2.0;
            std.log.info("Coverage G {d:.2}", .{value});
            break :blk value;
        }

        std.debug.assert(false);
        break :blk 0.0;
    };
    return coverage;
}

fn rasterize(allocator: Allocator, dimensions: Dimensions2D(u32), vertices: []Vertex, scale: f32) !Bitmap {
    std.log.info("Glyph bbox {d} x {d} -- {d} x {d}", .{ min_x, min_y, max_x, max_y });
    const bitmap_pixel_count = @intCast(usize, dimensions.width) * dimensions.height;
    var bitmap = Bitmap{
        .width = @intCast(u32, dimensions.width),
        .height = @intCast(u32, dimensions.height),
        .pixels = try allocator.alloc(u8, bitmap_pixel_count),
    };
    std.mem.set(u8, bitmap.pixels, 0);

    std.log.info("Rasterizing image of dimensions: {d} x {d}", .{ bitmap.width, bitmap.height });
    const printf = std.debug.print;

    var scanline_buffer_a: [64]Intersection = undefined;
    var scanline_buffer_b: [64]Intersection = undefined;

    printVertices(vertices, scale);

    for (vertices) |*vertex| {
        vertex.is_active = 0;
    }

    var current_scanline = IntersectionList{ .buffer = scanline_buffer_a[0..], .count = 0 };
    var previous_scanline = IntersectionList{ .buffer = scanline_buffer_b[0..], .count = 0 };

    var scanline_y_index: i32 = @intCast(i32, dimensions.height) - 1;
    while (scanline_y_index >= 0) : (scanline_y_index -= 1) {
        current_scanline.reset();
        const scanline_y = @intToFloat(f64, scanline_y_index);
        const image_y_index = (dimensions.height - 1) - @intCast(u32, scanline_y_index);

        printf("Scanline {d}, image Y {d}\n", .{ scanline_y_index, image_y_index });
        // Loop through all vertices and store line intersections
        var vertex_i: u32 = 1;
        while (vertex_i < vertices.len) : (vertex_i += 1) {
            const previous_vertex = vertices[vertex_i - 1];
            const current_vertex = vertices[vertex_i];
            const kind = @intToEnum(VMove, current_vertex.kind);

            if (kind == .move) {
                continue;
            }

            const point_a = Point(f64){
                .x = @intToFloat(f64, current_vertex.x) * scale,
                .y = @intToFloat(f64, current_vertex.y) * scale,
            };
            const point_b = Point(f64){
                .x = @intToFloat(f64, previous_vertex.x) * scale,
                .y = @intToFloat(f64, previous_vertex.y) * scale,
            };

            if (kind == .line) {
                printf("  Vertex (line) {d:^5.2} x {d:^5.2} --> {d:^5.2} x {d:^5.2} -- ", .{ point_a.x, point_a.y, point_b.x, point_b.y });
            } else if (kind == .curve) {
                printf("  Vertex (curve) A({d:.2}, {d:.2}) --> B({d:.2}, {d:.2}) C ({d:.2}, {d:.2}) -- ", .{
                    point_b.x,
                    point_b.y,
                    point_a.x,
                    point_a.y,
                    @intToFloat(f64, current_vertex.control1_x) * scale,
                    @intToFloat(f64, current_vertex.control1_y) * scale,
                });
            }

            const is_horizontal = current_vertex.y == previous_vertex.y;
            if (is_horizontal) {
                printf("REJECT - Horizonatal line\n", .{});
                continue;
            }

            const is_outsize_y_range = blk: {
                const max_yy = max(f64, point_a.y, point_b.y);
                const min_yy = min(f64, point_a.y, point_b.y);
                std.debug.assert(max_yy >= min_yy);
                if (scanline_y > max_yy or scanline_y < min_yy) {
                    break :blk true;
                }
                if (kind == .curve) {
                    const bezier = BezierQuadratic{ .a = point_a, .b = point_b, .control = .{
                        .x = @intToFloat(f64, current_vertex.control1_x) * scale,
                        .y = @intToFloat(f64, current_vertex.control1_y) * scale,
                    } };
                    const inflection_y = quadradicBezierInflectionPoint(bezier).y;
                    const is_middle_higher = (inflection_y > max_yy) and scanline_y > inflection_y;
                    const is_middle_lower = (inflection_y < min_yy) and scanline_y < inflection_y;
                    if (is_middle_higher or is_middle_lower) {
                        break :blk true;
                    }
                }
                break :blk false;
            };

            if (is_outsize_y_range) {
                printf("REJECT - Outsize Y range\n", .{});
                continue;
            }

            switch (kind) {
                .line => {
                    if (point_a.x == point_b.x) {
                        try current_scanline.add(.{ .x_position = point_a.x, .outline_index = vertex_i }, "vertical");
                        continue;
                    }
                    try current_scanline.add(.{ .x_position = horizontalPlaneIntersection(scanline_y, point_a, point_b), .outline_index = vertex_i }, "line");
                },
                .curve => {
                    const bezier = BezierQuadratic{ .a = point_b, .b = point_a, .control = .{
                        .x = @intToFloat(f64, current_vertex.control1_x) * scale,
                        .y = @intToFloat(f64, current_vertex.control1_y) * scale,
                    } };
                    const optional_intersection_points = quadraticBezierPlaneIntersections3(bezier, scanline_y);
                    if (optional_intersection_points[0]) |first_intersection| {
                        try current_scanline.add(.{ .x_position = first_intersection.x, .t = first_intersection.t, .outline_index = vertex_i }, "curve 1");
                        {
                            // Sanity test
                            // const intersection_point = quadraticBezierPoint(bezier, first_intersection.t);
                            // std.log.info("Curve intersection at t {d}. Expected {d}, {d}. Got {d}, {d}", .{
                            //     first_intersection.t,
                            //     first_intersection.x,
                            //     scanline_y,
                            //     intersection_point.x,
                            //     intersection_point.y,
                            // });
                            // std.debug.assert(@floatToInt(u32, @floor(intersection_point.y)) == scanline_y_index);
                            // std.debug.assert(intersection_point.x == first_intersection.x);
                        }
                        if (optional_intersection_points[1]) |second_intersection| {
                            try current_scanline.add(.{ .x_position = second_intersection.x, .t = second_intersection.t, .outline_index = vertex_i }, "curve 2");
                            {
                                // Sanity test
                                // const intersection_point = quadraticBezierPoint(bezier, second_intersection.t);
                                // std.log.info("Curve intersection at t {d}. Expected {d}, {d}. Got {d}, {d}", .{
                                //     second_intersection.t,
                                //     second_intersection.x,
                                //     scanline_y,
                                //     intersection_point.x,
                                //     intersection_point.y,
                                // });
                                // std.debug.assert(@floatToInt(u32, @floor(intersection_point.y)) == scanline_y_index);
                                // std.debug.assert(intersection_point.x == second_intersection.x);
                            }
                        }
                    } else if (optional_intersection_points[1]) |second_intersection| {
                        try current_scanline.add(.{
                            .x_position = second_intersection.x,
                            .t = second_intersection.t,
                            .outline_index = vertex_i,
                        }, "curve 2 only");
                        {
                            // Sanity test
                            // const intersection_point = quadraticBezierPoint(bezier, second_intersection.t);
                            // std.log.info("Curve intersection at t {d}. Expected {d}, {d}. Got {d}, {d}", .{
                            //     second_intersection.t,
                            //     second_intersection.x,
                            //     scanline_y,
                            //     intersection_point.x,
                            //     intersection_point.y,
                            // });
                            // std.debug.assert(@floatToInt(u32, @floor(intersection_point.y)) == scanline_y_index);
                            // std.debug.assert(intersection_point.x == second_intersection.x);
                        }
                    }
                },
                else => {
                    std.log.warn("Kind: {}", .{kind});
                    continue;
                },
            }
        }

        if (current_scanline.count == 0) {
            std.log.warn("No intersections found", .{});
            continue;
        }
        std.debug.assert(current_scanline.count % 2 == 0);

        {
            // We have to sort by x ascending
            var step: usize = 1;
            while (step < current_scanline.count) : (step += 1) {
                const key = current_scanline.buffer[step];
                var x = @intCast(i64, step) - 1;
                while (x >= 0 and current_scanline.buffer[@intCast(usize, x)].x_position > key.x_position) : (x -= 1) {
                    current_scanline.buffer[@intCast(usize, x) + 1] = current_scanline.buffer[@intCast(usize, x)];
                }
                current_scanline.buffer[@intCast(usize, x + 1)] = key;
            }
        }

        // What's the plan for curves..
        // Perhaps something simple first that I can expand upon
        // Let's try the long way first. Get the accurate coverage of a curve
        // I suppose you could try and subdivide it
        // I do like the idea of keeping a reference to the last intersection.
        // If you need more accuracy you can get the point at the t inbetween current and previous
        // The only issue would be the first intersection point

        // 1. Ends inside the pixel
        // 2. Is inflection point (Keep a reference to curve boundries and inflection point)

        var i: u32 = 0;
        while (i < (current_scanline.count - 1)) : (i += 2) {
            const scanline_start = current_scanline.buffer[i];
            const scanline_end = current_scanline.buffer[i + 1];

            const line_start: LineSegment = blk: {
                const current_vertex = vertices[scanline_start.outline_index];
                const previous_vertex = vertices[scanline_start.outline_index - 1];
                const left = if (current_vertex.x < previous_vertex.x) current_vertex else previous_vertex;
                const right = if (current_vertex.x >= previous_vertex.x) current_vertex else previous_vertex;
                break :blk .{ .left = Point(f64){
                    .x = @intToFloat(f64, left.x) * scale,
                    .y = @intToFloat(f64, left.y) * scale,
                }, .right = Point(f64){
                    .x = @intToFloat(f64, right.x) * scale,
                    .y = @intToFloat(f64, right.y) * scale,
                } };
            };

            const line_end: LineSegment = blk: {
                const current_vertex = vertices[scanline_end.outline_index];
                const previous_vertex = vertices[scanline_end.outline_index - 1];
                const left = if (current_vertex.x < previous_vertex.x) current_vertex else previous_vertex;
                const right = if (current_vertex.x >= previous_vertex.x) current_vertex else previous_vertex;
                break :blk .{ .left = Point(f64){
                    .x = @intToFloat(f64, left.x) * scale,
                    .y = @intToFloat(f64, left.y) * scale,
                }, .right = Point(f64){
                    .x = @intToFloat(f64, right.x) * scale,
                    .y = @intToFloat(f64, right.y) * scale,
                } };
            };

            std.debug.print("Intersection pair:\n", .{});
            std.debug.print("  V {d} Curve {} P1 ({d:.2}, {d:.2}) -> P2 ({d:.2}, {d:.2}): intersection {d}\n", .{
                scanline_start.outline_index,
                scanline_start.isCurve(),
                line_start.left.x,
                line_start.left.y,
                line_start.right.x,
                line_start.right.y,
                scanline_start.x_position,
            });
            std.debug.print("  V {d} Curve {} P1 ({d:.2}, {d:.2}) -> P2 ({d:.2}, {d:.2}): intersection {d}\n", .{
                scanline_end.outline_index,
                scanline_end.isCurve(),
                line_end.left.x,
                line_end.left.y,
                line_end.right.x,
                line_end.right.y,
                scanline_end.x_position,
            });

            // TODO: Check here if there is a horizonal line between the start
            //       and end intersections. Do the fill and return

            std.debug.assert(scanline_start.x_position <= scanline_end.x_position);
            std.log.info("Intersection {d}: {d} -- {d}", .{ i, scanline_start.x_position, scanline_end.x_position });

            std.log.info("Doing leading anti-aliasing", .{});
            var full_fill_index_start: u32 = std.math.maxInt(u32);
            {
                //
                // Do leading anti-aliasing
                //
                const start_x = @floatToInt(u32, @floor(scanline_start.x_position));
                var entry_intersection = Point(f64){
                    .y = 0.0,
                    .x = @rem(scanline_start.x_position, 1.0),
                };

                if (scanline_start.isCurve()) {
                    // Scenarios:
                    //   1. Start of curve (End point)
                    //   2. Start of curve (Inflection point)
                    //   3. Continuation of curve
                    //      We can use current horizontal intersection with previous intersection to approximate
                    //      coverage. For more accuracy, we can calculate additional t values between both intersections

                    // Check if we have a previous intersection
                    const previous_entry_point_opt: ?Point(f64) = blk: {
                        const target_outline_i = scanline_start.outline_index;
                        for (previous_scanline.bufferSlice()) |intersection, intersection_i| {
                            // NOTE: There could be two previous intersections for the current curve
                            //       We need to make sure to select the "entry" one.
                            //       I.e The first intersection that begins the fill range
                            if ((intersection_i % 2 == 0) and intersection.outline_index == target_outline_i) {
                                break :blk Point(f64){
                                    .x = intersection.x_position,
                                    .y = 1.0,
                                };
                            }
                        }
                        break :blk null;
                    };

                    if (previous_entry_point_opt) |previous_entry_point| {
                        // The x-value might lie to the right or the left of our current pixel
                        // If that's the case, we need to use it to calculate another point
                        // that lies within our pixel boundry
                        const x_pixel = @floatToInt(u32, @floor(previous_entry_point.x));
                        const is_positive = (previous_entry_point.x > scanline_start.x_position);
                        if (x_pixel == start_x) {
                            // std.debug.assert(false);
                            std.log.info("SIMPLE CURVE", .{});
                            const exit_intersection = Point(f64){
                                .x = @rem(previous_entry_point.x, 1.0),
                                .y = previous_entry_point.y,
                            };
                            const coverage = calculateCoverage(entry_intersection, exit_intersection, is_positive);
                            bitmap.pixels[@intCast(usize, start_x) + (image_y_index * dimensions.width)] = @floatToInt(u8, 255.0 * coverage);
                            full_fill_index_start = start_x + 1;
                        } else {
                            // Last horizontal intersection was 1 or more pixels to left or right
                            // We need to apply AA for all pixels
                            std.log.info("Current x {d:.2}, Previous {d:.2}", .{ scanline_start.x_position, previous_entry_point.x });
                            const increment: i32 = if (is_positive) 1 else -1;
                            const aa_start_x = @intCast(i32, start_x);
                            const aa_end_x = @intCast(i32, x_pixel) + increment;
                            // Entry point for pixel
                            var start_point = Point(f64){
                                .x = @rem(scanline_start.x_position, 1.0),
                                .y = 0.0,
                            };
                            // Total distance between entry point and previous intersection
                            // As we advance to the left, this total diff calculation
                            // will be one pixel less
                            const total_x_diff: f64 = abs(scanline_start.x_position - previous_entry_point.x);
                            const remaining_x = if (is_positive) 1.0 - start_point.x else start_point.x;
                            std.log.info("Range {d} -> {d}. Positive {} Distance {d:.2}", .{ aa_start_x, aa_end_x - increment, is_positive, total_x_diff });
                            std.debug.assert(total_x_diff >= 0.0);
                            var x: i32 = aa_start_x;
                            while (x != aa_end_x) : (x += increment) {
                                const iteration = if (is_positive) @intToFloat(f64, x - aa_start_x) else @intToFloat(f64, aa_start_x - x);
                                const exit_y: f64 = @rem(remaining_x + iteration, total_x_diff) / total_x_diff;
                                std.log.info("{d}, Remaining x {d:.2}, exit_y {d:.2}", .{ iteration, remaining_x, exit_y });
                                std.debug.assert(exit_y <= 1.0);
                                std.debug.assert(exit_y >= 0.0);
                                const exit_point = Point(f64){
                                    .x = if (is_positive) 1.0 else 0.0,
                                    .y = exit_y,
                                };
                                const coverage = calculateCoverage(start_point, exit_point, is_positive);
                                // const adjusted_coverage = if(is_positive) coverage else 1.0 - coverage;
                                bitmap.pixels[@intCast(usize, x) + (image_y_index * dimensions.width)] = @floatToInt(u8, 255.0 * coverage);
                                start_point = Point(f64){
                                    .x = if (is_positive) 0.0 else 1.0,
                                    .y = exit_point.y,
                                };
                            }
                            full_fill_index_start = start_x + 1;
                        }
                    } else {
                        //
                        // New curve. Check for endpoint, otherwise assume it's an inflection point
                        //
                        const end_point_opt = blk: {
                            const left = line_start.left;
                            const right = line_start.right;
                            if (@floatToInt(u32, @floor(left.x)) == start_x and @floatToInt(u32, @floor(left.y)) == scanline_y_index) {
                                break :blk line_start.left;
                            }
                            if (@floatToInt(u32, @floor(right.x)) == start_x and @floatToInt(u32, @floor(right.y)) == scanline_y_index) {
                                break :blk line_start.right;
                            }
                            break :blk null;
                        };

                        if (end_point_opt) |end_point| {
                            std.log.info("End point found for curve", .{});
                            // This will require three points
                            // 1. Current intersecion
                            // 2. End point of current contour
                            // 3. Exiting intersection of connecting contour
                            _ = end_point;
                            // TODO: Implement
                            bitmap.pixels[@intCast(usize, start_x) + (image_y_index * dimensions.width)] = 155;
                            full_fill_index_start = start_x + 1;
                        } else {
                            // 
                            // This is the first time encountering a curve
                            //

                            const t_start = scanline_start.t.?;
                            // TODO: Factor in t_end

                            var contour_index = scanline_start.outline_index;
                            var x = @floatToInt(i32, @floor(scanline_start.x_position));
                            const t_increment: f64 = 0.01;
                            const end_x = @floatToInt(u32, @floor(scanline_end.x_position));
                            var current_t = t_start + t_increment;
                            // We can assume the curve is positive, and we end the leading AA phase when it turns negative
                            // while(contour_index <= scanline_end.outline_index) : (contour_index += 1) {
                            var coverage_accumulator: f64 = 0;
                            const current_contour = vertices[contour_index];
                            
                            // NOTE: Order of bezier points matter! Otherwise you'll get inverted results when 
                            //       calculating an intersection point at t. Therefore we have to use contour 
                            //       order instead of sorting by x (I.e using line_start)
                            // TODO: contour_index could be 0
                            const sv1 = vertices[contour_index];
                            const sv2 = vertices[contour_index - 1];
                            var bezier = BezierQuadratic {
                                .a = .{ .x = @intToFloat(f64, sv2.x) * scale, .y = @intToFloat(f64, sv2.y) * scale },
                                .b = .{ .x = @intToFloat(f64, sv1.x) * scale, .y = @intToFloat(f64, sv1.y) * scale },
                                .control = .{
                                    .x = @intToFloat(f64, current_contour.control1_x) * scale,
                                    .y = @intToFloat(f64, current_contour.control1_y) * scale,
                                }
                            };

                            var last_point = Point(f64) {
                                .x = scanline_start.x_position - @intToFloat(f64, x),
                                .y = 0.0,
                            };

                            std.log.info("Initial last_point {d} {d} with t {d}", .{last_point.x, last_point.y, t_start});

                            std.debug.assert(last_point.x <= 1.0);
                            std.debug.assert(last_point.x >= 0.0);

                            std.debug.assert(last_point.y <= 1.0);
                            std.debug.assert(last_point.y >= 0.0);

                            // Rasterize curve from left to right or right to left
                            //  Assumed conditions:
                            //  1. The curve stays within the y boundries for the scanline (Between 0 and 1)
                            //  2. Positive winding order
                            while(x <= end_x and x >= 0) {
                                std.debug.assert(last_point.y <= 1.0);
                                std.debug.assert(last_point.y >= 0.0);

                                std.log.info("T: {d} Contour: {d}", .{current_t, contour_index});

                                if(current_t > 1.0) {
                                    current_t = 1.0;
                                }

                                var sampled_point = quadraticBezierPoint(bezier, current_t);
                                std.log.info("Sampled: point {d} {d} at x {d}", .{sampled_point.x, sampled_point.y, x});
                                if(@floatToInt(u32, @floor(sampled_point.y)) != scanline_y_index) {
                                    break;
                                }

                                const x_pixel = @floatToInt(u32, @floor(sampled_point.x));
                                std.debug.assert(@floatToInt(u32, @floor(sampled_point.y)) == scanline_y_index);

                                sampled_point.x = sampled_point.x - @intToFloat(f64, x);
                                sampled_point.y = @rem(sampled_point.y, 1.0 + std.math.floatMin(f64));
                                std.log.info("Sampled point {d} {d} at x {d}", .{sampled_point.x, sampled_point.y, x});

                                const is_positive = (sampled_point.x > last_point.x);
                                
                                if(x_pixel == x) {
                                    const corner_point = Point(f64) {
                                        .x = if(is_positive) 1.0 else 0.0,
                                        .y = 0.0,
                                    };
                                    //
                                    // p1 = corner_point, p2 = last_point, p3 = sampled_point
                                    // ((p1.x * p2.y) + (p2.x * p3.y) + (p3.x * p1.y) - (p1.y * p2.x) - (p2.x * p3.y) - (p3.y * p1.x)) / 2.0
                                    //
                                    const calc_1 = corner_point.x * last_point.y;
                                    const calc_2 = last_point.x * sampled_point.y;
                                    const calc_3 = sampled_point.x * corner_point.y;
                                    const calc_4 = corner_point.y * last_point.x;
                                    const calc_5 = last_point.x * sampled_point.y;
                                    const calc_6 = sampled_point.y * corner_point.x;
                                    coverage_accumulator += abs(calc_1 + calc_2 + calc_3 - calc_4 - calc_5 - calc_6) / 2.0;
                                    std.debug.assert(coverage_accumulator <= 1.0);
                                    std.debug.assert(coverage_accumulator >= 0.0);
                                    last_point = sampled_point;
                                } else {
                                    // 
                                    // Our sampled point lies in the pixel to the left or right of our current
                                    // We'll interpolate a new sampled point between last_point and sampled_point
                                    // that lies on the y boundry (0.0 or 1.0). This will be the last point used to calculate 
                                    // the total coverage for our current pixel. We'll set the pixel value and then
                                    // use our original sampled value, along with our interpolated point to 
                                    // calulate the first area triangle for our new pixel.
                                    //
                                    // Linear interpolatation is used to estimate the y value that corresponds to our new x value
                                    //
                                    const change_in_y = sampled_point.y - last_point.y;
                                    const remaining_x = if(is_positive) 1.0 - last_point.x else last_point.x;
                                    const range_x = abs(sampled_point.x - last_point.x);
                                    std.debug.assert(range_x > 0.0);

                                    const x_percentage = remaining_x / range_x;
                                    std.debug.assert(x_percentage <= 1.0);
                                    std.debug.assert(x_percentage >= 0.0);
                                    const interpolated_between_point = Point(f64) {
                                        .x = if(is_positive) 1.0 else 0.0,
                                        .y = last_point.y + (change_in_y * x_percentage),
                                    };
                                    std.debug.assert(interpolated_between_point.y <= 1.0);
                                    std.debug.assert(interpolated_between_point.y >= 0.0);

                                    const corner_point = Point(f64) {
                                        .x = if(is_positive) 1.0 else 0.0,
                                        .y = 0.0,
                                    };

                                    {
                                        const calc_1 = corner_point.x * last_point.y;
                                        const calc_2 = last_point.x * interpolated_between_point.y;
                                        const calc_3 = interpolated_between_point.x * corner_point.y;
                                        const calc_4 = corner_point.y * last_point.x;
                                        const calc_5 = last_point.x * interpolated_between_point.y;
                                        const calc_6 = interpolated_between_point.y * corner_point.x;
                                        coverage_accumulator += abs(calc_1 + calc_2 + calc_3 - calc_4 - calc_5 - calc_6) / 2.0;
                                    }

                                    std.debug.assert(coverage_accumulator <= 1.0);
                                    std.debug.assert(coverage_accumulator >= 0.0);

                                    if(!is_positive) {
                                        coverage_accumulator = 1.0 - coverage_accumulator;
                                    }
                                    std.log.info("Coverage: {d}", .{coverage_accumulator});
                                    bitmap.pixels[@intCast(usize, x) + (image_y_index * dimensions.width)] = @floatToInt(u8, 255.0 * coverage_accumulator);

                                    x = if(is_positive) x + 1 else x - 1;

                                    // We're guarding here as next x could be beyond our valid AA range.
                                    // NOTE: If this branch isn't taken, this is the last iteration
                                    if(x <= end_x and x >= 0) {
                                        last_point = Point(f64) {
                                            .x = if(is_positive) 0.0 else 1.0,
                                            .y = interpolated_between_point.y,
                                        };
                                        const calc_1 = corner_point.x * last_point.y;
                                        const calc_2 = last_point.x * sampled_point.y;
                                        const calc_3 = sampled_point.x * corner_point.y;
                                        const calc_4 = corner_point.y * last_point.x;
                                        const calc_5 = last_point.x * sampled_point.y;
                                        const calc_6 = sampled_point.y * corner_point.x;
                                        coverage_accumulator = abs(calc_1 + calc_2 + calc_3 - calc_4 - calc_5 - calc_6) / 2.0;
                                    }
                                }
                                if(current_t >= 1.0) {
                                    current_t = t_increment;
                                    // TODO: This doesn't work. We can't do anything with .move vertices nor do we 
                                    //       have support for lines
                                    contour_index = (contour_index + 1) % @intCast(u32, vertices.len);
                                    const previous_contour = if(contour_index == 0) vertices.len - 1 else contour_index - 1;
                                    const v1 = vertices[contour_index];

                                    if(@intToEnum(VMove, v1.kind) == .move) {
                                        std.debug.assert(false);
                                        break;
                                    }

                                    if(@intToEnum(VMove, v1.kind) == .line) {
                                        // TODO:
                                        bitmap.pixels[@intCast(usize, x) + (image_y_index * dimensions.width)] = 150;
                                        break;
                                    }

                                    const v2 = vertices[previous_contour];
                                    bezier = BezierQuadratic {
                                        .a = .{ .x = @intToFloat(f64, v2.x) * scale, .y = @intToFloat(f64, v2.y) * scale },
                                        .b = .{ .x = @intToFloat(f64, v1.x) * scale, .y = @intToFloat(f64, v1.y) * scale },
                                        .control = .{
                                            .x = @intToFloat(f64, v1.control1_x) * scale,
                                            .y = @intToFloat(f64, v1.control1_y) * scale,
                                        }
                                    };
                                } else {
                                    current_t += t_increment;
                                }
                            }
                            std.debug.assert(x >= 0);
                            const start_x_pixel = @floatToInt(u32, @floor(scanline_start.x_position));
                            full_fill_index_start = if(x >= start_x_pixel) @intCast(u32, x) else start_x_pixel + 1;
                        }
                    }
                } else if (line_start.isVertical()) {
                    std.log.info("  Vertical found", .{});
                    const coverage = @floatToInt(u8, (1.0 - @rem(scanline_start.x_position, 1.0)) * 255);
                    bitmap.pixels[start_x + (image_y_index * dimensions.width)] = coverage;
                    full_fill_index_start = start_x + 1;
                } else {
                    //
                    // Non-vertical line
                    //
                    const is_positive: bool = line_start.left.y < line_start.right.y;
                    std.log.info("Leading AA iteration. is_positive {}", .{is_positive});
                    const end_x: i64 = if (is_positive) @floatToInt(u32, @floor(scanline_end.x_position)) else -1;
                    const increment: i32 = if (is_positive) 1 else -1;
                    const left = line_start.left;
                    const right = line_start.right;
                    var current_x: i64 = start_x;
                    while (current_x != end_x) : (current_x += increment) {
                        std.debug.assert(current_x >= 0);
                        std.log.info("Pixel coverage iteration. ENTRY {d:.2} {d:.2}", .{
                            entry_intersection.x, entry_intersection.y,
                        });
                        const exit_intersection = blk: {
                            const slope = abs((left.y - right.y) / (left.x - right.x));
                            const exit_height = if (is_positive) ((1.0 - entry_intersection.x) * slope) else (entry_intersection.x * slope);
                            std.log.info("Exit height: {d}", .{exit_height});
                            // NOTE: Working around a stage 1 compiler bug
                            //       A break inside the initial if statement triggers a broken
                            //       LLVM module error
                            var x: f64 = 0.0;
                            var y: f64 = 0.0;
                            const remaining_y = 1.0 - entry_intersection.y;
                            if (exit_height > remaining_y) {
                                const y_per_x = remaining_y / slope;
                                const result = if (is_positive) entry_intersection.x + y_per_x else entry_intersection.x - y_per_x;
                                std.log.info("Calculated exit x: {d}", .{result});
                                std.debug.assert(result <= 1.0);
                                std.debug.assert(result >= 0.0);
                                x = result;
                                y = 1.0;
                            } else {
                                x = if (is_positive) 1.0 else 0.0;
                                y = exit_height;
                            }
                            if (x != entry_intersection.x) {
                                const new_slope = abs((y - entry_intersection.y) / (x - entry_intersection.x));
                                std.log.info("Old slope {d} New {d}", .{ slope, new_slope });
                                std.debug.assert(abs(slope - new_slope) < 0.0001);
                            }
                            break :blk Point(f64){
                                .x = x,
                                .y = y,
                            };
                        };
                        const coverage = calculateCoverage(entry_intersection, exit_intersection, is_positive);
                        const x_pos = @intCast(usize, current_x);
                        bitmap.pixels[x_pos + (image_y_index * dimensions.width)] = @floatToInt(u8, 255.0 * coverage);

                        std.log.info("Current: {d} end {d}", .{ current_x, end_x });

                        if (((current_x + increment) == end_x) or exit_intersection.y == 1.0) {
                            std.log.info("full_fill_index_start set", .{});
                            full_fill_index_start = if (is_positive) (@intCast(u32, current_x) + 1) else (start_x + 1);
                            break;
                        }

                        entry_intersection = .{
                            .x = if (is_positive) 0.0 else 1.0,
                            .y = exit_intersection.y,
                        };
                    }
                }
            }

            std.log.info("Doing trailing anti-aliasing", .{});
            var full_fill_index_end: u32 = 0;
            {
                //
                // Do trailing anti-aliasing
                //
                const start_x = @floatToInt(u32, @floor(scanline_end.x_position));
                if (scanline_end.isCurve()) {
                    std.log.info("  Curve found", .{});
                    const coverage = @floatToInt(u8, (@rem(scanline_end.x_position, 1.0)) * 255);
                    bitmap.pixels[start_x + (image_y_index * dimensions.width)] = coverage;
                    full_fill_index_end = if (start_x == 0) (full_fill_index_start + 1) else start_x - 1;
                } else if (line_end.isVertical()) {
                    std.log.info("  Vertical found", .{});
                    const coverage = @floatToInt(u8, (@rem(scanline_end.x_position, 1.0)) * 255);
                    bitmap.pixels[start_x + (image_y_index * dimensions.width)] = coverage;
                    full_fill_index_end = if (start_x == 0) (full_fill_index_start + 1) else start_x - 1;
                } else {
                    const direction = line_end.direction();
                    const is_positive: bool = (direction == .positive);
                    const end_x = if (direction == .positive) dimensions.width else full_fill_index_start;
                    const increment: i32 = if (direction == .positive) 1 else -1;
                    const left = line_end.left;
                    const right = line_end.right;
                    var entry_intersection = Point(f64){
                        .y = 0.0,
                        .x = @rem(scanline_end.x_position, 1.0),
                    };
                    var current_x: i64 = start_x;
                    while (current_x != end_x) : (current_x += increment) {
                        const exit_intersection = blk: {
                            const m = abs((left.y - right.y) / (left.x - right.x));
                            const exit_height = if (direction == .positive) ((1.0 - entry_intersection.x) * m) else (entry_intersection.x * m);
                            // NOTE: Working around a stage 1 compiler bug
                            //       A break inside the initial if statement triggers a broken
                            //       LLVM module error
                            var x: f64 = 0.0;
                            var y: f64 = 0.0;
                            if (exit_height > (1.0 - entry_intersection.y)) {
                                const run = (1.0 - entry_intersection.y) / m;
                                const result = if (direction == .positive) entry_intersection.x + run else entry_intersection.x - run;
                                std.log.info("Calculated exit x: {d}", .{result});
                                std.debug.assert(result <= 1.0);
                                std.debug.assert(result >= 0.0);
                                x = result;
                                y = 1.0;
                            } else {
                                x = if (direction == .positive) 1.0 else 0.0;
                                y = exit_height;
                            }

                            const new_slope = abs((y - entry_intersection.y) / (x - entry_intersection.x));
                            std.log.info("Old slope {d} New {d}", .{ m, new_slope });
                            std.debug.assert(abs(m - new_slope) < 0.0001);

                            break :blk Point(f64){
                                .x = x,
                                .y = y,
                            };
                        };
                        const coverage = calculateCoverage(entry_intersection, exit_intersection, is_positive);
                        bitmap.pixels[@intCast(usize, current_x) + (image_y_index * dimensions.width)] = @floatToInt(u8, 255.0 * (1.0 - coverage));

                        if (((current_x + increment) == end_x) or exit_intersection.y == 1.0) {
                            full_fill_index_end = if (is_positive) start_x - 1 else @intCast(u32, current_x - 1);
                            break;
                        }

                        entry_intersection = .{
                            .x = if (direction == .positive) 0.0 else 1.0,
                            .y = exit_intersection.y,
                        };
                    }
                }
            }

            std.log.info("Doing inner fill {d} -> {d}", .{ full_fill_index_start, full_fill_index_end });
            // std.debug.assert(full_fill_index_start <= full_fill_index_end);

            std.debug.assert(full_fill_index_start >= 0);
            std.debug.assert(full_fill_index_end < dimensions.width);

            //
            // Fill all pixels between aliased zones
            //

            var current_x = full_fill_index_start;
            while (current_x <= full_fill_index_end) : (current_x += 1) {
                bitmap.pixels[current_x + (image_y_index * dimensions.width)] = 255;
            }
        }
        const temp_buffer = previous_scanline.buffer;
        previous_scanline = current_scanline;
        current_scanline.buffer = temp_buffer;
        current_scanline.reset();
    }
    return bitmap;
}

fn rasterize2(allocator: Allocator, dimensions: Dimensions2D(u32), vertices: []Vertex, scale: f32) !Bitmap {
    {
        // quadraticBezierPlaneIntersection2
        const b = BezierQuadratic{
            .a = .{ .x = 0.0, .y = 0.0 },
            .b = .{
                .x = 12,
                .y = 8,
            },
            .control = .{
                .x = 0,
                .y = 8,
            },
        };
        std.debug.assert(quadraticBezierPlaneIntersection2(6, b) == 3);
    }

    {
        const b = BezierQuadratic{
            .a = .{ .x = 0.0, .y = 0.0 },
            .b = .{
                .x = 20,
                .y = 0,
            },
            .control = .{
                .x = 10,
                .y = 10,
            },
        };
        std.debug.assert(quadradicBezierInflectionPoint(b).x == 10);
        std.debug.assert(quadradicBezierInflectionPoint(b).y == 5);
    }

    {
        const b = BezierQuadratic{
            .a = .{ .x = 0.0, .y = 0.0 },
            .b = .{
                .x = 20,
                .y = 0,
            },
            .control = .{
                .x = 14,
                .y = -8,
            },
        };
        std.debug.assert(quadradicBezierInflectionPoint(b).x == 12);
        std.debug.assert(quadradicBezierInflectionPoint(b).y == -4);
    }

    {
        const b = BezierQuadratic{
            .a = .{ .x = 10.0, .y = 0.0 },
            .b = .{
                .x = 200,
                .y = 15,
            },
            .control = .{
                .x = 40,
                .y = 40,
            },
        };
        std.debug.assert(quadraticBezierPoint(0.0, b).x == 10.0);
        std.debug.assert(quadraticBezierPoint(1.0, b).x == 200.0);
    }

    {
        const b = BezierQuadratic{
            .a = .{ .x = 0.0, .y = 0.0 },
            .b = .{
                .x = 20.0,
                .y = 0.0,
            },
            .control = .{
                .x = 10.0,
                .y = 10.0,
            },
        };
        std.debug.assert(quadraticBezierPlaneIntersection(3.75, b) == 15.0);
    }

    {
        const point_a = Point(f64){
            .x = 3.0,
            .y = 8.0,
        };
        const point_b = Point(f64){
            .x = 7.0,
            .y = 12.0,
        };
        std.debug.assert(horizontalPlaneIntersection(0.0, point_a, point_b) == -5.0);
    }

    const scale_x = scale;
    const scale_y = scale;

    // std.log.info("Scale xy {d} {d}", .{scale_x, scale_y});

    // glyph_dimensions.width = @intCast(u32, max_x - min_x);
    // glyph_dimensions.height = @intCast(u32, max_y - min_y);

    // const bitmap_aspect = @intToFloat(f32, dimensions.width) / @intToFloat(f32, dimensions.height);
    // const glyph_aspect = @intToFloat(f32, glyph_dimensions.width) / @intToFloat(f32, glyph_dimensions.height);

    // std.log.info("Bitmap aspect: {d}", .{bitmap_aspect});
    // std.log.info("Glyph aspect: {d}", .{glyph_aspect});
    std.log.info("Glyph bbox {d} x {d} -- {d} x {d}", .{ min_x, min_y, max_x, max_y });
    // std.log.info("Scaled {d} x {d} -- {d} x {d}", .{@intToFloat(f32, min_x) * scale.x, @intToFloat(f32, min_y) * scale.x, @intToFloat(f32, max_x) * scale.x, @intToFloat(f32, max_y) * scale.x});

    // std.debug.assert(bitmap_aspect == glyph_aspect);

    var bitmap: Bitmap = .{
        .width = @intCast(u32, dimensions.width),
        .height = @intCast(u32, dimensions.height),
        .pixels = (try allocator.alloc(u8, @intCast(usize, dimensions.width) * dimensions.height)),
    };

    std.mem.set(u8, bitmap.pixels, 0);

    std.log.info("Rasterizing image of dimensions: {d} x {d}", .{ bitmap.width, bitmap.height });

    // std.log.info("Calc full w size: {d} real {d}", .{dimensions.width * scale_x, glyph_dimensions.width});
    // std.log.info("Calc full h size: {d} real {d}", .{dimensions.height * scale_y, glyph_dimensions.height});

    // std.debug.assert(dimensions.width * scale_x == glyph_dimensions.width);
    // std.debug.assert(dimensions.height * scale_y == glyph_dimensions.height);

    const printf = std.debug.print;

    var y: u32 = 0;
    while (y < dimensions.height) : (y += 1) {
        var intersections: [256]f64 = [1]f64{0} ** 256;
        var intersection_count: u16 = 0;

        printf("Y axis {d}\n", .{y});

        // Loop through all vertices and store line intersections
        var vertex_i: u32 = 1;
        while (vertex_i < vertices.len) : (vertex_i += 1) {
            const previous_vertex = vertices[vertex_i - 1];
            const current_vertex = vertices[vertex_i];
            const kind = @intToEnum(VMove, current_vertex.kind);

            const point_a = Point(f64){ .x = @intToFloat(f64, current_vertex.x) * scale_x, .y = @intToFloat(f64, current_vertex.y) * scale_y };
            const point_b = Point(f64){ .x = @intToFloat(f64, previous_vertex.x) * scale_x, .y = @intToFloat(f64, previous_vertex.y) * scale_y };

            if (kind == .line) {
                printf("  Vertex (line) {d:.4} x {d:.4} --> {d:.4} x {d:.4} -- ", .{ point_a.x, point_a.y, point_b.x, point_b.y });
            } else if (kind == .curve) {
                printf("  Vertex (curve) A({d:.2}, {d:.2}) --> B({d:.2}, {d:.2}) C ({d:.2}, {d:.2}) -- ", .{
                    point_a.x,
                    point_a.y,
                    point_b.x,
                    point_b.y,
                    @intToFloat(f64, current_vertex.control1_x) * scale_x,
                    @intToFloat(f64, current_vertex.control1_y) * scale_y,
                });
            }

            const is_horizontal = current_vertex.y == previous_vertex.y;
            const is_outsize_y_range = blk: {
                const max_yy = max(f64, point_a.y, point_b.y);
                const min_yy = min(f64, point_a.y, point_b.y);
                std.debug.assert(max_yy >= min_yy);
                const y_float = @intToFloat(f64, y);
                if (y_float > max_yy or y_float < min_yy) {
                    break :blk true;
                }
                if (kind == .curve) {
                    const bezier = BezierQuadratic{ .a = point_a, .b = point_b, .control = .{
                        .x = @intToFloat(f64, current_vertex.control1_x) * scale_x,
                        .y = @intToFloat(f64, current_vertex.control1_y) * scale_y,
                    } };
                    const inflection_y = quadradicBezierInflectionPoint(bezier).y;
                    const is_middle_higher = (inflection_y > max_yy) and y_float > inflection_y;
                    const is_middle_lower = (inflection_y < min_yy) and y_float < inflection_y;
                    if (is_middle_higher or is_middle_lower) {
                        break :blk true;
                    }
                }
                break :blk false;
            };

            if (is_horizontal) {
                printf("REJECT - Horizonatal line\n", .{});
                continue;
            }

            if (is_outsize_y_range) {
                printf("REJECT - Outsize Y range\n", .{});
                continue;
            }

            if (kind == .line and point_a.x == point_b.x) {
                intersections[intersection_count] = point_a.x;
                printf("Horizontal Intersection added at X index {d}\n", .{intersections[intersection_count]});
                intersection_count += 1;
                continue;
            }

            if (kind == .curve) {
                const bezier = BezierQuadratic{ .a = point_a, .b = point_b, .control = .{
                    .x = @intToFloat(f64, current_vertex.control1_x) * scale_x,
                    .y = @intToFloat(f64, current_vertex.control1_y) * scale_y,
                } };
                const optional_intersection_points = quadraticBezierPlaneIntersections3(bezier, @intToFloat(f64, y));

                // The plane will intersect twice with the part of the bezier curve that is between the highest / lowest
                // end point, and the inflection point

                // const curve_intersection_count: u16 = blk: {
                //     const y_float = @intToFloat(f64, y);
                //     const inflection_y = quadradicBezierInflectionPoint(bezier).y;

                //     std.log.info("Inflection y: {d}", .{inflection_y});

                //     const max_yy = max(f64, point_a.y, point_b.y);
                //     const min_yy = min(f64, point_a.y, point_b.y);

                //     // Inflection point lies between the end points.
                //     // This means it will not change direction mid-curve
                //     if(inflection_y >= min_yy and inflection_y <= max_yy) {
                //         break :blk 1;
                //     }

                //     const is_above = (inflection_y > bezier.a.y and inflection_y > bezier.b.y);
                //     if(is_above and y_float >= max_yy and y_float <= inflection_y) {
                //         break :blk 2;
                //     }

                //     if(!is_above and y_float <= min_yy and y_float >= inflection_y) {
                //         break :blk 2;
                //     }

                //     break :blk 1;
                // };

                if (optional_intersection_points[0]) |first_intersection| {
                    intersections[intersection_count] = first_intersection;
                    printf("Intersection (curve) added at X {d}\n", .{intersections[intersection_count]});
                    intersection_count += 1;
                    if (optional_intersection_points[1]) |second_intersection| {
                        intersections[intersection_count] = second_intersection;
                        printf("Intersection (curve) added at X {d}\n", .{intersections[intersection_count]});
                        intersection_count += 1;
                    }
                } else if (optional_intersection_points[1]) |second_intersection| {
                    intersections[intersection_count] = second_intersection;
                    printf("Intersection (curve) added at X {d}\n", .{intersections[intersection_count]});
                    intersection_count += 1;
                }

                if (intersections[intersection_count] < 0) {
                    std.debug.assert(false);
                    intersections[intersection_count] = 0.0;
                }
            } else if (kind == .line) {
                intersections[intersection_count] = horizontalPlaneIntersection(@intToFloat(f64, y), point_a, point_b);
                if (intersections[intersection_count] < 0.0) {
                    printf("Intersection out of range\n", .{});
                    continue;
                }
                std.debug.assert(intersections[intersection_count] >= 0.0);
                std.debug.assert(intersections[intersection_count] <= @intToFloat(f64, dimensions.width));
                printf("Intersections added at X value {d}\n", .{intersections[intersection_count]});
                intersection_count += 1;
            } else {
                std.log.info("Kind: {}", .{kind});
                continue;
                // unreachable;
            }
        }

        {
            // We have to sort
            var step: usize = 1;
            while (step < intersection_count) : (step += 1) {
                const key = intersections[step];
                var x: i64 = @intCast(i64, step) - 1;
                while (x >= 0 and intersections[@intCast(usize, x)] > key) {
                    intersections[@intCast(usize, x) + 1] = intersections[@intCast(usize, x)];
                    x -= 1;
                }
                intersections[@intCast(usize, x + 1)] = key;
            }
        }

        if (intersection_count == 0) {
            continue;
        }
        std.debug.assert(intersection_count % 2 == 0);

        var i: u32 = 0;
        while (i < (intersection_count - 1)) : (i += 2) {
            std.debug.assert(intersections[i] <= intersections[i + 1]);

            std.log.info("Intersection {d}: {d} -- {d}", .{ i, intersections[i], intersections[i + 1] });

            var start_pixel_index = @floatToInt(u32, @floor(intersections[i]));
            const start_pixel_AA_value = @floatToInt(u8, (1.0 - @rem(intersections[i], 1.0)) * 255);
            var end_pixel_index = @floatToInt(u32, @floor(intersections[i + 1]));
            const end_pixel_AA_value = @floatToInt(u8, @rem(intersections[i + 1], 1.0) * 255);

            if (start_pixel_index == end_pixel_index) {
                const average_AA = @divFloor(@intCast(u16, start_pixel_AA_value) + end_pixel_AA_value, 2);
                bitmap.pixels[start_pixel_index + (y * dimensions.width)] = @intCast(u8, average_AA);
                continue;
            }

            if (start_pixel_index >= dimensions.width) {
                std.log.err("Invalid start pixel: {d} {d}", .{ start_pixel_index, @intToFloat(f64, start_pixel_index) / scale_x });
                start_pixel_index = dimensions.width - 1;
                std.debug.assert(false);
            }

            if (end_pixel_index >= dimensions.width) {
                std.log.info("Invalid end pixel index {d} max is {d}", .{ end_pixel_index, dimensions.width - 1 });
                std.log.err("Invalid end pixel: {d} {d}", .{ end_pixel_index, @intToFloat(f64, end_pixel_index) / scale_x });
                // end_pixel_index = dimensions.width - 1;
                std.debug.assert(false);
                end_pixel_index = dimensions.width - 1;
            }

            std.debug.assert(start_pixel_index < dimensions.width);
            std.debug.assert(end_pixel_index < dimensions.width);

            bitmap.pixels[start_pixel_index + (y * dimensions.width)] = start_pixel_AA_value;
            bitmap.pixels[end_pixel_index + (y * dimensions.width)] = end_pixel_AA_value;

            std.debug.assert(end_pixel_index >= start_pixel_index);

            var x = start_pixel_index + 1;
            while (x < end_pixel_index) : (x += 1) {
                bitmap.pixels[x + (y * dimensions.width)] = 255;
            }
        }
    }
    return bitmap;
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

const Head = packed struct {
    version: f32,
    font_revision: f32,
    checksum_adjustment: u32,
    magic_number: u32, // 0x5F0F3CF5
    flags: u16,
    units_per_em: u16,
    created: i64,
    modified: i64,
    x_min: i16,
    y_min: i16,
    x_max: i16,
    y_max: i16,
    mac_style: u16,
    lowest_rec_PPEM: u16,
    font_direction_hint: i16,
    index_to_loc_format: i16,
    glyph_data_format: i16,
};

const cff_magic_number: u32 = 0x5F0F3CF5;

const PlatformID = enum(u8) { unicode = 0, max = 1, iso = 2, microsoft = 3 };

const CmapIndex = struct {
    version: u16,
    subtables_count: u16,
};

const CMAPPlatformID = enum(u16) {
    unicode = 0,
    macintosh,
    reserved,
    microsoft,
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

const CMAPSubtable = struct {
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
                // unreachable;
            },
            .macintosh => {
                // unreachable;
            },
            .reserved => {
                // unreachable;
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

pub fn initializeFont(allocator: Allocator, data: []u8) !FontInfo {
    _ = allocator;

    var cmap: u32 = 0;

    var font_info: FontInfo = .{
        .data = data[0..],
        .loca = .{},
        .head = .{},
        .hhea = .{},
        .hmtx = .{},
        .glyf = .{},
        .kern = .{},
        .gpos = .{},
        .svg = .{},
        .maxp = .{},
        .cff = .{
            .data = undefined,
        },
        .char_strings = .{ .data = undefined },
        .gsubrs = .{ .data = undefined },
        .subrs = .{ .data = undefined },
        .userdata = undefined,
        .font_dicts = .{ .data = undefined },
        .fd_select = .{ .data = undefined },
    };

    // TODO: What is the real allocation size?
    // font_info.cff = try allocator.alloc(u8, 0);

    {
        const offset_subtable = OffsetSubtable.fromBigEndianBytes(@intToPtr(*align(4) [@sizeOf(OffsetSubtable)]u8, @ptrToInt(data.ptr)));
        assert(offset_subtable.tables_count < 20);

        var i: u32 = 0;
        while (i < offset_subtable.tables_count) : (i += 1) {
            const entry_addr = @intToPtr(*align(4) [@sizeOf(TableDirectory)]u8, @ptrToInt(data.ptr + @sizeOf(OffsetSubtable)) + (@sizeOf(TableDirectory) * i));
            if (TableDirectory.fromBigEndianBytes(entry_addr)) |table_directory| {
                var found: bool = false;
                for (TableTypeList) |valid_tag, valid_tag_i| {
                    // This is a little silly as we're doing a string comparision
                    // And then doing a somewhat unnessecary int comparision / jump
                    if (eql(u8, valid_tag, table_directory.tag[0..])) {
                        found = true;
                        switch (@intToEnum(TableType, @intCast(u4, valid_tag_i))) {
                            .cmap => {
                                cmap = table_directory.offset;
                            },
                            .loca => {
                                font_info.loca.offset = table_directory.offset;
                                font_info.loca.length = table_directory.length;
                            },
                            .head => {
                                font_info.head.offset = table_directory.offset;
                                font_info.head.length = table_directory.length;
                            },
                            .glyf => {
                                font_info.glyf.offset = table_directory.offset;
                                font_info.glyf.length = table_directory.length;
                            },
                            .hhea => {
                                font_info.hhea.offset = table_directory.offset;
                                font_info.hhea.length = table_directory.length;
                            },
                            .hmtx => {
                                font_info.hmtx.offset = table_directory.offset;
                                font_info.hmtx.length = table_directory.length;
                            },
                            .kern => {
                                font_info.loca.offset = table_directory.offset;
                                font_info.loca.length = table_directory.length;
                            },
                            .gpos => {
                                font_info.gpos.offset = table_directory.offset;
                                font_info.gpos.length = table_directory.length;
                            },
                            .maxp => {
                                font_info.maxp.offset = table_directory.offset;
                                font_info.maxp.length = table_directory.length;
                            },
                        }
                    }
                }

                found = false;
            } else {
                log.warn("Failed to load table directory", .{});
            }
        }
    }

    font_info.glyph_count = bigToNative(u16, @intToPtr(*u16, @ptrToInt(data.ptr) + font_info.maxp.offset + 4).*);

    std.log.info("Glyphs found: {d}", .{font_info.glyph_count});

    font_info.index_to_loc_format = bigToNative(u16, @intToPtr(*u16, @ptrToInt(data.ptr) + font_info.head.offset + 50).*);

    if (cmap == 0) {
        return error.RequiredFontTableCmapMissing;
    }

    if (font_info.head.offset == 0) {
        return error.RequiredFontTableHeadMissing;
    }

    if (font_info.hhea.offset == 0) {
        return error.RequiredFontTableHheaMissing;
    }

    if (font_info.hmtx.offset == 0) {
        return error.RequiredFontTableHmtxMissing;
    }

    const head = @intToPtr(*Head, @ptrToInt(data.ptr) + font_info.head.offset).*;
    assert(toNative(u32, head.magic_number, .Big) == 0x5F0F3CF5);

    // Let's read CMAP tables
    var cmap_index_table = @intToPtr(*CmapIndex, @ptrToInt(data.ptr + cmap)).*;

    cmap_index_table.version = toNative(u16, cmap_index_table.version, .Big);
    cmap_index_table.subtables_count = toNative(u16, cmap_index_table.subtables_count, .Big);

    assert(@sizeOf(CMAPPlatformID) == 2);
    assert(@sizeOf(CMAPPlatformSpecificID) == 2);

    font_info.cmap_encoding_table_offset = blk: {
        var cmap_subtable_index: u32 = 0;
        while (cmap_subtable_index < cmap_index_table.subtables_count) : (cmap_subtable_index += 1) {
            assert(@sizeOf(CmapIndex) == 4);
            assert(@sizeOf(CMAPSubtable) == 8);

            const cmap_subtable_addr: [*]u8 = @intToPtr([*]u8, @ptrToInt(data.ptr) + cmap + @sizeOf(CmapIndex) + (cmap_subtable_index * @sizeOf(CMAPSubtable)));
            const cmap_subtable = CMAPSubtable.fromBigEndianBytes(cmap_subtable_addr[0..@sizeOf(CMAPSubtable)]).?;

            if (cmap_subtable.platform_id == .microsoft and cmap_subtable.platform_specific_id.unicode != .other) {
                break :blk cmap + cmap_subtable.offset;
            }
        }

        unreachable;
    };

    const encoding_format: u16 = toNative(u16, @intToPtr(*u16, @ptrToInt(data.ptr) + font_info.cmap_encoding_table_offset).*, .Big);
    _ = encoding_format;

    // Load CFF

    // if(font_info.glyf != 0) {
    // if(font_info.loca == 0) { return error.RequiredFontTableCmapMissing; }
    // } else {

    // var buffer: Buffer = undefined;
    // var top_dict: Buffer = undefined;
    // var top_dict_idx: Buffer = undefined;

    // var cstype: u32 = 2;
    // var char_strings: u32 = 0;
    // var fdarrayoff: u32 = 0;
    // var fdselectoff: u32 = 0;
    // var cff: u32 = findTable(data, font_start, "CFF ");

    // if(!cff) {
    // return error.RequiredFontTableCffMissing;
    // }

    // font_info.font_dicts = Buffer.create();
    // font_info.fd_select = Buffer.create();

    // }
    //

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
