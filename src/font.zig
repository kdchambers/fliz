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
const graphics = @import("graphics.zig");
const Scale2D = geometry.Scale2D;
const Shift2D = geometry.Shift2D;

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
    pixels: []graphics.RGBA(f32),
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

// TODO:
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
        .y1 = bigToNative(i16, @intToPtr(*i16, base_index + 8).*), // max_y
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
    const dimensions = Dimensions2D(u32){
        .width = @intCast(u32, bounding_box.x1 - bounding_box.x0),
        .height = @intCast(u32, bounding_box.y1 - bounding_box.y0),
    };

    bitmap.width = @intCast(u32, bounding_box.x1 - bounding_box.x0);
    bitmap.height = @intCast(u32, bounding_box.y1 - bounding_box.y0);

    if (bitmap.width != 0 and bitmap.height != 0) {
        bitmap = try rasterize2(allocator, dimensions, vertices, scale.x);
    }

    return bitmap;
}

fn Point(comptime T: type) type {
    return packed struct {
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

// How about instead of one intersection at the y axis, you provide 10 intersections per pixel.
// The code I have for calculating intersections is already agnostic of line vs curve, etc

fn quadraticBezierPlaneIntersections(bezier: BezierQuadratic, horizontal_axis: f64) [2]?CurveYIntersection {
    const a = bezier.a.y;
    const b = bezier.control.y;
    const c = bezier.b.y;

    const term_a = a - (2 * b) + c;
    const term_b = 2 * (b - a);
    const term_c = a - horizontal_axis;

    const sqrt_calculation = std.math.sqrt((term_b * term_b) - (4 * term_a * term_c));

    const first_intersection_t = ((-term_b) + sqrt_calculation) / (2 * term_a);
    const second_intersection_t = ((-term_b) - sqrt_calculation) / (2 * term_a);

    // std.log.info("T values {d} {d}", .{ first_intersection_t, second_intersection_t });

    const is_first_valid = (first_intersection_t <= 1.0 and first_intersection_t >= 0.0);
    const is_second_valid = (second_intersection_t <= 1.0 and second_intersection_t >= 0.0);

    return .{
        if (is_first_valid) CurveYIntersection{ .t = first_intersection_t, .x = quadraticBezierPoint(bezier, first_intersection_t).x } else null,
        if (is_second_valid) CurveYIntersection{ .t = second_intersection_t, .x = quadraticBezierPoint(bezier, second_intersection_t).x } else null,
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
    return @fabs(((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) - (p2.x * p1.y + p3.x * p2.y + p4.x * p3.y + p1.x * p4.y)) / 2.0);
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

        if (self.buffer[self.count].t) |t| {
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

fn triangleArea(p1: Point(f64), p2: Point(f64), p3: Point(f64)) f64 {
    // |x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|
    return @fabs((p1.x * (p2.y - p3.y)) + (p2.x * (p3.y - p1.y)) + (p3.x * (p1.y - p2.y))) / 2.0;
}

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
            return 0.5 * @fabs(self.entry.x - self.exit.x) * self.exit.y;
        }
        if (entry.y == 0.0 and exit.y == 1.0) {
            std.log.info("Coverage B", .{});
            return (exit.x + entry.x) / 2.0;
        }
        if (entry.x == 0.0 and exit.y == 1.0) {
            std.log.info("Coverage C", .{});
            return 1.0 - (0.5 * @fabs(self.entry.x - self.exit.x) * self.exit.y);
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
            return 0.5 * @fabs(self.entry.x - self.exit.x) * self.exit.y;
        }

        if (entry.x == 1.0 and exit.y == 1.0) {
            std.log.info("Coverage F", .{});
            return 0.5 * @fabs(self.entry.x - self.exit.x) * self.exit.y;
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

        // std.debug.print("  ENTRY ({d:.2}, {d:.2}) EXIT ({d:.2}, {d:.2})\n", .{
        //     entry.x,
        //     entry.y,
        //     exit.x,
        //     exit.y,
        // });

        // Bottom -> Right
        if (entry.y == 0.0 and exit.x == 1.0) {
            std.debug.assert(is_positive);
            const value = 0.5 * @fabs(entry.x - exit.x) * exit.y;
            // std.log.info("Coverage A {d:.2}", .{value});
            break :blk value;
        }

        // Bottom -> Top
        if (entry.y == 0.0 and exit.y == 1.0) {
            const base = ((exit.x + entry.x) / 2.0);
            const value = if (is_positive) 1.0 - base else 1.0 - base;
            // std.log.info("Coverage B {d:.2}", .{value});
            break :blk value;
        }

        // Left -> Top
        if (entry.x == 0.0 and exit.y == 1.0) {
            std.debug.assert(is_positive);
            const value = 1.0 - (0.5 * @fabs(entry.x - exit.x) * exit.y);
            // std.log.info("Coverage C {d:.2}", .{value});
            break :blk value;
        }

        // Left -> Right
        if (entry.x == 0.0 and exit.x == 1.0) {
            // std.debug.assert(false);
            const value = (exit.y + entry.y) / 2.0;
            // std.log.info("Coverage D {d:.2}", .{value});
            break :blk value;
        }

        // Bottom -> Left
        if (entry.y == 0 and exit.x == 0.0) {
            std.debug.assert(!is_positive);
            const value = 1.0 - (0.5 * @fabs(entry.x - exit.x) * exit.y);
            // std.log.info("Coverage E {d:.2}", .{value});
            break :blk value;
        }

        // Right -> Top
        if (entry.x == 1.0 and exit.y == 1.0) {
            std.debug.assert(!is_positive);
            const value = (0.5 * @fabs(entry.x - exit.x) * exit.y);
            // std.log.info("Coverage F {d:.2}", .{value});
            break :blk value;
        }

        // Right -> Left
        if (entry.x == 1.0 and exit.x == 0.0) {
            const value = (exit.y + entry.y) / 2.0;
            // std.log.info("Coverage G {d:.2}", .{value});
            break :blk value;
        }

        std.debug.assert(false);
        break :blk 0.0;
    };
    return coverage;
}

const CurveIntersectionType = enum(u8) {
    pub const left = 0b0000_0001;
    pub const right = 0b0000_0010;
    pub const top = 0b0000_0100;
    pub const bottom = 0b0000_1000;

    right_top = right | top,
    right_bottom = right | bottom,
    right_left = right | left,
    top_bottom = top | bottom,
    top_left = top | left,
    left_bottom = left | bottom,

    _,

    // right_top = right | top,
    // right_bottom = right | bottom,
    // right_left = right | left,
    // top_bottom = top | bottom,
    // top_left = top | left,
    // left_bottom = left | bottom,

    // convex_right_top = convex | right | top,
    // convex_right_bottom = convex | right | bottom,
    // convex_right_left = convex | right | left,
    // convex_top_bottom = convex | top | bottom,
    // convex_top_left = convex | top | left,
    // convex_left_bottom = convex | left | bottom,

    // concave_right_top = concave | right | top,
    // concave_right_bottom = concave | right | bottom,
    // concave_right_left = concave | right | left,
    // concave_top_bottom = concave | top | bottom,
    // concave_top_left = concave | top | left,
    // concave_left_bottom = concave | left | bottom,
};

fn calculateLineSegmentSlope(point_a: Point(f64), point_b: Point(f64)) f64 {
    return (point_a.y - point_b.y) / (point_a.x - point_b.x);
}

// Needs to take a fill direction vector
// Hmm, I wonder could I rotate curves and then calculate the average x / y
// like I do for left -> right intersections..
fn pixelCurveCoverage(points: []Point(f64)) f64 {
    comptime {
        const T = CurveIntersectionType;
        std.debug.assert(@intToEnum(T, T.right | T.top) == T.right_top);
        std.debug.assert(@intToEnum(T, T.right | T.bottom) == T.right_bottom);
        std.debug.assert(@intToEnum(T, T.right | T.left) == T.right_left);
        std.debug.assert(@intToEnum(T, T.top | T.bottom) == T.top_bottom);
        std.debug.assert(@intToEnum(T, T.top | T.left) == T.top_left);
        std.debug.assert(@intToEnum(T, T.left | T.bottom) == T.left_bottom);
    }

    const first = points[0];
    const last = points[points.len - 1];

    std.log.info("First {d:.4} {d:.4} Last {d:.4} {d:.4}", .{ first.x, first.y, last.x, last.y });

    std.debug.assert(points.len > 1);

    if (points.len == 2) {
        const is_to_right = if (last.x > first.x) true else false;
        const is_positive = if ((is_to_right and last.y > first.y)) true else false;
        return calculateCoverage(first, last, is_positive);
    }

    for (points) |point, point_i| {
        const printf = std.debug.print;
        printf("P {d:.2}: {d:.4}, {d:.4}\n", .{ point_i, point.x, point.y });
        std.debug.assert(point.x >= 0.0);
        std.debug.assert(point.x <= 1.0);
        std.debug.assert(point.y >= 0.0);
        std.debug.assert(point.y <= 1.0);
    }

    const averaged_point = Point(f64){
        .x = (first.x + last.x) / 2.0,
        .y = (first.y + last.y) / 2.0,
    };

    // TODO: Find the closest point to averaged_point.
    //       Middle index is quite inaccurate
    const middle_point = points[@divTrunc(points.len, 2)];

    var intersection_type_int: u8 = 0;
    if (first.x == 0.0 or last.x == 0.0) intersection_type_int |= CurveIntersectionType.left;
    if (first.x == 1.0 or last.x == 1.0) intersection_type_int |= CurveIntersectionType.right;
    if (first.y == 0.0 or last.y == 0.0) intersection_type_int |= CurveIntersectionType.bottom;
    if (first.y == 1.0 or last.y == 1.0) intersection_type_int |= CurveIntersectionType.top;

    switch (@intToEnum(CurveIntersectionType, intersection_type_int)) {
        .right_top => {
            // Averaged point -> Middle point forms a positive line
            const is_concave = calculateLineSegmentSlope(averaged_point, middle_point) > 0;
            std.log.info("Is concave: {}", .{is_concave});
            if (is_concave) {
                // 2 squares + calc triangle fan
                std.debug.assert(first.x == 1.0 or last.x == 1.0);
                std.debug.assert(first.y == 1.0 or last.y == 1.0);
                const fixed_point = Point(f64){
                    .x = if (first.x == 1.0) last.x else first.x,
                    .y = if (first.y == 1.0) last.y else first.y,
                };
                const bottom = fixed_point.y;
                const left = fixed_point.x;
                const intersection = left * bottom;
                var coverage = left + bottom - intersection;
                std.debug.assert(coverage >= 0.0);
                std.debug.assert(coverage <= 1.0);
                var i: usize = 0;
                while (i < (points.len - 1)) : (i += 1) {
                    coverage += triangleArea(points[i], points[i + 1], fixed_point);
                }
                return 1.0 - coverage;
            }
            // Triangle fan from Top Right
            var coverage: f64 = 0;
            var i: usize = 0;
            while (i < (points.len - 1)) : (i += 1) {
                coverage += triangleArea(points[i], points[i + 1], .{ .x = 1.0, .y = 1.0 });
            }
            return coverage;
        },
        .right_bottom => {
            // Averaged point -> Middle point forms a negative line
            const is_concave = calculateLineSegmentSlope(averaged_point, middle_point) < 0;
            std.log.info("Is concave: {}", .{is_concave});
            if (is_concave) {
                // 2 squares + calc triangle fan
                std.debug.assert(first.x == 1.0 or last.x == 1.0);
                std.debug.assert(first.y == 0.0 or last.y == 0.0);
                const fixed_point = Point(f64){
                    .x = if (first.x == 1.0) last.x else first.x,
                    .y = if (first.y == 0.0) last.y else first.y,
                };
                const left = fixed_point.x;
                const top = (1.0 - fixed_point.y);
                const intersection = left * top;
                var coverage = left + top - intersection;
                std.debug.assert(coverage >= 0.0);
                std.debug.assert(coverage <= 1.0);
                var i: usize = 0;
                while (i < (points.len - 1)) : (i += 1) {
                    coverage += triangleArea(points[i], points[i + 1], fixed_point);
                }
                return 1.0 - coverage;
            }
            // Triangle fan from Bottom Right
            var coverage: f64 = 0;
            var i: usize = 0;
            while (i < (points.len - 1)) : (i += 1) {
                coverage += triangleArea(points[i], points[i + 1], .{ .x = 1.0, .y = 0.0 });
            }
            return coverage;
        },
        .right_left => {
            // Regardless of whether convex or concave,
            // we're averaging the y values and returning the bottom half
            const right_to_left = first.x > last.x;
            var total_y: f64 = 0;
            for (points) |point| {
                total_y += point.y;
            }
            const average = (total_y / @intToFloat(f64, points.len));
            return if (right_to_left) 1.0 - average else average;
        },
        .top_bottom => {
            // Regardless of whether convex or concave,
            // we're averaging the x values and returning the right half
            var total_x: f64 = 0;
            for (points) |point| {
                total_x += point.x;
            }
            return 1.0 - (total_x / @intToFloat(f64, points.len));
        },
        .top_left => {
            // Averaged point -> Middle point forms a negative line
            const is_concave = calculateLineSegmentSlope(averaged_point, middle_point) < 0;
            std.log.info("Is concave: {}", .{is_concave});
            if (is_concave) {
                // Triangle fan from Top Left
                var coverage: f64 = 0;
                var i: usize = 0;
                while (i < (points.len - 1)) : (i += 1) {
                    coverage += triangleArea(points[i], points[i + 1], .{ .x = 0.0, .y = 1.0 });
                }
                return 1.0 - coverage;
            }
            // 2 squares + calc triangle fan
            std.debug.assert(first.x == 0.0 or last.x == 0.0);
            std.debug.assert(first.y == 1.0 or last.y == 1.0);
            const fixed_point = Point(f64){
                .x = if (first.x == 0.0) last.x else first.x,
                .y = if (first.y == 1.0) last.y else first.y,
            };
            const bottom = fixed_point.y;
            const right = 1.0 - (fixed_point.x);
            const intersection = right * bottom;
            var coverage = bottom + right - intersection;
            std.log.info("Base coverage: {d}", .{coverage});
            std.debug.assert(coverage >= 0.0);
            std.debug.assert(coverage <= 1.0);
            var i: usize = 0;
            while (i < (points.len - 1)) : (i += 1) {
                // std.log.info("Area coverage: ({d:.4}, {d:.4}) ({d:.4}, {d:.4}) ({d:.4}, {d:.4}): {d:.6}", .{ points[i].x, points[i].y, points[i + 1].x, points[i + 1].y, fixed_point.x, fixed_point.y, triangleArea(points[i], points[i + 1], fixed_point) });
                coverage += triangleArea(points[i], points[i + 1], fixed_point);
            }
            std.log.info("Full coverage: {d}", .{coverage});
            return coverage;
        },
        .left_bottom => {
            // Averaged point -> Middle point forms a positive line
            const is_concave = calculateLineSegmentSlope(averaged_point, middle_point) > 0;
            std.log.info("Is concave: {}", .{is_concave});
            if (is_concave) {
                // Triangle fan from Bottom Left
                var coverage: f64 = 0;
                var i: usize = 0;
                while (i < (points.len - 1)) : (i += 1) {
                    coverage += triangleArea(points[i], points[i + 1], .{ .x = 0.0, .y = 0.0 });
                }
                return 1.0 - coverage;
            }
            // 2 squares + calc triangle fan
            std.debug.assert(first.x == 0.0 or last.x == 0.0);
            std.debug.assert(first.y == 0.0 or last.y == 0.0);
            const fixed_point = Point(f64){
                .x = if (first.x == 0.0) last.x else first.x,
                .y = if (first.y == 0.0) last.y else first.y,
            };
            const top = 1.0 - fixed_point.y;
            const right = 1.0 - fixed_point.x;
            const intersection = top * right;
            var coverage = top + right - intersection;
            std.debug.assert(coverage >= 0.0);
            std.debug.assert(coverage <= 1.0);
            var i: usize = 0;
            while (i < (points.len - 1)) : (i += 1) {
                coverage += triangleArea(points[i], points[i + 1], fixed_point);
            }
            return coverage;
        },
        else => {
            // TODO:
            // std.debug.assert(false);
            return 0.5;
        },
    }
    return 0;
}

fn clamp(value: f64, val_min: f64, val_max: f64) f64 {
    std.debug.assert(val_min < val_max);
    if (value > val_max) {
        return val_max;
    }
    if (value < val_min) {
        return val_min;
    }
    return value;
}

/// Given a point within a pixel and a slope, calculate where it leaves the pixel
fn pixelLineIntersection(x_per_y: f64, point: Point(f64)) Point(f64) {
    std.debug.assert(x_per_y != 0);

    const x_per_unit: f64 = x_per_y;
    const y_per_unit: f64 = if (x_per_y > 0) 1 else -1;

    std.log.info("x_per_y: {d}", .{x_per_y});

    const y_per_x = 1 / x_per_y;

    const units_to_left = -point.x / x_per_y;
    const units_to_right = (1.0 - point.x) / x_per_y;
    const units_to_top = (1.0 - point.y);
    const units_to_bottom = -point.y;

    std.debug.print("units_to_left: {d}\n", .{units_to_left});
    std.debug.print("units_to_right: {d}\n", .{units_to_right});
    std.debug.print("units_to_top: {d}\n", .{units_to_top});
    std.debug.print("units_to_bottom: {d}\n", .{units_to_bottom});

    var best_scale: f64 = std.math.floatMax(f64);
    if (units_to_left > 0 and units_to_left < best_scale)
        best_scale = units_to_left;
    if (units_to_right > 0 and units_to_right < best_scale)
        best_scale = units_to_right;
    if (units_to_top > 0 and units_to_top < best_scale)
        best_scale = units_to_top;
    if (units_to_bottom > 0 and units_to_bottom < best_scale)
        best_scale = units_to_bottom;

    std.debug.print("Best scale: {d}\n", .{best_scale});
    std.debug.print("x_per_y: {d}\n", .{x_per_y});
    std.debug.print("y_per_x: {d}\n", .{y_per_x});

    std.debug.print("Result: {d} {d}\n", .{ point.x + (best_scale), point.y + (best_scale * y_per_x) });

    return Point(f64){
        .x = point.x + (best_scale * x_per_unit), // need a best scale of 1
        .y = point.y + (best_scale * y_per_unit),
    };
}

test "pixelLineIntersection" {
    {
        const point = Point(f64){
            .x = 0.5,
            .y = 0.5,
        };
        const exit = pixelLineIntersection(2.0, point);
        try std.testing.expect(exit.x == 1.0);
        try std.testing.expect(exit.y == 0.75);
    }

    {
        const point = Point(f64){
            .x = 0.5,
            .y = 0.5,
        };
        const exit = pixelLineIntersection(-2.0, point);
        try std.testing.expect(exit.x == 0.0);
        try std.testing.expect(exit.y == 0.25);
    }

    {
        const point = Point(f64){
            .x = 0.5,
            .y = 0.5,
        };
        const exit = pixelLineIntersection(-1.0, point);
        try std.testing.expect(exit.x == 0.0);
        try std.testing.expect(exit.y == 0.0);
    }

    {
        const point = Point(f64){
            .x = 0.0,
            .y = 0.0,
        };

        const exit = pixelLineIntersection(0.75, point);
        try std.testing.expect(exit.x == 0.75);
        try std.testing.expect(exit.y == 1.0);
    }
}

// Maybe I could calculate all of the intersections for each scanline upfront before I try and rasterize
// I could mark all of the vertex intersections and use that as a starting point
// That could make a lot of the code simpler as between connections wouldn't have to deal with those
// edge cases

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

// I should sample points at a min distance
// Can aim for 1/8 of a pixel.
// The default is the sample 10 times alone y axis
// But if the overall distance is too large, I can take additional values between t's

const FillSegment = struct {
    // The issue with this is it misses horizontal lines
    start_points: []Point(f64),
    end_points: []Point(f64),
};

fn sortAscending(T: type, slice: []T) []T {
    var step: usize = 1;
    while (step < slice.len) : (step += 1) {
        const key = slice[step];
        var x = @intCast(i64, step) - 1;
        while (x >= 0 and slice[@intCast(usize, x)] > key) : (x -= 1) {
            slice[@intCast(usize, x) + 1] = slice[@intCast(usize, x)];
        }
        slice[@intCast(usize, x + 1)] = key;
    }
    return slice;
}

// How would you write function, that takes a set of points within y 0 - 1 and rasterizes the line

// You could make points always form a shape.
// If start and end outlines go through the bottom y axis,
// just connect both the points to form sort of a rectangle
// Then you can check for that and fill the inner square at the beginning

// Requirements:
// All points lie between y 0.0 and 1.0 inclusively
// There are either 2 or 4 y intersection points
// There are at least 2 points that lies between each pixel bounds
fn rasterizeLineSegment(line: []graphics.RGBA(f32), points: []Point(f64)) void {
    const index_last: usize = points.len - 1;

    std.log.info("Rastering line segment with p0 {d}, {d} and pl {d}, {d}", .{
        points[0].x,
        points[0].y,
        points[index_last].x,
        points[index_last].y,
    });

    printPoints(points);

    std.debug.assert(points[0].y == 0.0 or points[0].y == 1.0);
    std.debug.assert(points[index_last].y == 0.0 or points[index_last].y == 1.0);

    const line_start_index: usize = if (points[0].x < points[index_last].x) 0 else index_last;
    const line_end_index: usize = if (line_start_index == 0) index_last else 0;

    // It's required that first and last point are y intersects
    // However, it's possible to have two more
    var aux_intersects = [2]?usize{ null, null };

    //
    // Find auxilary y intersection points
    //
    var point_index: usize = 1;
    while (point_index < points.len) : (point_index += 1) {
        const point = points[point_index];
        if (point.y == 0.0 or point.y == 1.0) {
            std.debug.assert(aux_intersects[0] == null or aux_intersects[1] == null);
            if (aux_intersects[0] == null) {
                aux_intersects[0] = point_index;
            } else {
                aux_intersects[1] = point_index;
            }
        }
    }

    //
    // The presence of two auxilary y intersection points means that we
    // have a region in the middle that can be filled
    //
    if (aux_intersects[0] != null and aux_intersects[1] != null) {
        const first = aux_intersects[0].?;
        const second = aux_intersects[1].?;
        const left = if (points[first].x < points[second].x) first else second;
        const right = if (points[first].x > points[second].x) first else second;

        const fill_start = if (points[left].x > points[line_start_index].x) left else line_start_index;
        const fill_end = if (points[right].x < points[line_end_index].x) right else line_end_index;

        var current_pixel = @floatToInt(usize, @ceil(points[fill_start].x));
        while (current_pixel < @floatToInt(usize, @floor(points[fill_end].x))) : (current_pixel += 1) {
            line[current_pixel] = pixelFill(1.0);
        }
        rasterizeLineSegment(line, points[0..fill_start]); // left side
        rasterizeLineSegment(line, points[fill_end..]); // right side
    } else {
        // Maybe find the middle (inflection) pixel, fill it using y average
        // and then rasterize left and right seperately

        // Find the highest y

        // Intersection only has two points
        // find min and max x
        // Loop through and collect each point that lies in a pixel
        // When finished send all to calculateCoverage function
        var min_x_pixel: usize = std.math.maxInt(usize);
        var max_x_pixel: usize = 0;
        for (points) |point| {
            if (point.x > @intToFloat(f64, max_x_pixel)) max_x_pixel = @floatToInt(usize, @floor(point.x));
            if (point.x < @intToFloat(f64, min_x_pixel)) min_x_pixel = @floatToInt(usize, @floor(point.x));
        }
        var current_pixel = min_x_pixel;
        point_index = 0;
        var point_buffer: [64]Point(f64) = undefined;
        //
        // Sort by x and calculate coverage for that pixel
        //
        printPoints(points);
        while (current_pixel <= max_x_pixel) : (current_pixel += 1) {
            var point_count: usize = 0;
            while (point_index < points.len) : (point_index += 1) {
                const relative_x: f64 = points[point_index].x - @intToFloat(f64, current_pixel);
                if (relative_x >= 0.0 and relative_x <= 1.0) {
                    point_buffer[point_count] = .{
                        .x = relative_x,
                        .y = points[point_index].y,
                    };
                    point_count += 1;
                }
            }
            std.debug.assert(point_count >= 2);
            const coverage = pixelCurveCoverage(point_buffer[0..point_count]);
            line[current_pixel] = pixelFill(coverage);
        }
    }
}

fn pointMinX(a: Point(f64), b: Point(f64)) Point(f64) {
    return if (a.x < b.x) a else b;
}

fn pointMaxX(a: Point(f64), b: Point(f64)) Point(f64) {
    return if (a.x > b.x) a else b;
}

fn isNormalized(value: f64) bool {
    std.debug.assert(value <= 1.0);
    std.debug.assert(value >= 0.0);
    return (value <= 1.0) and (value >= 0.0);
}

fn clampTo(value: f64, edge: f64, threshold: f64) f64 {
    if (@fabs(value - edge) <= threshold) {
        return edge;
    }
    return value;
}

fn rasterize2(allocator: Allocator, dimensions: Dimensions2D(u32), vertices: []Vertex, scale: f32) !Bitmap {
    const bitmap_pixel_count = @intCast(usize, dimensions.width) * dimensions.height;
    var bitmap = Bitmap{
        .width = @intCast(u32, dimensions.width),
        .height = @intCast(u32, dimensions.height),
        .pixels = try allocator.alloc(graphics.RGBA(f32), bitmap_pixel_count),
    };
    const null_pixel = graphics.RGBA(f32){ .r = 0.0, .g = 0.0, .b = 0.0, .a = 0.0 };
    std.mem.set(graphics.RGBA(f32), bitmap.pixels, null_pixel);

    // TODO: Wrap in struct with add function
    var points_count: usize = 0;
    var points_buffer: [256]Point(f64) = undefined;

    var scanline_y_index: usize = 0;
    while (scanline_y_index < dimensions.height) : (scanline_y_index += 1) {
        const scanline_y = @intToFloat(f64, scanline_y_index);
        var scanline_y_relative: f64 = 0;
        const sub_scanline_increment: f64 = 0.1;
        while (scanline_y_relative <= 1.0) : (scanline_y_relative += sub_scanline_increment) {
            std.debug.assert(isNormalized(scanline_y_relative));
            const abs_scanline = scanline_y + scanline_y_relative;
            var vertex_i: usize = 1;
            while (vertex_i < vertices.len) : (vertex_i += 1) {
                const previous_vertex = vertices[vertex_i - 1];
                const current_vertex = vertices[vertex_i];
                const kind = @intToEnum(VMove, current_vertex.kind);

                //
                // TODO: Should switch on kind from the beginning
                //

                if (kind == .move) {
                    continue;
                }

                const point_a = Point(f64){
                    .x = @intToFloat(f64, previous_vertex.x) * scale,
                    .y = @intToFloat(f64, previous_vertex.y) * scale,
                };

                const point_b = Point(f64){
                    .x = @intToFloat(f64, current_vertex.x) * scale,
                    .y = @intToFloat(f64, current_vertex.y) * scale,
                };

                const printf = std.debug.print;

                if (kind == .line) {
                    printf("  Vertex (line) {d:^5.2} x {d:^5.2} --> {d:^5.2} x {d:^5.2}\n", .{ point_a.x, point_a.y, point_b.x, point_b.y });
                } else if (kind == .curve) {
                    printf("  Vertex (curve) A({d:.2}, {d:.2}) --> B({d:.2}, {d:.2}) C ({d:.2}, {d:.2})\n", .{
                        point_b.x,
                        point_b.y,
                        point_a.x,
                        point_a.y,
                        @intToFloat(f64, current_vertex.control1_x) * scale,
                        @intToFloat(f64, current_vertex.control1_y) * scale,
                    });
                }

                std.debug.assert(!(point_a.x == point_b.x and point_a.y == point_b.y));

                // TODO: Only valid if it's a line
                const is_horizontal = current_vertex.y == previous_vertex.y;
                if (kind == .line and is_horizontal) {
                    if (point_a.y >= abs_scanline and point_a.y < (abs_scanline + sub_scanline_increment)) {
                        //
                        // This horizontal line lies between current and next sub scanline
                        //
                        const x_start = pointMinX(point_a, point_b);
                        const x_end = pointMaxX(point_a, point_b);
                        const relative_y = point_a.y - scanline_y;
                        std.debug.assert(isNormalized(relative_y));
                        const last_x = @floatToInt(usize, @floor(x_end.x));
                        points_buffer[points_count] = .{ .x = x_start.x, .y = relative_y };
                        points_count += 1;
                        var current_x_pixel = @floatToInt(usize, @floor(x_start.x)) + 1;
                        while (current_x_pixel <= last_x) : (current_x_pixel += 1) {
                            const current_x = @intToFloat(f64, current_x_pixel);
                            points_buffer[points_count] = .{ .x = current_x, .y = relative_y };
                            points_count += 1;
                        }
                        points_buffer[points_count] = .{ .x = x_end.x, .y = relative_y };
                        points_count += 1;

                        std.log.info("Horizontal line", .{});
                        printPoints(points_buffer[0..points_count]);
                    }
                    continue;
                }

                const is_outsize_y_range = blk: {
                    const max_yy = @maximum(point_a.y, point_b.y);
                    const min_yy = @minimum(point_a.y, point_b.y);
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
                    // printf("REJECT - Outsize Y range\n", .{});
                    continue;
                }

                std.debug.assert(current_vertex.y != previous_vertex.y);

                switch (kind) {
                    .line => {
                        const is_vertical = (point_a.x == point_b.x);
                        points_buffer[points_count] = .{
                            .x = if (is_vertical) point_a.x else horizontalPlaneIntersection(abs_scanline, point_a, point_b),
                            .y = scanline_y_relative,
                        };
                        points_count += 1;
                        std.debug.assert(is_vertical == true);

                        std.log.info("Vertical ({}) line", .{is_vertical});
                        printPoints(points_buffer[0..points_count]);
                    },
                    .curve => {
                        const bezier = BezierQuadratic{ .a = point_b, .b = point_a, .control = .{
                            .x = @intToFloat(f64, current_vertex.control1_x) * scale,
                            .y = @intToFloat(f64, current_vertex.control1_y) * scale,
                        } };
                        const optional_intersection_points = quadraticBezierPlaneIntersections(bezier, scanline_y);
                        if (optional_intersection_points[0]) |first_intersection| {
                            points_buffer[points_count] = .{
                                .x = first_intersection.x,
                                .y = scanline_y_relative,
                            };
                            points_count += 1;
                            if (optional_intersection_points[1]) |second_intersection| {
                                const x_diff_threshold = 0.001;
                                if (@fabs(second_intersection.x - first_intersection.x) > x_diff_threshold) {
                                    points_buffer[points_count] = .{
                                        .x = second_intersection.x,
                                        .y = scanline_y_relative,
                                    };
                                    points_count += 1;
                                }
                            }
                        } else if (optional_intersection_points[1]) |second_intersection| {
                            points_buffer[points_count] = .{
                                .x = second_intersection.x,
                                .y = scanline_y_relative,
                            };
                            points_count += 1;
                        }
                    },
                    else => {
                        std.log.warn("Kind: {}", .{kind});
                        continue;
                    },
                }
            }
        }

        printPoints(points_buffer[0..points_count]);

        // Sort all points by x ascending
        var step: usize = 1;
        while (step < points_count) : (step += 1) {
            const key = points_buffer[step];
            var i = @intCast(i64, step) - 1;
            while (i >= 0 and points_buffer[@intCast(usize, i)].x > key.x) : (i -= 1) {
                points_buffer[@intCast(usize, i) + 1] = points_buffer[@intCast(usize, i)];
            }
            points_buffer[@intCast(usize, i + 1)] = key;
        }

        printPoints(points_buffer[0..points_count]);

        {
            //
            // Protect against minor floating point inaccuracies
            //
            points_buffer[0].y = clampTo(points_buffer[0].y, 1.0, 0.00001);
            points_buffer[0].y = clampTo(points_buffer[0].y, 0.0, 0.00001);

            const last_index = points_count - 1;
            points_buffer[last_index].y = clampTo(points_buffer[last_index].y, 1.0, 0.00001);
            points_buffer[last_index].y = clampTo(points_buffer[last_index].y, 0.0, 0.00001);

            // First (leftmost) and last (rightmost) points should be y intersections
            const first = points_buffer[0];
            const last = points_buffer[last_index];

            std.debug.assert(first.y == 1.0 or first.y == 0.0);
            std.debug.assert(last.y == 1.0 or last.y == 0.0);
        }

        //
        // We split our points into seperate fill regions across the y axis and rasterize line segment
        //
        var points_processed_count: usize = 1;
        var intersection_point_count: usize = 0;
        const line_index_y_offset = scanline_y_index * dimensions.width;
        while (points_processed_count < points_count) : (points_processed_count += 1) {
            // Save the y of the first point in the fill region
            // The fill region will end at the same y value
            const start_y_intersection = points_buffer[points_processed_count].y;
            const fill_region_start_index = points_processed_count;
            std.debug.assert(start_y_intersection == 1.0 or start_y_intersection == 0.0);
            points_processed_count += 1;
            while (points_buffer[points_processed_count].y != start_y_intersection) {
                intersection_point_count += 1;
                points_processed_count += 1;
            }

            while (points_buffer[points_processed_count].y == start_y_intersection) {
                intersection_point_count += 1;
                points_processed_count += 1;
            }

            if (points_buffer[points_processed_count].y != start_y_intersection) {
                points_processed_count -= 1;
                intersection_point_count -= 1;
            }

            std.debug.assert(intersection_point_count > 1);

            var line = bitmap.pixels[line_index_y_offset .. line_index_y_offset + (dimensions.width - 1)];
            const current_point = points_buffer[points_processed_count];
            std.log.info("Current point: {d}, {d}", .{ current_point.x, current_point.y });

            std.debug.assert(current_point.y == 1.0 or current_point.y == 0.0);

            const intersection_points_buffer = points_buffer[fill_region_start_index .. points_processed_count + 1];

            std.debug.assert(current_point.y == intersection_points_buffer[intersection_points_buffer.len - 1].y);
            std.debug.assert(current_point.x == intersection_points_buffer[intersection_points_buffer.len - 1].x);

            std.debug.assert(intersection_points_buffer.len > 1);

            printPoints(intersection_points_buffer);

            rasterizeLineSegment(line, intersection_points_buffer);
            intersection_point_count = 0;
        }
        points_count = 0;
    }
    return bitmap;
}

fn rasterize(allocator: Allocator, dimensions: Dimensions2D(u32), vertices: []Vertex, scale: f32) !Bitmap {
    std.log.info("Glyph bbox {d} x {d} -- {d} x {d}", .{
        min_x,
        min_y,
        max_x,
        max_y,
    });
    const bitmap_pixel_count = @intCast(usize, dimensions.width) * dimensions.height;
    var bitmap = Bitmap{
        .width = @intCast(u32, dimensions.width),
        .height = @intCast(u32, dimensions.height),
        .pixels = try allocator.alloc(graphics.RGBA(f32), bitmap_pixel_count),
    };
    const null_pixel = graphics.RGBA(f32){
        .r = 0.0,
        .g = 0.0,
        .b = 0.0,
        .a = 0.0,
    };
    std.mem.set(graphics.RGBA(f32), bitmap.pixels, null_pixel);

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
                const max_yy = @maximum(point_a.y, point_b.y);
                const min_yy = @minimum(point_a.y, point_b.y);
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
                    //
                    // How do we
                    //
                    const bezier = BezierQuadratic{ .a = point_b, .b = point_a, .control = .{
                        .x = @intToFloat(f64, current_vertex.control1_x) * scale,
                        .y = @intToFloat(f64, current_vertex.control1_y) * scale,
                    } };
                    const optional_intersection_points = quadraticBezierPlaneIntersections(bezier, scanline_y);
                    if (optional_intersection_points[0]) |first_intersection| {
                        try current_scanline.add(.{ .x_position = first_intersection.x, .t = first_intersection.t, .outline_index = vertex_i }, "curve 1");
                        if (optional_intersection_points[1]) |second_intersection| {
                            const x_diff_threshold = 0.001;
                            if (@fabs(second_intersection.x - first_intersection.x) > x_diff_threshold) {
                                try current_scanline.add(.{ .x_position = second_intersection.x, .t = second_intersection.t, .outline_index = vertex_i }, "curve 2");
                            }
                        }
                    } else if (optional_intersection_points[1]) |second_intersection| {
                        try current_scanline.add(.{
                            .x_position = second_intersection.x,
                            .t = second_intersection.t,
                            .outline_index = vertex_i,
                        }, "curve 2 only");
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

            // std.debug.print("Intersection pair:\n", .{});
            // std.debug.print("  V {d} Curve {} P1 ({d:.2}, {d:.2}) -> P2 ({d:.2}, {d:.2}): intersection {d}\n", .{
            //     scanline_start.outline_index,
            //     scanline_start.isCurve(),
            //     line_start.left.x,
            //     line_start.left.y,
            //     line_start.right.x,
            //     line_start.right.y,
            //     scanline_start.x_position,
            // });
            // std.debug.print("  V {d} Curve {} P1 ({d:.2}, {d:.2}) -> P2 ({d:.2}, {d:.2}): intersection {d}\n", .{
            //     scanline_end.outline_index,
            //     scanline_end.isCurve(),
            //     line_end.left.x,
            //     line_end.left.y,
            //     line_end.right.x,
            //     line_end.right.y,
            //     scanline_end.x_position,
            // });

            // TODO: Check here if there is a horizonal line between the start
            //       and end intersections. Do the fill and return

            std.debug.assert(scanline_start.x_position <= scanline_end.x_position);
            std.log.info("Intersection {d}: {d} -- {d}", .{ i, scanline_start.x_position, scanline_end.x_position });

            std.log.info("Doing leading anti-aliasing", .{});
            var full_fill_index_start: usize = std.math.maxInt(usize);
            {
                //
                // Do leading anti-aliasing
                //
                const start_x = @floatToInt(usize, @floor(scanline_start.x_position));
                var entry_intersection = Point(f64){
                    .y = 0.0,
                    .x = @rem(scanline_start.x_position, 1.0),
                };

                if (scanline_start.isCurve()) {

                    // I could get the intersection at next y

                    //
                    // Do leading AA with only an intersection point and a connecting curve
                    // This is a non-optimized catch-all case for rasterizing a curve contour
                    //
                    var point_buffer: [500]Point(f64) = undefined;

                    for (point_buffer) |*point| {
                        point.* = Point(f64){
                            .x = 99999,
                            .y = 99999,
                        };
                    }

                    var current_x_pixel = start_x;
                    const scanline_end_x_pixel = @floatToInt(usize, @floor(scanline_end.x_position));
                    // TODO: Previous point is a more precise name
                    var last_point_rel = Point(f64){
                        .x = scanline_start.x_position - @intToFloat(f64, current_x_pixel),
                        .y = 0.0,
                    };

                    point_buffer[0] = last_point_rel;
                    var point_count: usize = 1;

                    var contour_index: usize = scanline_start.outline_index;
                    const last_contour_index = scanline_end.outline_index;
                    const last_t_opt = scanline_end.t;
                    // TODO:
                    std.debug.assert(contour_index > 0);
                    const sv1 = vertices[contour_index];
                    const sv2 = vertices[contour_index - 1];
                    var bezier = BezierQuadratic{ .a = .{ .x = @intToFloat(f64, sv2.x) * scale, .y = @intToFloat(f64, sv2.y) * scale }, .b = .{ .x = @intToFloat(f64, sv1.x) * scale, .y = @intToFloat(f64, sv1.y) * scale }, .control = .{
                        .x = @intToFloat(f64, sv1.control1_x) * scale,
                        .y = @intToFloat(f64, sv1.control1_y) * scale,
                    } };
                    const t_start = scanline_start.t.?;
                    var t_increment: f64 = 0.005;

                    if (quadraticBezierPoint(bezier, t_start + t_increment).y < scanline_y) {
                        std.debug.assert(false);
                    }

                    var current_t = t_start + t_increment;
                    var is_line: bool = false;

                    var itr: usize = 0;
                    loop_leading_aa: while (itr < 1000) : (itr += 1) {
                        const sampled_point: Point(f64) = blk: {
                            if (!is_line) {
                                break :blk quadraticBezierPoint(bezier, current_t);
                            }

                            std.debug.assert(false);

                            const previous_contour = if (contour_index == 0) vertices.len - 1 else contour_index - 1;
                            const v1 = vertices[previous_contour];
                            const v2 = vertices[contour_index];
                            const start_point = Point(f64){
                                .x = @intToFloat(f32, v1.x) * scale,
                                .y = @intToFloat(f32, v1.y) * scale,
                            };
                            const end_point = Point(f64){
                                .x = @intToFloat(f32, v2.x) * scale,
                                .y = @intToFloat(f32, v2.y) * scale,
                            };
                            const base_x = @intToFloat(f64, current_x_pixel);
                            const base_y = scanline_y;
                            if (start_point.x == end_point.x) {
                                std.log.info("line: vertical", .{});
                                const x = base_x + start_point.x;
                                break :blk if (start_point.y > end_point.y) Point(f64){ .x = x, .y = base_y } else Point(f64){ .x = x, .y = base_y + 1.0 };
                            }
                            const x_per_y = ((start_point.y - end_point.y) / (start_point.x - end_point.x));
                            std.debug.assert(x_per_y != 0);
                            const relative_exit_point = pixelLineIntersection(x_per_y, last_point_rel);

                            std.log.info("Sampled line x {d} -> {d}", .{ relative_exit_point.x, relative_exit_point.x + @intToFloat(f32, current_x_pixel) });

                            break :blk .{
                                .x = relative_exit_point.x + @intToFloat(f32, current_x_pixel),
                                .y = relative_exit_point.y + scanline_y,
                            };
                        };

                        std.log.info("Sampled point {d}, {d}", .{ sampled_point.x, sampled_point.y });

                        const sampled_pixel_x = @floatToInt(usize, @floor(sampled_point.x));
                        var sampled_point_rel = Point(f64){
                            .x = sampled_point.x - @intToFloat(f64, current_x_pixel),
                            .y = sampled_point.y - scanline_y,
                        };
                        const sampled_pixel_y = @floatToInt(u32, @floor(sampled_point.y));

                        // Sampled pixel is above scanline (End Anti-aliasing condition)
                        if (sampled_pixel_y > scanline_y_index) {
                            const change_in_y = sampled_point_rel.y - last_point_rel.y;
                            const change_in_x = sampled_point_rel.x - last_point_rel.x;
                            const remaining_y = 1.0 - last_point_rel.y;
                            const y_percentage = remaining_y / change_in_y;
                            std.debug.assert(remaining_y <= change_in_y);
                            const interpolated_between_point = Point(f64){
                                .x = clamp(last_point_rel.x + (change_in_x * y_percentage), 0.0 + std.math.floatMin(f64), 1.0 - std.math.floatMin(f64)),
                                .y = 1.0,
                            };
                            std.debug.assert(interpolated_between_point.x <= 1.0);
                            std.debug.assert(interpolated_between_point.x >= 0.0);
                            point_buffer[point_count] = interpolated_between_point;
                            const coverage = pixelCurveCoverage(point_buffer[0 .. point_count + 1]);
                            std.log.info("Reached top of scanline. Interp point {d} {d}. Setting {d} -> {d:.4}", .{
                                interpolated_between_point.x,
                                interpolated_between_point.y,
                                current_x_pixel,
                                coverage,
                            });
                            bitmap.pixels[current_x_pixel + (image_y_index * dimensions.width)] = pixelCurve(coverage);
                            break;
                        }

                        // Sampled pixel is below scanline (End Anti-aliasing condition)
                        if (sampled_pixel_y != scanline_y_index) {
                            std.log.info("Reached bottom of scanline. Last {d}, {d}. Sampled {d}, {d}", .{
                                last_point_rel.x,
                                last_point_rel.y,
                                sampled_point_rel.x,
                                sampled_point_rel.y,
                            });
                            // Not need to interpolate a new point
                            if (last_point_rel.y == 0.0) {
                                const coverage = pixelCurveCoverage(point_buffer[0..point_count]);
                                bitmap.pixels[current_x_pixel + (image_y_index * dimensions.width)] = pixelCurve(coverage);
                                break;
                            }
                            const change_in_y = last_point_rel.y - sampled_point_rel.y;
                            const change_in_x = sampled_point_rel.x - last_point_rel.x;
                            const remaining_y = last_point_rel.y;
                            const y_percentage = remaining_y / change_in_y;
                            std.debug.assert(remaining_y <= change_in_y);
                            std.debug.assert(change_in_x != 0.0);
                            std.debug.assert(y_percentage != 0.0);
                            const interpolated_between_point = Point(f64){
                                .x = last_point_rel.x + (change_in_x * y_percentage),
                                .y = 0.0,
                            };
                            std.debug.assert(interpolated_between_point.x <= 1.0);
                            std.debug.assert(interpolated_between_point.x >= 0.0);
                            std.debug.assert(interpolated_between_point.x != point_buffer[0].x or interpolated_between_point.y != point_buffer[0].y);
                            point_buffer[point_count] = interpolated_between_point;
                            const coverage = pixelCurveCoverage(point_buffer[0 .. point_count + 1]);
                            std.log.info("Reached bottom of scanline. Interp point {d} {d}. Setting {d} -> {d:.4}", .{
                                interpolated_between_point.x,
                                interpolated_between_point.y,
                                current_x_pixel,
                                coverage,
                            });
                            bitmap.pixels[current_x_pixel + (image_y_index * dimensions.width)] = pixelCurve(coverage);
                            break;
                        }

                        // Advance to right pixel
                        if (sampled_pixel_x > current_x_pixel) {
                            std.debug.assert(sampled_pixel_x == (current_x_pixel + 1));
                            std.debug.assert(sampled_pixel_x <= scanline_end_x_pixel);
                            // Interpolate point on the right x boundry, insert and calc coverage for current pixel
                            const change_in_y: f64 = sampled_point_rel.y - last_point_rel.y;
                            const remaining_x: f64 = 1.0 - last_point_rel.x;
                            const change_in_x: f64 = sampled_point_rel.x - last_point_rel.x;
                            std.debug.assert(change_in_x > 0.0);
                            const x_percentage: f64 = remaining_x / change_in_x;
                            std.debug.assert(x_percentage <= 1.0);
                            std.debug.assert(x_percentage >= 0.0);
                            const interpolated_between_point = Point(f64){
                                .x = 1.0,
                                .y = last_point_rel.y + (change_in_y * x_percentage),
                            };
                            point_buffer[point_count] = interpolated_between_point;
                            const coverage = pixelCurveCoverage(point_buffer[0 .. point_count + 1]);
                            std.log.info("Sampled pixel to right. Interp point {d} {d}. Setting {d} -> {d:.4}", .{
                                interpolated_between_point.x,
                                interpolated_between_point.y,
                                current_x_pixel,
                                coverage,
                            });
                            bitmap.pixels[current_x_pixel + (image_y_index * dimensions.width)] = pixelCurve(coverage);

                            // Reset points buffer, our sampled point will be added as the first element below
                            point_buffer[0] = Point(f64){ .x = 0.0, .y = interpolated_between_point.y };
                            point_count = 1;
                            current_x_pixel = sampled_pixel_x;
                            sampled_point_rel.x -= 1.0;
                        }

                        // Advance to left pixel
                        if (sampled_pixel_x < current_x_pixel) {
                            std.debug.assert(sampled_pixel_x == (current_x_pixel - 1));
                            std.debug.assert(sampled_pixel_x >= 0);
                            std.debug.assert(sampled_point_rel.x < 0);
                            const change_in_y: f64 = sampled_point_rel.y - last_point_rel.y;
                            const remaining_x: f64 = last_point_rel.x;
                            const change_in_x: f64 = last_point_rel.x - sampled_point_rel.x;
                            std.debug.assert(change_in_x > 0.0);
                            const x_percentage: f64 = remaining_x / change_in_x;
                            std.debug.assert(x_percentage <= 1.0);
                            std.debug.assert(x_percentage >= 0.0);
                            const interpolated_between_point = Point(f64){
                                .x = 0.0,
                                .y = last_point_rel.y + (change_in_y * x_percentage),
                            };

                            point_buffer[point_count] = interpolated_between_point;
                            const coverage = pixelCurveCoverage(point_buffer[0 .. point_count + 1]);
                            std.log.info("Sampled pixel to left. Interp point {d} {d}. Coverage: Setting {d} -> {d:.4}", .{
                                interpolated_between_point.x,
                                interpolated_between_point.y,
                                current_x_pixel,
                                coverage,
                            });
                            bitmap.pixels[current_x_pixel + (image_y_index * dimensions.width)] = pixelCurve(coverage);
                            point_buffer[0] = Point(f64){ .x = 1.0, .y = interpolated_between_point.y };
                            point_count = 1;
                            current_x_pixel = sampled_pixel_x;
                            sampled_point_rel.x += 1.0;
                        }

                        // std.log.info("Sample {d:.2} {d:.4} {d:.4}", .{point_count, sampled_point_rel.x, sampled_point_rel.y});

                        std.debug.assert(sampled_point_rel.x <= 1.0);
                        std.debug.assert(sampled_point_rel.x >= 0.0);
                        std.debug.assert(sampled_point_rel.y <= 1.0);
                        std.debug.assert(sampled_point_rel.y >= 0.0);

                        point_buffer[point_count] = sampled_point_rel;
                        point_count += 1;
                        current_t += t_increment;

                        last_point_rel = sampled_point_rel;

                        // T out of bounds, clip to final valid and continue
                        if (last_t_opt) |last_t| {
                            if (contour_index == last_contour_index and current_t > last_t) {
                                if (current_t >= (last_t + t_increment)) {
                                    break;
                                }
                                current_t = last_t;
                                continue;
                            }
                        }

                        if (current_t > (1.0 - t_increment)) {
                            current_t = 0;
                            contour_index = (contour_index + 1) % @intCast(u32, vertices.len);
                            // TODO: Handle gone past end contour index
                            const previous_contour = if (contour_index == 0) vertices.len - 1 else contour_index - 1;
                            // TODO: v1 and v2 are confusing and error prone
                            const v1 = vertices[contour_index];
                            if (@intToEnum(VMove, v1.kind) == .move) {
                                std.debug.assert(false);
                                break;
                            }
                            if (@intToEnum(VMove, v1.kind) == .line) {
                                std.log.info("Switching to line contour", .{});

                                const v2 = vertices[contour_index];
                                const start_point = Point(f64){
                                    .x = @intToFloat(f32, v2.x) * scale,
                                    .y = @intToFloat(f32, v2.y) * scale,
                                };
                                const end_point = Point(f64){
                                    .x = @intToFloat(f32, v1.x) * scale,
                                    .y = @intToFloat(f32, v1.y) * scale,
                                };

                                //
                                // If line is horizontal, we can quickly rasterize the entire contour
                                // and proceed to the next one if necessary
                                //
                                if (start_point.y == end_point.y) {
                                    const left_to_right = start_point.x > end_point.x;
                                    const line_end_pixel = @floatToInt(i64, @floor(end_point.x));
                                    // If the previous point is above the horizontal line, the top part is to be filled
                                    const coverage = if (last_point_rel.y > start_point.y) 1.0 - start_point.y else start_point.y;
                                    while (true) {
                                        bitmap.pixels[current_x_pixel + (image_y_index * dimensions.width)] = pixelFill(coverage);
                                        if (current_x_pixel == line_end_pixel) {
                                            contour_index = (contour_index + 1) % @intCast(u32, vertices.len);
                                            const next_end_point = Point(f64){
                                                .x = @intToFloat(f32, vertices[contour_index].x) * scale,
                                                .y = @intToFloat(f32, vertices[contour_index].y) * scale,
                                            };
                                            if (next_end_point.y < start_point.y) {
                                                // Next contour trends downwards, therefore not our problem.
                                            } else {
                                                // TODO: Cry and/or implement
                                                std.debug.assert(false);
                                            }
                                            break;
                                        }
                                        current_x_pixel = if (left_to_right) current_x_pixel + 1 else current_x_pixel - 1;
                                    }
                                }

                                // To complete the coverage for this pixel, we're going to take 4 points
                                // and calculate the area of the resulting quadrilateral
                                // 1. Curve entry (point[0])
                                // 2. Line exit
                                // 3. Connection point
                                // 4. Corner point

                                const exit_point: Point(f64) = blk: {
                                    if (start_point.y == end_point.y) {
                                        const y = start_point.y;
                                        break :blk if (start_point.x > end_point.x) Point(f64){ .x = 0.0, .y = y } else Point(f64){ .x = 1.0, .y = y };
                                    }
                                    if (start_point.x == end_point.x) {
                                        const x = start_point.x;
                                        break :blk if (start_point.y > end_point.y) Point(f64){ .x = x, .y = 0.0 } else Point(f64){ .x = x, .y = 1.0 };
                                    }
                                    const x_per_y = ((start_point.y - end_point.y) / (start_point.x - end_point.x));
                                    std.debug.assert(x_per_y != 0);
                                    break :blk pixelLineIntersection(x_per_y, last_point_rel);
                                };

                                const connection_point = start_point;
                                const entry_side = sideOfPixel(point_buffer[0]);
                                const exit_side = sideOfPixel(exit_point);

                                var coverage: f64 = 0.0;

                                switch (entry_side) {
                                    .left => {
                                        switch (exit_side) {
                                            .left => {
                                                coverage = triangleArea(point_buffer[0], exit_point, connection_point);
                                            },
                                            .right => {
                                                const bottom_left = Point(f64){ .x = 0.0, .y = 0.0 };
                                                const bottom_right = Point(f64){ .x = 1.0, .y = 0.0 };
                                                coverage = triangleArea(point_buffer[0], connection_point, bottom_left);
                                                coverage += triangleArea(connection_point, bottom_right, bottom_left);
                                                coverage += triangleArea(connection_point, exit_point, bottom_right);
                                                const fill_below: bool = (point_buffer[0].y < connection_point.y);
                                                if (!fill_below) {
                                                    coverage = 1.0 - coverage;
                                                }
                                            },
                                            .top => {
                                                coverage = 0.5;
                                            },
                                            .bottom => {
                                                coverage = 0.5;
                                            },
                                        }
                                    },
                                    .right => {
                                        switch (exit_side) {
                                            .left => {
                                                coverage = 0.5;
                                            },
                                            .right => {
                                                coverage = 0.5;
                                            },
                                            .top => {
                                                coverage = 0.5;
                                            },
                                            .bottom => {
                                                coverage = 0.5;
                                            },
                                        }
                                    },
                                    .top => {
                                        switch (exit_side) {
                                            .left => {
                                                coverage = 0.5;
                                            },
                                            .right => {
                                                coverage = 0.5;
                                            },
                                            .top => {
                                                coverage = 0.5;
                                            },
                                            .bottom => {
                                                coverage = 0.5;
                                            },
                                        }
                                    },
                                    .bottom => {
                                        switch (exit_side) {
                                            .left => {
                                                coverage = 0.5;
                                            },
                                            .right => {
                                                coverage = 0.5;
                                            },
                                            .top => {
                                                coverage = 0.5;
                                            },
                                            .bottom => {
                                                coverage = 0.5;
                                            },
                                        }
                                    },
                                }

                                bitmap.pixels[current_x_pixel + (image_y_index * dimensions.width)] = pixelFill(coverage);

                                switch (exit_side) {
                                    .left => {
                                        current_x_pixel -= 1;
                                        point_buffer[0] = .{
                                            .x = 1.0,
                                            .y = exit_point.y,
                                        };
                                    },
                                    .right => {
                                        current_x_pixel += 1;
                                        point_buffer[0] = .{
                                            .x = 0.0,
                                            .y = exit_point.y,
                                        };
                                    },
                                    .top => break :loop_leading_aa,
                                    .bottom => break :loop_leading_aa,
                                }

                                last_point_rel = point_buffer[0];
                                point_count = 1;

                                // const slope: f64 = 2.0;
                                // const line_start_point = Point(f64) {
                                //     .x = v1.x * scale,
                                //     .y = v1.y * scale,
                                // };
                                // How to calculate coverage for both line and curve within pixel..
                                // IDEA! You can scale the curve samples so that they fit into an imaginary
                                // pixel and do coverage as normal. Then scale down the coverage, calculate
                                // the area for the line segment and add
                                // It's a massive pain, but technically you could calculate how much the line
                                // intersects with the imaginary pixel, and subtract it.
                                // For now.. You can add an assert I think. Very unlikely to have that shape
                                // Hmm. Maybe you should think about what kind of crazy bezier curves could technically
                                // be within a pixel. Almost a circle.
                                // const exit_rel = pixelLineIntersection(slope, line_start_point);
                                // Need to be able to calculate the bounding box of the bezier at a t range
                                // Then to cut the pixel, you need to be able move one side inwards so that it
                                // can intersect with the endpoint without going inside the bounding box
                                // If you cut 0.5 off say horizontally, multiply all x values by 2
                                // You can actually do the same for the line, seeing as it helps to view things
                                // as entry and exit intersections to a pixel

                                // Also.. To see how the rest of the code fares, you can just set the pixel to a random
                                // value and break.

                                // Also x2.. Modify this function to return a colored image, that way you can use it to debug
                                // what parts of the code are responsable for drawing what parts..
                                is_line = true;
                                // continue;
                                continue;
                            }
                            is_line = false;
                            const v2 = vertices[previous_contour];
                            bezier = BezierQuadratic{ .a = .{ .x = @intToFloat(f64, v2.x) * scale, .y = @intToFloat(f64, v2.y) * scale }, .b = .{ .x = @intToFloat(f64, v1.x) * scale, .y = @intToFloat(f64, v1.y) * scale }, .control = .{
                                .x = @intToFloat(f64, v1.control1_x) * scale,
                                .y = @intToFloat(f64, v1.control1_y) * scale,
                            } };
                        }
                    }
                    const start_x_pixel = @floatToInt(usize, @floor(scanline_start.x_position));
                    full_fill_index_start = if (current_x_pixel >= start_x_pixel) (current_x_pixel + 1) else start_x_pixel + 1;
                    if (full_fill_index_start > scanline_end_x_pixel) {
                        full_fill_index_start = scanline_end_x_pixel;
                    }
                } else if (line_start.isVertical()) {
                    bitmap.pixels[start_x + (image_y_index * dimensions.width)] = pixelFill(1.0 - @rem(scanline_start.x_position, 1.0));
                    full_fill_index_start = start_x + 1;
                } else {
                    //
                    // Non-vertical line
                    //
                    const is_positive: bool = line_start.left.y < line_start.right.y;
                    // std.log.info("Leading AA iteration. is_positive {}", .{is_positive});
                    const end_x: i64 = if (is_positive) @floatToInt(u32, @floor(scanline_end.x_position)) else -1;
                    const increment: i32 = if (is_positive) 1 else -1;
                    const left = line_start.left;
                    const right = line_start.right;
                    var current_x = @intCast(i64, start_x);
                    while (current_x != end_x) : (current_x += increment) {
                        std.debug.assert(current_x >= 0);
                        // std.log.info("Pixel coverage iteration. ENTRY {d:.2} {d:.2}", .{
                        //     entry_intersection.x, entry_intersection.y,
                        // });
                        const exit_intersection = blk: {
                            const slope = @fabs((left.y - right.y) / (left.x - right.x));
                            const exit_height = if (is_positive) ((1.0 - entry_intersection.x) * slope) else (entry_intersection.x * slope);
                            // std.log.info("Exit height: {d}", .{exit_height});
                            // NOTE: Working around a stage 1 compiler bug
                            //       A break inside the initial if statement triggers a broken
                            //       LLVM module error
                            var x: f64 = 0.0;
                            var y: f64 = 0.0;
                            const remaining_y = 1.0 - entry_intersection.y;
                            if (exit_height > remaining_y) {
                                const y_per_x = remaining_y / slope;
                                const result = if (is_positive) entry_intersection.x + y_per_x else entry_intersection.x - y_per_x;
                                // std.log.info("Calculated exit x: {d}", .{result});
                                std.debug.assert(result <= 1.0);
                                std.debug.assert(result >= 0.0);
                                x = result;
                                y = 1.0;
                            } else {
                                x = if (is_positive) 1.0 else 0.0;
                                y = exit_height;
                            }
                            if (x != entry_intersection.x) {
                                const new_slope = @fabs((y - entry_intersection.y) / (x - entry_intersection.x));
                                // std.log.info("Old slope {d} New {d}", .{ slope, new_slope });
                                std.debug.assert(@fabs(slope - new_slope) < 0.0001);
                            }
                            break :blk Point(f64){
                                .x = x,
                                .y = y,
                            };
                        };
                        const coverage = calculateCoverage(entry_intersection, exit_intersection, is_positive);
                        const x_pos = @intCast(usize, current_x);
                        bitmap.pixels[x_pos + (image_y_index * dimensions.width)] = pixelFill(coverage); //@floatToInt(u8, 255.0 * coverage);

                        // std.log.info("Current: {d} end {d}", .{ current_x, end_x });

                        if (((current_x + increment) == end_x) or exit_intersection.y == 1.0) {
                            // std.log.info("full_fill_index_start set", .{});
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

            // std.log.info("Doing trailing anti-aliasing", .{});
            var full_fill_index_end: usize = 0;
            {
                //
                // Do trailing anti-aliasing
                //
                const start_x = @floatToInt(u32, @floor(scanline_end.x_position));
                std.log.info("Start_x for end AA {d}", .{start_x});
                if (scanline_end.isCurve()) {
                    // std.log.info("  Curve found", .{});
                    // const coverage = @floatToInt(u8, (@rem(scanline_end.x_position, 1.0)) * 255);
                    bitmap.pixels[start_x + (image_y_index * dimensions.width)] = pixelFill(@rem(scanline_end.x_position, 1.0)); //coverage;
                    full_fill_index_end = if (start_x == 0) 0 else start_x - 1;
                } else if (line_end.isVertical()) {
                    // std.log.info("  Vertical found", .{});
                    // const coverage = @floatToInt(u8, (@rem(scanline_end.x_position, 1.0)) * 255);
                    bitmap.pixels[start_x + (image_y_index * dimensions.width)] = pixelFill(@rem(scanline_end.x_position, 1.0)); //coverage;
                    full_fill_index_end = if (start_x == 0) 0 else start_x - 1;
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
                            const m = @fabs((left.y - right.y) / (left.x - right.x));
                            const exit_height = if (direction == .positive) ((1.0 - entry_intersection.x) * m) else (entry_intersection.x * m);
                            // NOTE: Working around a stage 1 compiler bug
                            //       A break inside the initial if statement triggers a broken
                            //       LLVM module error
                            var x: f64 = 0.0;
                            var y: f64 = 0.0;
                            if (exit_height > (1.0 - entry_intersection.y)) {
                                const run = (1.0 - entry_intersection.y) / m;
                                const result = if (direction == .positive) entry_intersection.x + run else entry_intersection.x - run;
                                // std.log.info("Calculated exit x: {d}", .{result});
                                std.debug.assert(result <= 1.0);
                                std.debug.assert(result >= 0.0);
                                x = result;
                                y = 1.0;
                            } else {
                                x = if (direction == .positive) 1.0 else 0.0;
                                y = exit_height;
                            }

                            const new_slope = @fabs((y - entry_intersection.y) / (x - entry_intersection.x));
                            // std.log.info("Old slope {d} New {d}", .{ m, new_slope });
                            std.debug.assert(@fabs(m - new_slope) < 0.0001);

                            break :blk Point(f64){
                                .x = x,
                                .y = y,
                            };
                        };
                        const coverage = calculateCoverage(entry_intersection, exit_intersection, is_positive);
                        bitmap.pixels[@intCast(usize, current_x) + (image_y_index * dimensions.width)] = pixelFill(1.0 - coverage); // @floatToInt(u8, 255.0 * (1.0 - coverage));

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

            std.debug.assert(full_fill_index_start >= @floatToInt(u32, @floor(scanline_start.x_position)));
            // std.debug.assert(full_fill_index_start <= @floatToInt(u32, @floor(scanline_end.x_position)));

            std.debug.assert(full_fill_index_end <= @floatToInt(u32, @floor(scanline_end.x_position)));
            // std.debug.assert(full_fill_index_end >= @floatToInt(u32, @floor(scanline_start.x_position)));

            std.debug.assert(full_fill_index_start >= 0);
            std.debug.assert(full_fill_index_end < dimensions.width);

            //
            // Fill all pixels between aliased zones
            //

            var current_x = full_fill_index_start;
            while (current_x <= full_fill_index_end) : (current_x += 1) {
                bitmap.pixels[current_x + (image_y_index * dimensions.width)] = pixelFill(1.0); // 255;
            }
        }
        const temp_buffer = previous_scanline.buffer;
        previous_scanline = current_scanline;
        current_scanline.buffer = temp_buffer;
        current_scanline.reset();
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

    std.debug.assert(font_info.index_to_loc_format == 0 or font_info.index_to_loc_format == 1);

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

    font_info.index_to_loc_format = bigToNative(u16, @intToPtr(*u16, @ptrToInt(data.ptr) + font_info.head.offset + 50).*);

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
