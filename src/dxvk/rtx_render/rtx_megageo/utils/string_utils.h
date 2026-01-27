/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include "../subdivision/subdivision_surface.h"

// Fast string parsing helpers for OBJ file loading

inline bool IsWhiteSpace(char c)
{
    return ((c == ' ') || (c == '\t') || (c == '\r'));
}

inline bool IsNewLine(char c) { return c == '\n'; }

inline bool IsDigit(char c) { return ((c >= '0') && (c <= '9')); }

inline bool IsExponent(char c) { return ((c == 'e') || (c == 'E')); }

inline char const* SkipWhiteSpace(char const* ptr)
{
    while (IsWhiteSpace(*ptr))
        ++ptr;
    return ptr;
}

inline char const* SkipLine(char const* ptr)
{
    while (!IsNewLine(*ptr++))
        ;
    return ptr;
}

inline char const* SkipWord(char const* ptr)
{
    while ((!IsWhiteSpace(*ptr)) && (*ptr != '\n') && (*ptr != '\0'))
        ++ptr;
    return ptr;
}

inline char const* SkipWords(char const* ptr, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        ptr = SkipWord(ptr);
        ptr = SkipWhiteSpace(ptr);
    }
    return ptr;
}

inline char const* FindSubstring(char const* str, char const* substr,
    uint32_t num)
{
    while ((*str != '\0') && (num >= 0))
    {
        if (!std::memcmp(str, substr, num))
            return str;
        ++str;
    }
    return nullptr;
}

char const* ParseInt(char const* ptr, int* value);

char const* ParseDouble(char const* ptr, double* value);
char const* ParseFloat(char const* ptr, float* value);

char const* ParseString(char const* ptr, std::string* value);

std::string ReadASCIIFile(char const* filepath);
char const* sgets(char* s, int size, char** stream);

std::unique_ptr<uint8_t[]> ReadBigFile(std::filesystem::path const& filepath,
    uint64_t* size = nullptr);

std::istream& operator>>(std::istream& is, Vector3& v);
std::ostream& operator<<(std::ostream& os, const Vector3& v);
std::ostream& operator<<(std::ostream& os, const box3& v);
