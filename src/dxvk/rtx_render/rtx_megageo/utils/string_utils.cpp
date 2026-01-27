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

#include "string_utils.h"
#include "../../../util/log/log.h"

#include <fstream>
#include <sstream>
#include <chrono>

namespace fs = std::filesystem;

char const* ParseInt(char const* ptr, int* value)
{
    ptr = SkipWhiteSpace(ptr);

    int sign = 1;
    if (*ptr == '-')
    {
        sign = -1;
        ++ptr;
    }

    int num = 0;
    while (IsDigit(*ptr))
        num = 10 * num + (*ptr++ - '0');

    *value = sign * num;
    return ptr;
}

char const* ParseDouble(char const* ptr, double* value)
{
    static double const kPowersPos[] = {
        1.0e0,  1.0e1,  1.0e2,  1.0e3,  1.0e4,  1.0e5,  1.0e6,
        1.0e7,  1.0e8,  1.0e9,  1.0e10, 1.0e11, 1.0e12, 1.0e13,
        1.0e14, 1.0e15, 1.0e16, 1.0e17, 1.0e18, 1.0e19,
    };

    static double const kPowersNeg[] = {
        1.0e0,   1.0e-1,  1.0e-2,  1.0e-3,  1.0e-4,  1.0e-5,  1.0e-6,
        1.0e-7,  1.0e-8,  1.0e-9,  1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13,
        1.0e-14, 1.0e-15, 1.0e-16, 1.0e-17, 1.0e-18, 1.0e-19,
    };
    static constexpr uint8_t npowers = sizeof(kPowersPos) / sizeof(double);

    double sign = 1.0;

    ptr = SkipWhiteSpace(ptr);

    if (*ptr == '-')
    {
        sign = -1;
        ++ptr;
    }
    else if (*ptr == '+')
    {
        ++ptr;
    }

    double num = 0.0;
    while (IsDigit(*ptr))
        num = 10.0 * num + (double)(*ptr++ - '0');

    if (*ptr == '.')
        ++ptr;

    double frac = 0.0, div = 1.0;
    while (IsDigit(*ptr))
    {
        frac = 10.0 * frac + (double)(*ptr++ - '0');
        div *= 10.0;
    }
    num += frac / div;

    if (IsExponent(*ptr))
    {
        ptr++;
        double const* powers = nullptr;
        if (*ptr == '+')
        {
            powers = kPowersPos;
            ++ptr;
        }
        else if (*ptr == '-')
        {
            powers = kPowersNeg;
            ++ptr;
        }
        else
        {
            powers = kPowersPos;
        }

        int e = 0;
        while (IsDigit(*ptr))
            e = 10 * e + (*ptr++ - '0');

        num *= (e >= npowers) ? 0.0 : powers[e];
    }

    *value = sign * num;

    return ptr;
}

char const* ParseFloat(char const* ptr, float* value)
{
    double d;
    ptr = ParseDouble(ptr, &d);
    *value = static_cast<float>(d);
    return ptr;
}

std::unique_ptr<uint8_t[]> ReadBigFile(fs::path const& filepath,
    uint64_t* size)
{
    if (std::ifstream file(filepath, std::ios::binary); file.is_open())
    {
        auto tstart = std::chrono::steady_clock::now();

        std::streampos start = file.tellg();
        file.seekg(0, std::ios::end);
        std::streampos end = file.tellg();
        file.seekg(0, std::ios::beg);
        uint64_t length = end - start;

        if (length > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
            return nullptr;

        std::unique_ptr<uint8_t[]> data = std::make_unique<uint8_t[]>(length + 1);
        file.read((char*)data.get(), length);

        data[length] = '\0';

        if (size)
            *size = length;

        {
            auto tstop = std::chrono::steady_clock::now();
            std::chrono::duration<float> elapsed = tstop - tstart;
            dxvk::Logger::info(dxvk::str::format("Read (", elapsed.count(), " seconds) '", filepath.generic_string(), "'"));
        }

        if (file.good())
            return std::move(data);
    }
    dxvk::Logger::err(dxvk::str::format("Error reading '", filepath.generic_string(), "'"));
    return nullptr;
}

char const* ParseString(char const* ptr, std::string* value)
{
    char const* wordEnd = SkipWord(ptr);
    *value = std::string(ptr, wordEnd);
    return wordEnd;
}

std::string ReadASCIIFile(char const* filepath)
{
    std::ifstream ifs(filepath);

    if (!ifs)
        throw std::runtime_error(std::string("Cannot find: ") + filepath);

    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();

    std::string s = ss.str();
    if (s.empty())
        throw std::runtime_error(std::string("Read error: ") + filepath);

    return std::move(s);
}

char const* sgets(char* s, int size, char** stream)
{
    for (int i = 0; i < size; ++i)
    {
        if ((*stream)[i] == '\n' || (*stream)[i] == '\0')
        {
            memcpy(s, *stream, i);
            s[i] = '\0';

            if ((*stream)[i] == '\0')
                return 0;
            else
            {
                (*stream) += i + 1;
                return s;
            }
        }
    }
    return 0;
}

std::istream& operator>>(std::istream& is, Vector3& v)
{
    // parse [x,y,z]
    char st;
    is >> st >> v.x >> st >> v.y >> st >> v.z >> st;
    return is;
}

std::ostream& operator<<(std::ostream& os, const Vector3& v)
{
    os << "[" << v.x << "," << v.y << "," << v.z << "]";
    return os;
}

std::ostream& operator<<(std::ostream& os, const box3& b)
{
    os << "[" << b.m_mins[0] << "," << b.m_mins[1] << "," << b.m_mins[2] << "] --> ["
       << b.m_maxs[0] << "," << b.m_maxs[1] << "," << b.m_maxs[2] << "]";
    return os;
}
