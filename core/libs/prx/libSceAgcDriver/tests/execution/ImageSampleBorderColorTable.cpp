#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "Recompiler.hpp"
#include "VulkanTestDevice.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using AgcDriver::Graphics::Require;
using ShaderRecompiler::ShaderStage;

constexpr std::uint32_t Threads = 128;
constexpr std::uint32_t Words = 16;
constexpr std::uint32_t ResultWord = 4;
constexpr std::uint32_t Results = 12;
constexpr std::int32_t Width = 13;
constexpr std::int32_t Height = 7;
constexpr std::uint32_t Format8888UNorm = 56;
constexpr std::uint32_t Format8888UInt = 60;
constexpr std::uint32_t Type2D = 9;
constexpr std::uint32_t UnnormalizedPointClampToBorder = 0x000080b6u;
constexpr std::uint32_t NormalizedPointClampToBorder = 0x000000b6u;
constexpr std::uint32_t BorderColorTable = 0xc0000000u;
alignas(256) std::array<std::uint32_t, Threads * Words> Buffer{};
alignas(256) std::array<std::uint8_t, 16384> Texels{};
alignas(256) std::array<std::uint32_t, 16> Table{};
alignas(256) std::array<std::uint32_t, 16> OtherTable{};

alignas(256) constexpr std::array<std::uint32_t, 40> Code{
    0x34020086, 0xe0301000, 0x80000201, 0xe0301004, 0x80000301, 0xbf8c3f70, 0x7e0802ff, 0x402ccccd,
    0xf09c0f08, 0x00610802, 0xf0900f08, 0x00610c02, 0xf0800f08, 0x00611002, 0xbf8c3f70, 0xe0701010,
    0x80000801, 0xe0701014, 0x80000901, 0xe0701018, 0x80000a01, 0xe070101c, 0x80000b01, 0xe0701020,
    0x80000c01, 0xe0701024, 0x80000d01, 0xe0701028, 0x80000e01, 0xe070102c, 0x80000f01, 0xe0701030,
    0x80001001, 0xe0701034, 0x80001101, 0xe0701038, 0x80001201, 0xe070103c, 0x80001301, 0xbf810000,
};

constexpr std::array<const char*, 3> Instructions{"image_sample_lz", "image_sample_l 2.7", "image_sample"};

struct Coordinate {
    float u;
    float v;
};

const std::vector<Coordinate> Coordinates{
    {0.5f, 0.5f}, {12.5f, 6.5f}, {6.25f, 3.75f}, {-1.0f, 0.5f}, {-0.25f, 3.5f}, {13.0f, 3.5f}, {20.5f, 2.5f},
    {5.5f, -0.5f}, {5.5f, 7.0f}, {-3.0f, -3.0f}, {14.0f, 9.0f},
};

std::array<double, 4> TexelOf(std::int32_t x, std::int32_t y) {
    return {240.0 * ((x + y) & 1), 16.0 * x + 16.0, 32.0 * y + 32.0, 240.0};
}

void FillTexture() {
    const auto mips = AgcDriver::Graphics::ComputeMipLayout(AgcDriver::Graphics::TextureTileMode::kLinear, Format8888UNorm, Width, Height, 1);
    Require(AgcDriver::Graphics::ComputeSurfaceSize(mips, 1) <= Texels.size(), "border color table: the image does not fit the texel storage");
    std::fill(Texels.begin(), Texels.end(), std::uint8_t{0xeeu});
    for (std::uint32_t y = 0; y < mips[0].height; ++y) {
        for (std::uint32_t x = 0; x < mips[0].width; ++x) {
            auto* texel = &Texels[mips[0].tiledOffset + static_cast<std::uint64_t>(y) * mips[0].pitchBytes + x * 4u];
            const auto value = TexelOf(static_cast<std::int32_t>(x), static_cast<std::int32_t>(y));
            for (std::uint32_t component = 0; component < 4u; ++component) texel[component] = static_cast<std::uint8_t>(value[component]);
        }
    }
}

void SetEntry(std::array<std::uint32_t, 16>& table, std::uint32_t row, const std::array<float, 4>& color) {
    for (std::uint32_t component = 0; component < 4u; ++component) table[row * 4u + component] = std::bit_cast<std::uint32_t>(color[component]);
}

std::uint64_t Base(const std::array<std::uint32_t, 16>& table) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(table.data()));
}

std::array<std::uint32_t, 4> BufferDescriptor(const void* data, std::uint32_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(data);
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu), bytes, 0x01016facu};
}

std::array<std::uint32_t, 8> TextureDescriptor(std::uint32_t format) {
    const auto address = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(Texels.data()));
    return {
        static_cast<std::uint32_t>(address >> 8u),
        static_cast<std::uint32_t>((address >> 40u) & 0xffu) | (format << 20u) | ((static_cast<std::uint32_t>(Width - 1) & 3u) << 30u),
        (static_cast<std::uint32_t>(Width - 1) >> 2u) | (static_cast<std::uint32_t>(Height - 1) << 14u),
        0xfacu | (Type2D << 28u),
        0u,
        0u,
        0u,
        0u,
    };
}

std::array<std::uint32_t, 4> TableSampler(std::uint32_t row, std::uint32_t word0 = UnnormalizedPointClampToBorder) {
    return {word0, 0u, 0u, BorderColorTable | row};
}

using Samples = std::vector<std::array<std::uint32_t, Results>>;

Samples Run(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, 8>& texture, const std::array<std::uint32_t, 4>& sampler, std::optional<std::uint64_t> table, const std::vector<Coordinate>& coordinates) {
    Samples samples;
    const bool normalized = (sampler[0] & 0x8000u) == 0u;
    Buffer.fill(0xdeadbeefu);
    for (std::uint32_t lane = 0; lane < Threads; ++lane) {
        const auto coordinate = lane < coordinates.size() ? coordinates[lane] : Coordinate{0.5f, 0.5f};
        Buffer[lane * Words] = std::bit_cast<std::uint32_t>(normalized ? coordinate.u / static_cast<float>(Width) : coordinate.u);
        Buffer[lane * Words + 1u] = std::bit_cast<std::uint32_t>(normalized ? coordinate.v / static_cast<float>(Height) : coordinate.v);
    }
    std::vector<std::uint32_t> userData(16, 0u);
    const auto buffer = BufferDescriptor(Buffer.data(), static_cast<std::uint32_t>(Buffer.size() * 4u));
    std::copy(buffer.begin(), buffer.end(), userData.begin());
    std::copy(texture.begin(), texture.end(), userData.begin() + 4);
    std::copy(sampler.begin(), sampler.end(), userData.begin() + 12);
    const std::span<const std::uint32_t> code(Code);
    const std::array<ShaderRecompiler::MemoryRegion, 1> memory{{{reinterpret_cast<std::uintptr_t>(code.data()), std::as_bytes(code)}}};
    const ShaderRecompiler::ShaderComputeStageInfo compute{{Threads, 1, 1}, 0u, {false, false, false}, false, 1};
    ShaderRecompiler::RecompileRequest request{
        {ShaderStage::Compute, reinterpret_cast<std::uintptr_t>(code.data()), code, 0, {}},
        {32, 0, userData, compute, std::nullopt, std::nullopt, memory},
        device.Target(),
        {0, 0, 0, 128}
    };
    request.useCache = false;
    const auto result = ShaderRecompiler::Recompile(request);
    device.Dispatch(result, 1, 1, 1, {}, reinterpret_cast<std::uintptr_t>(code.data()), nullptr, nullptr, table);
    device.WaitIdle();
    for (std::uint32_t lane = 0; lane < coordinates.size(); ++lane) {
        std::array<std::uint32_t, Results> values{};
        std::copy_n(&Buffer[lane * Words + ResultWord], Results, values.begin());
        samples.push_back(values);
    }
    return samples;
}

void Check(const char* what, const std::array<float, 4>& border, const Samples& samples) {
    Require(samples.size() == Coordinates.size(), std::string(what) + ": a dispatch lost samples");
    for (std::size_t index = 0; index < Coordinates.size(); ++index) {
        const auto x = static_cast<std::int32_t>(std::floor(Coordinates[index].u));
        const auto y = static_cast<std::int32_t>(std::floor(Coordinates[index].v));
        const bool outside = x < 0 || x >= Width || y < 0 || y >= Height;
        for (std::uint32_t instruction = 0; instruction < Instructions.size(); ++instruction) {
            for (std::uint32_t component = 0; component < 4u; ++component) {
                const auto value = static_cast<double>(std::bit_cast<float>(samples[index][instruction * 4u + component]));
                const auto expected = outside ? static_cast<double>(border[component]) : TexelOf(x, y)[component] / 255.0;
                const auto tolerance = outside ? 0.5 / 255.0 : 0.25 / 255.0;
                Require(std::fabs(value - expected) <= tolerance, std::string(what) + ", " + Instructions[instruction] + ": (" + std::to_string(Coordinates[index].u) + ", " + std::to_string(Coordinates[index].v) + ") component " + std::to_string(component) + " is " + std::to_string(value) + ", expected " + std::to_string(expected));
            }
        }
    }
}

void ExpectFailure(AgcDriver::VulkanDevice& device, const std::array<std::uint32_t, 8>& texture, const std::array<std::uint32_t, 4>& sampler, std::optional<std::uint64_t> table, std::string_view reason, const char* what) {
    try {
        static_cast<void>(Run(device, texture, sampler, table, {{0.5f, 0.5f}}));
    } catch (const std::exception& error) {
        Require(std::string_view(error.what()).find(reason) != std::string_view::npos, std::string(what) + ": unexpected error: " + error.what());
        return;
    }
    throw std::runtime_error(std::string(what) + " was accepted");
}

}

int main() {
    try {
        const auto device = OpenVulkanTestDevice();
        if (!device) return VulkanTestSkipped;
        FillTexture();
        const auto texture = TextureDescriptor(Format8888UNorm);
        const std::array<float, 4> first{1.0f, 0.0f, 1.0f, 0.0f};
        const std::array<float, 4> second{0.25f, 0.5f, 0.75f, 1.0f};
        const std::array<float, 4> third{0.125f, 0.0f, 0.875f, 0.5f};
        const std::array<float, 4> rewritten{0.1f, 0.2f, 0.3f, 0.4f};
        const std::array<float, 4> other{0.6f, 0.7f, 0.8f, 0.9f};
        SetEntry(Table, 0, first);
        SetEntry(Table, 1, second);
        SetEntry(Table, 2, third);
        SetEntry(OtherTable, 1, other);
        const std::array<std::uint32_t, 4> captured{0x3e19999au, 0x3f000000u, 0x3d4ccccdu, 0x00000000u};
        std::copy(captured.begin(), captured.end(), Table.begin() + 12);
        Check("entry 1", second, Run(*device, texture, TableSampler(1), Base(Table), Coordinates));
        Check("entry 2", third, Run(*device, texture, TableSampler(2), Base(Table), Coordinates));
        Check("entry 0", first, Run(*device, texture, TableSampler(0), Base(Table), Coordinates));
        Check("entry 3, as PPSA21567 stores its entry 0", {0.15f, 0.5f, 0.05f, 0.0f}, Run(*device, texture, TableSampler(3), Base(Table), Coordinates));
        Check("entry 2, normalized coordinates", third, Run(*device, texture, TableSampler(2, NormalizedPointClampToBorder), Base(Table), Coordinates));
        SetEntry(Table, 1, rewritten);
        Check("entry 1 after the table changed", rewritten, Run(*device, texture, TableSampler(1), Base(Table), Coordinates));
        Check("entry 1 of another table", other, Run(*device, texture, TableSampler(1), Base(OtherTable), Coordinates));
        ExpectFailure(*device, texture, TableSampler(1), std::nullopt, "was never written", "a table border colour without a table base");
        ExpectFailure(*device, texture, TableSampler(1), Base(Table) | (std::uint64_t{1} << 48u), "above the 48-bit address", "a table base past 48 bits");
        ExpectFailure(*device, TextureDescriptor(Format8888UInt), TableSampler(1), Base(Table), "integer-format texture", "an integer texture through a table border colour");
        std::puts("image sample border color table tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
