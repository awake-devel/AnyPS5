#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <string>

namespace AgcDriver::Graphics {

namespace {

std::atomic<std::uint32_t> liveCustomBorderSamplers{0};

}

    Sampler::Sampler(const Context& context, const GuestSamplerResource& descriptor) : context(context) {
        Require(!descriptor.anisotropyEnable || context.samplerAnisotropy, "guest sampler descriptor requests anisotropic filtering which the device does not support");
        Require(descriptor.maxAnisotropy <= context.limits.maxSamplerAnisotropy, "guest sampler descriptor requests an anisotropy ratio beyond the device limit");
        Require(descriptor.lodBias >= -context.limits.maxSamplerLodBias && descriptor.lodBias <= context.limits.maxSamplerLodBias, "guest sampler descriptor requests a LOD bias beyond the device limit");
        Require(!(descriptor.unnormalizedCoordinates && descriptor.compareEnable), "guest sampler descriptor with unnormalized coordinates enables depth comparison, which is not implemented");

        VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        VkSamplerReductionModeCreateInfoEXT reduction{VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO_EXT, nullptr, descriptor.reductionMode};
        if (descriptor.reductionMode != VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE_EXT) {
            Require(context.samplerFilterMinmax, "guest sampler descriptor uses a min or max reduction which the device does not support");
            Require(!descriptor.compareEnable, "guest sampler descriptor combines a min or max reduction with depth comparison, which is not implemented");
            info.pNext = &reduction;
            requiresFilterMinmax = descriptor.magFilter == VK_FILTER_LINEAR || descriptor.minFilter == VK_FILTER_LINEAR;
        }
        info.magFilter = descriptor.magFilter;
        info.minFilter = descriptor.minFilter;
        info.mipmapMode = descriptor.mipmapMode;
        info.addressModeU = descriptor.addressModeU;
        info.addressModeV = descriptor.addressModeV;
        info.addressModeW = descriptor.addressModeW;
        info.mipLodBias = descriptor.lodBias;
        info.anisotropyEnable = descriptor.anisotropyEnable ? VK_TRUE : VK_FALSE;
        info.maxAnisotropy = descriptor.maxAnisotropy;
        info.compareEnable = descriptor.compareEnable ? VK_TRUE : VK_FALSE;
        info.compareOp = descriptor.compareOp;
        info.minLod = descriptor.minLod;
        info.maxLod = descriptor.maxLod;
        info.borderColor = descriptor.borderColor;
        info.unnormalizedCoordinates = descriptor.unnormalizedCoordinates ? VK_TRUE : VK_FALSE;
        const auto createSampler = context.Function<PFN_vkCreateSampler>("vkCreateSampler");
        VkSamplerCustomBorderColorCreateInfoEXT customBorder{VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
        if (descriptor.borderColorTable) {
            Require(context.customBorderColor, "guest sampler descriptor uses a border color table, which needs VK_EXT_custom_border_color with customBorderColorWithoutFormat");
            Require(descriptor.borderColor == VK_BORDER_COLOR_FLOAT_CUSTOM_EXT, "guest sampler descriptor with a border color table must use a custom float border color");
            Require(descriptor.customBorderColor.has_value(), "guest sampler descriptor reads a border color table entry that was not supplied");
            std::copy(descriptor.customBorderColor->begin(), descriptor.customBorderColor->end(), customBorder.customBorderColor.uint32);
            customBorder.format = VK_FORMAT_UNDEFINED;
            customBorder.pNext = info.pNext;
            info.pNext = &customBorder;
            if (liveCustomBorderSamplers.fetch_add(1) >= context.maxCustomBorderColorSamplers) {
                liveCustomBorderSamplers.fetch_sub(1);
                Require(false, "guest sampler descriptors with a border color table exceed the device limit of " + std::to_string(context.maxCustomBorderColorSamplers) + " custom border color samplers");
            }
            customBorderColor = true;
        }
        const auto created = createSampler(context.device, &info, nullptr, &sampler);
        if (created != VK_SUCCESS && customBorderColor) liveCustomBorderSamplers.fetch_sub(1);
        Check(created, "vkCreateSampler");
    }

    Sampler::~Sampler() {
        release();
    }

    void Sampler::release() noexcept {
        if (sampler) context.Function<PFN_vkDestroySampler>("vkDestroySampler")(context.device, sampler, nullptr);
        if (customBorderColor) liveCustomBorderSamplers.fetch_sub(1);
    }

    VkSampler Sampler::Handle() const {
        return sampler;
    }

    bool Sampler::RequiresFilterMinmax() const {
        return requiresFilterMinmax;
    }

    bool Sampler::CustomBorderColor() const {
        return customBorderColor;
    }

    SamplerCache::SamplerCache(std::size_t capacity) : capacity(std::max<std::size_t>(capacity, 1)) {}

    std::shared_ptr<Sampler> SamplerCache::Get(const Context& context, std::span<const std::uint32_t> words, bool compareEnable, bool unnormalizedProven, std::span<const std::uint32_t> borderColor) {
        Require(words.size() == 4, "guest sampler descriptor must contain 4 dwords");
        Require(borderColor.empty() || borderColor.size() == 4, "a border color table entry must contain 4 dwords");
        std::array<std::uint32_t, 9> key{words[0], words[1], words[2], words[3], (compareEnable ? 1u : 0u) | (unnormalizedProven ? 2u : 0u)};
        std::copy(borderColor.begin(), borderColor.end(), key.begin() + 5);
        std::lock_guard lock(mutex);
        ++clock;
        if (const auto found = entries.find(key); found != entries.end()) {
            ++hits;
            found->second.lastUse = clock;
            return found->second.sampler;
        }
        ++misses;
        auto resource = DecodeSamplerResource(words, unnormalizedProven);
        resource.compareEnable = compareEnable;
        Require(resource.borderColorTable == !borderColor.empty(), "a border color table entry must be given exactly for a sampler that reads one");
        if (resource.borderColorTable) resource.customBorderColor = std::array<std::uint32_t, 4>{borderColor[0], borderColor[1], borderColor[2], borderColor[3]};
        auto sampler = std::make_shared<Sampler>(context, resource);
        // The cap keeps live samplers well below the device's limit (NVIDIA: ~4000); a set in flight
        // still holds the evicted sampler through its own shared_ptr.
        while (entries.size() >= capacity) {
            const auto oldest = std::min_element(entries.begin(), entries.end(), [](const auto& left, const auto& right) { return left.second.lastUse < right.second.lastUse; });
            entries.erase(oldest);
        }
        entries.emplace(key, Entry{sampler, clock});
        return sampler;
    }

    void RequireFilterMinmax(const Context& context, VkFormat format, std::uint32_t samplerMask, std::span<const std::shared_ptr<Sampler>> samplers) {
        bool filtered = false;
        for (std::uint32_t element = 0; element < 32u; ++element) {
            if (((samplerMask >> element) & 1u) == 0u) continue;
            if (element >= samplers.size()) Require(false, "a sampled texture is paired with sampler element " + std::to_string(element) + ", which its shader does not bind");
            filtered = filtered || samplers[element]->RequiresFilterMinmax();
        }
        if (!filtered) return;
        VkFormatProperties properties{};
        context.formatProperties(context.physical, format, &properties);
        if ((properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT_EXT) == 0) Require(false, "a sampled texture whose format " + std::to_string(format) + " does not support min/max filtering is sampled through a min or max reduction sampler with linear filtering");
    }

}
