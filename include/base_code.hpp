/*
sudo apt install libglm-dev cmake libxcb-dri3-0 libxcb-present0 libpciaccess0 libpng-dev libxcb-keysyms1-dev libxcb-dri3-dev libx11-dev g++ gcc g++-multilib libmirclient-dev libwayland-dev libxrandr-dev libxcb-randr0-dev libxcb-ewmh-dev git python3 bison libx11-xcb-dev liblz4-dev libzstd-dev
*/
#ifndef BASE_CODE_HPP
#define BASE_CODE_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <glm/glm.hpp>
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>
#include <glm/glm.hpp>
#include <array>

static constexpr uint32_t WIDTH = 800;
static constexpr uint32_t HEIGHT = 600;

static const std::string MODEL_PATH = "models/viking_room.obj";
static const std::string TEXTURE_PATH = "textures/viking_room.png";



struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
/*
• VK_VERTEX_INPUT_RATE_VERTEX: Move to the next data entry after each
vertex
• VK_VERTEX_INPUT_RATE_INSTANCE: Move to the next data entry after
each instance
*/
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return bindingDescription;
    }
    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;// position in vertex shader;
/*
float: VK_FORMAT_R32_SFLOAT
vec2: VK_FORMAT_R32G32_SFLOAT
vec3: VK_FORMAT_R32G32B32_SFLOAT
vec4: VK_FORMAT_R32G32B32A32_SFLOAT
*/
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;

        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);
        return attributeDescriptions;
    }
    bool operator==(const Vertex& other) const
    {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }
};

static const int MAX_FRAMES_IN_FLIGHT = 2;

static const std::vector<const char*> validationLayers =
{
    "VK_LAYER_KHRONOS_validation"
};

static const std::vector<const char*> deviceExtensions = 
{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

static VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr
    (instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }else
    {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

static void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr
    (instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
    {
        func(instance, debugMessenger, pAllocator);
    }
}

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    inline bool isComplete(){   return graphicsFamily.has_value() && presentFamily.has_value();}
};

struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct UniformBufferObject
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

class HelloTriangleApplication
{
    public:
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

        void run()
        {
            initWindow();
            initVulkan();
            mainLoop();
            cleanup();
        }
    private:
        void initWindow();
        void initVulkan();
        void createInstance();
        void mainLoop() noexcept(false);
        void cleanup();
        bool checkValidationLayerSupport();
        void setupDebugMessenger();
        void createSurface();
        void pickPhysicalDevice();
        bool isDeviceSuitable(VkPhysicalDevice physicalDevice);
        [[__nodiscard__]] QueueFamilyIndices findQueueFamilies(VkPhysicalDevice physicalDevice) noexcept(true);
        void createLogicalDevice();
        bool checkDeviceExtensionSupport(VkPhysicalDevice physicalDevice);
        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice physicalDevice);
        VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
        VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresntModes);
        VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
        void createSwapChain();
        void createImageViews();
        static std::vector<char> readFile(const std::string& filename);
        VkShaderModule createShaderModule(const std::vector<char>& shaderCode);
        void createGraphicsPipeline();
        void createRenderPass();
        void createFrameBuffers();
        void createCommandPool();
        void createCommandBuffers();
        void drawFrame() noexcept(false);
        void createSyncObjects();
        void recreateSwapChain();
        void cleanupSwapChain();
        void createVertexBuffer();
        void createIndexBuffer();
        void createUniformBuffers();
        void createDescriptorPool();
        void createColorResources();
        void createDescriptorSetLayout();
        void createDescriptorSets();
        uint32_t mipLevels;
        void createTextureImage();
        void createTextureImageView();
        void createTextureSampler();
        void createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
                         VkSampleCountFlagBits numSamples, VkFormat format, VkImageTiling tiling,
                         VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image,
                         VkDeviceMemory& imageMemory);
        VkCommandBuffer beginSingleTimeCommands();
        void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
        void transitionImageLayout(VkImage image, VkFormat format,
                                   VkImageLayout oldLayout, VkImageLayout newLayout, uint32_t mipLevels);
        void endSingleTimeCommands(VkCommandBuffer commandBuffer);
        void updateUniformBuffer(uint32_t currentImage);

        VkImageView createImageView(VkImage image, VkFormat format, 
                                    VkImageAspectFlags aspectFlags, uint32_t mipLevels);
        void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
        void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                          VkBuffer &buffer, VkDeviceMemory &bufferMemory);
        void createDepthResources();
        VkFormat findDepthFormat();
        void loadModel();
        void generateMipmaps(VkImage image, VkFormat imageFormat, int32_t texWidth, int32_t texHeight, uint32_t mipLevels);
        bool hasStencilComponent(VkFormat format);
        VkFormat findSupportedFormat(const std::vector<VkFormat>&candidates, 
                                     VkImageTiling tiling, VkFormatFeatureFlags features);
        uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
        VkSampleCountFlagBits getMaxUsableSampleCount();
        std::vector<const char*> getRequiredExtensions();
        const uint32_t WIDTH = 800;
        const uint32_t HEIGHT = 600;

        GLFWwindow *window;
        VkInstance instance;
        VkDebugUtilsMessengerEXT debugMessenger;
        VkSurfaceKHR surface;
        VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
        VkDevice device;
        VkSwapchainKHR swapChain;
        std::vector<VkImage> swapChainImages;
        std::vector<VkImageView> swapChainImageViews;
        VkFormat swapChainImageFormat;
        VkExtent2D swapChainExtent;
        VkRenderPass renderPass;
        VkDescriptorSetLayout descriptorSetLayout;
        VkPipelineLayout pipelineLayout;
        VkPipeline graphicsPipeline;
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        VkBuffer vertexBuffer;
        VkDeviceMemory vertexBufferMemory;
        VkBuffer indexBuffer;
        VkDeviceMemory indexBufferMemory;
        VkDescriptorPool descriptorPool;
        VkImage textureImage;
        VkImageView textureImageView;
        VkSampler textureSampler;
        VkDeviceMemory textureImageMemory;
        std::vector<VkDescriptorSet> descriptorSets;
        VkImage depthImage;
        VkDeviceMemory depthImageMemory;
        VkImageView depthImageView;


        VkImage colorImage;
        VkDeviceMemory colorImageMemory;
        VkImageView colorImageView;
        std::vector<VkBuffer> uniformBuffers;
        std::vector<VkDeviceMemory> uniformBuffersMemory;
        std::vector<VkFramebuffer> swapChainFrameBuffers;
        VkCommandPool commandPool;
        std::vector<VkCommandBuffer> commandBuffers;
        std::vector<VkSemaphore> imageAvailableSemaphores;
        std::vector<VkSemaphore> renderFinishedSemaphores;
        std::vector<VkFence> inFlightFences;
        std::vector<VkFence> imagesInFlight;
        size_t currentFrame = 0;
        VkSampleCountFlagBits msaaSamples = VK_SAMPLE_COUNT_1_BIT;
        bool framebufferResized = false;
/*
Device queues are implicitly cleaned up when the device is destroyed, so we
don’t need to do anything in cleanup.
*/
        VkQueue graphicsQueue;
        VkQueue presentQueue;
        static void framebufferResizeCallback(GLFWwindow *window, int width, int height) noexcept(false);

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
                    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                    VkDebugUtilsMessageTypeFlagsEXT messageType,
                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                    void* pUserData)
        {
            std::cerr << "validation layer: " << pCallbackData->pMessage <<
            std::endl;
            /*
            The callback returns a boolean that indicates if the Vulkan call that triggered
            the validation layer message should be aborted. If the callback returns true, then
            the call is aborted with the VK_ERROR_VALIDATION_FAILED_EXT error. This is
            normally only used to test the validation layers themselves, so you should always
            return VK_FALSE.
            */
            return VK_FALSE;
        }
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
};

#endif