#ifndef TEENSY41_RAWHID_H
#define TEENSY41_RAWHID_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class Config;
struct hid_device_;
typedef struct hid_device_ hid_device;

constexpr size_t Teensy41RawHidPacketSize = 64;
constexpr uint16_t Teensy41RawHidHostMagic = 0x3448;
constexpr uint16_t Teensy41RawHidDeviceMagic = 0x4834;
constexpr uint8_t Teensy41RawHidVersion = 1;

constexpr uint8_t Teensy41RawHidButtonLeft = 0x01;
constexpr uint8_t Teensy41RawHidButtonRight = 0x02;
constexpr uint8_t Teensy41RawHidButtonMiddle = 0x04;
constexpr uint8_t Teensy41RawHidButtonBack = 0x08;
constexpr uint8_t Teensy41RawHidButtonForward = 0x10;

enum class Teensy41RawHidCommand : uint8_t
{
    Move = 1,
    Buttons = 2,
};

enum class Teensy41RawHidEvent : uint8_t
{
    Button = 1,
    Status = 2,
};

#pragma pack(push, 1)
struct Teensy41RawHidPacket
{
    uint16_t magic = Teensy41RawHidHostMagic;
    uint8_t version = Teensy41RawHidVersion;
    uint8_t command = static_cast<uint8_t>(Teensy41RawHidCommand::Move);
    uint8_t buttonMask = 0;
    int16_t dx = 0;
    int16_t dy = 0;
    int16_t wheel = 0;
    int16_t wheelH = 0;
    uint32_t sequence = 0;
    uint8_t reserved[47] = {};
};

struct Teensy41RawHidHostPacket
{
    uint16_t magic = Teensy41RawHidHostMagic;
    uint8_t version = Teensy41RawHidVersion;
    uint8_t command = static_cast<uint8_t>(Teensy41RawHidCommand::Move);
    uint8_t buttonMask = 0;
    int16_t dx = 0;
    int16_t dy = 0;
    int16_t wheel = 0;
    int16_t wheelH = 0;
    uint32_t sequence = 0;
    uint8_t reserved[47] = {};
};

struct Teensy41RawHidDevicePacket
{
    uint16_t magic = Teensy41RawHidDeviceMagic;
    uint8_t version = Teensy41RawHidVersion;
    uint8_t event = static_cast<uint8_t>(Teensy41RawHidEvent::Button);
    uint8_t buttonId = 0;
    uint8_t pressed = 0;
    uint8_t hostButtonMask = 0;
    uint32_t sequenceAck = 0;
    uint8_t reserved[53] = {};
};
#pragma pack(pop)

static_assert(sizeof(Teensy41RawHidPacket) == Teensy41RawHidPacketSize, "Teensy41RawHidPacket must stay 64 bytes.");
static_assert(sizeof(Teensy41RawHidHostPacket) == Teensy41RawHidPacketSize, "Teensy41RawHidHostPacket must stay 64 bytes.");
static_assert(sizeof(Teensy41RawHidDevicePacket) == Teensy41RawHidPacketSize, "Teensy41RawHidDevicePacket must stay 64 bytes.");
static_assert(sizeof(Teensy41RawHidPacket) == 64, "Teensy41RawHidPacket must stay 64 bytes.");
static_assert(sizeof(Teensy41RawHidHostPacket) == 64, "Teensy41RawHidHostPacket must stay 64 bytes.");
static_assert(sizeof(Teensy41RawHidDevicePacket) == 64, "Teensy41RawHidDevicePacket must stay 64 bytes.");

class Teensy41RawHid
{
public:
    explicit Teensy41RawHid(const Config& config);
    ~Teensy41RawHid();

    Teensy41RawHid(const Teensy41RawHid&) = delete;
    Teensy41RawHid& operator=(const Teensy41RawHid&) = delete;

    bool isOpen() const;
    bool move(int dx, int dy, int wheel = 0, int wheelH = 0);
    bool press();
    bool release();
    bool setHostButtons(uint8_t mask);

    bool aimingActive() const;
    bool shootingActive() const;
    bool zoomingActive() const;

private:
    struct HidDeviceDeleter
    {
        void operator()(hid_device* device) const;
    };

    bool open();
    bool sendPacket(Teensy41RawHidCommand command, int dx, int dy, int wheel, int wheelH, uint8_t buttonMask);
    void readerLoop();
    void applyButtonEvent(uint8_t buttonId, bool pressed);

    std::unique_ptr<hid_device, HidDeviceDeleter> device_;
    mutable std::mutex writeMutex_;
    std::thread readerThread_;
    std::atomic<bool> connected_{ false };
    std::atomic<bool> stopReader_{ false };
    std::atomic<bool> aimingActive_{ false };
    std::atomic<bool> shootingActive_{ false };
    std::atomic<bool> zoomingActive_{ false };
    uint32_t sequence_ = 0;
    std::atomic<uint8_t> hostButtons_{ 0 };
    uint16_t usagePage_ = 0xFFAB;
    uint16_t usageId_ = 0x0200;
    int openIndex_ = 0;
    int packetTimeoutMs_ = 2;
    int reconnectIntervalMs_ = 500;
    std::string serialFilter_;
    std::string vidFilter_;
    std::string pidFilter_;
};

#endif // TEENSY41_RAWHID_H
