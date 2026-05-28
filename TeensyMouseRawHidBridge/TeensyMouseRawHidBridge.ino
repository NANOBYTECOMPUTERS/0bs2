#include <Arduino.h>
#include <USBHost_t36.h>

#if !defined(RAWHID_INTERFACE)
#error "Select a Teensy USB Type that includes RawHID."
#endif

#if !defined(MOUSE_INTERFACE)
#error "Select a Teensy USB Type that includes Mouse."
#endif

USBHost myusb;
USBHub hub1(myusb);
USBHub hub2(myusb);
USBHIDParser hid1(myusb);
USBHIDParser hid2(myusb);
USBHIDParser hid3(myusb);
USBHIDParser hid4(myusb);
MouseController mouse1(myusb);

static constexpr uint16_t HOST_MAGIC = 0x3448;
static constexpr uint16_t DEVICE_MAGIC = 0x4834;
static constexpr uint8_t PROTOCOL_VERSION = 1;
static constexpr const char PROTOCOL_SOURCE[] = "Teensy41RawHid";

enum : uint8_t {
  COMMAND_MOVE = 1,
  COMMAND_BUTTONS = 2,
  EVENT_BUTTON = 1,
  EVENT_STATUS = 2,
  PC_BUTTON_LEFT = 0x01,
  PC_BUTTON_RIGHT = 0x02,
  PC_BUTTON_MIDDLE = 0x04,
  PC_BUTTON_BACK = 0x08,
  PC_BUTTON_FORWARD = 0x10,
};

struct __attribute__((packed)) RawHidHostPacket {
  uint16_t magic;
  uint8_t version;
  uint8_t command;
  uint8_t buttonMask;
  int16_t dx;
  int16_t dy;
  int16_t wheel;
  int16_t wheelH;
  uint32_t sequence;
  uint8_t reserved[47];
};

struct __attribute__((packed)) RawHidDeviceEventPacket {
  uint16_t magic;
  uint8_t version;
  uint8_t event;
  uint8_t buttonId;
  uint8_t pressed;
  uint8_t hostButtonMask;
  uint32_t sequenceAck;
  uint8_t reserved[53];
};

static_assert(sizeof(RawHidHostPacket) == 64, "RawHID host packet must be 64 bytes");
static_assert(sizeof(RawHidDeviceEventPacket) == 64, "RawHID device packet must be 64 bytes");

uint8_t hostButtons = 0;
uint8_t pcButtons = 0;
uint32_t lastSequence = 0;

int8_t clampMouseStep(int value) {
  if (value > 127) return 127;
  if (value < -127) return -127;
  return (int8_t)value;
}

void moveMouseInSteps(int dx, int dy, int wheel = 0, int wheelH = 0) {
  while (dx != 0 || dy != 0 || wheel != 0 || wheelH != 0) {
    int8_t stepX = clampMouseStep(dx);
    int8_t stepY = clampMouseStep(dy);
    int8_t stepWheel = clampMouseStep(wheel);
    int8_t stepWheelH = clampMouseStep(wheelH);

    Mouse.move(stepX, stepY, stepWheel, stepWheelH);

    dx -= stepX;
    dy -= stepY;
    wheel -= stepWheel;
    wheelH -= stepWheelH;
  }
}

void setMouseButtons(uint8_t buttons) {
  Mouse.set_buttons(
    buttons & MOUSE_LEFT,
    buttons & MOUSE_MIDDLE,
    buttons & MOUSE_RIGHT,
    buttons & MOUSE_BACK,
    buttons & MOUSE_FORWARD
  );
}

void applyButtons() {
  Mouse.set_buttons(
    (hostButtons & MOUSE_LEFT) || (pcButtons & PC_BUTTON_LEFT),
    (hostButtons & MOUSE_MIDDLE) || (pcButtons & PC_BUTTON_MIDDLE),
    (hostButtons & MOUSE_RIGHT) || (pcButtons & PC_BUTTON_RIGHT),
    (hostButtons & MOUSE_BACK) || (pcButtons & PC_BUTTON_BACK),
    (hostButtons & MOUSE_FORWARD) || (pcButtons & PC_BUTTON_FORWARD)
  );
}

uint8_t buttonIdForMask(uint8_t mask) {
  if (mask == MOUSE_LEFT) return 1;
  if (mask == MOUSE_RIGHT) return 2;
  if (mask == MOUSE_MIDDLE) return 3;
  if (mask == MOUSE_BACK) return 4;
  if (mask == MOUSE_FORWARD) return 5;
  return 0;
}

void emitButtonTransition(uint8_t previousButtons, uint8_t currentButtons, uint8_t mask) {
  bool wasDown = (previousButtons & mask) != 0;
  bool isDown = (currentButtons & mask) != 0;
  if (wasDown == isDown) return;

  RawHidDeviceEventPacket packet = {};
  packet.magic = DEVICE_MAGIC;
  packet.version = PROTOCOL_VERSION;
  packet.event = EVENT_BUTTON;
  packet.buttonId = buttonIdForMask(mask);
  packet.pressed = isDown ? 1 : 0;
  packet.hostButtonMask = currentButtons;
  packet.sequenceAck = lastSequence;

  if (packet.buttonId != 0) {
    RawHID.send(&packet, 0);
  }
}

void emitButtonTransitions(uint8_t previousButtons, uint8_t currentButtons) {
  emitButtonTransition(previousButtons, currentButtons, MOUSE_LEFT);
  emitButtonTransition(previousButtons, currentButtons, MOUSE_RIGHT);
  emitButtonTransition(previousButtons, currentButtons, MOUSE_MIDDLE);
  emitButtonTransition(previousButtons, currentButtons, MOUSE_BACK);
  emitButtonTransition(previousButtons, currentButtons, MOUSE_FORWARD);
}

bool validHostPacket(const RawHidHostPacket &packet) {
  return packet.magic == HOST_MAGIC && packet.version == PROTOCOL_VERSION;
}

void handleRawHidHostPacket() {
  RawHidHostPacket packet = {};
  int received = RawHID.recv(&packet, 0);
  if (received != 64 || !validHostPacket(packet)) return;

  lastSequence = packet.sequence;
  if (packet.command == COMMAND_MOVE) {
    moveMouseInSteps(packet.dx, packet.dy, packet.wheel, packet.wheelH);
  }
  pcButtons = packet.buttonMask;
  applyButtons();
}

void forwardHostMouse() {
  if (!mouse1.available()) return;

  int dx = mouse1.getMouseX();
  int dy = mouse1.getMouseY();
  int wheel = mouse1.getWheel();
  int wheelH = mouse1.getWheelH();
  uint8_t buttons = mouse1.getButtons();

  moveMouseInSteps(dx, dy, wheel, wheelH);

  if (buttons != hostButtons) {
    emitButtonTransitions(hostButtons, buttons);
    hostButtons = buttons;
    applyButtons();
  }

  mouse1.mouseDataClear();
}

void setup() {
  (void)PROTOCOL_SOURCE;

  pinMode(LED_BUILTIN, OUTPUT);
  myusb.begin();
  Mouse.begin();
  applyButtons();
}

void loop() {
  myusb.Task();
  handleRawHidHostPacket();
  forwardHostMouse();
}
