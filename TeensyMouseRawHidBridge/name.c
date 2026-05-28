// Teensyduino USB string overrides must live in a separate .c file.
// The serial string is left to the Teensy core so each board keeps a unique ID.

#include "usb_names.h"

#define MANUFACTURER_NAME {'G','e','n','e','r','i','c'}
#define MANUFACTURER_NAME_LEN 7

#define PRODUCT_NAME {'U','S','B',' ','H','I','D',' ','M','o','u','s','e'}
#define PRODUCT_NAME_LEN 13

struct usb_string_descriptor_struct usb_string_manufacturer_name = {
        2 + MANUFACTURER_NAME_LEN * 2,
        3,
        MANUFACTURER_NAME
};

struct usb_string_descriptor_struct usb_string_product_name = {
        2 + PRODUCT_NAME_LEN * 2,
        3,
        PRODUCT_NAME
};
