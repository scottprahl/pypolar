/*
 * Arduino Controller for Gaertner L116A Ellipsometer
 *
 * The L116A measures the intensity at every 5° of analyzer rotation for
 * a total of 72 values over one revolution of the analyzer.  Internally
 * the L116 intensity is digitized by a 12-bit ADC.
 *
 * This program waits for a byte to be sent from the host computer to the Arduino
 * The Arduino responds with 2 bytes for each of the 72 analyzer positions (lo,hi)
 * for a total of 144 bytes.  The initial value corresponds to the analyzer at
 * an angle of 0° and the last value corresponds to 355°
 *
 * Scott Prahl
 * May 2020
 * version 4
 */

const int BITS_IN_A2D = 12;           // Ellipsometer does internal 12-bit A2D conversion
const int PIN_BIT_ZERO = 2;           // Pins 2 - 14 get bits from LC116
const int PIN_A2D_DONE = A0;          // Pin that signals A/D conversion is finished
const int PIN_START_BUTTON = A1;      // Pin to physical button to start data capture
const int PIN_ANALYZER_AT_ZERO = A2;  // Pin with signal that analyzer is at 0°

// This holds the current intensity reading of the ellipsometer
int intensity[72];

void setup()
{
    Serial.begin(38400);

    for (int i=0; i < BITS_IN_A2D; i++)
        pinMode(PIN_BIT_ZERO + i, INPUT);

    pinMode(PIN_A2D_DONE, INPUT);
    pinMode(PIN_START_BUTTON, INPUT);
    pinMode(PIN_ANALYZER_AT_ZERO, INPUT);
}

void loop()
{
    wait_for_rising_edge(PIN_ANALYZER_AT_ZERO);

    for (int i = 0; i < 72; i++) {
        wait_for_falling_edge(PIN_A2D_DONE);
        intensity[i] = read_ellipsometer_intensity();
    }

    // If data is requested by computer
    if (Serial.available() > 0) {
        read_and_discard_all_bytes_in_buffer();
        
        // send 144 bytes of data over the Serial interface
        for (int i=0; i < 72; i++) {
            Serial.write(byte(intensity[i]>>8));
            Serial.write(byte(0x00FF & intensity[i]));
        }
    }
}

// read digitized bits over 12 wires and return an integer 0-4095
int read_ellipsometer_intensity(void)
{
    int value = 0;

    for (int i=0; i < BITS_IN_A2D; i++)
        value = (value << 1) + digitalRead(PIN_BIT_ZERO + i);

    // all the bits are flipped
    return 4095 - value;
}

void read_and_discard_all_bytes_in_buffer(void)
{
    while (Serial.available()) Serial.read();
}

void wait_for_rising_edge(int pin)
{
    // wait until pin is 0
    while (digitalRead(pin) != LOW) {
//       Serial.println("RE ... waiting for pin to be low");
    }

    // wait for pin transition to 1
    while (digitalRead(pin) != HIGH) {
//       Serial.println("RE ... waiting for pin to go high");
    }
}

void wait_for_falling_edge(int pin)
{
    // wait until pin is 1
    while (digitalRead(pin) != HIGH) {
//        Serial.println("FE ... waiting for pin to be high");
    }

    // wait for pin transition to 0
    while (digitalRead(pin) != LOW) {
//        Serial.println("FE ... waiting for pin to go low");
    }
}
