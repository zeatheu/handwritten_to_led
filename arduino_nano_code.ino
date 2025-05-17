// Pin definitions for seven-segment display (a–g)
const int segA = 2;  // Segment a
const int segB = 3;  // Segment b
const int segC = 4;  // Segment c
const int segD = 5;  // Segment d
const int segE = 6;  // Segment e
const int segF = 7;  // Segment f
const int segG = 8;  // Segment g

// Change to true for common anode, false for common cathode
const bool IS_COMMON_ANODE = false;

// Digit patterns (1 = segment on for cathode, 0 = segment on for anode)
const byte digits[10] = {
  B1111110, // 0
  B0110000, // 1
  B1101101, // 2
  B1111001, // 3
  B0110011, // 4
  B1011011, // 5
  B1011111, // 6
  B1110000, // 7
  B1111111, // 8
  B1111011  // 9
};

void setup() {
  Serial.begin(9600);
  // Set segment pins as outputs
  
  pinMode(segA, OUTPUT);
  pinMode(segB, OUTPUT);
  pinMode(segC, OUTPUT);
  pinMode(segD, OUTPUT);
  pinMode(segE, OUTPUT);
  pinMode(segF, OUTPUT);
  pinMode(segG, OUTPUT);
  // Initialize display to off
  displayDigit(-1);
}

void loop() {
  if (Serial.available()) {
    char ch = Serial.read();
    Serial.print("Received: ");
    Serial.println(ch);
    if (ch >= '0' && ch <= '9') {
      int num = ch - '0';
      displayDigit(num);
    }
  }
}

void displayDigit(int num) {
  // Turn off all segments if num is invalid
  if (num < 0 || num > 9) {
    digitalWrite(segA, IS_COMMON_ANODE ? HIGH : LOW);
    digitalWrite(segB, IS_COMMON_ANODE ? HIGH : LOW);
    digitalWrite(segC, IS_COMMON_ANODE ? HIGH : LOW);
    digitalWrite(segD, IS_COMMON_ANODE ? HIGH : LOW);
    digitalWrite(segE, IS_COMMON_ANODE ? HIGH : LOW);
    digitalWrite(segF, IS_COMMON_ANODE ? HIGH : LOW);
    digitalWrite(segG, IS_COMMON_ANODE ? HIGH : LOW);
    return;
  }

  // Write segment states for the digit
  byte pattern = digits[num];
  digitalWrite(segA, IS_COMMON_ANODE ? !bitRead(pattern, 6) : bitRead(pattern, 6));
  digitalWrite(segB, IS_COMMON_ANODE ? !bitRead(pattern, 5) : bitRead(pattern, 5));
  digitalWrite(segC, IS_COMMON_ANODE ? !bitRead(pattern, 4) : bitRead(pattern, 4));
  digitalWrite(segD, IS_COMMON_ANODE ? !bitRead(pattern, 3) : bitRead(pattern, 3));
  digitalWrite(segE, IS_COMMON_ANODE ? !bitRead(pattern, 2) : bitRead(pattern, 2));
  digitalWrite(segF, IS_COMMON_ANODE ? !bitRead(pattern, 1) : bitRead(pattern, 1));
  digitalWrite(segG, IS_COMMON_ANODE ? !bitRead(pattern, 0) : bitRead(pattern, 0));
}