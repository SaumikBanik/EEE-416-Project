// ============================================================================
//  SMART GLOVE - ESP32 DATA-COLLECTION FIRMWARE  V3.1.2
// ----------------------------------------------------------------------------
//  What this program does
//   * The glove runs from a battery and talks to the laptop only over
//     Wi-Fi/TCP. No USB cable is needed while collecting data.
//   * The MPU6050 is read as a plain sensor (no DMP, no FIFO) 200 times per
//     second in its own task. A Mahony filter on the ESP32 turns the readings
//     into an orientation (quaternion). The orientation lives in ESP32 memory,
//     so a sensor glitch costs one retried read, not a dead sensor.
//   * Every sensor read is checked. Frozen data, lost sensor settings and bus
//     faults are detected and repaired automatically. If the orientation may
//     no longer be continuous, the "IMU epoch" number changes and the laptop
//     program starts a new calibration session.
//   * One 5-second neutral calibration is used for the participant/session's
//     flex baseline, accelerometer baseline, gyro zero-rate bias and orientation
//     reference. There is no separate gyro-calibration pose.
//   * Captures at 25 Hz: 5 flex sensors (8 reads per finger spread over one
//     20 ms mains cycle, which strongly reduces 50 Hz hum) plus the orientation
//     sampled at the CENTRE of that same 20 ms window. It records first and
//     transmits afterwards.
//   * Every fault is reported to the laptop over TCP (Serial is optional).
//
//  Needs: Arduino IDE + "esp32 by Espressif Systems" boards package 3.x.
//  No extra libraries (Wire, WiFi and Preferences come with the ESP32 core).
//  Board: "ESP32 Dev Module" (or your ESP32 board).
// ----------------------------------------------------------------------------
//  CHANGE IN 3.1.2
//   * The calibration stillness gate (BIAS_MAX_STD_DPS / BIAS_MAX_DEV_DPS) is
//     now a sanity check rather than a stillness test, so a normal hand hold
//     is no longer rejected with CAL_MOVING. Whether the hold was good enough
//     is decided (and can be overridden) by the laptop program, which records
//     the numbers either way.
//
//  CHANGES IN 3.1.1 (all fixes applied to 3.1.0)
//   0. BUILD FIX: enum CfgResult moved into the type section (section 3).
//      The Arduino IDE injects generated prototypes just above the FIRST
//      function definition, so any type used in a return value must be
//      declared before that point, otherwise you get
//      "'CfgResult' does not name a type".
//   1. Flex/orientation time skew removed: the IMU snapshot is taken at the
//      centroid of the ADC window instead of after it (was ~9 ms late).
//   2. A failed re-seed no longer leaves a pending request that would fire
//      later and bump the epoch in the middle of a trial.
//   3. Calibration now keeps the 200 Hz motion peak as well as the 25 Hz one.
//   4. ERR lines print sample=-1 (not 65535) when no sample failed.
//   5. Unused averageQuat() removed.
//   6. WHO_AM_I is now verified during configuration and reported.
//   7. All snprintf line buffers enlarged so a STATS/STATUS line can never be
//      truncated (a truncated line loses its '\n' and desyncs the parser).
//   8. The IMU task's pacing reference is reset after a failed re-init too.
//   9. CAL_MOVING is emitted through the normal ERR path (one ERR format).
//  10. TCP write failures are detected; a truncated block drops the link so
//      the laptop sees a clean disconnect instead of a silent short block.
//  11. Wire.requestFrom() overload made unambiguous; tick-rate assumption is
//      now checked at compile time.
// ============================================================================

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <Preferences.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "esp_system.h"
#include "esp_attr.h"
#if __has_include("esp_random.h")
#include "esp_random.h"
#endif

// The 200 Hz IMU loop uses pdMS_TO_TICKS(5); with a slower tick that rounds
// down to 0 ticks and the task would spin instead of pacing itself.
static_assert(configTICK_RATE_HZ == 1000, "This firmware assumes a 1000 Hz FreeRTOS tick");

// ============================================================================
// 1. SETTINGS YOU MUST CHECK
// ============================================================================

static const char*    WIFI_SSID     = "DESKTOP-D7LMP4F 8069";
static const char*    WIFI_PASSWORD = "8N9577=m";
static const char*    SERVER_IP     = "192.168.137.1";   // laptop hotspot address (see ipconfig)
static const uint16_t SERVER_PORT   = 5000;

static const uint8_t FLEX_PINS[5] = {32, 33, 34, 35, 36};  // thumb, index, middle, ring, little (all ADC1)
static const int     SDA_PIN      = 21;
static const int     SCL_PIN      = 22;
static const int     LED_PIN      = 2;     // on-board status LED; set to -1 if your board has none
static const int     RAIL_PIN     = -1;    // set to 39 if you fitted the 3.3 V monitor (100k + 100k divider)
static const float   RAIL_DIVIDER = 2.0f;  // a 100k/100k divider halves the voltage

// ============================================================================
// 2. FIXED SETTINGS (these match the protocol - do not change casually)
// ============================================================================

static const char*    FW_NAME            = "SMART_GLOVE_V3";
static const char*    FW_VERSION         = "3.1.2";

static const uint32_t SAMPLE_INTERVAL_US = 40000;   // 25 Hz
static const uint16_t N_TRIAL = 50, N_CAL = 125, N_FIST = 75, N_CHECK = 50, N_DRIFT = 125;
static const uint16_t MAX_SAMPLES        = 125;
static const uint8_t  ADC_ROUNDS         = 8;       // 8 reads per finger per sample ...
static const uint32_t ADC_ROUND_US       = 2500;    // ... 2.5 ms apart = one 20 ms mains cycle

// The averaged flex value has its centre of mass at (ADC_ROUNDS-1)/2 rounds,
// i.e. 8.75 ms into the window. Snapshotting the IMU right after round 3 lands
// within ~1 ms of that, so the flex columns and the quaternion columns of a row
// describe the same instant.
static const uint8_t  IMU_SNAP_ROUND     = (ADC_ROUNDS / 2) - 1;
static_assert(ADC_ROUNDS >= 2, "ADC_ROUNDS must be at least 2");

static const uint32_t I2C_HZ             = 100000;  // conservative bus speed; wiring quality is still the main reliability factor
static const uint8_t  MPU_ADDR           = 0x68;
static const uint8_t  WHO_EXPECTED       = 0x68;    // genuine MPU6050; clones report other values
static const uint32_t IMU_PERIOD_US      = 5000;    // 200 Hz sensor loop
static const float    GYRO_LSB_PER_DPS   = 16.4f;   // +-2000 deg/s range
static const float    ACC_LSB_PER_G      = 8192.0f; // +-4 g range
static const float    MAHONY_KP          = 1.0f;    // tilt correction strength (about 1 s time constant)
static const float    ACC_GATE_G         = 0.15f;   // use gravity only when |a| is within 1 +- 0.15 g
// Largest heading error a data gap may cause before the orientation frame is
// declared broken. 1.0 deg was strict enough that a single missed sample during
// a fast gesture (>200 deg/s) forced a full re-calibration; 2.0 deg keeps the
// protection while tolerating isolated I2C hiccups. Set it back to 1.0f if you
// prefer the stricter behaviour.
static const float    GAP_BUDGET_DEG     = 2.0f;
static const float    GAP_FLOOR_DPS      = 50.0f;   // rotation speed assumed during an unseen gap
static const uint8_t  FROZEN_LIMIT       = 3;       // identical readings in a row = frozen sensor
static const uint8_t  FAIL_LIMIT         = 3;       // failed reads in a row = sensor fault
static const uint32_t CFG_CHECK_US       = 100000;  // re-check the sensor settings every 100 ms
static const uint32_t FRESH_US           = 15000;   // an IMU sample older than this is "stale"
static const uint32_t POST_GUARD_US      = 30000;   // after a capture, wait this long for late fault flags
static const uint16_t BIAS_SAMPLES       = 1000;    // 5 s x 200 Hz, collected during the SAME neutral hold
// These are a SANITY gate, not a stillness test: they only reject a hold so
// disturbed that the measured "zero-rate bias" would really be hand rotation.
// A normal, unforced human hold passes. Judging whether the hold was good
// enough is the laptop program's job, and the operator can accept a marginal
// one there. Tighten these only if you need the strictest possible yaw.
static const float    BIAS_MAX_STD_DPS   = 3.0f;    // was 1.0 - too strict for a real hand
static const float    BIAS_MAX_DEV_DPS   = 15.0f;   // was 5.0

static const float    DEG2RAD            = 0.01745329252f;
static const float    RAD2DEG            = 57.2957795131f;

// MPU6050 registers
enum : uint8_t {
  R_SMPLRT_DIV = 0x19, R_CONFIG = 0x1A, R_GYRO_CONFIG = 0x1B, R_ACCEL_CONFIG = 0x1C,
  R_FIFO_EN = 0x23, R_INT_ENABLE = 0x38, R_DATA = 0x3B, R_SIGNAL_RESET = 0x68,
  R_USER_CTRL = 0x6A, R_PWR_MGMT_1 = 0x6B, R_WHO_AM_I = 0x75
};
// SMPLRT_DIV=0 (1 kHz), CONFIG=3 (about 42 Hz low-pass), GYRO=+-2000 deg/s, ACCEL=+-4 g
static const uint8_t CFG_EXPECTED[4] = {0x00, 0x03, 0x18, 0x08};
static const uint8_t PWR_EXPECTED    = 0x01;   // awake, clock from the gyro PLL

// ============================================================================
// 3. DATA TYPES AND SHARED STATE
// ----------------------------------------------------------------------------
//  IMPORTANT (Arduino .ino build): every type used as a function return type
//  must be declared in THIS section, above the first function definition.
//  The IDE inserts its generated prototypes there, and a type declared later
//  in the file produces "'X' does not name a type".
// ============================================================================

struct Quat { float w, x, y, z; };

enum CfgResult { CFG_OK, CFG_READ_FAIL, CFG_WRONG };

// Written by the IMU task, read by the main loop (always under imuMux).
struct ImuState {
  Quat     q;
  int16_t  ax, ay, az, gx, gy, gz, tempRaw;
  uint32_t sampleUs;          // micros() of the newest valid sample
  bool     healthy;           // false while the sensor is faulty / being repaired
  uint32_t epoch;             // changes whenever orientation continuity may be lost
  uint32_t i2cErrors, faults, reinits, frozenEvents, configLost, gapInvalidations, recoveryFails;
  uint32_t whoUnexpected;     // configured OK but WHO_AM_I was not 0x68
  uint8_t  who;               // WHO_AM_I value
  bool     capActive;         // capture statistics (reset by the main loop)
  float    capMaxDps;
  uint32_t capMissed;
  bool     biasBusy;          // collect full-rate gyro bias during the 5 s neutral calibration
  uint16_t biasN;
  double   bSum[3], bSq[3];
  float    bMin[3], bMax[3];
  bool     reqTestReset;      // request from the main loop: simulate a sensor reset
  uint32_t reseedRequest;     // main loop asks the IMU task to adopt a new calibration seed
  uint32_t reseedAck;
  Quat     reseedQ;
};

struct Sample {
  uint32_t t_ms;
  uint16_t flex[5];
  Quat     q;
  int16_t  gx, gy, gz, ax, ay, az;
  float    roll, pitch, yaw;
};

struct CapStats {
  float    maxDps;
  uint32_t missed, lateMaxUs;
  float    tempC, railStartMv, railEndMv;
  uint16_t failSample;
  uint32_t failAgeUs;
  bool     hasCalBias;
  float    calBiasDps[3];
  float    calBiasStdDps;
  float    calBiasMaxDevDps;
};

enum SwReason : uint32_t { SW_NONE = 0, SW_IMU_STALL = 1, SW_TEST_REBOOT = 2 };

static const uint16_t NO_FAIL_SAMPLE = 0xFFFF;   // "no sample failed" sentinel

static ImuState          imu;
static portMUX_TYPE      imuMux = portMUX_INITIALIZER_UNLOCKED;
static volatile uint32_t imuHeartbeatMs = 0;
static float             biasDps[3] = {0.0f, 0.0f, 0.0f};   // protected by imuMux
static char              biasSource[10] = "none";

static Sample buf[MAX_SAMPLES];

// Calibration reference (main loop only)
static Quat     qRef     = {1.0f, 0.0f, 0.0f, 0.0f};
static bool     refValid = false;
static uint32_t refEpoch = 0;
static char     refId[24] = "-";

// Identity / reset information
static uint32_t    bootId    = 0;
static const char* resetText = "UNKNOWN";
static const char* swText    = "NONE";
#define RTC_MAGIC 0x5A17C0DEUL
RTC_NOINIT_ATTR static uint32_t rtcMagic;
RTC_NOINIT_ATTR static uint32_t rtcReason;

// Network
static WiFiClient client;
static IPAddress  serverIp;
static char       lineBuf[96];
static uint8_t    lineLen = 0;
static uint32_t   lastTcpTryMs = 0, wifiLostSinceMs = 0;
static bool       txOk = true;          // false once a TCP write has been short/failed

// Forward declarations (so this file also compiles as plain C++)
static bool i2cWrite(uint8_t reg, uint8_t val);
static bool i2cRead(uint8_t reg, uint8_t* out, uint8_t n);
static void imuTask(void* arg);
static ImuState snapshotImu();
static void softReboot(uint32_t reason);
static bool sendText(const char* s);
static void sendErrEx(uint32_t seq, const char* code, const CapStats* st, const char* extra);
static void sendErr(uint32_t seq, const char* code, const CapStats* st);

// ============================================================================
// 4. QUATERNION MATH  (same conventions as the V2 firmware)
// ============================================================================

static Quat qNormalize(Quat q) {
  float n = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n < 1e-8f) return Quat{1.0f, 0.0f, 0.0f, 0.0f};
  return Quat{q.w / n, q.x / n, q.y / n, q.z / n};
}

static Quat qConj(const Quat& q) { return Quat{q.w, -q.x, -q.y, -q.z}; }

static Quat qMul(const Quat& a, const Quat& b) {
  return Quat{a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
              a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
              a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
              a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

// roll = X, pitch = Y, yaw = Z (ZYX order), degrees
static void quatToRPY(const Quat& q, float& roll, float& pitch, float& yaw) {
  roll = atan2f(2.0f * (q.w * q.x + q.y * q.z), 1.0f - 2.0f * (q.x * q.x + q.y * q.y)) * RAD2DEG;
  float s = 2.0f * (q.w * q.y - q.z * q.x);
  s = s > 1.0f ? 1.0f : (s < -1.0f ? -1.0f : s);
  pitch = asinf(s) * RAD2DEG;
  yaw = atan2f(2.0f * (q.w * q.z + q.x * q.y), 1.0f - 2.0f * (q.y * q.y + q.z * q.z)) * RAD2DEG;
}

// Orientation from gravity only (heading set to 0). a = accelerometer (any unit).
static Quat seedFromAccel(float ax, float ay, float az) {
  float roll = atan2f(ay, az);
  float pitch = atan2f(-ax, sqrtf(ay * ay + az * az));
  float cr = cosf(roll * 0.5f), sr = sinf(roll * 0.5f);
  float cp = cosf(pitch * 0.5f), sp = sinf(pitch * 0.5f);
  return Quat{cr * cp, sr * cp, cr * sp, -sr * sp};
}

// One Mahony step. g in rad/s (bias already removed), a in g, dt in s.
// q rotates sensor-frame vectors into the world frame (world z = up).
static Quat mahonyStep(Quat q, float gx, float gy, float gz, float ax, float ay, float az, float dt) {
  float an = sqrtf(ax * ax + ay * ay + az * az);
  if (an > (1.0f - ACC_GATE_G) && an < (1.0f + ACC_GATE_G)) {
    ax /= an; ay /= an; az /= an;
    // world "up" seen from the sensor, according to the current estimate
    float vx = 2.0f * (q.x * q.z - q.w * q.y);
    float vy = 2.0f * (q.w * q.x + q.y * q.z);
    float vz = q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z;
    // error = measured x estimated
    gx += MAHONY_KP * (ay * vz - az * vy);
    gy += MAHONY_KP * (az * vx - ax * vz);
    gz += MAHONY_KP * (ax * vy - ay * vx);
  }
  float h = 0.5f * dt;
  Quat d = {(-q.x * gx - q.y * gy - q.z * gz) * h,
            ( q.w * gx + q.y * gz - q.z * gy) * h,
            ( q.w * gy - q.x * gz + q.z * gx) * h,
            ( q.w * gz + q.x * gy - q.y * gx) * h};
  return qNormalize(Quat{q.w + d.w, q.x + d.x, q.y + d.y, q.z + d.z});
}

// Could a gap of `missingS` seconds without data have caused more heading
// error than the budget? Worst case = gap x fastest plausible rotation.
static bool gapBreaksContinuity(float missingS, float rateBeforeDps, float rateAfterDps) {
  if (missingS <= 0.0f) return false;
  float worst = rateBeforeDps;
  if (rateAfterDps > worst) worst = rateAfterDps;
  if (GAP_FLOOR_DPS > worst) worst = GAP_FLOOR_DPS;
  return missingS * worst > GAP_BUDGET_DEG;
}

// roll/pitch/yaw of q relative to the reference: q_rel = conj(qRef) * q
static void relativeRPY(const Quat& ref, const Quat& q, float& r, float& p, float& y) {
  quatToRPY(qNormalize(qMul(qConj(ref), qNormalize(q))), r, p, y);
}

// ============================================================================
// 5. MPU6050 LOW-LEVEL ACCESS  (every read and write is checked)
// ============================================================================

static bool i2cWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool i2cRead(uint8_t reg, uint8_t* out, uint8_t n) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  size_t got = Wire.requestFrom((uint16_t)MPU_ADDR, (size_t)n, true);   // unambiguous overload
  if (got != n) {
    while (Wire.available()) Wire.read();
    return false;
  }
  for (uint8_t i = 0; i < n; i++) out[i] = (uint8_t)Wire.read();
  return true;
}

// Frees a stuck bus: clock SCL nine times, then send START + STOP.
static void i2cBusRecover() {
  Wire.end();
  pinMode(SDA_PIN, INPUT_PULLUP);
  pinMode(SCL_PIN, OUTPUT_OPEN_DRAIN);
  for (int i = 0; i < 9; i++) {
    digitalWrite(SCL_PIN, LOW);  delayMicroseconds(10);
    digitalWrite(SCL_PIN, HIGH); delayMicroseconds(10);
  }
  pinMode(SDA_PIN, OUTPUT_OPEN_DRAIN);
  digitalWrite(SDA_PIN, LOW);  delayMicroseconds(10);
  digitalWrite(SCL_PIN, HIGH); delayMicroseconds(10);
  digitalWrite(SDA_PIN, HIGH); delayMicroseconds(10);
  Wire.begin(SDA_PIN, SCL_PIN, I2C_HZ);
  Wire.setTimeOut(20);
}

static CfgResult mpuConfigCheck() {
  uint8_t cfg[4], pwr;
  if (!i2cRead(R_SMPLRT_DIV, cfg, 4) || !i2cRead(R_PWR_MGMT_1, &pwr, 1)) return CFG_READ_FAIL;
  return (memcmp(cfg, CFG_EXPECTED, 4) == 0 && pwr == PWR_EXPECTED) ? CFG_OK : CFG_WRONG;
}

// Full reset + configuration with known values (no read-modify-write), then verify.
// whoOut receives WHO_AM_I; unexpected receives true if it is not 0x68.
static bool mpuConfigure(uint8_t* whoOut, bool* unexpected) {
  *whoOut = 0;
  *unexpected = false;
  if (!i2cWrite(R_PWR_MGMT_1, 0x80)) return false;         // chip reset
  vTaskDelay(pdMS_TO_TICKS(100));
  if (!i2cWrite(R_SIGNAL_RESET, 0x07)) return false;       // reset gyro/accel/temp signal paths
  vTaskDelay(pdMS_TO_TICKS(100));
  const uint8_t regs[8][2] = {
    {R_PWR_MGMT_1, PWR_EXPECTED}, {R_SMPLRT_DIV, CFG_EXPECTED[0]}, {R_CONFIG, CFG_EXPECTED[1]},
    {R_GYRO_CONFIG, CFG_EXPECTED[2]}, {R_ACCEL_CONFIG, CFG_EXPECTED[3]},
    {R_INT_ENABLE, 0x00}, {R_FIFO_EN, 0x00}, {R_USER_CTRL, 0x00}};
  for (uint8_t i = 0; i < 8; i++)
    if (!i2cWrite(regs[i][0], regs[i][1])) return false;
  vTaskDelay(pdMS_TO_TICKS(50));

  // Identity check: 0x00 / 0xFF means nothing is really answering on the bus.
  uint8_t who = 0;
  if (!i2cRead(R_WHO_AM_I, &who, 1)) return false;
  *whoOut = who;
  if (who == 0x00 || who == 0xFF) return false;
  if (who != WHO_EXPECTED) *unexpected = true;             // clone part: usable, but recorded

  return mpuConfigCheck() == CFG_OK;
}

// ============================================================================
// 6. IMU TASK  (200 Hz, core 1, higher priority than the main loop)
// ============================================================================

static void imuTask(void* arg) {
  (void)arg;
  bool     running = false, seeded = false, havePrev = false, everConfigured = false;
  uint8_t  fails = 0, frozen = 0;
  int16_t  prev[6] = {0, 0, 0, 0, 0, 0};
  uint32_t lastValidUs = 0, lastCfgUs = 0, lastInitTryMs = 0;
  float    lastRateDps = 0.0f;
  Quat     q = {1.0f, 0.0f, 0.0f, 0.0f};
  TickType_t wake = xTaskGetTickCount();

  for (;;) {
    imuHeartbeatMs = millis();
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(IMU_PERIOD_US / 1000));

    // --- requests from the main loop ---
    bool testReset, doReseed = false;
    uint32_t reseedReq = 0;
    Quat requestedQ = {1.0f, 0.0f, 0.0f, 0.0f};
    portENTER_CRITICAL(&imuMux);
    testReset = imu.reqTestReset;
    imu.reqTestReset = false;
    if (imu.reseedRequest != imu.reseedAck) {
      doReseed = true;
      reseedReq = imu.reseedRequest;
      requestedQ = imu.reseedQ;
    }
    portEXIT_CRITICAL(&imuMux);
    if (testReset) i2cWrite(R_PWR_MGMT_1, 0x80);            // fault test: reset the sensor chip
    if (doReseed && running) {
      q = qNormalize(requestedQ);
      seeded = true;
      lastValidUs = micros();
      lastRateDps = 0.0f;
      portENTER_CRITICAL(&imuMux);
      imu.q = q;
      imu.epoch++;                         // a deliberate new orientation frame for this calibration
      imu.reseedAck = reseedReq;
      portEXIT_CRITICAL(&imuMux);
    }

    // --- (re)initialise the sensor when needed ---
    if (!running) {
      if (millis() - lastInitTryMs < 200) continue;
      lastInitTryMs = millis();
      uint8_t who = 0;
      bool unexpectedWho = false;
      bool ok = mpuConfigure(&who, &unexpectedWho);
      if (!ok) {
        i2cBusRecover();
        ok = mpuConfigure(&who, &unexpectedWho);
      }
      portENTER_CRITICAL(&imuMux);
      if (ok) {
        if (everConfigured) imu.reinits++;                     // recovery count excludes normal startup
        imu.epoch++;                                           // every new sensor state starts a new orientation epoch
        imu.who = who;
        if (unexpectedWho) imu.whoUnexpected++;
      } else {
        imu.who = who;
        imu.recoveryFails++;
      }
      portEXIT_CRITICAL(&imuMux);
      if (ok) {
        everConfigured = true;
        running = true; seeded = false; havePrev = false; fails = 0; frozen = 0; lastCfgUs = micros();
      }
      // Reset the pacing reference after ANY init attempt (success or failure):
      // both paths block for a few hundred ms and would otherwise leave
      // vTaskDelayUntil() catching up with a burst of zero-delay iterations.
      wake = xTaskGetTickCount();
      continue;
    }

    // --- read one sample (one immediate retry on failure) ---
    uint8_t b[14];
    bool ok = i2cRead(R_DATA, b, 14);
    if (!ok) {
      portENTER_CRITICAL(&imuMux); imu.i2cErrors++; portEXIT_CRITICAL(&imuMux);
      ok = i2cRead(R_DATA, b, 14);
    }
    bool fault = false;
    if (!ok) {
      portENTER_CRITICAL(&imuMux);
      imu.i2cErrors++;
      if (imu.capActive) imu.capMissed++;
      portEXIT_CRITICAL(&imuMux);
      if (++fails >= FAIL_LIMIT) fault = true;
      else continue;
    }

    int16_t v[7] = {0, 0, 0, 0, 0, 0, 0};
    uint32_t now = micros();
    if (!fault) {
      fails = 0;
      for (int i = 0; i < 7; i++) v[i] = (int16_t)((b[2 * i] << 8) | b[2 * i + 1]);  // ax ay az t gx gy gz
      int16_t m6[6] = {v[0], v[1], v[2], v[4], v[5], v[6]};
      if (havePrev && memcmp(m6, prev, sizeof(m6)) == 0) {
        if (++frozen >= FROZEN_LIMIT) {
          fault = true;
          portENTER_CRITICAL(&imuMux); imu.frozenEvents++; portEXIT_CRITICAL(&imuMux);
        }
      } else {
        frozen = 0;
      }
      memcpy(prev, m6, sizeof(m6));
      havePrev = true;
    }
    if (!fault && (uint32_t)(now - lastCfgUs) >= CFG_CHECK_US) {
      lastCfgUs = now;
      CfgResult c = mpuConfigCheck();
      if (c == CFG_WRONG) {
        fault = true;
        portENTER_CRITICAL(&imuMux); imu.configLost++; portEXIT_CRITICAL(&imuMux);
      } else if (c == CFG_READ_FAIL) {
        portENTER_CRITICAL(&imuMux); imu.i2cErrors++; portEXIT_CRITICAL(&imuMux);
      }
    }
    if (fault) {                      // stop, and re-initialise on the next tick
      running = false;
      lastInitTryMs = 0;
      portENTER_CRITICAL(&imuMux);
      imu.healthy = false;
      imu.faults++;
      portEXIT_CRITICAL(&imuMux);
      continue;
    }

    // --- convert units, update the filter ---
    float bx, by, bz;
    portENTER_CRITICAL(&imuMux);
    bx = biasDps[0]; by = biasDps[1]; bz = biasDps[2];
    portEXIT_CRITICAL(&imuMux);
    float gxd = v[4] / GYRO_LSB_PER_DPS - bx;
    float gyd = v[5] / GYRO_LSB_PER_DPS - by;
    float gzd = v[6] / GYRO_LSB_PER_DPS - bz;
    float rate = sqrtf(gxd * gxd + gyd * gyd + gzd * gzd);
    float axg = v[0] / ACC_LSB_PER_G, ayg = v[1] / ACC_LSB_PER_G, azg = v[2] / ACC_LSB_PER_G;

    bool invalidate = false;
    if (!seeded) {
      q = seedFromAccel(axg, ayg, azg);   // fresh start: tilt from gravity, heading 0
      seeded = true;
    } else {
      float dt = (uint32_t)(now - lastValidUs) * 1e-6f;
      float missing = dt - IMU_PERIOD_US * 1e-6f;          // time with no data beyond normal spacing
      if (missing > 0.5f * IMU_PERIOD_US * 1e-6f)          // at least one sample was missed
        invalidate = gapBreaksContinuity(missing, lastRateDps, rate);
      if (dt > 0.05f) dt = 0.05f;
      q = mahonyStep(q, gxd * DEG2RAD, gyd * DEG2RAD, gzd * DEG2RAD, axg, ayg, azg, dt);
    }
    lastValidUs = now;
    lastRateDps = rate;

    // --- publish ---
    portENTER_CRITICAL(&imuMux);
    imu.q = q;
    imu.ax = v[0]; imu.ay = v[1]; imu.az = v[2];
    imu.tempRaw = v[3];
    imu.gx = v[4]; imu.gy = v[5]; imu.gz = v[6];
    imu.sampleUs = now;
    imu.healthy = true;
    if (invalidate) { imu.epoch++; imu.gapInvalidations++; }
    if (imu.capActive && rate > imu.capMaxDps) imu.capMaxDps = rate;
    if (imu.biasBusy && imu.biasN < BIAS_SAMPLES) {
      float raw[3] = {v[4] / GYRO_LSB_PER_DPS, v[5] / GYRO_LSB_PER_DPS, v[6] / GYRO_LSB_PER_DPS};
      for (int k = 0; k < 3; k++) {
        imu.bSum[k] += raw[k];
        imu.bSq[k] += (double)raw[k] * raw[k];
        if (imu.biasN == 0 || raw[k] < imu.bMin[k]) imu.bMin[k] = raw[k];
        if (imu.biasN == 0 || raw[k] > imu.bMax[k]) imu.bMax[k] = raw[k];
      }
      imu.biasN++;
    }
    portEXIT_CRITICAL(&imuMux);
  }
}

static ImuState snapshotImu() {
  ImuState s;
  portENTER_CRITICAL(&imuMux);
  s = imu;
  portEXIT_CRITICAL(&imuMux);
  return s;
}

static bool refUsable(const ImuState& s) { return refValid && refEpoch == s.epoch; }

// ============================================================================
// 7. GYRO BIAS STORAGE (updated by the combined 5 s calibration)
// ============================================================================

static void loadBias() {
  Preferences p;
  if (!p.begin("glove", true)) return;
  float x = p.getFloat("bgx", NAN), y = p.getFloat("bgy", NAN), z = p.getFloat("bgz", NAN);
  p.end();
  if (!isnan(x) && !isnan(y) && !isnan(z)) {
    biasDps[0] = x; biasDps[1] = y; biasDps[2] = z;
    strcpy(biasSource, "flash");
  }
}

static void saveBias(float x, float y, float z) {
  Preferences p;
  if (!p.begin("glove", false)) return;
  p.putFloat("bgx", x); p.putFloat("bgy", y); p.putFloat("bgz", z);
  p.end();
}

// ============================================================================
// 8. SMALL HELPERS
// ============================================================================

static void ledSet(bool on) { if (LED_PIN >= 0) digitalWrite(LED_PIN, on ? HIGH : LOW); }

// LED: fast blink = sensor fault, slow blink = waiting for laptop,
//      steady on = connected and healthy, off = recording.
static void ledService() {
  if (LED_PIN < 0) return;
  ImuState s = snapshotImu();
  uint32_t t = millis();
  bool on;
  if (!s.healthy)                 on = (t / 100) % 2;
  else if (!client.connected())   on = (t / 500) % 2;
  else                            on = true;
  ledSet(on);
}

static float readRailMv() {
  if (RAIL_PIN < 0) return -1.0f;
  analogReadMilliVolts(RAIL_PIN);                         // discard: first read after switching channel
  uint32_t sum = 0;
  for (int i = 0; i < 16; i++) sum += analogReadMilliVolts(RAIL_PIN);
  return (sum / 16.0f) * RAIL_DIVIDER;
}

// Wait until micros() reaches target. Sleeps when far away, spins when close.
static void waitUntilUs(uint32_t target) {
  for (;;) {
    int32_t remain = (int32_t)(target - micros());
    if (remain <= 0) return;
    if (remain > 2500) vTaskDelay(1);
  }
}

// Returns false (and latches txOk = false) if the line could not be fully sent.
static bool sendText(const char* s) {
  if (!client.connected()) { txOk = false; return false; }
  size_t len = strlen(s);
  size_t n = client.write((const uint8_t*)s, len);
  if (n != len) { txOk = false; return false; }
  return true;
}

// A half-sent block would look like a valid short block to the laptop.
// Dropping the link makes the failure unambiguous; the next connect re-sends HELLO.
static void dropIfTxFailed() {
  if (!txOk) {
    client.stop();
    Serial.println("TCP write failed - connection dropped");
  }
}

static const char* resetReasonText(esp_reset_reason_t r) {
  switch (r) {
    case ESP_RST_POWERON:   return "POWER_ON";
    case ESP_RST_EXT:       return "EXTERNAL_PIN";
    case ESP_RST_SW:        return "SOFTWARE";
    case ESP_RST_PANIC:     return "CRASH";
    case ESP_RST_INT_WDT:   return "INT_WATCHDOG";
    case ESP_RST_TASK_WDT:  return "TASK_WATCHDOG";
    case ESP_RST_WDT:       return "WATCHDOG";
    case ESP_RST_DEEPSLEEP: return "DEEP_SLEEP";
    case ESP_RST_BROWNOUT:  return "BROWNOUT";
    default:                return "OTHER";
  }
}

static void softReboot(uint32_t reason) {
  rtcMagic = RTC_MAGIC;
  rtcReason = reason;
  if (client.connected()) client.stop();
  delay(50);
  ESP.restart();
}

// One ERR format for every error, so the laptop can parse key=value pairs.
// `extra` adds error-specific fields (already formatted as ",key=value").
static void sendErrEx(uint32_t seq, const char* code, const CapStats* st, const char* extra) {
  ImuState s = snapshotImu();
  int sampleField = -1;
  uint32_t ageField = 0;
  if (st && st->failSample != NO_FAIL_SAMPLE) {
    sampleField = (int)st->failSample;
    ageField = st->failAgeUs;
  }
  char line[400];
  snprintf(line, sizeof line,
           "ERR,%lu,%s,sample=%d,age_us=%lu,epoch=%lu,imu=%s,i2c_err=%lu,faults=%lu,reinits=%lu,"
           "frozen=%lu,cfg_lost=%lu,gap_inval=%lu%s\n",
           (unsigned long)seq, code, sampleField, (unsigned long)ageField,
           (unsigned long)s.epoch, s.healthy ? "OK" : "FAULT",
           (unsigned long)s.i2cErrors, (unsigned long)s.faults, (unsigned long)s.reinits,
           (unsigned long)s.frozenEvents, (unsigned long)s.configLost,
           (unsigned long)s.gapInvalidations, extra ? extra : "");
  sendText(line);
  dropIfTxFailed();
}

static void sendErr(uint32_t seq, const char* code, const CapStats* st) {
  sendErrEx(seq, code, st, nullptr);
}

// ============================================================================
// 9. CAPTURE (no network traffic while sampling)
// ============================================================================

// Returns nullptr on success, otherwise an error code.
static const char* captureBlock(uint16_t n, bool relative, CapStats& st, uint32_t& epochOut) {
  memset(&st, 0, sizeof st);
  st.failSample = NO_FAIL_SAMPLE;
  ImuState s = snapshotImu();
  if (!s.healthy) return "IMU_FAULT";
  if ((uint32_t)(micros() - s.sampleUs) > FRESH_US) return "IMU_STALE";
  if (relative && !refUsable(s)) return refValid ? "REF_INVALID" : "NO_REFERENCE";
  uint32_t epoch0 = s.epoch;
  epochOut = epoch0;

  st.railStartMv = readRailMv();
  portENTER_CRITICAL(&imuMux);
  imu.capActive = true; imu.capMaxDps = 0.0f; imu.capMissed = 0;
  portEXIT_CRITICAL(&imuMux);
  ledSet(false);                                   // LED off while recording

  const char* err = nullptr;
  uint32_t t0 = micros() + 2000;
  for (uint16_t i = 0; i < n; i++) {
    uint32_t slot = t0 + (uint32_t)i * SAMPLE_INTERVAL_US;
    uint32_t sum[5] = {0, 0, 0, 0, 0};
    uint32_t actualStart = 0;
    ImuState x{};                       // orientation sampled at the centre of this window
    uint32_t snapAge = 0;
    for (uint8_t k = 0; k < ADC_ROUNDS; k++) {
      uint32_t target = slot + (uint32_t)k * ADC_ROUND_US;
      waitUntilUs(target);
      uint32_t nowUs = micros();
      if (k == 0) actualStart = nowUs;
      uint32_t late = nowUs - target;
      if (late > st.lateMaxUs) st.lateMaxUs = late;
      for (uint8_t c = 0; c < 5; c++) {
        analogRead(FLEX_PINS[c]);                   // discard: first read after switching channel
        sum[c] += analogRead(FLEX_PINS[c]);
      }
      if (k == IMU_SNAP_ROUND) {                    // time-align q with the flex average
        x = snapshotImu();
        snapAge = micros() - x.sampleUs;
      }
    }

    // Validate the snapshot that produced the data ...
    if (!x.healthy)              err = "IMU_FAULT";
    else if (x.epoch != epoch0)  err = "IMU_EPOCH_CHANGED";
    else if (snapAge > FRESH_US) err = "IMU_STALE";
    // ... and the state at the end of the window, to catch a fault that
    // started in the second half of the sample.
    if (!err) {
      ImuState y = snapshotImu();
      if (!y.healthy)             err = "IMU_FAULT";
      else if (y.epoch != epoch0) err = "IMU_EPOCH_CHANGED";
      if (err) snapAge = micros() - y.sampleUs;
    }
    if (err) { st.failSample = i; st.failAgeUs = snapAge; break; }

    Sample& o = buf[i];
    o.t_ms = (actualStart - t0) / 1000UL;
    for (uint8_t c = 0; c < 5; c++) o.flex[c] = (uint16_t)((sum[c] + ADC_ROUNDS / 2) / ADC_ROUNDS);
    o.q = x.q;
    o.gx = x.gx; o.gy = x.gy; o.gz = x.gz;
    o.ax = x.ax; o.ay = x.ay; o.az = x.az;
    if (relative) relativeRPY(qRef, o.q, o.roll, o.pitch, o.yaw);
    else          o.roll = o.pitch = o.yaw = 0.0f;
    feedLoopWDT();
  }

  // A fault is only flagged a few samples after it starts (e.g. frozen data).
  // Wait briefly so a fault that began during the last sample is not missed.
  if (!err) {
    waitUntilUs(micros() + POST_GUARD_US);
    ImuState g = snapshotImu();
    if (!g.healthy)             err = "IMU_FAULT";
    else if (g.epoch != epoch0) err = "IMU_EPOCH_CHANGED";
    if (err) { st.failSample = n; st.failAgeUs = micros() - g.sampleUs; }
  }

  portENTER_CRITICAL(&imuMux);
  imu.capActive = false;
  st.maxDps = imu.capMaxDps;
  st.missed = imu.capMissed;
  portEXIT_CRITICAL(&imuMux);
  if (!err && st.missed > 0) {
    err = "IMU_MISSED_SAMPLE";                  // raw trial is kept out of accepted data
    st.failSample = n;
    st.failAgeUs = 0;
  }
  st.tempC = snapshotImu().tempRaw / 340.0f + 36.53f;
  st.railEndMv = readRailMv();
  return err;
}

// Sends the block only after the capture is complete.
static void sendBlock(const char* kind, uint32_t seq, uint16_t n, uint8_t trialId,
                      const CapStats& st, uint32_t epoch, bool withRef) {
  char out[1460];
  size_t used = 0;
  auto flush = [&]() {
    if (!used) return;
    if (client.connected()) {
      size_t w = client.write((const uint8_t*)out, used);
      if (w != used) txOk = false;
    } else {
      txOk = false;
    }
    used = 0;
  };
  auto add = [&](const char* line) {
    size_t len = strlen(line);
    if (used + len > sizeof out) flush();
    memcpy(out + used, line, len);
    used += len;
  };
  char line[320];
  snprintf(line, sizeof line, "BEGIN,%s,%lu,n=%u,ref=%s,epoch=%lu,boot=%08lX\n", kind,
           (unsigned long)seq, (unsigned)n, refValid ? refId : "-", (unsigned long)epoch, (unsigned long)bootId);
  add(line);
  for (uint16_t i = 0; i < n; i++) {
    const Sample& s = buf[i];
    snprintf(line, sizeof line,
             "%lu,%u,%u,%u,%u,%u,%u,%.7f,%.7f,%.7f,%.7f,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f\n",
             (unsigned long)s.t_ms, (unsigned)trialId,
             (unsigned)s.flex[0], (unsigned)s.flex[1], (unsigned)s.flex[2], (unsigned)s.flex[3], (unsigned)s.flex[4],
             s.q.w, s.q.x, s.q.y, s.q.z, s.gx, s.gy, s.gz, s.ax, s.ay, s.az, s.roll, s.pitch, s.yaw);
    add(line);
  }
  if (withRef) {
    snprintf(line, sizeof line, "REF,%lu,%.8f,%.8f,%.8f,%.8f\n", (unsigned long)seq, qRef.w, qRef.x, qRef.y, qRef.z);
    add(line);
  }
  if (st.hasCalBias) {
    snprintf(line, sizeof line,
             "STATS,%lu,max_dps=%.2f,imu_missed=%lu,late_max_us=%lu,temp_c=%.2f,rail_mv=%.0f/%.0f,"
             "bias_dps=%.4f/%.4f/%.4f,bias_std_dps=%.4f,bias_max_dev_dps=%.4f\n",
             (unsigned long)seq, st.maxDps, (unsigned long)st.missed, (unsigned long)st.lateMaxUs,
             st.tempC, st.railStartMv, st.railEndMv, st.calBiasDps[0], st.calBiasDps[1], st.calBiasDps[2],
             st.calBiasStdDps, st.calBiasMaxDevDps);
  } else {
    snprintf(line, sizeof line, "STATS,%lu,max_dps=%.2f,imu_missed=%lu,late_max_us=%lu,temp_c=%.2f,rail_mv=%.0f/%.0f\n",
             (unsigned long)seq, st.maxDps, (unsigned long)st.missed, (unsigned long)st.lateMaxUs,
             st.tempC, st.railStartMv, st.railEndMv);
  }
  add(line);
  snprintf(line, sizeof line, "END,%s,%lu\n", kind, (unsigned long)seq);
  add(line);
  flush();
  dropIfTxFailed();
}

// ============================================================================
// 10. COMMANDS FROM THE LAPTOP   ("<seq> <COMMAND> [argument]")
// ============================================================================

// Withdraw a re-seed request that was never acknowledged, so it cannot be
// applied later (which would bump the epoch in the middle of a trial).
static void cancelReseed() {
  portENTER_CRITICAL(&imuMux);
  imu.reseedAck = imu.reseedRequest;
  portEXIT_CRITICAL(&imuMux);
}

static bool reseedOrientation(const Quat& seed, uint32_t& epochOut) {
  uint32_t req;
  portENTER_CRITICAL(&imuMux);
  req = imu.reseedRequest + 1;
  if (req == 0) req = 1;
  imu.reseedQ = qNormalize(seed);
  imu.reseedRequest = req;
  portEXIT_CRITICAL(&imuMux);

  uint32_t t0 = millis();
  while (millis() - t0 < 1000) {
    delay(5);
    feedLoopWDT();
    ImuState s = snapshotImu();
    if (!s.healthy) { cancelReseed(); return false; }
    if (s.reseedAck == req) { epochOut = s.epoch; return true; }
  }
  cancelReseed();
  return false;
}

static void cmdCapture(uint32_t seq, const char* kind, uint16_t n, bool needRef, uint8_t trialId) {
  CapStats st;
  uint32_t epoch = 0;
  ImuState s = snapshotImu();
  bool relative = needRef || refUsable(s);
  const char* err = captureBlock(n, relative, st, epoch);
  if (err) { sendErr(seq, err, &st); return; }
  sendBlock(kind, seq, n, trialId, st, epoch, false);
}

static void cmdCalibrate(uint32_t seq, const char* id) {
  CapStats st;
  uint32_t captureEpoch = 0;
  refValid = false;                        // the old reference is dropped even if this one fails

  // One physical 5 s neutral hold is used for ALL participant/session baselines.
  // The recorded dataset remains 25 Hz, while gyro bias is accumulated internally
  // at the full 200 Hz IMU rate during those exact same 5 seconds.
  portENTER_CRITICAL(&imuMux);
  imu.biasBusy = true; imu.biasN = 0;
  for (int k = 0; k < 3; k++) { imu.bSum[k] = 0; imu.bSq[k] = 0; imu.bMin[k] = 0; imu.bMax[k] = 0; }
  portEXIT_CRITICAL(&imuMux);

  const char* err = captureBlock(N_CAL, false, st, captureEpoch);
  // Give the 200 Hz task a few milliseconds to deliver the final bias samples.
  uint32_t wait0 = millis();
  ImuState bs = snapshotImu();
  while (!err && bs.healthy && bs.biasN < BIAS_SAMPLES && millis() - wait0 < 150) {
    delay(5); feedLoopWDT(); bs = snapshotImu();
  }
  portENTER_CRITICAL(&imuMux); imu.biasBusy = false; portEXIT_CRITICAL(&imuMux);
  if (err) { sendErr(seq, err, &st); return; }
  bs = snapshotImu();
  if (!bs.healthy || bs.biasN < (uint16_t)(BIAS_SAMPLES * 0.90f)) {
    sendErr(seq, "CAL_BIAS_INCOMPLETE", &st);
    return;
  }

  double aSum[3] = {0, 0, 0};
  for (uint16_t i = 0; i < N_CAL; i++) {
    aSum[0] += buf[i].ax; aSum[1] += buf[i].ay; aSum[2] += buf[i].az;
  }

  float mean[3], sdMax = 0.0f, devMax = 0.0f;
  for (int k = 0; k < 3; k++) {
    mean[k] = (float)(bs.bSum[k] / bs.biasN);
    double var = bs.bSq[k] / bs.biasN - (double)mean[k] * mean[k];
    float sd = var > 0 ? (float)sqrt(var) : 0.0f;
    if (sd > sdMax) sdMax = sd;
    float d1 = fabsf(bs.bMax[k] - mean[k]), d2 = fabsf(bs.bMin[k] - mean[k]);
    if (d1 > devMax) devMax = d1;
    if (d2 > devMax) devMax = d2;
  }

  if (sdMax > BIAS_MAX_STD_DPS || devMax > BIAS_MAX_DEV_DPS) {
    char extra[96];
    snprintf(extra, sizeof extra, ",bias_std_dps=%.4f,bias_max_dev_dps=%.4f", sdMax, devMax);
    sendErrEx(seq, "CAL_MOVING", &st, extra);
    return;
  }

  // Store the session's gyro bias. It is also kept in flash for safe restart diagnostics;
  // every new/recovered session still performs this 5 s calibration again.
  portENTER_CRITICAL(&imuMux);
  biasDps[0] = mean[0]; biasDps[1] = mean[1]; biasDps[2] = mean[2];
  portEXIT_CRITICAL(&imuMux);
  saveBias(mean[0], mean[1], mean[2]);
  strcpy(biasSource, "cal5s");

  // Re-seed the live Mahony filter from the average gravity direction of the SAME hold.
  // Heading is intentionally set to zero because the MPU6050 has no magnetometer.
  Quat seed = seedFromAccel((float)(aSum[0] / N_CAL), (float)(aSum[1] / N_CAL), (float)(aSum[2] / N_CAL));
  uint32_t newEpoch = 0;
  if (!reseedOrientation(seed, newEpoch)) {
    sendErr(seq, "CAL_RESEED_FAILED", &st);
    return;
  }
  ImuState now = snapshotImu();
  if (!now.healthy || now.epoch != newEpoch) {
    sendErr(seq, "CAL_RESEED_FAILED", &st);
    return;
  }
  qRef = qNormalize(now.q);

  size_t j = 0;
  for (size_t i = 0; id[i] && j < sizeof(refId) - 1; i++) {
    char c = id[i];
    if (isalnum((unsigned char)c) || c == '_' || c == '-') refId[j++] = c;
  }
  refId[j] = 0;
  if (j == 0) strcpy(refId, "REF");
  refEpoch = newEpoch;
  refValid = true;

  // Make the saved calibration quaternion columns consistent with the newly seeded
  // orientation frame. Raw flex/gyro/accel counts are left untouched.
  // The 200 Hz peak measured during the hold is kept: a fast twitch between two
  // 25 Hz samples is invisible in the buffered data but is exactly what the
  // stillness check needs to see.
  float peak200 = st.maxDps;
  st.maxDps = 0.0f;
  for (uint16_t i = 0; i < N_CAL; i++) {
    float gxd = buf[i].gx / GYRO_LSB_PER_DPS - mean[0];
    float gyd = buf[i].gy / GYRO_LSB_PER_DPS - mean[1];
    float gzd = buf[i].gz / GYRO_LSB_PER_DPS - mean[2];
    float rate = sqrtf(gxd * gxd + gyd * gyd + gzd * gzd);
    if (rate > st.maxDps) st.maxDps = rate;
    buf[i].q = seedFromAccel((float)buf[i].ax, (float)buf[i].ay, (float)buf[i].az);
    relativeRPY(qRef, buf[i].q, buf[i].roll, buf[i].pitch, buf[i].yaw);
  }
  if (peak200 > st.maxDps) st.maxDps = peak200;
  st.hasCalBias = true;
  st.calBiasDps[0] = mean[0]; st.calBiasDps[1] = mean[1]; st.calBiasDps[2] = mean[2];
  st.calBiasStdDps = sdMax;
  st.calBiasMaxDevDps = devMax;

  sendBlock("CAL", seq, N_CAL, 0, st, newEpoch, true);
}

static void cmdStatus(uint32_t seq) {
  ImuState s = snapshotImu();
  float bx, by, bz;
  portENTER_CRITICAL(&imuMux);
  bx = biasDps[0]; by = biasDps[1]; bz = biasDps[2];
  portEXIT_CRITICAL(&imuMux);
  uint16_t f[5];
  for (int c = 0; c < 5; c++) { analogRead(FLEX_PINS[c]); f[c] = analogRead(FLEX_PINS[c]); }
  char line[560];
  snprintf(line, sizeof line,
           "STATUS,%lu,fw=%s,boot=%08lX,uptime_s=%lu,epoch=%lu,ref=%s,ref_ok=%d,imu=%s,i2c_err=%lu,faults=%lu,"
           "reinits=%lu,frozen=%lu,cfg_lost=%lu,gap_inval=%lu,rec_fail=%lu,who_bad=%lu,bias_src=%s,"
           "bias_dps=%.3f/%.3f/%.3f,temp_c=%.2f,rssi=%d,rail_mv=%.0f,flex=%u/%u/%u/%u/%u,heap=%lu,who=0x%02X\n",
           (unsigned long)seq, FW_VERSION, (unsigned long)bootId, (unsigned long)(millis() / 1000),
           (unsigned long)s.epoch, refValid ? refId : "-", refUsable(s) ? 1 : 0, s.healthy ? "OK" : "FAULT",
           (unsigned long)s.i2cErrors, (unsigned long)s.faults, (unsigned long)s.reinits,
           (unsigned long)s.frozenEvents, (unsigned long)s.configLost, (unsigned long)s.gapInvalidations,
           (unsigned long)s.recoveryFails, (unsigned long)s.whoUnexpected, biasSource, bx, by, bz,
           s.tempRaw / 340.0f + 36.53f, (int)WiFi.RSSI(), readRailMv(), f[0], f[1], f[2], f[3], f[4],
           (unsigned long)ESP.getFreeHeap(), s.who);
  sendText(line);
  dropIfTxFailed();
}

static void handleCommand(char* line) {
  char* p = line;
  while (*p == ' ') p++;
  uint32_t seq = strtoul(p, &p, 10);
  while (*p == ' ') p++;
  char* cmd = p;
  char* arg = strchr(p, ' ');
  if (arg) { *arg++ = 0; while (*arg == ' ') arg++; } else { arg = p + strlen(p); }
  for (char* c = arg + strlen(arg); c > arg && (c[-1] == '\r' || c[-1] == ' '); c--) c[-1] = 0;
  for (char* c = cmd + strlen(cmd); c > cmd && c[-1] == '\r'; c--) c[-1] = 0;

  char out[200];
  if (!strcmp(cmd, "PING")) {
    ImuState s = snapshotImu();
    snprintf(out, sizeof out, "PONG,%lu,epoch=%lu,ref=%s,ref_ok=%d,imu=%s,boot=%08lX\n", (unsigned long)seq,
             (unsigned long)s.epoch, refValid ? refId : "-", refUsable(s) ? 1 : 0, s.healthy ? "OK" : "FAULT",
             (unsigned long)bootId);
    sendText(out);
    dropIfTxFailed();
  } else if (!strcmp(cmd, "STATUS"))  { cmdStatus(seq);
  } else if (!strcmp(cmd, "CAL"))     { cmdCalibrate(seq, arg);
  } else if (!strcmp(cmd, "FIST"))    { cmdCapture(seq, "FIST", N_FIST, false, 0);
  } else if (!strcmp(cmd, "CHECK"))   { cmdCapture(seq, "CHECK", N_CHECK, true, 0);
  } else if (!strcmp(cmd, "DRIFT"))   { cmdCapture(seq, "DRIFT", N_DRIFT, true, 0);
  } else if (!strcmp(cmd, "TRIAL")) {
    long t = strtol(arg, nullptr, 10);
    if (t < 1 || t > 99) { sendErr(seq, "INVALID_TRIAL_NUMBER", nullptr); return; }
    cmdCapture(seq, "TRIAL", N_TRIAL, true, (uint8_t)t);
  } else if (!strcmp(cmd, "TEST") && !strcmp(arg, "MPU_RESET")) {
    portENTER_CRITICAL(&imuMux); imu.reqTestReset = true; portEXIT_CRITICAL(&imuMux);
    snprintf(out, sizeof out, "TEST,%lu,OK,MPU_RESET\n", (unsigned long)seq);
    sendText(out);
    dropIfTxFailed();
  } else if (!strcmp(cmd, "TEST") && !strcmp(arg, "REBOOT")) {
    snprintf(out, sizeof out, "TEST,%lu,OK,REBOOT\n", (unsigned long)seq);
    sendText(out);
    softReboot(SW_TEST_REBOOT);
  } else {
    sendErr(seq, "UNKNOWN_COMMAND", nullptr);
  }
}

// ============================================================================
// 11. NETWORK (never blocks the IMU task)
// ============================================================================

static void sendHello() {
  ImuState s = snapshotImu();
  char line[320];
  snprintf(line, sizeof line,
           "HELLO,%s,fw=%s,boot=%08lX,reset=%s,sw=%s,epoch=%lu,ref_ok=%d,bias=%s,who=0x%02X,rail=%d,"
           "gyro_lsb_per_dps=%.1f,acc_lsb_per_g=%.0f\n",
           FW_NAME, FW_VERSION, (unsigned long)bootId, resetText, swText, (unsigned long)s.epoch,
           refUsable(s) ? 1 : 0, biasSource, s.who, RAIL_PIN >= 0 ? 1 : 0, GYRO_LSB_PER_DPS, ACC_LSB_PER_G);
  sendText(line);
}

static void networkService() {
  if (WiFi.status() != WL_CONNECTED) {
    if (wifiLostSinceMs == 0) wifiLostSinceMs = millis();
    if (millis() - wifiLostSinceMs > 20000) {          // auto-reconnect did not work: start again
      WiFi.disconnect();
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
      wifiLostSinceMs = millis();
    }
    return;
  }
  wifiLostSinceMs = 0;

  if (!client.connected()) {
    if (millis() - lastTcpTryMs < 1000) return;
    lastTcpTryMs = millis();
    client.stop();
    if (client.connect(serverIp, SERVER_PORT, 1000)) {
      client.setNoDelay(true);
      lineLen = 0;
      txOk = true;                                      // fresh link: clear the write-failure latch
      sendHello();
    }
    return;
  }

  while (client.available()) {
    int c = client.read();
    if (c < 0) break;
    if (c == '\n') {
      lineBuf[lineLen] = 0;
      if (lineLen > 0) handleCommand(lineBuf);
      lineLen = 0;
      if (!client.connected()) break;                   // a command may have dropped the link
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = (char)c;
    } else {
      lineLen = 0;                                      // over-long line: drop it
    }
  }
}

// ============================================================================
// 12. SETUP AND LOOP
// ============================================================================

void setup() {
  Serial.begin(115200);                                 // optional: only useful when USB is plugged in
  resetText = resetReasonText(esp_reset_reason());
  if (rtcMagic == RTC_MAGIC) {
    swText = rtcReason == SW_IMU_STALL ? "IMU_TASK_STALL" : rtcReason == SW_TEST_REBOOT ? "TEST_REBOOT" : "OTHER";
  }
  rtcMagic = 0;

  if (LED_PIN >= 0) { pinMode(LED_PIN, OUTPUT); ledSet(false); }

  analogReadResolution(12);
  for (uint8_t c = 0; c < 5; c++) { analogSetPinAttenuation(FLEX_PINS[c], ADC_11db); analogRead(FLEX_PINS[c]); }
  if (RAIL_PIN >= 0) { analogSetPinAttenuation(RAIL_PIN, ADC_11db); analogRead(RAIL_PIN); }

  loadBias();

  memset(&imu, 0, sizeof imu);
  imu.q = Quat{1.0f, 0.0f, 0.0f, 0.0f};
  Wire.begin(SDA_PIN, SCL_PIN, I2C_HZ);
  Wire.setTimeOut(20);
  imuHeartbeatMs = millis();
  xTaskCreatePinnedToCore(imuTask, "imu", 6144, nullptr, 3, nullptr, 1);

  serverIp.fromString(SERVER_IP);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  bootId = esp_random() ^ (uint32_t)micros();           // radio is on now, so the random source is good

  enableLoopWDT();                                      // reboot if the main loop ever hangs
  Serial.printf("%s %s boot=%08lX reset=%s\n", FW_NAME, FW_VERSION, (unsigned long)bootId, resetText);
}

void loop() {
  if (millis() - imuHeartbeatMs > 2000) softReboot(SW_IMU_STALL);   // IMU task stopped: restart
  networkService();
  ledService();
  delay(1);
}
