// ============================================================================
// SMART GLOVE - STANDALONE ESP32 F5 LOGISTIC-REGRESSION + BUTTON + LED + AUDIO
// ----------------------------------------------------------------------------
// Normal operation (no laptop, Wi-Fi or Python required):
//   1) Power on -> LED shows calibration required.
//   2) Wear glove in the same neutral pose used for data collection.
//   3) Double-click button -> 1.5 s settle + 5 s neutral calibration.
//   4) Slow LED blink -> ready. Form one of the 10 trained static gestures.
//   5) Single-click -> 2 s / 50-sample capture -> F5 features -> StandardScaler
//      -> multinomial Logistic Regression -> G01..G10 prediction.
//   6) ESP32 commands DFPlayer Mini to play /mp3/0001.mp3 ... /mp3/0010.mp3.
//
// Core sensor acquisition, F5 feature mathematics and model inference are derived
// from the previously validated SmartGlove_Live_F5_LR_calibration_v3 firmware.
// ============================================================================

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>   // retained only because legacy diagnostic helpers below reference WiFi types; Wi-Fi is NOT started
#include <Preferences.h>
#include <DFRobotDFPlayerMini.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include "esp_system.h"
#include "esp_attr.h"
#include "model_params.h"
#if __has_include("esp_random.h")
#include "esp_random.h"
#endif

// ============================================================================
// 1. STANDALONE HARDWARE SETTINGS
// ============================================================================

// Existing glove sensor pins
static const uint8_t FLEX_PINS[5] = {32, 33, 34, 35, 36};  // thumb, index, middle, ring, little
static const int     SDA_PIN       = 21;
static const int     SCL_PIN       = 22;

// New standalone interface pins
static const int     BUTTON_PIN    = 27;  // button between GPIO27 and GND; INPUT_PULLUP
static const int     LED_PIN       = 4;   // GPIO4 -> 220 ohm -> LED anode; LED cathode -> GND
static const int     DFPLAYER_RX_PIN = 26; // ESP32 RX2  <- DFPlayer TX
static const int     DFPLAYER_TX_PIN = 25; // ESP32 TX2  -> 1k ohm -> DFPlayer RX

// Optional old 3.3 V rail monitor. Leave -1 if not fitted.
static const int     RAIL_PIN      = -1;
static const float   RAIL_DIVIDER  = 2.0f;

// Button behavior
static const uint32_t BUTTON_DEBOUNCE_MS = 30;
static const uint32_t DOUBLE_CLICK_MS     = 350;

// Audio
static const uint8_t  DFPLAYER_VOLUME     = 24;   // 0..30
static const uint32_t AUDIO_TIMEOUT_MS     = 7000; // fallback if play-finished event is missed

// Legacy network constants only keep unused diagnostic functions compilable.
// setup() never starts Wi-Fi and loop() never calls networkService().
static const char*    WIFI_SSID     = "";
static const char*    WIFI_PASSWORD = "";
static const char*    SERVER_IP     = "0.0.0.0";
static const uint16_t SERVER_PORT   = 5000;

// ============================================================================
// 2. FIXED SETTINGS (these match the protocol - do not change casually)
// ============================================================================

static const char*    FW_NAME            = "SMART_GLOVE_STANDALONE_F5_LR";
static const char*    FW_VERSION         = "2.0.0";

static const uint32_t SAMPLE_INTERVAL_US = 40000;   // 25 Hz
static const uint16_t N_TRIAL = 50, N_CAL = 125, N_FIST = 75, N_CHECK = 50, N_DRIFT = 125;
static const uint16_t MAX_SAMPLES        = 125;
static const uint8_t  ADC_ROUNDS         = 8;       // 8 reads per finger per sample ...
static const uint32_t ADC_ROUND_US       = 2500;    // ... 2.5 ms apart = one 20 ms mains cycle

static const uint32_t I2C_HZ             = 100000;  // conservative bus speed; wiring quality is still the main reliability factor
static const uint8_t  MPU_ADDR           = 0x68;
static const uint32_t IMU_PERIOD_US      = 5000;    // 200 Hz sensor loop
static const float    GYRO_LSB_PER_DPS   = 16.4f;   // +-2000 deg/s range
static const float    ACC_LSB_PER_G      = 8192.0f; // +-4 g range
static const float    MAHONY_KP          = 1.0f;    // tilt correction strength (about 1 s time constant)
static const float    ACC_GATE_G         = 0.15f;   // use gravity only when |a| is within 1 +- 0.15 g
static const float    GAP_BUDGET_DEG     = 1.0f;    // largest heading error a data gap may cause
static const float    GAP_FLOOR_DPS      = 50.0f;   // rotation speed assumed during an unseen gap
static const uint8_t  FROZEN_LIMIT       = 3;       // identical readings in a row = frozen sensor
static const uint8_t  FAIL_LIMIT         = 3;       // failed reads in a row = sensor fault
static const uint32_t CFG_CHECK_US       = 100000;  // re-check the sensor settings every 100 ms
static const uint32_t FRESH_US           = 15000;   // an IMU sample older than this is "stale"
static const uint32_t POST_GUARD_US      = 30000;   // after a capture, wait this long for late fault flags
static const uint16_t BIAS_SAMPLES       = 1000;    // 5 s x 200 Hz, collected during the SAME neutral hold
static const float    BIAS_MAX_STD_DPS   = 1.0f;    // legacy data-collection threshold (unused by live calibration)
static const float    BIAS_MAX_DEV_DPS   = 5.0f;    // legacy data-collection threshold (unused by live calibration)

// Live calibration stability policy (v1.0.1).
// GOOD:   std <= 5 dps AND max deviation <= 20 dps -> accept normally.
// WARNING: above GOOD but within 8/30 -> accept, but report a warning.
// BAD:    std > 8 dps OR max deviation > 30 dps -> reject and recalibrate.
// This keeps a hard safety limit while avoiding repeated rejection of reasonably
// steady human participants during live use.
static const float    LIVE_CAL_GOOD_STD_DPS = 5.0f;
static const float    LIVE_CAL_GOOD_DEV_DPS = 20.0f;
static const float    LIVE_CAL_HARD_STD_DPS = 8.0f;
static const float    LIVE_CAL_HARD_DEV_DPS = 30.0f;
static const uint32_t LIVE_LATE_MAX_US      = 3000;

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
// ============================================================================

struct Quat { float w, x, y, z; };

// Written by the IMU task, read by the main loop (always under imuMux).
struct ImuState {
  Quat     q;
  int16_t  ax, ay, az, gx, gy, gz, tempRaw;
  uint32_t sampleUs;          // micros() of the newest valid sample
  bool     healthy;           // false while the sensor is faulty / being repaired
  uint32_t epoch;             // changes whenever orientation continuity may be lost
  uint32_t i2cErrors, faults, reinits, frozenEvents, configLost, gapInvalidations, recoveryFails;
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

static ImuState          imu;
static portMUX_TYPE      imuMux = portMUX_INITIALIZER_UNLOCKED;
static volatile uint32_t imuHeartbeatMs = 0;
static float             biasDps[3] = {0.0f, 0.0f, 0.0f};   // protected by imuMux
static char              biasSource[10] = "none";

struct Sample {
  uint32_t t_ms;
  uint16_t flex[5];
  Quat     q;
  int16_t  gx, gy, gz, ax, ay, az;
  float    roll, pitch, yaw;
};
static Sample buf[MAX_SAMPLES];

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

// Calibration reference (main loop only)
static Quat     qRef     = {1.0f, 0.0f, 0.0f, 0.0f};
static bool     refValid = false;
static uint32_t refEpoch = 0;

static char     refId[24] = "-";

// Live F5 deployment state.
// F5 uses current-session neutral flex means, neutral acceleration means,
// and current-session full-rate gyro bias. No fist reference is used.
static bool     liveCalValid = false;
static uint32_t liveCalEpoch = 0;
static float    liveNeutralFlex[5] = {0, 0, 0, 0, 0};
static float    liveNeutralAccelG[3] = {0, 0, 0};
static float    liveGyroBiasDps[3] = {0, 0, 0};

static bool     liveCaptureReady = false;
static uint32_t liveCaptureEpoch = 0;
static float    liveFeatures[MODEL_N_FEATURES] = {0};
static float    liveScores[MODEL_N_CLASSES] = {0};

// Identity / reset information
static uint32_t    bootId    = 0;
static const char* resetText = "UNKNOWN";
static const char* swText    = "NONE";
#define RTC_MAGIC 0x5A17C0DEUL
RTC_NOINIT_ATTR static uint32_t rtcMagic;
RTC_NOINIT_ATTR static uint32_t rtcReason;
enum SwReason : uint32_t { SW_NONE = 0, SW_IMU_STALL = 1, SW_TEST_REBOOT = 2 };

// Network
static WiFiClient client;
static IPAddress  serverIp;
static char       lineBuf[96];
static uint8_t    lineLen = 0;
static uint32_t   lastTcpTryMs = 0, wifiLostSinceMs = 0;

// Standalone UI/audio state
static HardwareSerial dfSerial(2);
static DFRobotDFPlayerMini dfPlayer;
static bool dfReady = false;

static const int UI_NEED_CAL = 0;
static const int UI_READY    = 1;
static const int UI_AUDIO    = 2;
static int uiState = UI_NEED_CAL;

static const int CAP_LED_NONE    = 0;
static const int CAP_LED_CAL     = 1;
static const int CAP_LED_GESTURE = 2;
static int captureLedMode = CAP_LED_NONE;

static const int BTN_NONE   = 0;
static const int BTN_SINGLE = 1;
static const int BTN_DOUBLE = 2;

static bool buttonRawLast = HIGH;
static bool buttonStable = HIGH;
static uint32_t buttonChangedMs = 0;
static uint8_t clickCount = 0;
static uint32_t firstClickMs = 0;

static uint32_t audioStartedMs = 0;
static int lastPredictedClass = -1;

// Forward declarations (so this file also compiles as plain C++)
static bool i2cWrite(uint8_t reg, uint8_t val);
static bool i2cRead(uint8_t reg, uint8_t* out, uint8_t n);
static void imuTask(void* arg);
static ImuState snapshotImu();
static void softReboot(uint32_t reason);

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

// Sign-aligned average of quaternions (q and -q are the same orientation).
static Quat averageQuat(const Sample* s, uint16_t n) {
  Quat a = qNormalize(s[0].q);
  double w = 0, x = 0, y = 0, z = 0;
  for (uint16_t i = 0; i < n; i++) {
    Quat q = qNormalize(s[i].q);
    float sg = (a.w * q.w + a.x * q.x + a.y * q.y + a.z * q.z) < 0.0f ? -1.0f : 1.0f;
    w += sg * q.w; x += sg * q.x; y += sg * q.y; z += sg * q.z;
  }
  return qNormalize(Quat{(float)(w / n), (float)(x / n), (float)(y / n), (float)(z / n)});
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
  size_t got = Wire.requestFrom((uint8_t)MPU_ADDR, (size_t)n, true);
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

static const int CFG_OK = 0;
static const int CFG_READ_FAIL = 1;
static const int CFG_WRONG = 2;

static int mpuConfigCheck() {
  uint8_t cfg[4], pwr;
  if (!i2cRead(R_SMPLRT_DIV, cfg, 4) || !i2cRead(R_PWR_MGMT_1, &pwr, 1)) return CFG_READ_FAIL;
  return (memcmp(cfg, CFG_EXPECTED, 4) == 0 && pwr == PWR_EXPECTED) ? CFG_OK : CFG_WRONG;
}

// Full reset + configuration with known values (no read-modify-write), then verify.
static bool mpuConfigure() {
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
      bool ok = mpuConfigure();
      if (!ok) { i2cBusRecover(); ok = mpuConfigure(); }
      uint8_t who = 0;
      if (ok) i2cRead(R_WHO_AM_I, &who, 1);
      portENTER_CRITICAL(&imuMux);
      if (ok) {
        if (everConfigured) imu.reinits++;                     // recovery count excludes normal startup
        imu.epoch++;                                           // every new sensor state starts a new orientation epoch
        imu.who = who;
      } else {
        imu.recoveryFails++;
      }
      portEXIT_CRITICAL(&imuMux);
      if (ok) {
        everConfigured = true;
        running = true; seeded = false; havePrev = false; fails = 0; frozen = 0; lastCfgUs = micros();
      }
      wake = xTaskGetTickCount();                            // no catch-up burst after the slow re-init
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

    int16_t v[7];
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
      int c = mpuConfigCheck();
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

static void sendText(const char* s) {
  if (client.connected()) client.write((const uint8_t*)s, strlen(s));
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

static void sendErr(uint32_t seq, const char* code, const CapStats* st) {
  ImuState s = snapshotImu();
  char line[240];
  snprintf(line, sizeof line,
           "ERR,%lu,%s,sample=%d,age_us=%lu,epoch=%lu,imu=%s,i2c_err=%lu,faults=%lu,reinits=%lu,"
           "frozen=%lu,cfg_lost=%lu,gap_inval=%lu\n",
           (unsigned long)seq, code, st ? (int)st->failSample : -1,
           (unsigned long)(st ? st->failAgeUs : 0), (unsigned long)s.epoch, s.healthy ? "OK" : "FAULT",
           (unsigned long)s.i2cErrors, (unsigned long)s.faults, (unsigned long)s.reinits,
           (unsigned long)s.frozenEvents, (unsigned long)s.configLost, (unsigned long)s.gapInvalidations);
  sendText(line);
}

// ============================================================================
// 9. CAPTURE (no network traffic while sampling)
// ============================================================================

// Returns nullptr on success, otherwise an error code.
static const char* captureBlock(uint16_t n, bool relative, CapStats& st, uint32_t& epochOut) {
  memset(&st, 0, sizeof st);
  st.failSample = 0xFFFF;
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
  // Standalone LED feedback while the blocking capture loop is running.
  if (captureLedMode == CAP_LED_GESTURE) ledSet(true);
  else ledSet(false);

  const char* err = nullptr;
  uint32_t t0 = micros() + 2000;
  for (uint16_t i = 0; i < n; i++) {
    if (captureLedMode == CAP_LED_CAL) ledSet(((millis() / 100U) % 2U) == 0U);
    else if (captureLedMode == CAP_LED_GESTURE) ledSet(true);
    uint32_t slot = t0 + (uint32_t)i * SAMPLE_INTERVAL_US;
    uint32_t sum[5] = {0, 0, 0, 0, 0};
    uint32_t actualStart = 0;
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
    }
    ImuState x = snapshotImu();
    uint32_t age = micros() - x.sampleUs;
    if (!x.healthy)              err = "IMU_FAULT";
    else if (x.epoch != epoch0)  err = "IMU_EPOCH_CHANGED";
    else if (age > FRESH_US)     err = "IMU_STALE";
    if (err) { st.failSample = i; st.failAgeUs = age; break; }

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
  auto flush = [&]() { if (used) { if (client.connected()) client.write((const uint8_t*)out, used); used = 0; } };
  auto add = [&](const char* line) {
    size_t len = strlen(line);
    if (used + len > sizeof out) flush();
    memcpy(out + used, line, len);
    used += len;
  };
  char line[200];
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
}


// ============================================================================
// 9B. LIVE F5 FEATURE EXTRACTION + LOGISTIC REGRESSION
// ============================================================================

static float roundDecimals(float x, float scale) {
  return roundf(x * scale) / scale;
}

static void sortFloat(float* a, int n) {
  // Insertion sort is small, deterministic and sufficient for n=50.
  for (int i = 1; i < n; i++) {
    float key = a[i];
    int j = i - 1;
    while (j >= 0 && a[j] > key) {
      a[j + 1] = a[j];
      j--;
    }
    a[j + 1] = key;
  }
}

static float median50(const float* src) {
  float a[N_TRIAL];
  for (int i = 0; i < N_TRIAL; i++) a[i] = src[i];
  sortFloat(a, N_TRIAL);
  return 0.5f * (a[24] + a[25]);
}

static float percentile95_50(const float* src) {
  float a[N_TRIAL];
  for (int i = 0; i < N_TRIAL; i++) a[i] = src[i];
  sortFloat(a, N_TRIAL);
  // NumPy percentile(method="linear"), n=50:
  // h=(50-1)*0.95=46.55 -> a[46] + 0.55*(a[47]-a[46]).
  return a[46] + 0.55f * (a[47] - a[46]);
}

static float sampleSd50(const float* src) {
  double sum = 0.0;
  for (int i = 0; i < N_TRIAL; i++) sum += src[i];
  double mean = sum / (double)N_TRIAL;
  double ss = 0.0;
  for (int i = 0; i < N_TRIAL; i++) {
    double d = (double)src[i] - mean;
    ss += d * d;
  }
  return (float)sqrt(ss / (double)(N_TRIAL - 1));  // ddof=1, exactly as training
}

static bool computeF5Features(float out[MODEL_N_FEATURES]) {
  if (!liveCalValid || !liveCaptureReady) return false;

  float tmp[N_TRIAL];

  // 1-5: trial flex median minus current-session neutral flex mean.
  for (int c = 0; c < 5; c++) {
    for (int i = 0; i < N_TRIAL; i++) tmp[i] = (float)buf[i].flex[c];
    out[c] = median50(tmp) - liveNeutralFlex[c];
  }

  // Component-wise median trial acceleration in g.
  float aMed[3];
  for (int axis = 0; axis < 3; axis++) {
    for (int i = 0; i < N_TRIAL; i++) {
      int16_t raw = axis == 0 ? buf[i].ax : (axis == 1 ? buf[i].ay : buf[i].az);
      tmp[i] = (float)raw / ACC_LSB_PER_G;
    }
    aMed[axis] = median50(tmp);
  }

  float an = sqrtf(aMed[0]*aMed[0] + aMed[1]*aMed[1] + aMed[2]*aMed[2]);
  float n0 = sqrtf(liveNeutralAccelG[0]*liveNeutralAccelG[0] +
                   liveNeutralAccelG[1]*liveNeutralAccelG[1] +
                   liveNeutralAccelG[2]*liveNeutralAccelG[2]);
  if (!(an > 0.0f) || !(n0 > 0.0f)) return false;

  // 6-8: u - u0, the exact F5 relative-gravity definition used in training.
  out[5] = aMed[0] / an - liveNeutralAccelG[0] / n0;
  out[6] = aMed[1] / an - liveNeutralAccelG[1] / n0;
  out[7] = aMed[2] / an - liveNeutralAccelG[2] / n0;

  // 9-10: acceleration-magnitude median and sample SD in g.
  float accMag[N_TRIAL];
  for (int i = 0; i < N_TRIAL; i++) {
    float ax = (float)buf[i].ax / ACC_LSB_PER_G;
    float ay = (float)buf[i].ay / ACC_LSB_PER_G;
    float az = (float)buf[i].az / ACC_LSB_PER_G;
    accMag[i] = sqrtf(ax*ax + ay*ay + az*az);
  }
  out[8] = median50(accMag);
  out[9] = sampleSd50(accMag);

  // 11-12: gyro magnitude AFTER axis-wise current-session bias subtraction.
  float gyroMag[N_TRIAL];
  for (int i = 0; i < N_TRIAL; i++) {
    float gx = (float)buf[i].gx / GYRO_LSB_PER_DPS - liveGyroBiasDps[0];
    float gy = (float)buf[i].gy / GYRO_LSB_PER_DPS - liveGyroBiasDps[1];
    float gz = (float)buf[i].gz / GYRO_LSB_PER_DPS - liveGyroBiasDps[2];
    gyroMag[i] = sqrtf(gx*gx + gy*gy + gz*gz);
  }
  out[10] = median50(gyroMag);
  out[11] = percentile95_50(gyroMag);

  for (int j = 0; j < MODEL_N_FEATURES; j++) {
    if (!isfinite(out[j])) return false;
  }
  return true;
}

static int predictF5(const float feat[MODEL_N_FEATURES],
                     float scoreOut[MODEL_N_CLASSES]) {
  float z[MODEL_N_FEATURES];
  for (int j = 0; j < MODEL_N_FEATURES; j++) {
    z[j] = (feat[j] - MODEL_SCALER_MEAN[j]) / MODEL_SCALER_SCALE[j];
  }

  int best = 0;
  for (int k = 0; k < MODEL_N_CLASSES; k++) {
    float s = MODEL_INTERCEPT[k];
    for (int j = 0; j < MODEL_N_FEATURES; j++) {
      s += MODEL_COEF[k][j] * z[j];
    }
    scoreOut[k] = s;
    if (k == 0 || s > scoreOut[best]) best = k;
  }
  return best;
}

static bool liveCalibrationStillValid() {
  if (!liveCalValid) return false;
  ImuState s = snapshotImu();
  return s.healthy && s.epoch == liveCalEpoch;
}

static void cmdLiveCalibrate(uint32_t seq) {
  liveCalValid = false;
  liveCaptureReady = false;

  CapStats st;
  uint32_t captureEpoch = 0;

  // Same physical 5 s neutral hold as data collection:
  // 125 saved-rate samples plus full-rate 200 Hz gyro bias accumulation.
  portENTER_CRITICAL(&imuMux);
  imu.biasBusy = true;
  imu.biasN = 0;
  for (int k = 0; k < 3; k++) {
    imu.bSum[k] = 0;
    imu.bSq[k] = 0;
    imu.bMin[k] = 0;
    imu.bMax[k] = 0;
  }
  portEXIT_CRITICAL(&imuMux);

  const char* err = captureBlock(N_CAL, false, st, captureEpoch);

  uint32_t wait0 = millis();
  ImuState bs = snapshotImu();
  while (!err && bs.healthy && bs.biasN < BIAS_SAMPLES && millis() - wait0 < 150) {
    delay(5);
    feedLoopWDT();
    bs = snapshotImu();
  }
  portENTER_CRITICAL(&imuMux);
  imu.biasBusy = false;
  portEXIT_CRITICAL(&imuMux);

  if (err) {
    sendErr(seq, err, &st);
    return;
  }

  bs = snapshotImu();
  if (!bs.healthy || bs.biasN < (uint16_t)(BIAS_SAMPLES * 0.90f)) {
    sendErr(seq, "CAL_BIAS_INCOMPLETE", &st);
    return;
  }

  float bias[3];
  float sdMax = 0.0f, devMax = 0.0f;
  for (int k = 0; k < 3; k++) {
    bias[k] = (float)(bs.bSum[k] / bs.biasN);
    double var = bs.bSq[k] / bs.biasN - (double)bias[k] * bias[k];
    float sd = var > 0.0 ? (float)sqrt(var) : 0.0f;
    if (sd > sdMax) sdMax = sd;
    float d1 = fabsf(bs.bMax[k] - bias[k]);
    float d2 = fabsf(bs.bMin[k] - bias[k]);
    if (d1 > devMax) devMax = d1;
    if (d2 > devMax) devMax = d2;
  }

  // Three-level calibration decision:
  //   GOOD    -> accept normally
  //   WARNING -> accept, but tell the controller that motion was elevated
  //   BAD     -> reject and require another calibration
  const bool calHardFail =
      (sdMax > LIVE_CAL_HARD_STD_DPS) || (devMax > LIVE_CAL_HARD_DEV_DPS);
  const bool calWarning =
      (sdMax > LIVE_CAL_GOOD_STD_DPS) || (devMax > LIVE_CAL_GOOD_DEV_DPS);

  if (calHardFail) {
    char line[300];
    snprintf(line, sizeof line,
             "ERR,%lu,CAL_MOVING,bias_std_dps=%.4f,bias_max_dev_dps=%.4f,"
             "hard_std_limit=%.1f,hard_dev_limit=%.1f\n",
             (unsigned long)seq, sdMax, devMax,
             LIVE_CAL_HARD_STD_DPS, LIVE_CAL_HARD_DEV_DPS);
    sendText(line);
    return;
  }

  // Match the precision that the training pipeline received from the
  // original saved calibration_summary.csv:
  // flex means 3 decimals, accel means 6 decimals, firmware gyro bias 4 decimals.
  for (int c = 0; c < 5; c++) {
    double sum = 0.0;
    for (int i = 0; i < N_CAL; i++) sum += buf[i].flex[c];
    liveNeutralFlex[c] = roundDecimals((float)(sum / (double)N_CAL), 1000.0f);
  }

  double aSum[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < N_CAL; i++) {
    aSum[0] += buf[i].ax;
    aSum[1] += buf[i].ay;
    aSum[2] += buf[i].az;
  }
  for (int k = 0; k < 3; k++) {
    float g = (float)(aSum[k] / (double)N_CAL / (double)ACC_LSB_PER_G);
    liveNeutralAccelG[k] = roundDecimals(g, 1000000.0f);
    liveGyroBiasDps[k] = roundDecimals(bias[k], 10000.0f);
  }

  // The IMU task also uses this same current-session bias.
  portENTER_CRITICAL(&imuMux);
  biasDps[0] = liveGyroBiasDps[0];
  biasDps[1] = liveGyroBiasDps[1];
  biasDps[2] = liveGyroBiasDps[2];
  portEXIT_CRITICAL(&imuMux);
  saveBias(liveGyroBiasDps[0], liveGyroBiasDps[1], liveGyroBiasDps[2]);
  strcpy(biasSource, "live5s");

  liveCalEpoch = captureEpoch;
  liveCalValid = true;

  char line[520];
  snprintf(line, sizeof line,
           "CALOK,%lu,epoch=%lu,quality=%s,flex=%.3f/%.3f/%.3f/%.3f/%.3f,"
           "acc=%.6f/%.6f/%.6f,bias=%.4f/%.4f/%.4f,"
           "bias_std=%.4f,bias_dev=%.4f,good_std_limit=%.1f,good_dev_limit=%.1f\n",
           (unsigned long)seq, (unsigned long)liveCalEpoch,
           calWarning ? "WARN" : "GOOD",
           liveNeutralFlex[0], liveNeutralFlex[1], liveNeutralFlex[2],
           liveNeutralFlex[3], liveNeutralFlex[4],
           liveNeutralAccelG[0], liveNeutralAccelG[1], liveNeutralAccelG[2],
           liveGyroBiasDps[0], liveGyroBiasDps[1], liveGyroBiasDps[2],
           sdMax, devMax, LIVE_CAL_GOOD_STD_DPS, LIVE_CAL_GOOD_DEV_DPS);
  sendText(line);
}

static void cmdLiveCapture(uint32_t seq) {
  liveCaptureReady = false;

  if (!liveCalibrationStillValid()) {
    sendErr(seq, "NEED_CAL", nullptr);
    return;
  }

  CapStats st;
  uint32_t epoch = 0;
  const char* err = captureBlock(N_TRIAL, false, st, epoch);
  if (err) {
    sendErr(seq, err, &st);
    return;
  }

  if (st.lateMaxUs > LIVE_LATE_MAX_US) {
    sendErr(seq, "TIMING_LATE", &st);
    return;
  }

  if (epoch != liveCalEpoch) {
    sendErr(seq, "NEED_CAL", &st);
    return;
  }

  liveCaptureEpoch = epoch;
  liveCaptureReady = true;

  char line[240];
  snprintf(line, sizeof line,
           "CAPTURED,%lu,n=%u,epoch=%lu,max_dps=%.2f,late_max_us=%lu,temp_c=%.2f\n",
           (unsigned long)seq, (unsigned)N_TRIAL, (unsigned long)epoch,
           st.maxDps, (unsigned long)st.lateMaxUs, st.tempC);
  sendText(line);
}

static void cmdLiveDiscard(uint32_t seq) {
  liveCaptureReady = false;
  char line[64];
  snprintf(line, sizeof line, "DISCARDED,%lu\n", (unsigned long)seq);
  sendText(line);
}

static void cmdLivePredict(uint32_t seq) {
  if (!liveCaptureReady) {
    sendErr(seq, "NO_CAPTURE", nullptr);
    return;
  }

  if (!computeF5Features(liveFeatures)) {
    liveCaptureReady = false;
    sendErr(seq, "FEATURE_ERROR", nullptr);
    return;
  }

  int best = predictF5(liveFeatures, liveScores);

  char line[1200];
  int n = snprintf(line, sizeof line,
                   "PRED,%lu,gesture=%s,class_name=%s,features=",
                   (unsigned long)seq, MODEL_CLASS_IDS[best], MODEL_CLASS_NAMES[best]);

  for (int j = 0; j < MODEL_N_FEATURES && n > 0 && n < (int)sizeof(line) - 40; j++) {
    n += snprintf(line + n, sizeof(line) - n, "%s%.8g", j ? "/" : "", liveFeatures[j]);
  }

  if (n > 0 && n < (int)sizeof(line) - 20) {
    n += snprintf(line + n, sizeof(line) - n, ",scores=");
    for (int k = 0; k < MODEL_N_CLASSES && n > 0 && n < (int)sizeof(line) - 40; k++) {
      n += snprintf(line + n, sizeof(line) - n, "%s%.8g", k ? "/" : "", liveScores[k]);
    }
  }

  if (n > 0 && n < (int)sizeof(line) - 2) {
    snprintf(line + n, sizeof(line) - n, "\n");
  } else {
    strcpy(line, "ERR,0,RESPONSE_TOO_LONG\n");
  }

  sendText(line);
  liveCaptureReady = false;  // one captured window can be predicted only once
}



// ============================================================================
// 10. STANDALONE BUTTON / LED / DFPLAYER APPLICATION
// ============================================================================

static void waitWithWdt(uint32_t ms) {
  uint32_t t0 = millis();
  while ((uint32_t)(millis() - t0) < ms) {
    feedLoopWDT();
    delay(5);
  }
}

static void flashLed(uint8_t count, uint16_t onMs = 90, uint16_t offMs = 90) {
  for (uint8_t i = 0; i < count; i++) {
    ledSet(true);  waitWithWdt(onMs);
    ledSet(false); waitWithWdt(offMs);
  }
}

static void resetButtonDetector() {
  buttonRawLast = digitalRead(BUTTON_PIN);
  buttonStable = buttonRawLast;
  buttonChangedMs = millis();
  clickCount = 0;
  firstClickMs = 0;
}

// Returns BTN_SINGLE, BTN_DOUBLE or BTN_NONE. Click is counted on RELEASE.
static int pollButton() {
  uint32_t now = millis();
  bool raw = digitalRead(BUTTON_PIN);

  if (raw != buttonRawLast) {
    buttonRawLast = raw;
    buttonChangedMs = now;
  }

  if ((uint32_t)(now - buttonChangedMs) >= BUTTON_DEBOUNCE_MS && raw != buttonStable) {
    buttonStable = raw;

    if (buttonStable == HIGH) { // released
      if (clickCount == 0) {
        clickCount = 1;
        firstClickMs = now;
      } else if (clickCount == 1 && (uint32_t)(now - firstClickMs) <= DOUBLE_CLICK_MS) {
        clickCount = 0;
        return BTN_DOUBLE;
      }
    }
  }

  if (clickCount == 1 && (uint32_t)(now - firstClickMs) > DOUBLE_CLICK_MS) {
    clickCount = 0;
    return BTN_SINGLE;
  }
  return BTN_NONE;
}

static void updateIdleLed() {
  ImuState s = snapshotImu();
  uint32_t t = millis();

  if (!s.healthy) {
    // Sensor fault: very fast continuous blink.
    ledSet(((t / 100U) % 2U) == 0U);
    return;
  }

  if (uiState == UI_NEED_CAL) {
    // Two short flashes every 1.5 s = calibration required.
    uint32_t p = t % 1500U;
    bool on = (p < 110U) || (p >= 240U && p < 350U);
    ledSet(on);
  } else if (uiState == UI_READY) {
    // One slow pulse each second = ready to form a gesture.
    ledSet((t % 1000U) < 180U);
  } else {
    // During audio playback, keep LED off after the prediction flashes.
    ledSet(false);
  }
}

static int performStandaloneCalibration() {
  liveCalValid = false;
  liveCaptureReady = false;

  ImuState pre = snapshotImu();
  if (!pre.healthy) {
    Serial.println("CAL ERROR: MPU6050 is not healthy.");
    return -1;
  }

  Serial.println("CAL: hold the neutral pose still. 1.5 s settling...");
  uint32_t settle0 = millis();
  while ((uint32_t)(millis() - settle0) < 1500U) {
    ledSet(((millis() / 100U) % 2U) == 0U);
    if (!snapshotImu().healthy) {
      ledSet(false);
      Serial.println("CAL ERROR: IMU fault during settling.");
      return -1;
    }
    feedLoopWDT();
    delay(5);
  }

  CapStats st;
  uint32_t captureEpoch = 0;

  portENTER_CRITICAL(&imuMux);
  imu.biasBusy = true;
  imu.biasN = 0;
  for (int k = 0; k < 3; k++) {
    imu.bSum[k] = 0;
    imu.bSq[k] = 0;
    imu.bMin[k] = 0;
    imu.bMax[k] = 0;
  }
  portEXIT_CRITICAL(&imuMux);

  Serial.println("CAL: recording 5 s neutral calibration...");
  captureLedMode = CAP_LED_CAL;
  const char* err = captureBlock(N_CAL, false, st, captureEpoch);
  captureLedMode = CAP_LED_NONE;
  ledSet(false);

  uint32_t wait0 = millis();
  ImuState bs = snapshotImu();
  while (!err && bs.healthy && bs.biasN < BIAS_SAMPLES && (uint32_t)(millis() - wait0) < 150U) {
    delay(5);
    feedLoopWDT();
    bs = snapshotImu();
  }

  portENTER_CRITICAL(&imuMux);
  imu.biasBusy = false;
  portEXIT_CRITICAL(&imuMux);

  if (err) {
    Serial.print("CAL ERROR: "); Serial.println(err);
    return -1;
  }

  bs = snapshotImu();
  if (!bs.healthy || bs.biasN < (uint16_t)(BIAS_SAMPLES * 0.90f)) {
    Serial.println("CAL ERROR: gyro bias sampling incomplete.");
    return -1;
  }

  float bias[3];
  float sdMax = 0.0f, devMax = 0.0f;
  for (int k = 0; k < 3; k++) {
    bias[k] = (float)(bs.bSum[k] / bs.biasN);
    double var = bs.bSq[k] / bs.biasN - (double)bias[k] * bias[k];
    float sd = var > 0.0 ? (float)sqrt(var) : 0.0f;
    if (sd > sdMax) sdMax = sd;
    float d1 = fabsf(bs.bMax[k] - bias[k]);
    float d2 = fabsf(bs.bMin[k] - bias[k]);
    if (d1 > devMax) devMax = d1;
    if (d2 > devMax) devMax = d2;
  }

  const bool hardFail = (sdMax > LIVE_CAL_HARD_STD_DPS) || (devMax > LIVE_CAL_HARD_DEV_DPS);
  const bool warning  = (sdMax > LIVE_CAL_GOOD_STD_DPS) || (devMax > LIVE_CAL_GOOD_DEV_DPS);

  Serial.printf("CAL QC: gyro_std=%.4f dps, max_dev=%.4f dps\n", sdMax, devMax);

  if (hardFail) {
    Serial.println("CAL REJECTED: too much movement. Hold still and double-click again.");
    return -1;
  }

  // Exact calibration precision used by the live F5 implementation.
  for (int c = 0; c < 5; c++) {
    double sum = 0.0;
    for (int i = 0; i < N_CAL; i++) sum += buf[i].flex[c];
    liveNeutralFlex[c] = roundDecimals((float)(sum / (double)N_CAL), 1000.0f);
  }

  double aSum[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < N_CAL; i++) {
    aSum[0] += buf[i].ax;
    aSum[1] += buf[i].ay;
    aSum[2] += buf[i].az;
  }
  for (int k = 0; k < 3; k++) {
    float g = (float)(aSum[k] / (double)N_CAL / (double)ACC_LSB_PER_G);
    liveNeutralAccelG[k] = roundDecimals(g, 1000000.0f);
    liveGyroBiasDps[k] = roundDecimals(bias[k], 10000.0f);
  }

  portENTER_CRITICAL(&imuMux);
  biasDps[0] = liveGyroBiasDps[0];
  biasDps[1] = liveGyroBiasDps[1];
  biasDps[2] = liveGyroBiasDps[2];
  portEXIT_CRITICAL(&imuMux);
  saveBias(liveGyroBiasDps[0], liveGyroBiasDps[1], liveGyroBiasDps[2]);
  strcpy(biasSource, "live5s");

  liveCalEpoch = captureEpoch;
  liveCalValid = true;

  if (!liveCalibrationStillValid()) {
    liveCalValid = false;
    Serial.println("CAL ERROR: calibration became invalid immediately (IMU epoch changed).");
    return -1;
  }

  Serial.printf("CAL OK (%s). epoch=%lu\n", warning ? "WARNING" : "GOOD", (unsigned long)liveCalEpoch);
  Serial.printf("Neutral flex: %.3f %.3f %.3f %.3f %.3f\n",
                liveNeutralFlex[0], liveNeutralFlex[1], liveNeutralFlex[2], liveNeutralFlex[3], liveNeutralFlex[4]);
  Serial.printf("Neutral accel(g): %.6f %.6f %.6f\n",
                liveNeutralAccelG[0], liveNeutralAccelG[1], liveNeutralAccelG[2]);
  Serial.printf("Gyro bias(dps): %.4f %.4f %.4f\n",
                liveGyroBiasDps[0], liveGyroBiasDps[1], liveGyroBiasDps[2]);

  return warning ? 1 : 0;
}

static void startPredictionAudio(int classIndex) {
  lastPredictedClass = classIndex;
  uint16_t track = (uint16_t)(classIndex + 1); // G01->0001.mp3 ... G10->0010.mp3

  if (!dfReady) {
    Serial.println("AUDIO WARNING: DFPlayer not available. Prediction is still valid.");
    uiState = UI_READY;
    return;
  }

  // DFRobot playMp3Folder(N) targets /mp3/000N.mp3.
  dfPlayer.playMp3Folder(track);
  audioStartedMs = millis();
  uiState = UI_AUDIO;
  Serial.printf("AUDIO: /mp3/%04u.mp3\n", (unsigned)track);
}

static void performStandaloneRecognition() {
  if (!liveCalibrationStillValid()) {
    liveCalValid = false;
    uiState = UI_NEED_CAL;
    Serial.println("PREDICT BLOCKED: calibration required.");
    flashLed(5, 70, 70);
    return;
  }

  CapStats st;
  uint32_t epoch = 0;
  liveCaptureReady = false;

  Serial.println("CAPTURE: hold gesture still for 2 seconds...");
  captureLedMode = CAP_LED_GESTURE;
  const char* err = captureBlock(N_TRIAL, false, st, epoch);
  captureLedMode = CAP_LED_NONE;
  ledSet(false);

  if (err) {
    Serial.print("CAPTURE ERROR: "); Serial.println(err);
    if (!snapshotImu().healthy || epoch != liveCalEpoch || strcmp(err, "IMU_EPOCH_CHANGED") == 0) {
      liveCalValid = false;
      uiState = UI_NEED_CAL;
    } else {
      uiState = UI_READY;
    }
    flashLed(5, 70, 70);
    return;
  }

  if (st.lateMaxUs > LIVE_LATE_MAX_US) {
    Serial.printf("CAPTURE ERROR: timing late (%lu us > %lu us). Retry gesture.\n",
                  (unsigned long)st.lateMaxUs, (unsigned long)LIVE_LATE_MAX_US);
    uiState = UI_READY;
    flashLed(5, 70, 70);
    return;
  }

  if (epoch != liveCalEpoch) {
    liveCalValid = false;
    uiState = UI_NEED_CAL;
    Serial.println("CAPTURE ERROR: IMU epoch changed; recalibration required.");
    flashLed(5, 70, 70);
    return;
  }

  liveCaptureEpoch = epoch;
  liveCaptureReady = true;

  if (!computeF5Features(liveFeatures)) {
    liveCaptureReady = false;
    uiState = UI_READY;
    Serial.println("PREDICT ERROR: F5 feature calculation failed.");
    flashLed(5, 70, 70);
    return;
  }

  int best = predictF5(liveFeatures, liveScores);
  liveCaptureReady = false;

  Serial.println("F5 features:");
  for (int j = 0; j < MODEL_N_FEATURES; j++) {
    Serial.printf("  %2d %-32s = %.8g\n", j + 1, MODEL_FEATURE_NAMES[j], liveFeatures[j]);
  }
  Serial.println("Class scores:");
  for (int k = 0; k < MODEL_N_CLASSES; k++) {
    Serial.printf("  %s %-10s : %.6f%s\n", MODEL_CLASS_IDS[k], MODEL_CLASS_NAMES[k], liveScores[k], k == best ? "  <-- predicted" : "");
  }
  Serial.printf("PREDICTION: %s - %s\n", MODEL_CLASS_IDS[best], MODEL_CLASS_NAMES[best]);

  // Two quick flashes = prediction completed / audio starting.
  flashLed(2, 90, 90);
  startPredictionAudio(best);
}

static void serviceAudio() {
  if (uiState != UI_AUDIO) return;

  if (dfReady && dfPlayer.available()) {
    uint8_t type = dfPlayer.readType();
    int value = dfPlayer.read();
    if (type == DFPlayerPlayFinished) {
      Serial.printf("AUDIO finished, track=%d. READY.\n", value);
      uiState = UI_READY;
      resetButtonDetector();
      return;
    }
  }

  if ((uint32_t)(millis() - audioStartedMs) > AUDIO_TIMEOUT_MS) {
    Serial.println("AUDIO timeout fallback -> READY.");
    uiState = UI_READY;
    resetButtonDetector();
  }
}

static bool initDfPlayer() {
  Serial.println("DFPlayer: starting UART2 at 9600 baud...");
  dfSerial.begin(9600, SERIAL_8N1, DFPLAYER_RX_PIN, DFPLAYER_TX_PIN);
  delay(1000);

  // ACK + reset. If it fails, sensing/classification remains usable; only audio is disabled.
  if (!dfPlayer.begin(dfSerial, true, true)) {
    Serial.println("DFPlayer WARNING: initialization failed. Check 5V/GND, TX/RX, resistor, speaker and microSD.");
    return false;
  }
  dfPlayer.setTimeOut(500);
  dfPlayer.volume(DFPLAYER_VOLUME);
  dfPlayer.EQ(DFPLAYER_EQ_NORMAL);
  dfPlayer.outputDevice(DFPLAYER_DEVICE_SD);
  delay(300);
  Serial.printf("DFPlayer: ready, volume=%u/30.\n", (unsigned)DFPLAYER_VOLUME);
  return true;
}

// ============================================================================
// 10. COMMANDS FROM THE LAPTOP   ("<seq> <COMMAND> [argument]")
// ============================================================================

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
    if (!s.healthy) return false;
    if (s.reseedAck == req) { epochOut = s.epoch; return true; }
  }
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
    char line[220];
    snprintf(line, sizeof line,
             "ERR,%lu,CAL_MOVING,bias_std_dps=%.4f,bias_max_dev_dps=%.4f,epoch=%lu\n",
             (unsigned long)seq, sdMax, devMax, (unsigned long)captureEpoch);
    sendText(line);
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
  char line[440];
  snprintf(line, sizeof line,
           "STATUS,%lu,fw=%s,boot=%08lX,uptime_s=%lu,epoch=%lu,ref=%s,ref_ok=%d,imu=%s,i2c_err=%lu,faults=%lu,"
           "reinits=%lu,frozen=%lu,cfg_lost=%lu,gap_inval=%lu,rec_fail=%lu,bias_src=%s,bias_dps=%.3f/%.3f/%.3f,"
           "temp_c=%.2f,rssi=%d,rail_mv=%.0f,flex=%u/%u/%u/%u/%u,heap=%lu,who=0x%02X\n",
           (unsigned long)seq, FW_VERSION, (unsigned long)bootId, (unsigned long)(millis() / 1000),
           (unsigned long)s.epoch, refValid ? refId : "-", refUsable(s) ? 1 : 0, s.healthy ? "OK" : "FAULT",
           (unsigned long)s.i2cErrors, (unsigned long)s.faults, (unsigned long)s.reinits,
           (unsigned long)s.frozenEvents, (unsigned long)s.configLost, (unsigned long)s.gapInvalidations,
           (unsigned long)s.recoveryFails, biasSource, bx, by, bz, s.tempRaw / 340.0f + 36.53f,
           (int)WiFi.RSSI(), readRailMv(), f[0], f[1], f[2], f[3], f[4],
           (unsigned long)ESP.getFreeHeap(), s.who);
  sendText(line);
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

  char out[160];
  if (!strcmp(cmd, "PING")) {
    ImuState s = snapshotImu();
    snprintf(out, sizeof out, "PONG,%lu,epoch=%lu,ref=%s,ref_ok=%d,imu=%s,boot=%08lX\n", (unsigned long)seq,
             (unsigned long)s.epoch, refValid ? refId : "-", refUsable(s) ? 1 : 0, s.healthy ? "OK" : "FAULT",
             (unsigned long)bootId);
    sendText(out);
  } else if (!strcmp(cmd, "LCAL"))    { cmdLiveCalibrate(seq);
  } else if (!strcmp(cmd, "LCAP"))    { cmdLiveCapture(seq);
  } else if (!strcmp(cmd, "LPRED"))   { cmdLivePredict(seq);
  } else if (!strcmp(cmd, "LDISCARD")){ cmdLiveDiscard(seq);
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
  char line[260];
  snprintf(line, sizeof line,
           "HELLO,%s,fw=%s,boot=%08lX,reset=%s,sw=%s,epoch=%lu,ref_ok=%d,bias=%s,who=0x%02X,rail=%d,"
           "gyro_lsb_per_dps=%.1f,acc_lsb_per_g=%.0f,mode=F5_LR\n",
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
    } else if (lineLen < sizeof(lineBuf) - 1) {
      lineBuf[lineLen++] = (char)c;
    } else {
      lineLen = 0;                                      // over-long line: drop it
    }
  }
}

// ============================================================================
// 12. STANDALONE SETUP AND LOOP
// ============================================================================

void setup() {
  Serial.begin(115200); // optional debug only; normal standalone use needs no laptop
  delay(150);

  resetText = resetReasonText(esp_reset_reason());
  if (rtcMagic == RTC_MAGIC) {
    swText = rtcReason == SW_IMU_STALL ? "IMU_TASK_STALL" : rtcReason == SW_TEST_REBOOT ? "TEST_REBOOT" : "OTHER";
  }
  rtcMagic = 0;

  pinMode(LED_PIN, OUTPUT);
  ledSet(false);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  resetButtonDetector();

  analogReadResolution(12);
  for (uint8_t c = 0; c < 5; c++) {
    analogSetPinAttenuation(FLEX_PINS[c], ADC_11db);
    analogRead(FLEX_PINS[c]);
  }
  if (RAIL_PIN >= 0) {
    analogSetPinAttenuation(RAIL_PIN, ADC_11db);
    analogRead(RAIL_PIN);
  }

  // A saved bias may help the IMU settle before this session's calibration,
  // but predictions remain locked until a fresh 5 s calibration succeeds.
  loadBias();

  memset(&imu, 0, sizeof imu);
  imu.q = Quat{1.0f, 0.0f, 0.0f, 0.0f};
  Wire.begin(SDA_PIN, SCL_PIN, I2C_HZ);
  Wire.setTimeOut(20);
  imuHeartbeatMs = millis();
  xTaskCreatePinnedToCore(imuTask, "imu", 6144, nullptr, 3, nullptr, 1);

  bootId = esp_random() ^ (uint32_t)micros();
  enableLoopWDT();

  Serial.println();
  Serial.printf("%s %s boot=%08lX reset=%s\n", FW_NAME, FW_VERSION, (unsigned long)bootId, resetText);
  Serial.println("Standalone mode: Wi-Fi/Python disabled.");
  Serial.println("Button: SINGLE=record/predict, DOUBLE=neutral calibration.");

  dfReady = initDfPlayer();

  // Wait briefly for the IMU task to initialize, but do not block forever.
  uint32_t t0 = millis();
  while ((uint32_t)(millis() - t0) < 2500U) {
    if (snapshotImu().healthy) break;
    updateIdleLed();
    feedLoopWDT();
    delay(10);
  }

  uiState = UI_NEED_CAL;
  Serial.println("READY FOR CALIBRATION: wear glove in neutral pose, then DOUBLE-CLICK.");
}

void loop() {
  if ((uint32_t)(millis() - imuHeartbeatMs) > 2000U) {
    softReboot(SW_IMU_STALL);
  }

  serviceAudio();

  // If a sensor reset/recovery changed the epoch, the old neutral calibration is no longer trusted.
  if (uiState == UI_READY && !liveCalibrationStillValid()) {
    liveCalValid = false;
    uiState = UI_NEED_CAL;
    Serial.println("Calibration invalidated by IMU state change. DOUBLE-CLICK to recalibrate.");
    resetButtonDetector();
  }

  updateIdleLed();

  // Button is intentionally ignored while audio is playing.
  if (uiState == UI_AUDIO) {
    delay(2);
    return;
  }

  int ev = pollButton();

  if (ev == BTN_DOUBLE) {
    Serial.println("BUTTON: DOUBLE -> calibration");
    int q = performStandaloneCalibration();
    if (q < 0) {
      uiState = UI_NEED_CAL;
      flashLed(5, 70, 70);
      Serial.println("CALIBRATION REQUIRED: double-click and try again.");
    } else {
      if (q == 0) flashLed(2, 90, 90); // GOOD
      else        flashLed(3, 90, 90); // WARNING accepted
      uiState = UI_READY;
      Serial.println("READY: form a trained gesture, then SINGLE-CLICK.");
    }
    resetButtonDetector();
  }
  else if (ev == BTN_SINGLE) {
    if (uiState != UI_READY || !liveCalibrationStillValid()) {
      liveCalValid = false;
      uiState = UI_NEED_CAL;
      Serial.println("BUTTON: SINGLE ignored -> calibration required first.");
      flashLed(5, 70, 70);
      resetButtonDetector();
    } else {
      Serial.println("BUTTON: SINGLE -> record + predict");
      performStandaloneRecognition();
      resetButtonDetector();
    }
  }

  delay(2);
}

