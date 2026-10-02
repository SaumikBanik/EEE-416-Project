SMARTGLOVE STANDALONE BUTTON + LED + DFPLAYER
================================================

FILES
-----
1. SmartGlove_Standalone_Button_LED_DFPlayer.ino
2. model_params.h

ARDUINO REQUIREMENTS
--------------------
- Board package: esp32 by Espressif Systems, version 3.x
- Library Manager: install "DFRobotDFPlayerMini" by DFRobot
- Board: ESP32 Dev Module
- Serial Monitor (optional debugging): 115200 baud

PIN MAP USED BY THIS SKETCH
---------------------------
Flex thumb/index/middle/ring/little: GPIO 32,33,34,35,36
MPU6050 SDA/SCL: GPIO 21/22
Push button: GPIO27 to GND (uses INPUT_PULLUP)
External LED: GPIO4 -> 220 ohm -> LED anode; LED cathode -> GND
ESP32 GPIO25 (TX2) -> 1 kOhm -> DFPlayer RX
ESP32 GPIO26 (RX2) <- DFPlayer TX
DFPlayer VCC -> 5 V
DFPlayer GND -> common GND
Speaker -> DFPlayer SPK1/SPK2 (speaker is NOT connected to GND)

MICROSD
-------
Format FAT32. Create /mp3 in root:
/mp3/0001.mp3 = G01 Fist
/mp3/0002.mp3 = G02 Excellent
/mp3/0003.mp3 = G03 Stop
/mp3/0004.mp3 = G04 Thumbs Up
/mp3/0005.mp3 = G05 This Way
/mp3/0006.mp3 = G06 Wait
/mp3/0007.mp3 = G07 Hey You!
/mp3/0008.mp3 = G08 Victory
/mp3/0009.mp3 = G09 Call Me
/mp3/0010.mp3 = G10 Attention

NORMAL USE
----------
1. Power on.
2. LED gives repeating double flash = calibration required.
3. Wear glove in the SAME neutral posture used during data collection.
4. DOUBLE-CLICK button.
5. 1.5 s settle + 5 s neutral calibration. LED fast-blinks during calibration.
6. Calibration result:
   - 2 quick flashes = GOOD
   - 3 quick flashes = accepted with WARNING
   - 5 fast flashes = rejected; stay still and double-click again
7. Slow LED pulse = READY.
8. Form one of the 10 trained static gestures and stabilize it.
9. SINGLE-CLICK once.
10. LED solid ON while 2 s / 50 samples are recorded.
11. ESP32 calculates F5, applies frozen StandardScaler parameters, runs LR and selects G01..G10.
12. Two quick flashes, then DFPlayer speaks the matching file.
13. Slow LED pulse returns = ready for the next gesture.
14. DOUBLE-CLICK from READY whenever a new person wears the glove or you want a fresh calibration.

IMPORTANT MODEL LIMITATION
--------------------------
The model is closed-set: it always chooses one of G01..G10. The button therefore acts as the trigger: only press SINGLE after intentionally forming one of the trained gestures. There is no trained REST/UNKNOWN class yet.

OPTIONAL SERIAL DEBUG
---------------------
Connect USB and open Serial Monitor at 115200. The sketch prints:
- calibration QC and session baselines
- the 12 F5 feature values
- all 10 logistic-regression class scores
- the final predicted gesture
- audio track number

DFPLAYER NOTES
--------------
- This code uses DFRobotDFPlayerMini and playMp3Folder(track).
- Volume defaults to 24/30. Change DFPLAYER_VOLUME near the top of the .ino if needed.
- If DFPlayer initialization fails, gesture sensing/classification still works; audio is skipped and the Serial Monitor prints a warning.
- AUDIO_TIMEOUT_MS is a 7 s fallback in case the DFPlayer play-finished event is missed.

DO NOT CHANGE FOR FINAL DEMO UNLESS NECESSARY
---------------------------------------------
- 25 Hz capture
- 2 s / 50 samples
- 5 s / 125-sample neutral calibration
- F5 feature order/formulas
- MODEL_SCALER_MEAN / MODEL_SCALER_SCALE
- MODEL_COEF / MODEL_INTERCEPT
- MPU6050 ranges and conversion factors

FIRST TEST SEQUENCE
-------------------
A. Power on with USB attached and Serial Monitor open.
B. Confirm DFPlayer says ready (or fix wiring/SD first).
C. Double-click in neutral pose and confirm CAL OK.
D. Test G01 Fist. Confirm Serial says G01 and speaker plays 0001.mp3.
E. Test G05 This Way and G08 Victory because they were a known difficult pair in development.
F. Try one additional participant after a fresh double-click calibration.
