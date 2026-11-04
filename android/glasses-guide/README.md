# Ocean Glasses (hackathon demo)

An Android app for Ray-Ban/Oakley Meta glasses. You face the sea, tap the glasses and ask "How
warm is it out there at 500 metres?". The app hears you through the glasses mic, takes the
phone's location and compass heading, asks the OceanScope backend (`POST /glasses/ask`,
`docs/PRD_BACKEND.md` §4.18), and speaks the answer through the glasses speakers.

The answer is for the first 0.25° model cell (about 27 km wide) at least 5 km out along your
heading that has water that deep, and it always says how far away that is. Every number comes
from the prediction store; no chat model writes the answer.

Hackathon demo only: Android only, sideloaded, built on Meta's Wearables Device Access Toolkit
1.0.1. The Speech and Inputs capabilities it uses are dev/beta-only toolkit APIs, so this app
can't be published as it is.

## What runs where

| Part | Real glasses | Mock glasses (no hardware) |
| --- | --- | --- |
| Question audio | Glasses mic, transcribed on the glasses (toolkit Speech) | Text from the "What the mock glasses hear" box, injected with `MockSpeechKit` |
| Start a question | Tap the touchpad or press the capture button (toolkit Inputs) | "Simulate touchpad tap" button (`MockInputKit.select()`) or Ask |
| Location and heading | Phone GPS and compass (hold the phone facing the sea) | Fake place: Puri beach facing 150° by default |
| Answer audio | Android text-to-speech, routed to the glasses over Bluetooth | Phone speaker |

Without glasses connected at all, Ask uses the phone mic (Android's speech recognizer).

## Build and run

1. Open `android/glasses-guide/` in Android Studio (Ladybug or newer) and let it sync. It needs
   JDK 17 and Android SDK 35.
2. Turn on Developer Mode on your phone, connect it with USB, and press Run (debug build).
3. In the app, keep "Use mock glasses" and "Fake location and heading" on, press
   **Connect glasses**, then **Simulate touchpad tap**. You should hear an answer in a few
   seconds (up to 90 s the first time, while Render wakes up).
4. Turn "Fake location and heading" off and try it at a window or outdoors with real GPS.

The backend must have `GLASSES_ENABLED=true` (`BEACH_ENABLED` is not needed). For a local
backend from the emulator, set the Backend URL to `http://10.0.2.2:7860/api/v1`.

## With real glasses (later)

Turn "Use mock glasses" off and press **Connect glasses**. That opens the Meta AI app to
register this app with your glasses (Developer Mode: `APPLICATION_ID` and `CLIENT_TOKEN` are
`0` in `app/build.gradle.kts`). Glasses must be on, unfolded and worn.

## If the build fails on a toolkit call

All toolkit code is in `GlassesLink.kt`. The calls were written from Meta's 1.0 Android API
reference on 2026-11-04 without a compile here (no Android SDK in that environment), so an
import path or a `DatResult` helper may need a small fix; the rest of the app uses only
standard Android APIs. Meta's sample apps:
https://github.com/meta/meta-wearables-dat-android/tree/main/samples
