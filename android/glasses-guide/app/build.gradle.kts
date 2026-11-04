plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "ai.oceanscope.glasses"
    compileSdk = 35

    defaultConfig {
        applicationId = "ai.oceanscope.glasses"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1-hackathon"
        // Developer Mode: both are 0 (Wearables toolkit Android integration guide, step 1).
        manifestPlaceholders["mwdatApplicationId"] = "0"
        manifestPlaceholders["mwdatClientToken"] = "0"
        buildConfigField(
            "String",
            "DEFAULT_BACKEND_URL",
            "\"https://oceanscope-backend.onrender.com/api/v1\"",
        )
    }

    buildFeatures {
        buildConfig = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation(libs.mwdat.core)
    // Speech (glasses mic -> on-device transcription) and Inputs (touchpad) are dev/beta-only
    // toolkit APIs: fine for a sideloaded hackathon build, not for a store release.
    implementation(libs.mwdat.speech)
    implementation(libs.mwdat.inputs)
    // Mock glasses for testing without hardware (debug UI in the app).
    implementation(libs.mwdat.mockdevice)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.activity.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
}
