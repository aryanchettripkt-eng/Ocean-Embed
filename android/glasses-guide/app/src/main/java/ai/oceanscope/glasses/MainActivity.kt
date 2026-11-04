package ai.oceanscope.glasses

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.text.InputType
import android.view.View
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.Switch
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

/**
 * One screen: connect the glasses, ask, hear the answer.
 *
 * Flow for one question: tap the glasses touchpad (or the Ask button) -> listen through the
 * glasses mic (phone mic if no glasses) -> phone GPS + compass -> POST /glasses/ask -> speak the
 * answer through the glasses speakers. The debug panel fakes everything the mock glasses can't:
 * the spoken question, the location and the heading.
 */
class MainActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    private lateinit var glasses: GlassesLink
    private lateinit var heading: Heading
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    private var busy = false

    private lateinit var statusView: TextView
    private lateinit var answerView: TextView
    private lateinit var detailView: TextView
    private lateinit var logView: TextView
    private lateinit var backendField: EditText
    private lateinit var mockSwitch: Switch
    private lateinit var mockQuestionField: EditText
    private lateinit var fakePlaceSwitch: Switch
    private lateinit var fakeLatField: EditText
    private lateinit var fakeLonField: EditText
    private lateinit var fakeHeadingField: EditText

    private val prefs by lazy { getSharedPreferences("glasses", Context.MODE_PRIVATE) }

    private val permissionRequest =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
            val missing = granted.filterValues { !it }.keys
            if (missing.isNotEmpty()) log("Permissions not granted: ${missing.joinToString()}")
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(buildUi())
        heading = Heading(this)
        tts = TextToSpeech(this, this)
        glasses = GlassesLink(this, lifecycleScope, ::log) { runOnUiThread { ask() } }
        glasses.initialize()
        glasses.handleIntent(intent)
        lifecycleScope.launch {
            glasses.registration.collect { log("Glasses registration: $it") }
        }
        requestPermissions()
        if (!heading.available) log("This phone has no compass: the nearest ocean cell is used")
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        glasses.handleIntent(intent)
    }

    override fun onResume() {
        super.onResume()
        heading.start()
    }

    override fun onPause() {
        heading.stop()
        super.onPause()
    }

    override fun onDestroy() {
        glasses.close()
        tts?.shutdown()
        super.onDestroy()
    }

    override fun onInit(status: Int) {
        ttsReady = status == TextToSpeech.SUCCESS
        if (ttsReady) tts?.language = Locale("en", "IN")
    }

    // --- actions ---------------------------------------------------------------------------

    private fun connectGlasses() {
        savePrefs()
        if (mockSwitch.isChecked) glasses.enableMock() else glasses.register()
        glasses.connect()
    }

    private fun ask() {
        if (busy) return
        busy = true
        savePrefs()
        lifecycleScope.launch {
            try {
                runQuestion()
            } finally {
                busy = false
            }
        }
    }

    private suspend fun runQuestion() {
        val mockQuestion = mockQuestionField.text.toString().trim().takeIf { mockSwitch.isChecked }

        status("Listening…")
        val question = when {
            glasses.ready -> glasses.listen(mockQuestion)
            !mockQuestion.isNullOrEmpty() -> mockQuestion
            hasPermission(Manifest.permission.RECORD_AUDIO) -> PhoneSpeech.listenOnce(this)
            else -> null
        }
        if (question.isNullOrBlank()) {
            say("Sorry, I didn't catch that.")
            status("Didn't catch a question")
            return
        }
        log("Heard: $question")

        status("Finding where you are…")
        val place = currentPlace()
        if (place == null) {
            say("I need your location to answer that.")
            status("No location (check the permission or use the fake place)")
            return
        }

        status("Asking OceanScope…")
        val backend = backendField.text.toString().ifBlank { BuildConfig.DEFAULT_BACKEND_URL }
        val result = withContext(Dispatchers.IO) {
            runCatching {
                OceanApi.ask(backend, question, place.lat, place.lon, place.headingDeg, place.accuracyDeg)
            }
        }
        result.onSuccess { a ->
            answerView.text = a.answerText
            detailView.text = buildString {
                append("You asked: ${a.question}\n")
                append(String.format(Locale.US, "Cell %.2f, %.2f · %.0f km %s · %s\n", a.lat, a.lon, a.distanceKm, a.direction, a.date))
                append(String.format(Locale.US, "%.1f °C at %.0f m", a.temperatureC, a.depthUsedM))
                if (a.synthetic) append(" · synthetic demo data")
                a.caveats.forEach { append("\n• $it") }
            }
            status("Answered")
            say(a.answerText)
        }.onFailure { e ->
            val spoken = when ((e as? OceanApiException)?.code) {
                "no_ocean_in_view" -> "I can't find the sea in that direction. Face the water and ask again."
                "out_of_region" -> "You're outside the area OceanScope covers."
                "glasses_disabled" -> "The glasses guide is switched off on the server."
                "rate_limited", "daily_cap_reached" -> "Too many questions for now. Try again later."
                else -> "I couldn't reach OceanScope just now."
            }
            answerView.text = spoken
            status("Error: ${e.message}")
            say(spoken)
        }
    }

    private data class Place(val lat: Double, val lon: Double, val headingDeg: Double?, val accuracyDeg: Double?)

    private suspend fun currentPlace(): Place? {
        if (fakePlaceSwitch.isChecked) {
            val lat = fakeLatField.text.toString().toDoubleOrNull() ?: return null
            val lon = fakeLonField.text.toString().toDoubleOrNull() ?: return null
            return Place(lat, lon, fakeHeadingField.text.toString().toDoubleOrNull(), null)
        }
        if (!hasPermission(Manifest.permission.ACCESS_FINE_LOCATION) &&
            !hasPermission(Manifest.permission.ACCESS_COARSE_LOCATION)
        ) {
            return null
        }
        val loc = PhoneLocation.current(this) ?: return null
        val h = heading.trueDeg(loc.latitude, loc.longitude, if (loc.hasAltitude()) loc.altitude else 0.0)
        return Place(loc.latitude, loc.longitude, h, heading.accuracyDeg)
    }

    // --- helpers ---------------------------------------------------------------------------

    private fun say(text: String) {
        if (ttsReady) tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "answer")
    }

    private fun status(text: String) {
        statusView.text = text
    }

    private fun log(text: String) {
        runOnUiThread {
            val lines = (logView.text.toString().lines() + text).takeLast(12)
            logView.text = lines.joinToString("\n").trim()
        }
    }

    private fun hasPermission(p: String) =
        ContextCompat.checkSelfPermission(this, p) == PackageManager.PERMISSION_GRANTED

    private fun requestPermissions() {
        val wanted = buildList {
            add(Manifest.permission.ACCESS_FINE_LOCATION)
            add(Manifest.permission.ACCESS_COARSE_LOCATION)
            add(Manifest.permission.RECORD_AUDIO)
            if (Build.VERSION.SDK_INT >= 31) add(Manifest.permission.BLUETOOTH_CONNECT)
        }.filterNot(::hasPermission)
        if (wanted.isNotEmpty()) permissionRequest.launch(wanted.toTypedArray())
    }

    private fun savePrefs() {
        prefs.edit()
            .putString("backend", backendField.text.toString())
            .putBoolean("mock", mockSwitch.isChecked)
            .putString("mockQuestion", mockQuestionField.text.toString())
            .putBoolean("fakePlace", fakePlaceSwitch.isChecked)
            .putString("fakeLat", fakeLatField.text.toString())
            .putString("fakeLon", fakeLonField.text.toString())
            .putString("fakeHeading", fakeHeadingField.text.toString())
            .apply()
    }

    // --- UI (built in code: one screen, no layout files) -------------------------------------

    private fun buildUi(): View {
        val pad = (16 * resources.displayMetrics.density).toInt()
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }
        fun label(text: String, size: Float = 14f) =
            TextView(this).apply { this.text = text; textSize = size }.also { col.addView(it) }
        fun field(hint: String, value: String, numeric: Boolean = false) = EditText(this).apply {
            this.hint = hint
            setText(value)
            if (numeric) {
                inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL or
                    InputType.TYPE_NUMBER_FLAG_SIGNED
            }
        }.also { col.addView(it) }
        fun button(text: String, onClick: () -> Unit) =
            Button(this).apply { this.text = text; setOnClickListener { onClick() } }.also { col.addView(it) }
        fun switch(text: String, on: Boolean) =
            Switch(this).apply { this.text = text; isChecked = on }.also { col.addView(it) }

        label("Ocean Glasses", 24f)
        label("Face the sea, tap the glasses (or Ask), and ask e.g. \"How warm is it out there at 500 metres?\"")
        statusView = label("Ready", 16f)
        button("Connect glasses") { connectGlasses() }
        button("Ask") { ask() }
        answerView = label("", 20f)
        detailView = label("", 13f)

        label("\nSettings and testing", 18f)
        backendField = field("Backend URL", prefs.getString("backend", BuildConfig.DEFAULT_BACKEND_URL)!!)
        mockSwitch = switch("Use mock glasses (no hardware)", prefs.getBoolean("mock", true))
        mockQuestionField = field(
            "What the mock glasses hear",
            prefs.getString("mockQuestion", "How warm is it out there at 500 metres?")!!,
        )
        button("Simulate touchpad tap") { glasses.simulateTap() }
        fakePlaceSwitch = switch("Fake location and heading", prefs.getBoolean("fakePlace", true))
        // Default: Puri beach, Odisha, facing the Bay of Bengal (south-east).
        fakeLatField = field("Latitude", prefs.getString("fakeLat", "19.7983")!!, numeric = true)
        fakeLonField = field("Longitude", prefs.getString("fakeLon", "85.8249")!!, numeric = true)
        fakeHeadingField = field("Heading, degrees from north", prefs.getString("fakeHeading", "150")!!, numeric = true)
        label("\nLog", 16f)
        logView = label("", 12f)

        return ScrollView(this).apply { addView(col) }
    }
}
