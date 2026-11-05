package ai.oceanscope.glasses

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.widget.EditText
import android.widget.FrameLayout
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
    private lateinit var answerCard: LinearLayout
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
        window.statusBarColor = OceanPalette.BG_TOP
        window.navigationBarColor = OceanPalette.BG_BOTTOM
        if (Build.VERSION.SDK_INT >= 23) {
            @Suppress("DEPRECATION")
            window.decorView.systemUiVisibility = View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR
        }
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
                if (!a.providerLabel.isNullOrBlank()) {
                    append("\n• AI explanation generated by ${a.providerLabel}; numbers come from the OceanScope API.")
                }
                a.caveats.forEach { append("\n• $it") }
            }
            detailView.visibility = View.VISIBLE
            answerCard.visibility = View.VISIBLE
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
            detailView.text = ""
            detailView.visibility = View.GONE
            answerCard.visibility = View.VISIBLE
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
        statusView.visibility = if (text.isBlank() || text == "Ready") View.GONE else View.VISIBLE
    }

    private fun log(text: String) {
        runOnUiThread {
            val lines = (logView.text.toString().lines().filter { it.isNotBlank() } + text).takeLast(12)
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

    // --- UI (Ocean Glasses seafoam/mint underwater aesthetic) -------------------------------

    private fun dp(v: Int): Int = (v * resources.displayMetrics.density).toInt()

    private fun pillDrawable(): GradientDrawable = GradientDrawable(
        GradientDrawable.Orientation.TOP_BOTTOM,
        intArrayOf(OceanPalette.TEAL_BTN_TOP, OceanPalette.TEAL_BTN_BOTTOM),
    ).apply {
        cornerRadius = dp(28).toFloat()
        setStroke(dp(1), Color.parseColor("#88E6DF"))
    }

    private fun glassCardDrawable(): GradientDrawable = GradientDrawable(
        GradientDrawable.Orientation.TOP_BOTTOM,
        intArrayOf(Color.parseColor("#F4FFFFFF"), Color.parseColor("#E6F4FCFA")),
    ).apply {
        cornerRadius = dp(24).toFloat()
        setStroke(dp(2), Color.WHITE)
    }

    private fun fieldDrawable(): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = dp(12).toFloat()
        setColor(OceanPalette.FIELD_FILL)
        setStroke(dp(1), OceanPalette.FIELD_STROKE)
    }

    private fun styleOceanSwitch(sw: Switch) {
        val trackW = dp(48)
        val trackH = dp(26)
        val thumbSize = dp(22)

        fun makeTrack(colorHex: String) = GradientDrawable().apply {
            shape = GradientDrawable.RECTANGLE
            cornerRadius = trackH / 2f
            setColor(Color.parseColor(colorHex))
            setSize(trackW, trackH)
        }
        val trackStates = android.graphics.drawable.StateListDrawable().apply {
            addState(intArrayOf(android.R.attr.state_checked), makeTrack("#24A89E"))
            addState(intArrayOf(), makeTrack("#BFDCD9"))
        }
        val thumbDrawable = GradientDrawable().apply {
            shape = GradientDrawable.OVAL
            setColor(Color.WHITE)
            setStroke(dp(2), Color.parseColor("#24A89E"))
            setSize(thumbSize, thumbSize)
        }
        sw.trackDrawable = trackStates
        sw.thumbDrawable = thumbDrawable
        sw.switchMinWidth = trackW
        sw.textOn = ""
        sw.textOff = ""
        sw.showText = false
    }

    private fun buildUi(): View {
        val root = FrameLayout(this)
        root.addView(
            OceanBackdropView(this),
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ),
        )

        val scroll = ScrollView(this).apply {
            isFillViewport = true
            overScrollMode = View.OVER_SCROLL_NEVER
        }

        val page = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            val statusBarH = resources.getIdentifier("status_bar_height", "dimen", "android")
                .let { if (it > 0) resources.getDimensionPixelSize(it) else dp(28) }
            setPadding(0, statusBarH + dp(4), 0, 0)
        }

        // 1. Top Header Bar: Wave logo + "Ocean Glasses" + circular gear button
        val headerRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(18), dp(6), dp(18), dp(2))
        }
        headerRow.addView(
            TopBarLogoView(this),
            LinearLayout.LayoutParams(dp(38), dp(28)).apply { rightMargin = dp(8) },
        )
        val titleView = TextView(this).apply {
            text = "Ocean Glasses"
            textSize = 22f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(OceanPalette.TEAL_DARK)
        }
        headerRow.addView(
            titleView,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
        )
        val gearBtn = GearButtonView(this).apply {
            setOnClickListener {
                scroll.smoothScrollTo(0, backendField.top)
            }
        }
        headerRow.addView(gearBtn, LinearLayout.LayoutParams(dp(42), dp(42)))
        page.addView(headerRow)

        // 2. Underwater Hero Illustration (glasses, silver fish, clownfish, corals, bubbles)
        page.addView(
            HeroIllustrationView(this),
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(138)).apply {
                bottomMargin = dp(4)
            },
        )

        // Content column with horizontal margins
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), 0, dp(16), 0)
        }

        fun pillButton(
            labelText: String,
            iconKind: OceanIconView.Kind,
            marginH: Int = dp(18),
            onClick: () -> Unit,
        ): LinearLayout {
            return LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER
                background = pillDrawable()
                elevation = dp(4).toFloat()
                isClickable = true
                isFocusable = true
                setPadding(dp(20), dp(11), dp(20), dp(11))
                setOnClickListener { onClick() }

                addView(
                    OceanIconView(context, iconKind, Color.WHITE),
                    LinearLayout.LayoutParams(dp(22), dp(22)).apply { rightMargin = dp(10) },
                )
                addView(
                    TextView(context).apply {
                        text = labelText
                        textSize = 15.5f
                        letterSpacing = 0.04f
                        setTypeface(typeface, Typeface.BOLD)
                        setTextColor(Color.WHITE)
                    },
                )
            }.also { btn ->
                content.addView(
                    btn,
                    LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        dp(48),
                    ).apply {
                        leftMargin = marginH
                        rightMargin = marginH
                        bottomMargin = dp(10)
                    },
                )
            }
        }

        // 3. Primary Pill Buttons: CONNECT GLASSES and ASK
        pillButton("CONNECT GLASSES", OceanIconView.Kind.BLUETOOTH) { connectGlasses() }
        pillButton("ASK", OceanIconView.Kind.CHAT_BUBBLE_WHITE) { ask() }

        // Status & Spoken Answer Card (compact when idle, expands smoothly when asking/answering)
        statusView = TextView(this).apply {
            text = "Ready"
            textSize = 13.5f
            gravity = Gravity.CENTER
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(OceanPalette.TEAL_TITLE)
            visibility = View.GONE
            setPadding(dp(8), dp(2), dp(8), dp(6))
        }
        content.addView(statusView)

        answerCard = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = glassCardDrawable()
            elevation = dp(3).toFloat()
            visibility = View.GONE
            setPadding(dp(16), dp(12), dp(16), dp(12))
        }
        answerView = TextView(this).apply {
            text = ""
            textSize = 15f
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(OceanPalette.TEXT_PRIMARY)
        }
        detailView = TextView(this).apply {
            text = ""
            textSize = 12.5f
            setTextColor(OceanPalette.TEXT_SECONDARY)
            setPadding(0, dp(6), 0, 0)
            visibility = View.GONE
        }
        answerCard.addView(answerView)
        answerCard.addView(detailView)
        content.addView(
            answerCard,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply {
                bottomMargin = dp(10)
            },
        )

        // 4. Frosted Glass Card: "Settings and testing"
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            background = glassCardDrawable()
            elevation = dp(4).toFloat()
            setPadding(dp(16), dp(16), dp(16), dp(16))
        }

        fun iconInputRow(
            iconKind: OceanIconView.Kind,
            hint: String,
            value: String,
            numeric: Boolean = false,
            singleLine: Boolean = true,
        ): EditText {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            row.addView(
                OceanIconView(this, iconKind, OceanPalette.TEAL_ICON),
                LinearLayout.LayoutParams(dp(26), dp(26)).apply { rightMargin = dp(12) },
            )
            val edit = EditText(this).apply {
                this.hint = hint
                setText(value)
                textSize = 14.5f
                setTextColor(OceanPalette.TEXT_PRIMARY)
                setHintTextColor(Color.parseColor("#7AA6A3"))
                background = fieldDrawable()
                setPadding(dp(14), dp(9), dp(14), dp(9))
                if (numeric) {
                    inputType = InputType.TYPE_CLASS_NUMBER or
                        InputType.TYPE_NUMBER_FLAG_DECIMAL or
                        InputType.TYPE_NUMBER_FLAG_SIGNED
                } else if (!singleLine) {
                    inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_MULTI_LINE
                    maxLines = 2
                } else {
                    inputType = InputType.TYPE_CLASS_TEXT
                }
            }
            row.addView(
                edit,
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
            )
            card.addView(
                row,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply { bottomMargin = dp(9) },
            )
            return edit
        }

        fun iconSwitchRow(
            iconKind: OceanIconView.Kind,
            labelText: String,
            checked: Boolean,
        ): Switch {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }
            row.addView(
                OceanIconView(this, iconKind, OceanPalette.TEAL_ICON),
                LinearLayout.LayoutParams(dp(26), dp(26)).apply { rightMargin = dp(12) },
            )
            val sw = Switch(this).apply {
                text = labelText
                isChecked = checked
                textSize = 14.5f
                setTextColor(OceanPalette.TEXT_PRIMARY)
                styleOceanSwitch(this)
            }
            row.addView(
                sw,
                LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
            )
            card.addView(
                row,
                LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                ).apply { bottomMargin = dp(9) },
            )
            return sw
        }

        // Card header: Wrench icon + "Settings and testing"
        val cardHeader = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        cardHeader.addView(
            OceanIconView(this, OceanIconView.Kind.WRENCH, OceanPalette.TEAL_ICON),
            LinearLayout.LayoutParams(dp(26), dp(26)).apply { rightMargin = dp(10) },
        )
        cardHeader.addView(
            TextView(this).apply {
                text = "Settings and testing"
                textSize = 18.5f
                setTypeface(typeface, Typeface.BOLD)
                setTextColor(OceanPalette.TEAL_TITLE)
            },
        )
        card.addView(
            cardHeader,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { bottomMargin = dp(12) },
        )

        // 5. Backend URL field
        backendField = iconInputRow(
            OceanIconView.Kind.SERVER_STACK,
            "Backend URL",
            prefs.getString("backend", BuildConfig.DEFAULT_BACKEND_URL)!!,
            singleLine = false,
        )

        // 6. Use mock glasses toggle
        mockSwitch = iconSwitchRow(
            OceanIconView.Kind.GLASSES,
            "Use mock glasses (no hardware)",
            prefs.getBoolean("mock", true),
        )

        // 7. Test question input
        mockQuestionField = iconInputRow(
            OceanIconView.Kind.CHAT_BUBBLE_TEAL,
            "What the mock glasses hear",
            prefs.getString("mockQuestion", "")!!.ifBlank { "How warm is it out there at 500 metres?" },
        )

        // 8. SIMULATE TOUCHPAD TAP button inside the card
        val simBtn = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            background = pillDrawable()
            elevation = dp(3).toFloat()
            isClickable = true
            isFocusable = true
            setPadding(dp(18), dp(10), dp(18), dp(10))
            setOnClickListener {
                if (!glasses.ready && mockSwitch.isChecked) connectGlasses()
                glasses.simulateTap()
                ask()
            }
            addView(
                OceanIconView(context, OceanIconView.Kind.TOUCH_HAND, Color.WHITE),
                LinearLayout.LayoutParams(dp(24), dp(24)).apply { rightMargin = dp(10) },
            )
            addView(
                TextView(context).apply {
                    text = "SIMULATE TOUCHPAD TAP"
                    textSize = 14.5f
                    letterSpacing = 0.04f
                    setTypeface(typeface, Typeface.BOLD)
                    setTextColor(Color.WHITE)
                },
            )
        }
        card.addView(
            simBtn,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                dp(46),
            ).apply {
                topMargin = dp(3)
                bottomMargin = dp(11)
            },
        )

        // 9. Fake location and heading toggle
        fakePlaceSwitch = iconSwitchRow(
            OceanIconView.Kind.LOCATION_PIN,
            "Fake location and heading",
            prefs.getBoolean("fakePlace", true),
        )

        // 10-12. Latitude, Longitude, Heading inputs
        fakeLatField = iconInputRow(
            OceanIconView.Kind.CROSSHAIR,
            "Latitude",
            prefs.getString("fakeLat", "19.7983")!!,
            numeric = true,
        )
        fakeLonField = iconInputRow(
            OceanIconView.Kind.CROSSHAIR,
            "Longitude",
            prefs.getString("fakeLon", "85.8249")!!,
            numeric = true,
        )
        fakeHeadingField = iconInputRow(
            OceanIconView.Kind.COMPASS_ARROW,
            "Heading, degrees from north",
            prefs.getString("fakeHeading", "150")!!,
            numeric = true,
        )

        // 13-14. Log section & Glasses registration status inside the card footer
        val logRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.TOP
            setPadding(0, dp(6), 0, dp(2))
        }
        logRow.addView(
            OceanIconView(this, OceanIconView.Kind.CLIPBOARD_LOG, OceanPalette.TEAL_ICON),
            LinearLayout.LayoutParams(dp(26), dp(26)).apply {
                topMargin = dp(2)
                rightMargin = dp(12)
            },
        )
        val logTextCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
        }
        logTextCol.addView(
            TextView(this).apply {
                text = "Log"
                textSize = 15.5f
                setTypeface(typeface, Typeface.BOLD)
                setTextColor(OceanPalette.TEAL_TITLE)
            },
        )
        logView = TextView(this).apply {
            text = ""
            textSize = 13f
            setTextColor(OceanPalette.TEXT_PRIMARY)
            setPadding(0, dp(2), 0, 0)
        }
        logTextCol.addView(logView)
        logRow.addView(
            logTextCol,
            LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f),
        )
        card.addView(logRow)

        content.addView(
            card,
            LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ),
        )
        page.addView(content)

        // Bottom Ocean Reef Footer Illustration
        page.addView(
            BottomReefView(this),
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dp(105)).apply {
                topMargin = dp(6)
            },
        )

        scroll.addView(page)
        root.addView(scroll)
        return root
    }
}
