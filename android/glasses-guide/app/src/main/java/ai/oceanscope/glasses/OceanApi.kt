package ai.oceanscope.glasses

import org.json.JSONArray
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.Locale

/** One answer from POST /glasses/ask + natural AI guide synthesis from POST /chat. */
data class GlassesAnswer(
    val answerText: String,
    val question: String,
    val temperatureC: Double,
    val depthUsedM: Double,
    val distanceKm: Double,
    val direction: String,
    val date: String,
    val lat: Double,
    val lon: Double,
    val caveats: List<String>,
    val synthetic: Boolean,
    val providerLabel: String? = null,
)

class OceanApiException(val code: String, message: String) : Exception(message)

/** Plain HttpURLConnection client: no extra dependencies. Call off the main thread. */
object OceanApi {
    // Render's free tier can take up to ~90 s to wake up.
    private const val CONNECT_TIMEOUT_MS = 20_000
    private const val READ_TIMEOUT_MS = 100_000
    private const val CHAT_READ_TIMEOUT_MS = 28_000

    private val GREETING_RE = Regex(
        """^\s*(?:hi+|hello+|hey+|yo+|namaste|hola|good\s+(?:morning|afternoon|evening)|how\s+are\s+you(?:\s+doing)?|what'?s\s+up|who\s+are\s+you|help)(?:\s+there|\s+oceanscope|\s+glasses)?\s*[!?.]*\s*$""",
        RegexOption.IGNORE_CASE,
    )

    fun ask(
        baseUrl: String,
        question: String,
        lat: Double,
        lon: Double,
        bearingDeg: Double?,
        headingAccuracyDeg: Double?,
    ): GlassesAnswer {
        val root = baseUrl.trim().trimEnd('/').let { if (it.endsWith("/api/v1")) it else "$it/api/v1" }
        val fields = buildList {
            add("lat" to fmt(lat, 5))
            add("lon" to fmt(lon, 5))
            if (bearingDeg != null) add("bearing_deg" to fmt(normalise(bearingDeg), 1))
            if (headingAccuracyDeg != null) {
                add("heading_accuracy_deg" to fmt(headingAccuracyDeg.coerceIn(0.0, 180.0), 1))
            }
            add("text" to question.take(300))
        }
        val body = fields.joinToString("&") { (k, v) -> "$k=" + URLEncoder.encode(v, "UTF-8") }
            .toByteArray(Charsets.UTF_8)

        val conn = URL("$root/glasses/ask").openConnection() as HttpURLConnection
        val baseAnswer: GlassesAnswer
        val firstLevelM: Double?
        try {
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.connectTimeout = CONNECT_TIMEOUT_MS
            conn.readTimeout = READ_TIMEOUT_MS
            conn.setRequestProperty("Content-Type", "application/x-www-form-urlencoded")
            conn.setRequestProperty("Accept", "application/json")
            conn.setFixedLengthStreamingMode(body.size)
            conn.outputStream.use { it.write(body) }

            val status = conn.responseCode
            val stream = if (status in 200..299) conn.inputStream else (conn.errorStream ?: conn.inputStream)
            val text = stream.bufferedReader().use { it.readText() }
            val json = runCatching { JSONObject(text) }.getOrNull()
            if (status !in 200..299 || json == null) {
                val err = json?.optJSONObject("error")
                throw OceanApiException(
                    err?.optString("code").takeUnless { it.isNullOrEmpty() } ?: "http_$status",
                    err?.optString("message").takeUnless { it.isNullOrEmpty() } ?: "HTTP $status",
                )
            }
            val caveats = json.optJSONArray("caveats")
            val levels = json.optJSONArray("levels_m")
            firstLevelM = if (levels != null && levels.length() > 0) levels.optDouble(0) else null
            baseAnswer = GlassesAnswer(
                answerText = json.getString("answer_text"),
                question = json.optString("question"),
                temperatureC = json.getDouble("temperature_c"),
                depthUsedM = json.getDouble("depth_used_m"),
                distanceKm = json.getDouble("distance_km"),
                direction = json.optString("direction"),
                date = json.optString("date"),
                lat = json.getDouble("lat"),
                lon = json.getDouble("lon"),
                caveats = List(caveats?.length() ?: 0) { caveats!!.getString(it) },
                synthetic = json.optBoolean("synthetic"),
            )
        } finally {
            conn.disconnect()
        }

        val cleanQ = baseAnswer.question.ifBlank { question }.trim()

        // 1. If the user just says "hi" / "hello" / "help", reply with a natural greeting
        // instead of reading out a long unrequested surface temperature dump.
        if (GREETING_RE.matches(cleanQ)) {
            val dist = String.format(Locale.US, "%.0f", baseAnswer.distanceKm)
            return baseAnswer.copy(
                answerText = "Hi! I'm your OceanScope glasses guide. You're facing the water about " +
                    "$dist km ${baseAnswer.direction} of you — ask me how warm it is at any depth, " +
                    "or how the temperature changes down to the deep ocean.",
            )
        }

        // 2. Otherwise, ask the website's AI Ocean Guide (POST /chat) for a short, natural
        // follow-up insight on that exact cell, and seamlessly blend it with the base answer.
        val chatResult = runCatching {
            askChatGuide(
                root = root,
                question = cleanQ,
                baseSpokenAnswer = baseAnswer.answerText,
                lat = baseAnswer.lat,
                lon = baseAnswer.lon,
                date = baseAnswer.date,
                depthM = firstLevelM ?: baseAnswer.depthUsedM,
                direction = baseAnswer.direction,
            )
        }.getOrNull()

        return if (chatResult != null) {
            val combined = mergeNaturally(baseAnswer.answerText, chatResult.first)
            baseAnswer.copy(
                answerText = combined,
                providerLabel = chatResult.second,
            )
        } else {
            baseAnswer
        }
    }

    /**
     * Calls POST /api/v1/chat with the located cell context and the app's base answer,
     * asking for one concise additional sentence that complements (never contradicts or
     * repeats) the app's base answer.
     */
    private fun askChatGuide(
        root: String,
        question: String,
        baseSpokenAnswer: String,
        lat: Double,
        lon: Double,
        date: String,
        depthM: Double,
        direction: String,
    ): Pair<String, String?>? {
        val contextObj = JSONObject().apply {
            put("lat", lat)
            put("lon", lon)
            put("date", date)
            put("depth_m", depthM)
        }
        val msgObj = JSONObject().apply {
            put("role", "user")
            put(
                "content",
                "User question: \"$question\"\n" +
                    "The glasses app already says: \"$baseSpokenAnswer\"\n" +
                    "Using get_temperature_profile at this point ($direction) and date, add ONE " +
                    "short, natural spoken sentence that directly answers the user's question or " +
                    "adds helpful context from the profile (for example comparing surface vs deep " +
                    "water or GLORYS reanalysis). Do not say 'the data doesn't cover that depth' " +
                    "and do not repeat the app's sentence.",
            )
        }
        val payload = JSONObject().apply {
            put("messages", JSONArray().put(msgObj))
            put("context", contextObj)
        }.toString().toByteArray(Charsets.UTF_8)

        val conn = URL("$root/chat").openConnection() as HttpURLConnection
        try {
            conn.requestMethod = "POST"
            conn.doOutput = true
            conn.connectTimeout = CONNECT_TIMEOUT_MS
            conn.readTimeout = CHAT_READ_TIMEOUT_MS
            conn.setRequestProperty("Content-Type", "application/json")
            conn.setRequestProperty("Accept", "application/json")
            conn.setFixedLengthStreamingMode(payload.size)
            conn.outputStream.use { it.write(payload) }

            val status = conn.responseCode
            if (status !in 200..299) return null
            val text = conn.inputStream.bufferedReader().use { it.readText() }
            val json = JSONObject(text)
            val grounded = json.optBoolean("grounded", true)
            val ans = json.optString("answer").trim()
                .replace(Regex("\\s+"), " ")
            if (!grounded || ans.isEmpty()) return null
            val provider = json.optString("provider_label").takeUnless { it.isNullOrEmpty() }
            return ans to provider
        } finally {
            conn.disconnect()
        }
    }

    /**
     * Joins the app's base spoken answer and the AI guide's addition into one cohesive paragraph.
     * Strips out the generic "Ask for a depth..." prompt when the AI already adds rich context,
     * and removes any accidental "the data doesn't cover..." sentence from the AI.
     */
    private fun mergeNaturally(baseText: String, chatAddition: String): String {
        val cleanedBase = baseText
            .replace("Ask for a depth, like 200 metres, to hear about deeper water.", "")
            .trim()
        val cleanedChat = chatAddition
            .split(Regex("(?<=[.!?])\\s+"))
            .filterNot { sentence ->
                val low = sentence.lowercase(Locale.US)
                low.contains("doesn't cover") || low.contains("does not cover") ||
                    low.contains("fixed model depths")
            }
            .joinToString(" ")
            .trim()

        return if (cleanedChat.isEmpty()) {
            baseText
        } else {
            "$cleanedBase $cleanedChat"
        }
    }

    /** The backend accepts bearings in [0, 360). */
    private fun normalise(deg: Double): Double {
        val b = ((deg % 360.0) + 360.0) % 360.0
        return if (b >= 359.95) 0.0 else b
    }

    private fun fmt(v: Double, places: Int): String = String.format(Locale.US, "%.${places}f", v)
}
