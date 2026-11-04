package ai.oceanscope.glasses

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.util.Locale

/** One answer from POST /glasses/ask (PRD_BACKEND §4.18). */
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
)

class OceanApiException(val code: String, message: String) : Exception(message)

/** Plain HttpURLConnection client: no extra dependencies. Call off the main thread. */
object OceanApi {
    // Render's free tier can take up to ~90 s to wake up.
    private const val CONNECT_TIMEOUT_MS = 20_000
    private const val READ_TIMEOUT_MS = 100_000

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
            return GlassesAnswer(
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
    }

    /** The backend accepts bearings in [0, 360). */
    private fun normalise(deg: Double): Double {
        val b = ((deg % 360.0) + 360.0) % 360.0
        return if (b >= 359.95) 0.0 else b
    }

    private fun fmt(v: Double, places: Int): String = String.format(Locale.US, "%.${places}f", v)
}
