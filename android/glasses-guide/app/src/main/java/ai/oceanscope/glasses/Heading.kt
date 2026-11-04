package ai.oceanscope.glasses

import android.content.Context
import android.hardware.GeomagneticField
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlin.math.abs

/**
 * Phone compass from the rotation-vector sensor. The glasses expose no heading, so the user
 * holds the phone facing the sea: either upright like taking a photo (we use the camera's
 * direction) or flat (we use the top edge's direction).
 */
class Heading(context: Context) : SensorEventListener {
    private val sensors = context.getSystemService(SensorManager::class.java)
    private val sensor: Sensor? = sensors?.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    /** Degrees from magnetic north, clockwise; null until the first reading. */
    @Volatile var magneticDeg: Double? = null
        private set

    /** Estimated accuracy in degrees; null if the phone doesn't report one. */
    @Volatile var accuracyDeg: Double? = null
        private set

    @Volatile private var statusAccuracyDeg: Double? = null

    val available: Boolean get() = sensor != null

    fun start() {
        sensor?.let { sensors?.registerListener(this, it, SensorManager.SENSOR_DELAY_UI) }
    }

    fun stop() {
        sensors?.unregisterListener(this)
    }

    /** True-north heading at this place (adds the magnetic declination), or null. */
    fun trueDeg(lat: Double, lon: Double, altitudeM: Double): Double? {
        val m = magneticDeg ?: return null
        val declination = GeomagneticField(
            lat.toFloat(), lon.toFloat(), altitudeM.toFloat(), System.currentTimeMillis(),
        ).declination
        return ((m + declination) % 360.0 + 360.0) % 360.0
    }

    override fun onSensorChanged(event: SensorEvent) {
        val r = FloatArray(9)
        SensorManager.getRotationMatrixFromVector(r, event.values)
        // r[8] is how much the screen faces up. Held upright, use the camera's direction.
        val m = if (abs(r[8]) < 0.7f) {
            FloatArray(9).also {
                SensorManager.remapCoordinateSystem(r, SensorManager.AXIS_X, SensorManager.AXIS_Z, it)
            }
        } else {
            r
        }
        val orientation = FloatArray(3)
        SensorManager.getOrientation(m, orientation)
        magneticDeg = (Math.toDegrees(orientation[0].toDouble()) + 360.0) % 360.0
        // values[4] is the heading accuracy in radians where the phone reports it.
        val reported = if (event.values.size > 4 && event.values[4] > 0f) {
            Math.toDegrees(event.values[4].toDouble())
        } else {
            null
        }
        accuracyDeg = reported ?: statusAccuracyDeg
    }

    override fun onAccuracyChanged(s: Sensor, accuracy: Int) {
        statusAccuracyDeg = when (accuracy) {
            SensorManager.SENSOR_STATUS_ACCURACY_HIGH -> 10.0
            SensorManager.SENSOR_STATUS_ACCURACY_MEDIUM -> 20.0
            SensorManager.SENSOR_STATUS_ACCURACY_LOW -> 45.0
            else -> 90.0
        }
    }
}
