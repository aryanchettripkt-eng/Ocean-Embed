package ai.oceanscope.glasses

import android.app.Activity
import android.content.Intent
import com.meta.wearable.dat.core.Wearables
import com.meta.wearable.dat.core.selectors.AutoDeviceSelector
import com.meta.wearable.dat.core.session.DeviceSession
import com.meta.wearable.dat.core.session.DeviceSessionState
import com.meta.wearable.dat.core.types.RegistrationState
import com.meta.wearable.dat.inputs.addInputs
import com.meta.wearable.dat.inputs.types.InputEvent
import com.meta.wearable.dat.mockdevice.MockDeviceKit
import com.meta.wearable.dat.mockdevice.api.GlassesModel
import com.meta.wearable.dat.mockdevice.api.MockDeviceKitConfig
import com.meta.wearable.dat.mockdevice.api.MockGlasses
import com.meta.wearable.dat.speech.Speech
import com.meta.wearable.dat.speech.addSpeech
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Everything that touches the Meta Wearables Device Access Toolkit (1.0.x) lives in this file,
 * so an API difference in a newer toolkit version is fixed here only.
 *
 * - The glasses mic is used through the toolkit's Speech capability (on-device transcription),
 *   so only text goes to our backend.
 * - A tap on the touchpad (Select) or a capture-button press starts a question.
 * - Answers are spoken with Android text-to-speech, which plays through the glasses speakers
 *   when they are connected over Bluetooth.
 * - Mock mode pairs simulated Ray-Ban Meta glasses (Mock Device Kit), so all of this runs with
 *   no hardware: a tap and a spoken question can be injected from the app's debug panel.
 */
class GlassesLink(
    private val activity: Activity,
    private val scope: CoroutineScope,
    private val log: (String) -> Unit,
    private val onTap: () -> Unit,
) {
    private var session: DeviceSession? = null
    private var speech: Speech? = null
    private var mock: MockGlasses? = null
    private val jobs = mutableListOf<Job>()

    val registration: StateFlow<RegistrationState> get() = Wearables.registrationState
    val ready: Boolean get() = speech != null
    val isMock: Boolean get() = mock != null

    /** Call once from onCreate. Safe to call again (ALREADY_INITIALIZED is ignored). */
    fun initialize() {
        runCatching { Wearables.initialize(activity.applicationContext) }
            .onFailure { log("Toolkit init failed: ${it.message}") }
    }

    /** Call from onCreate and onNewIntent: the Meta AI app returns here after registration. */
    fun handleIntent(intent: Intent) {
        runCatching {
            Wearables.handleIntent(intent) { request -> log("Registration request from Meta AI: $request") }
        }.onFailure { log("Intent not handled: ${it.message}") }
    }

    /** Opens the Meta AI app to register this app with the user's real glasses. */
    fun register() {
        runCatching { Wearables.startRegistration(activity) }
            .onFailure { log("Registration failed to start: ${it.message}") }
    }

    /** Pairs simulated Ray-Ban Meta glasses, powered on, unfolded and worn. */
    fun enableMock() {
        if (mock != null) return
        val kit = MockDeviceKit.getInstance(activity.applicationContext)
        kit.enable(MockDeviceKitConfig(initiallyRegistered = true, initialPermissionsGranted = true))
        val glasses = kit.pairGlasses(GlassesModel.RAYBAN_META).getOrNull()
        if (glasses == null) {
            log("Mock glasses could not be paired")
            return
        }
        glasses.powerOn()
        glasses.unfold()
        glasses.don()
        mock = glasses
        log("Mock Ray-Ban Meta glasses paired, on and worn")
    }

    /** Opens a session on the first available glasses and attaches Speech and Inputs. */
    fun connect() {
        if (session != null) return
        val s = Wearables.createSession(AutoDeviceSelector()).getOrNull()
        if (s == null) {
            log("No glasses session (are the glasses registered, on and worn?)")
            return
        }
        session = s
        jobs += scope.launch { s.errors.collect { log("Glasses error: $it") } }
        jobs += scope.launch {
            s.state.collect { state ->
                log("Glasses session: $state")
                when (state) {
                    DeviceSessionState.STARTED -> attach(s)
                    DeviceSessionState.STOPPED -> {
                        // A stopped session cannot restart; a new one is made on the next connect.
                        speech = null
                        session = null
                    }
                    else -> Unit
                }
            }
        }
        s.start()
    }

    private fun attach(s: DeviceSession) {
        if (speech == null) {
            speech = s.addSpeech().getOrNull()
            log(if (speech != null) "Glasses mic ready" else "Glasses mic unavailable")
        }
        val inputs = s.addInputs().getOrNull()
        if (inputs == null) {
            log("Glasses touchpad unavailable: use the Ask button")
            return
        }
        jobs += scope.launch {
            inputs.events.collect { event ->
                if (event is InputEvent.Select || event is InputEvent.Capture) onTap()
            }
        }
    }

    /**
     * Listens through the glasses mic until a final transcription or the timeout.
     * With mock glasses, `mockQuestion` is injected as what the user said.
     */
    suspend fun listen(mockQuestion: String?, timeoutMs: Long = 12_000): String? {
        val sp = speech ?: return null
        val before = sp.transcriptions.value
        if (sp.start().getOrNull() == null) {
            log("Glasses mic did not start: ${sp.errors.value} ${sp.errorDetails.value ?: ""}")
            return null
        }
        val kit = mock?.services?.speech
        if (kit != null && !mockQuestion.isNullOrBlank()) {
            scope.launch {
                delay(150); kit.simulateTranscription("", isFinal = false); delay(150)
                kit.simulateTranscription(mockQuestion, isFinal = true)
            }
        }
        val result = withTimeoutOrNull(timeoutMs) {
            sp.transcriptions.first { it != null && it !== before && it.isFinal }
        }
        sp.stop()
        return result?.text?.takeIf { it.isNotBlank() }
    }

    /** Debug panel: a tap on the mock glasses' touchpad. */
    fun simulateTap() {
        mock?.services?.input?.select() ?: log("Tap simulation needs mock glasses")
    }

    fun close() {
        jobs.forEach { it.cancel() }
        jobs.clear()
        runCatching { session?.stop() }
        session = null
        speech = null
    }
}
