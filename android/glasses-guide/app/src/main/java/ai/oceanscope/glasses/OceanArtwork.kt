package ai.oceanscope.glasses

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.LinearGradient
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.graphics.Shader
import android.view.View
import kotlin.math.cos
import kotlin.math.sin

/**
 * Custom vector artwork views matching the Ocean Glasses mint/seafoam underwater reference UI:
 * - [OceanBackdropView]: full-screen soft mint-to-seafoam gradient with subtle bubbles and wave bands.
 * - [TopBarLogoView]: double-wave turquoise/teal brand mark in the top-left header.
 * - [GearButtonView]: circular glass button with teal gear icon in the top-right header.
 * - [HeroIllustrationView]: underwater hero banner with 3D-styled aqua smart glasses, silver fish,
 *   clownfish, pink/yellow/teal coral & seaweed, wave crests, and floating bubbles.
 * - [BottomReefView]: bottom ocean floor reef scene with school of small fish, corals, and kelp.
 * - [OceanIconView]: crisp line-art icons for buttons and settings rows.
 */
object OceanPalette {
    val BG_TOP = Color.parseColor("#E9FAF7")
    val BG_MID = Color.parseColor("#CDF2EC")
    val BG_BOTTOM = Color.parseColor("#B6EAE3")
    val TEAL_DARK = Color.parseColor("#115E59")
    val TEAL_TITLE = Color.parseColor("#13635E")
    val TEAL_ICON = Color.parseColor("#248F88")
    val TEAL_BTN_TOP = Color.parseColor("#5DC8BF")
    val TEAL_BTN_BOTTOM = Color.parseColor("#279B95")
    val CARD_FILL = Color.parseColor("#EBFAFCFB")
    val CARD_STROKE = Color.parseColor("#FFFFFF")
    val FIELD_FILL = Color.parseColor("#F5FBFB")
    val FIELD_STROKE = Color.parseColor("#C7E2DF")
    val TEXT_PRIMARY = Color.parseColor("#1A3A3A")
    val TEXT_SECONDARY = Color.parseColor("#3E6B68")
    val CORAL_PINK = Color.parseColor("#F49595")
    val CORAL_GOLD = Color.parseColor("#F5CE62")
    val SEAWEED_TEAL = Color.parseColor("#6AC8BF")
    val SEAWEED_GREEN = Color.parseColor("#7CD4B8")
}

class OceanBackdropView(context: Context) : View(context) {
    private val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val wavePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val bubblePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.argb(90, 255, 255, 255)
    }
    private val bubbleFillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(45, 255, 255, 255)
    }
    private val path = Path()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        bgPaint.shader = LinearGradient(
            0f, 0f, 0f, h,
            intArrayOf(
                Color.parseColor("#ECFAF8"),
                Color.parseColor("#CCF1EC"),
                Color.parseColor("#D6F4F0"),
                Color.parseColor("#B5E9E2")
            ),
            floatArrayOf(0f, 0.25f, 0.65f, 1f),
            Shader.TileMode.CLAMP
        )
        canvas.drawRect(0f, 0f, w, h, bgPaint)

        // Subtle mid-screen wave wash
        path.reset()
        path.moveTo(0f, h * 0.12f)
        path.cubicTo(w * 0.28f, h * 0.09f, w * 0.68f, h * 0.15f, w, h * 0.11f)
        path.lineTo(w, h * 0.36f)
        path.cubicTo(w * 0.7f, h * 0.39f, w * 0.3f, h * 0.33f, 0f, h * 0.37f)
        path.close()
        wavePaint.color = Color.argb(55, 165, 232, 224)
        canvas.drawPath(path, wavePaint)

        // Ambient decorative bubbles along sides
        val bubbles = arrayOf(
            floatArrayOf(0.06f, 0.26f, 9f),
            floatArrayOf(0.04f, 0.28f, 5f),
            floatArrayOf(0.94f, 0.27f, 10f),
            floatArrayOf(0.96f, 0.29f, 5f),
            floatArrayOf(0.92f, 0.31f, 6f),
            floatArrayOf(0.05f, 0.62f, 7f),
            floatArrayOf(0.95f, 0.68f, 8f),
        )
        val d = resources.displayMetrics.density
        for (b in bubbles) {
            val cx = w * b[0]
            val cy = h * b[1]
            val r = b[2] * d * 0.45f
            canvas.drawCircle(cx, cy, r, bubbleFillPaint)
            canvas.drawCircle(cx, cy, r, bubblePaint)
        }
    }
}

class TopBarLogoView(context: Context) : View(context) {
    private val paint1 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#219B94")
    }
    private val paint2 = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        color = Color.parseColor("#65D6CB")
    }
    private val p1 = Path()
    private val p2 = Path()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        paint1.strokeWidth = h * 0.13f
        paint2.strokeWidth = h * 0.11f

        p1.reset()
        p1.moveTo(w * 0.08f, h * 0.68f)
        p1.cubicTo(w * 0.32f, h * 0.72f, w * 0.46f, h * 0.22f, w * 0.88f, h * 0.26f)
        canvas.drawPath(p1, paint1)

        p2.reset()
        p2.moveTo(w * 0.18f, h * 0.82f)
        p2.cubicTo(w * 0.44f, h * 0.84f, w * 0.56f, h * 0.38f, w * 0.94f, h * 0.42f)
        canvas.drawPath(p2, paint2)
    }
}

class GearButtonView(context: Context) : View(context) {
    private val circleFill = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.parseColor("#EAF8F6")
        setShadowLayer(8f, 0f, 3f, Color.argb(35, 17, 94, 89))
    }
    private val circleStroke = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = Color.WHITE
    }
    private val iconPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        color = Color.parseColor("#186963")
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val cx = width / 2f
        val cy = height / 2f
        val r = (minOf(width, height) / 2f) * 0.86f

        canvas.drawCircle(cx, cy, r, circleFill)
        canvas.drawCircle(cx, cy, r, circleStroke)

        iconPaint.strokeWidth = r * 0.11f
        val innerR = r * 0.22f
        val outerR = r * 0.44f
        canvas.drawCircle(cx, cy, innerR, iconPaint)
        canvas.drawCircle(cx, cy, outerR * 0.82f, iconPaint)

        for (i in 0 until 8) {
            val angle = Math.toRadians((i * 45).toDouble())
            val x1 = cx + (outerR * 0.78f) * cos(angle).toFloat()
            val y1 = cy + (outerR * 0.78f) * sin(angle).toFloat()
            val x2 = cx + (outerR * 1.05f) * cos(angle).toFloat()
            val y2 = cy + (outerR * 1.05f) * sin(angle).toFloat()
            canvas.drawLine(x1, y1, x2, y2, iconPaint)
        }
    }
}

/**
 * Top underwater hero scene matching the reference screenshot:
 * - Soft wave hills at top & bottom
 * - Distant school of tiny silhouettes in top-right
 * - Left: silver-blue fish + seafoam kelp + coral pink branch
 * - Center: 3D-styled Ocean Glasses with translucent seafoam lenses, white nose bridge, navy temples,
 *   and 3 teal accent rays above the right temple
 * - Right: orange/white clownfish + golden coral + teal seaweed
 */
class HeroIllustrationView(context: Context) : View(context) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private val path = Path()
    private val rect = RectF()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        // 1. Soft upper wave crest
        path.reset()
        path.moveTo(0f, h * 0.12f)
        path.cubicTo(w * 0.25f, h * 0.02f, w * 0.55f, h * 0.18f, w, h * 0.08f)
        path.lineTo(w, h)
        path.lineTo(0f, h)
        path.close()
        paint.style = Paint.Style.FILL
        paint.shader = LinearGradient(
            0f, 0f, 0f, h,
            Color.parseColor("#C7EFE9"),
            Color.parseColor("#DEF8F4"),
            Shader.TileMode.CLAMP
        )
        canvas.drawPath(path, paint)
        paint.shader = null

        // White foam wave line along the top crest
        strokePaint.color = Color.argb(180, 255, 255, 255)
        strokePaint.strokeWidth = h * 0.018f
        path.reset()
        path.moveTo(0f, h * 0.12f)
        path.cubicTo(w * 0.25f, h * 0.02f, w * 0.55f, h * 0.18f, w, h * 0.08f)
        canvas.drawPath(path, strokePaint)

        // 2. Soft seabed hills at base of hero
        path.reset()
        path.moveTo(0f, h * 0.80f)
        path.cubicTo(w * 0.22f, h * 0.72f, w * 0.42f, h * 0.88f, w * 0.65f, h * 0.78f)
        path.cubicTo(w * 0.82f, h * 0.71f, w * 0.92f, h * 0.82f, w, h * 0.76f)
        path.lineTo(w, h)
        path.lineTo(0f, h)
        path.close()
        paint.color = Color.parseColor("#B6EAE2")
        canvas.drawPath(path, paint)

        // 3. Left seaweed & pink coral
        drawKelp(canvas, w * 0.06f, h * 0.92f, h * 0.34f, Color.parseColor("#6AC7BE"), -1f)
        drawCoralBranch(canvas, w * 0.23f, h * 0.92f, h * 0.28f, OceanPalette.CORAL_PINK)

        // 4. Right golden coral & teal seaweed
        drawCoralBranch(canvas, w * 0.72f, h * 0.90f, h * 0.24f, OceanPalette.CORAL_GOLD)
        drawKelp(canvas, w * 0.92f, h * 0.94f, h * 0.36f, Color.parseColor("#5EC2B8"), 1f)

        // 5. Left silver-blue fish
        drawSilverFish(canvas, w * 0.18f, h * 0.33f, w * 0.11f)

        // 6. Right clownfish
        drawClownfish(canvas, w * 0.79f, h * 0.38f, w * 0.085f)

        // 7. Center smart glasses (angled 3D look with glowing mint lenses)
        drawSmartGlasses(canvas, w * 0.50f, h * 0.46f, w * 0.33f, h * 0.46f)

        // 8. Three teal accent rays above right top of glasses
        strokePaint.color = Color.parseColor("#20968E")
        strokePaint.strokeWidth = h * 0.022f
        canvas.drawLine(w * 0.62f, h * 0.15f, w * 0.625f, h * 0.06f, strokePaint)
        canvas.drawLine(w * 0.645f, h * 0.17f, w * 0.665f, h * 0.09f, strokePaint)
        canvas.drawLine(w * 0.665f, h * 0.21f, w * 0.695f, h * 0.15f, strokePaint)

        // 9. Small bubbles around the scene
        drawBubble(canvas, w * 0.11f, h * 0.22f, h * 0.025f)
        drawBubble(canvas, w * 0.31f, h * 0.24f, h * 0.018f)
        drawBubble(canvas, w * 0.85f, h * 0.24f, h * 0.022f)
        drawBubble(canvas, w * 0.76f, h * 0.58f, h * 0.016f)
    }

    private fun drawBubble(canvas: Canvas, cx: Float, cy: Float, r: Float) {
        paint.style = Paint.Style.FILL
        paint.color = Color.argb(70, 255, 255, 255)
        canvas.drawCircle(cx, cy, r, paint)
        strokePaint.strokeWidth = 2.5f
        strokePaint.color = Color.argb(150, 255, 255, 255)
        canvas.drawCircle(cx, cy, r, strokePaint)
    }

    private fun drawKelp(canvas: Canvas, baseX: Float, baseY: Float, height: Float, color: Int, dir: Float) {
        strokePaint.color = color
        strokePaint.strokeWidth = height * 0.14f
        // Center stem
        path.reset()
        path.moveTo(baseX, baseY)
        path.quadTo(baseX + dir * height * 0.18f, baseY - height * 0.5f, baseX + dir * height * 0.05f, baseY - height)
        canvas.drawPath(path, strokePaint)
        // Left leaf
        path.reset()
        path.moveTo(baseX, baseY - height * 0.15f)
        path.quadTo(baseX - height * 0.35f, baseY - height * 0.45f, baseX - height * 0.22f, baseY - height * 0.78f)
        canvas.drawPath(path, strokePaint)
        // Right leaf
        path.reset()
        path.moveTo(baseX, baseY - height * 0.1f)
        path.quadTo(baseX + height * 0.35f, baseY - height * 0.40f, baseX + height * 0.25f, baseY - height * 0.72f)
        canvas.drawPath(path, strokePaint)
    }

    private fun drawCoralBranch(canvas: Canvas, baseX: Float, baseY: Float, height: Float, color: Int) {
        strokePaint.color = color
        strokePaint.strokeWidth = height * 0.11f
        // Main stem
        path.reset()
        path.moveTo(baseX, baseY)
        path.lineTo(baseX, baseY - height * 0.85f)
        // Branches
        path.moveTo(baseX, baseY - height * 0.30f)
        path.quadTo(baseX - height * 0.32f, baseY - height * 0.42f, baseX - height * 0.36f, baseY - height * 0.70f)
        path.moveTo(baseX - height * 0.20f, baseY - height * 0.46f)
        path.lineTo(baseX - height * 0.16f, baseY - height * 0.78f)
        path.moveTo(baseX, baseY - height * 0.42f)
        path.quadTo(baseX + height * 0.30f, baseY - height * 0.52f, baseX + height * 0.34f, baseY - height * 0.82f)
        path.moveTo(baseX + height * 0.18f, baseY - height * 0.55f)
        path.lineTo(baseX + height * 0.14f, baseY - height * 0.88f)
        canvas.drawPath(path, strokePaint)
    }

    private fun drawSilverFish(canvas: Canvas, cx: Float, cy: Float, len: Float) {
        val h = len * 0.34f
        // Tail fin (left)
        path.reset()
        path.moveTo(cx - len * 0.42f, cy)
        path.lineTo(cx - len * 0.65f, cy - h * 0.65f)
        path.lineTo(cx - len * 0.56f, cy)
        path.lineTo(cx - len * 0.65f, cy + h * 0.65f)
        path.close()
        paint.style = Paint.Style.FILL
        paint.color = Color.parseColor("#6A8796")
        canvas.drawPath(path, paint)

        // Body
        rect.set(cx - len * 0.45f, cy - h * 0.5f, cx + len * 0.48f, cy + h * 0.5f)
        paint.shader = LinearGradient(
            rect.left, rect.top, rect.left, rect.bottom,
            Color.parseColor("#7E9AA8"),
            Color.parseColor("#D9E6EC"),
            Shader.TileMode.CLAMP
        )
        canvas.drawOval(rect, paint)
        paint.shader = null

        // Eye
        paint.color = Color.WHITE
        canvas.drawCircle(cx + len * 0.32f, cy - h * 0.08f, h * 0.16f, paint)
        paint.color = Color.parseColor("#263840")
        canvas.drawCircle(cx + len * 0.34f, cy - h * 0.08f, h * 0.09f, paint)
    }

    private fun drawClownfish(canvas: Canvas, cx: Float, cy: Float, len: Float) {
        val h = len * 0.52f
        // Tail fin (right, facing left)
        path.reset()
        path.moveTo(cx + len * 0.38f, cy)
        path.lineTo(cx + len * 0.60f, cy - h * 0.52f)
        path.lineTo(cx + len * 0.60f, cy + h * 0.52f)
        path.close()
        paint.style = Paint.Style.FILL
        paint.color = Color.parseColor("#F47B2A")
        canvas.drawPath(path, paint)

        // Orange body
        rect.set(cx - len * 0.46f, cy - h * 0.5f, cx + len * 0.44f, cy + h * 0.5f)
        paint.color = Color.parseColor("#F58229")
        canvas.drawOval(rect, paint)

        // Two white vertical stripes
        strokePaint.color = Color.WHITE
        strokePaint.strokeWidth = len * 0.11f
        canvas.drawLine(cx - len * 0.14f, cy - h * 0.42f, cx - len * 0.14f, cy + h * 0.42f, strokePaint)
        canvas.drawLine(cx + len * 0.14f, cy - h * 0.38f, cx + len * 0.14f, cy + h * 0.38f, strokePaint)

        // Eye (left side)
        paint.color = Color.WHITE
        canvas.drawCircle(cx - len * 0.30f, cy - h * 0.08f, h * 0.14f, paint)
        paint.color = Color.parseColor("#1E293B")
        canvas.drawCircle(cx - len * 0.31f, cy - h * 0.08f, h * 0.08f, paint)
    }

    private fun drawSmartGlasses(canvas: Canvas, cx: Float, cy: Float, gw: Float, gh: Float) {
        // Soft drop shadow beneath glasses
        rect.set(cx - gw * 0.42f, cy + gh * 0.38f, cx + gw * 0.44f, cy + gh * 0.54f)
        paint.style = Paint.Style.FILL
        paint.color = Color.argb(38, 18, 92, 86)
        canvas.drawOval(rect, paint)

        // Back temple arm (navy/slate strap curving behind)
        strokePaint.color = Color.parseColor("#476278")
        strokePaint.strokeWidth = gh * 0.11f
        path.reset()
        path.moveTo(cx - gw * 0.46f, cy - gh * 0.04f)
        path.cubicTo(
            cx - gw * 0.25f, cy - gh * 0.48f,
            cx + gw * 0.18f, cy - gh * 0.52f,
            cx + gw * 0.42f, cy - gh * 0.22f
        )
        canvas.drawPath(path, strokePaint)

        // Left temple hinge (light icy blue-white)
        strokePaint.color = Color.parseColor("#DCEBF0")
        strokePaint.strokeWidth = gh * 0.12f
        canvas.drawLine(cx - gw * 0.48f, cy - gh * 0.02f, cx - gw * 0.26f, cy + gh * 0.02f, strokePaint)
        // Temple ear tip
        strokePaint.color = Color.parseColor("#476278")
        strokePaint.strokeWidth = gh * 0.09f
        canvas.drawLine(cx - gw * 0.48f, cy - gh * 0.02f, cx - gw * 0.45f, cy + gh * 0.14f, strokePaint)

        // Translucent frame outer shell
        val leftLens = RectF(cx - gw * 0.26f, cy - gh * 0.18f, cx + gw * 0.12f, cy + gh * 0.40f)
        val rightLens = RectF(cx + gw * 0.15f, cy - gh * 0.24f, cx + gw * 0.46f, cy + gh * 0.32f)

        paint.style = Paint.Style.FILL
        paint.color = Color.parseColor("#D8F3EF")
        canvas.drawRoundRect(
            RectF(leftLens.left - 8f, leftLens.top - 8f, rightLens.right + 6f, leftLens.bottom + 4f),
            gh * 0.24f, gh * 0.24f, paint
        )

        // Glowing aqua/seafoam lenses
        val lensShader = LinearGradient(
            leftLens.left, leftLens.top, rightLens.right, leftLens.bottom,
            intArrayOf(
                Color.parseColor("#A5F2E4"),
                Color.parseColor("#6CE0CD"),
                Color.parseColor("#4DC6B4")
            ),
            floatArrayOf(0f, 0.55f, 1f),
            Shader.TileMode.CLAMP
        )
        paint.shader = lensShader
        canvas.drawRoundRect(leftLens, gh * 0.22f, gh * 0.22f, paint)
        canvas.drawRoundRect(rightLens, gh * 0.22f, gh * 0.22f, paint)
        paint.shader = null

        // White specular shine on lenses
        strokePaint.color = Color.argb(150, 255, 255, 255)
        strokePaint.strokeWidth = gh * 0.045f
        canvas.drawLine(
            leftLens.left + leftLens.width() * 0.22f,
            leftLens.top + leftLens.height() * 0.25f,
            leftLens.left + leftLens.width() * 0.45f,
            leftLens.top + leftLens.height() * 0.16f,
            strokePaint
        )
        canvas.drawLine(
            rightLens.left + rightLens.width() * 0.22f,
            rightLens.top + rightLens.height() * 0.25f,
            rightLens.left + rightLens.width() * 0.45f,
            rightLens.top + rightLens.height() * 0.16f,
            strokePaint
        )

        // Lens border highlight
        strokePaint.color = Color.parseColor("#96E6D9")
        strokePaint.strokeWidth = 3f
        canvas.drawRoundRect(leftLens, gh * 0.22f, gh * 0.22f, strokePaint)
        canvas.drawRoundRect(rightLens, gh * 0.22f, gh * 0.22f, strokePaint)
    }
}

/**
 * Bottom ocean reef footer matching the bottom of the reference screenshot:
 * - Rolling turquoise seabed hills
 * - School of 4 small teal fish swimming near the middle
 * - Left: tall seafoam kelp + golden coral branch
 * - Right: pink coral + tall seafoam kelp
 */
class BottomReefView(context: Context) : View(context) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.FILL }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
    }
    private val path = Path()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        if (w <= 0f || h <= 0f) return

        // Back seabed hill
        path.reset()
        path.moveTo(0f, h * 0.58f)
        path.cubicTo(w * 0.25f, h * 0.42f, w * 0.55f, h * 0.72f, w, h * 0.48f)
        path.lineTo(w, h)
        path.lineTo(0f, h)
        path.close()
        paint.color = Color.parseColor("#AEE6DF")
        canvas.drawPath(path, paint)

        // Front seabed hill
        path.reset()
        path.moveTo(0f, h * 0.76f)
        path.cubicTo(w * 0.30f, h * 0.64f, w * 0.65f, h * 0.85f, w, h * 0.68f)
        path.lineTo(w, h)
        path.lineTo(0f, h)
        path.close()
        paint.color = Color.parseColor("#9ADCD4")
        canvas.drawPath(path, paint)

        // Left plants: tall teal kelp + yellow coral
        drawPlant(canvas, w * 0.05f, h * 0.88f, h * 0.68f, Color.parseColor("#5DC4BA"))
        drawCoral(canvas, w * 0.16f, h * 0.90f, h * 0.46f, OceanPalette.CORAL_GOLD)
        drawCoral(canvas, w * 0.30f, h * 0.94f, h * 0.26f, Color.parseColor("#54BEB4"))

        // Right plants: pink coral + green-teal kelp
        drawCoral(canvas, w * 0.68f, h * 0.94f, h * 0.25f, Color.parseColor("#54BEB4"))
        drawCoral(canvas, w * 0.78f, h * 0.92f, h * 0.46f, OceanPalette.CORAL_PINK)
        drawPlant(canvas, w * 0.93f, h * 0.90f, h * 0.72f, Color.parseColor("#7AD4B4"))

        // Small school of 4 teal fish in center
        drawMiniFish(canvas, w * 0.40f, h * 0.36f, w * 0.042f)
        drawMiniFish(canvas, w * 0.46f, h * 0.42f, w * 0.038f)
        drawMiniFish(canvas, w * 0.52f, h * 0.33f, w * 0.036f)
        drawMiniFish(canvas, w * 0.57f, h * 0.28f, w * 0.035f)
    }

    private fun drawMiniFish(canvas: Canvas, cx: Float, cy: Float, size: Float) {
        paint.color = Color.parseColor("#35ABA2")
        val r = RectF(cx - size * 0.5f, cy - size * 0.22f, cx + size * 0.5f, cy + size * 0.22f)
        canvas.drawOval(r, paint)
        path.reset()
        path.moveTo(cx + size * 0.4f, cy)
        path.lineTo(cx + size * 0.72f, cy - size * 0.24f)
        path.lineTo(cx + size * 0.72f, cy + size * 0.24f)
        path.close()
        canvas.drawPath(path, paint)
    }

    private fun drawPlant(canvas: Canvas, x: Float, y: Float, h: Float, color: Int) {
        strokePaint.color = color
        strokePaint.strokeWidth = h * 0.13f
        path.reset()
        path.moveTo(x, y)
        path.quadTo(x - h * 0.12f, y - h * 0.5f, x, y - h)
        path.moveTo(x, y - h * 0.18f)
        path.quadTo(x - h * 0.34f, y - h * 0.46f, x - h * 0.24f, y - h * 0.78f)
        path.moveTo(x, y - h * 0.12f)
        path.quadTo(x + h * 0.34f, y - h * 0.42f, x + h * 0.24f, y - h * 0.72f)
        canvas.drawPath(path, strokePaint)
    }

    private fun drawCoral(canvas: Canvas, x: Float, y: Float, h: Float, color: Int) {
        strokePaint.color = color
        strokePaint.strokeWidth = h * 0.12f
        path.reset()
        path.moveTo(x, y)
        path.lineTo(x, y - h * 0.85f)
        path.moveTo(x, y - h * 0.30f)
        path.quadTo(x - h * 0.34f, y - h * 0.45f, x - h * 0.36f, y - h * 0.75f)
        path.moveTo(x, y - h * 0.38f)
        path.quadTo(x + h * 0.34f, y - h * 0.52f, x + h * 0.36f, y - h * 0.82f)
        canvas.drawPath(path, strokePaint)
    }
}

/**
 * Crisp line-art icons matching every icon in the reference screenshot.
 */
class OceanIconView(
    context: Context,
    private val kind: Kind,
    private val tintColor: Int = OceanPalette.TEAL_ICON,
) : View(context) {
    enum class Kind {
        BLUETOOTH,
        CHAT_BUBBLE_WHITE,
        WRENCH,
        SERVER_STACK,
        GLASSES,
        CHAT_BUBBLE_TEAL,
        TOUCH_HAND,
        LOCATION_PIN,
        CROSSHAIR,
        COMPASS_ARROW,
        CLIPBOARD_LOG
    }

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        color = tintColor
    }
    private val dotPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = tintColor
    }
    private val path = Path()
    private val rect = RectF()

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val w = width.toFloat()
        val h = height.toFloat()
        val s = minOf(w, h)
        if (s <= 0f) return
        paint.strokeWidth = maxOf(3f, s * 0.085f)
        paint.color = tintColor
        dotPaint.color = tintColor

        when (kind) {
            Kind.BLUETOOTH -> {
                path.reset()
                path.moveTo(w * 0.32f, h * 0.28f)
                path.lineTo(w * 0.68f, h * 0.72f)
                path.lineTo(w * 0.50f, h * 0.88f)
                path.lineTo(w * 0.50f, h * 0.12f)
                path.lineTo(w * 0.68f, h * 0.28f)
                path.lineTo(w * 0.32f, h * 0.72f)
                canvas.drawPath(path, paint)
            }
            Kind.CHAT_BUBBLE_WHITE, Kind.CHAT_BUBBLE_TEAL -> {
                rect.set(w * 0.12f, h * 0.16f, w * 0.88f, h * 0.76f)
                canvas.drawOval(rect, paint)
                // Tail at bottom-left
                path.reset()
                path.moveTo(w * 0.28f, h * 0.68f)
                path.lineTo(w * 0.16f, h * 0.86f)
                path.lineTo(w * 0.42f, h * 0.74f)
                canvas.drawPath(path, paint)
                // Three dots inside
                val r = s * 0.048f
                val cy = h * 0.46f
                canvas.drawCircle(w * 0.36f, cy, r, dotPaint)
                canvas.drawCircle(w * 0.50f, cy, r, dotPaint)
                canvas.drawCircle(w * 0.64f, cy, r, dotPaint)
            }
            Kind.WRENCH -> {
                path.reset()
                // Wrench handle + jaws
                path.moveTo(w * 0.22f, h * 0.78f)
                path.lineTo(w * 0.58f, h * 0.42f)
                canvas.drawPath(path, paint)
                canvas.drawCircle(w * 0.68f, h * 0.32f, s * 0.18f, paint)
                canvas.drawCircle(w * 0.22f, h * 0.78f, s * 0.08f, paint)
            }
            Kind.SERVER_STACK -> {
                val barH = h * 0.19f
                val gap = h * 0.06f
                var top = h * 0.16f
                for (i in 0 until 3) {
                    rect.set(w * 0.14f, top, w * 0.86f, top + barH)
                    canvas.drawRoundRect(rect, barH * 0.35f, barH * 0.35f, paint)
                    canvas.drawCircle(w * 0.30f, top + barH * 0.5f, s * 0.035f, dotPaint)
                    canvas.drawCircle(w * 0.44f, top + barH * 0.5f, s * 0.035f, dotPaint)
                    top += barH + gap
                }
            }
            Kind.GLASSES -> {
                val left = RectF(w * 0.08f, h * 0.34f, w * 0.45f, h * 0.68f)
                val right = RectF(w * 0.55f, h * 0.34f, w * 0.92f, h * 0.68f)
                canvas.drawRoundRect(left, s * 0.12f, s * 0.12f, paint)
                canvas.drawRoundRect(right, s * 0.12f, s * 0.12f, paint)
                canvas.drawLine(w * 0.45f, h * 0.44f, w * 0.55f, h * 0.44f, paint)
            }
            Kind.TOUCH_HAND -> {
                // Tap arcs at top
                rect.set(w * 0.30f, h * 0.08f, w * 0.62f, h * 0.34f)
                canvas.drawArc(rect, 200f, 140f, false, paint)
                // Index finger + hand outline
                path.reset()
                path.moveTo(w * 0.38f, h * 0.56f)
                path.lineTo(w * 0.38f, h * 0.24f)
                path.quadTo(w * 0.46f, h * 0.16f, w * 0.54f, h * 0.24f)
                path.lineTo(w * 0.54f, h * 0.48f)
                path.quadTo(w * 0.78f, h * 0.48f, w * 0.78f, h * 0.68f)
                path.quadTo(w * 0.78f, h * 0.88f, w * 0.52f, h * 0.88f)
                path.lineTo(w * 0.36f, h * 0.88f)
                path.quadTo(w * 0.22f, h * 0.76f, w * 0.24f, h * 0.58f)
                path.close()
                canvas.drawPath(path, paint)
            }
            Kind.LOCATION_PIN -> {
                path.reset()
                val cx = w * 0.5f
                val r = s * 0.28f
                val cy = h * 0.40f
                rect.set(cx - r, cy - r, cx + r, cy + r)
                path.addArc(rect, 150f, 240f)
                path.lineTo(cx, h * 0.88f)
                path.close()
                canvas.drawPath(path, paint)
                canvas.drawCircle(cx, cy, r * 0.38f, paint)
            }
            Kind.CROSSHAIR -> {
                val cx = w * 0.5f
                val cy = h * 0.5f
                val r = s * 0.26f
                canvas.drawCircle(cx, cy, r, paint)
                canvas.drawCircle(cx, cy, r * 0.28f, dotPaint)
                canvas.drawLine(cx, h * 0.10f, cx, cy - r, paint)
                canvas.drawLine(cx, cy + r, cx, h * 0.90f, paint)
                canvas.drawLine(w * 0.10f, cy, cx - r, cy, paint)
                canvas.drawLine(cx + r, cy, w * 0.90f, cy, paint)
            }
            Kind.COMPASS_ARROW -> {
                path.reset()
                path.moveTo(w * 0.84f, h * 0.18f)
                path.lineTo(w * 0.16f, h * 0.48f)
                path.lineTo(w * 0.48f, h * 0.56f)
                path.lineTo(w * 0.58f, h * 0.86f)
                path.close()
                canvas.drawPath(path, paint)
            }
            Kind.CLIPBOARD_LOG -> {
                rect.set(w * 0.20f, h * 0.20f, w * 0.80f, h * 0.88f)
                canvas.drawRoundRect(rect, s * 0.10f, s * 0.10f, paint)
                val clip = RectF(w * 0.36f, h * 0.12f, w * 0.64f, h * 0.25f)
                canvas.drawRoundRect(clip, s * 0.05f, s * 0.05f, paint)
                canvas.drawLine(w * 0.32f, h * 0.42f, w * 0.68f, h * 0.42f, paint)
                canvas.drawLine(w * 0.32f, h * 0.56f, w * 0.68f, h * 0.56f, paint)
                canvas.drawLine(w * 0.32f, h * 0.70f, w * 0.56f, h * 0.70f, paint)
            }
        }
    }
}

