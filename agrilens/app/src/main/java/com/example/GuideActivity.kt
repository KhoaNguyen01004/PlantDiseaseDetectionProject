package com.example

import android.content.Context
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class GuideActivity : AppCompatActivity() {

    override fun attachBaseContext(newBase: Context) {
        super.attachBaseContext(LanguageManager.apply(newBase))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        hideSystemBars()
        setContentView(buildContent())
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) {
            hideSystemBars()
        }
    }

    private fun buildContent(): View {
        val scrollView = ScrollView(this).apply {
            setBackgroundColor(color(0xF7F9F5))
            isFillViewport = true
            fitsSystemWindows = false
        }

        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(14), dp(16), dp(20))
        }

        content.addView(header())

        guideItems().forEach { item ->
            content.addView(card(item))
        }

        scrollView.addView(content)
        return scrollView
    }

    private fun header(): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 0, 0, dp(12))
        }

        val titleBlock = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        titleBlock.addView(text(getString(R.string.guide_title), 26f, 0x1D1B16, true))
        titleBlock.addView(text(getString(R.string.guide_subtitle), 14f, 0x5B6258, false))

        val back = text(getString(R.string.guide_back), 14f, 0x256D46, true).apply {
            gravity = Gravity.CENTER
            setPadding(dp(14), dp(10), dp(14), dp(10))
            background = rounded(0xEEF3EA, 0xCBD8C7, 12f)
            setOnClickListener { finish() }
        }

        row.addView(titleBlock)
        row.addView(back)
        return row
    }

    private fun card(item: GuideItem): View {
        val card = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(14), dp(14), dp(14), dp(14))
            background = rounded(0xFFFFFF, 0xDCE4D8, 12f)
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).apply {
                bottomMargin = dp(12)
            }
        }

        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        val title = text(item.name, 19f, 0x1D1B16, true).apply {
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }

        val category = text(item.category, 12f, 0xFFFFFF, true).apply {
            setPadding(dp(10), dp(5), dp(10), dp(5))
            background = rounded(0x256D46, 0x256D46, 16f)
        }

        header.addView(title)
        header.addView(category)

        card.addView(header)
        card.addView(section(getString(R.string.guide_growing_profile), item.profile))
        card.addView(section(getString(R.string.guide_model_tracks), item.diseases))
        return card
    }

    private fun section(title: String, lines: List<String>): View {
        val block = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, dp(10), 0, 0)
        }

        block.addView(text(title, 12f, 0x5B6258, true))

        lines.forEach { line ->
            block.addView(text("- $line", 13f, 0x2F352E, false).apply {
                setPadding(0, dp(3), 0, 0)
            })
        }

        return block
    }

    private fun text(value: String, size: Float, colorValue: Int, bold: Boolean): TextView {
        return TextView(this).apply {
            text = value
            textSize = size * AppSettings.getTextScale(this@GuideActivity)
            setTextColor(color(colorValue))
            if (bold) {
                setTypeface(typeface, android.graphics.Typeface.BOLD)
            }
        }
    }

    private fun hideSystemBars() {
        window.decorView.systemUiVisibility =
            View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY or
                View.SYSTEM_UI_FLAG_FULLSCREEN or
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE or
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION or
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
    }

    private fun guideItems(): List<GuideItem> {
        return if (LanguageManager.VIETNAMESE == LanguageManager.getLanguage(this)) {
            vietnameseGuideItems()
        } else {
            englishGuideItems()
        }
    }

    private fun englishGuideItems(): List<GuideItem> {
        return listOf(
            GuideItem(
                "Apple",
                "Fruit",
                listOf("Temperate orchard crop", "Needs open canopy airflow", "Inspect after wet spring weather"),
                listOf("Apple scab", "Black rot", "Cedar apple rust", "Healthy reference")
            ),
            GuideItem(
                "Blueberry",
                "Fruit",
                listOf("Acid-loving shrub", "Prefers steady moisture", "Leaf color changes can signal nutrient or disease stress"),
                listOf("Healthy reference")
            ),
            GuideItem(
                "Cherry",
                "Fruit",
                listOf("Stone fruit orchard crop", "Prune for airflow", "Watch for leaf spots after humid weather"),
                listOf("Powdery mildew", "Healthy reference")
            ),
            GuideItem(
                "Corn (Maize)",
                "Grain",
                listOf("Rapid-growing annual grass", "High nitrogen demand", "Lower leaves often show early infection"),
                listOf("Gray leaf spot", "Common rust", "Northern leaf blight", "Healthy reference")
            ),
            GuideItem(
                "Grape",
                "Fruit",
                listOf("Woody vine crop", "Dense canopy increases fungal risk", "Inspect leaves and fruit clusters"),
                listOf("Black rot", "Esca or black measles", "Isariopsis leaf spot", "Healthy reference")
            ),
            GuideItem(
                "Orange",
                "Fruit",
                listOf("Citrus tree", "Needs warm conditions and drainage", "Check new flush growth for citrus greening symptoms"),
                listOf("Huanglongbing or citrus greening")
            ),
            GuideItem(
                "Peach",
                "Fruit",
                listOf("Stone fruit tree", "Needs dormant pruning and good airflow", "Wet foliage increases bacterial spot risk"),
                listOf("Bacterial spot", "Healthy reference")
            ),
            GuideItem(
                "Bell Pepper",
                "Vegetable",
                listOf("Warm-season nightshade", "Avoid overhead watering", "Leaf spots spread faster on crowded plants"),
                listOf("Bacterial spot", "Healthy reference")
            ),
            GuideItem(
                "Potato",
                "Vegetable",
                listOf("Cool-season tuber crop", "Prefers loose, well-drained soil", "Blight often spreads quickly in humidity"),
                listOf("Early blight", "Late blight", "Healthy reference")
            ),
            GuideItem(
                "Raspberry",
                "Fruit",
                listOf("Cane berry crop", "Remove old canes after harvest", "Airflow helps prevent leaf and cane disease"),
                listOf("Healthy reference")
            ),
            GuideItem(
                "Soybean",
                "Legume",
                listOf("Warm-season field crop", "Canopy humidity increases disease pressure", "Scout lower and middle leaves"),
                listOf("Healthy reference")
            ),
            GuideItem(
                "Squash",
                "Vegetable",
                listOf("Vining cucurbit crop", "Needs full sun and room to spread", "Powdery mildew usually starts as white patches"),
                listOf("Powdery mildew")
            ),
            GuideItem(
                "Strawberry",
                "Fruit",
                listOf("Low-growing berry crop", "Mulch helps keep leaves cleaner", "Inspect older leaves for scorch or spots"),
                listOf("Leaf scorch", "Healthy reference")
            ),
            GuideItem(
                "Tomato",
                "Vegetable",
                listOf("Warm-season nightshade", "Needs airflow and consistent watering", "Lower leaves commonly show first symptoms"),
                listOf("Bacterial spot", "Early blight", "Late blight", "Leaf mold", "Septoria leaf spot", "Target spot", "Spider mites", "Yellow leaf curl virus", "Mosaic virus", "Healthy reference")
            )
        )
    }

    private fun vietnameseGuideItems(): List<GuideItem> {
        return listOf(
            GuideItem(
                "Táo",
                "Ăn quả",
                listOf("Hợp khí hậu mát", "Tán cây cần thoáng", "Sau mưa ẩm nên kiểm tra lá và quả"),
                listOf("Apple scab", "Black rot", "Cedar apple rust", "Lá khỏe")
            ),
            GuideItem(
                "Việt quất",
                "Ăn quả",
                listOf("Ưa đất chua", "Cần ẩm đều nhưng không úng", "Lá đổi màu có thể do dinh dưỡng hoặc bệnh"),
                listOf("Lá khỏe")
            ),
            GuideItem(
                "Anh đào",
                "Ăn quả",
                listOf("Cây ăn quả hạch", "Nên tỉa cho tán thoáng", "Thời tiết ẩm dễ phát sinh bệnh trên lá"),
                listOf("Powdery mildew", "Lá khỏe")
            ),
            GuideItem(
                "Bắp",
                "Ngũ cốc",
                listOf("Sinh trưởng nhanh", "Cần đủ dinh dưỡng", "Bệnh thường thấy trước ở lá già phía dưới"),
                listOf("Gray leaf spot", "Common rust", "Northern leaf blight", "Lá khỏe")
            ),
            GuideItem(
                "Nho",
                "Ăn quả",
                listOf("Dạng dây leo thân gỗ", "Tán quá rậm dễ bị nấm", "Nên kiểm tra cả lá và chùm quả"),
                listOf("Black rot", "Esca / Black measles", "Isariopsis leaf spot", "Lá khỏe")
            ),
            GuideItem(
                "Cam",
                "Có múi",
                listOf("Ưa khí hậu ấm", "Đất cần thoát nước tốt", "Theo dõi lộc non và màu gân lá"),
                listOf("HLB / vàng lá gân xanh")
            ),
            GuideItem(
                "Đào",
                "Ăn quả",
                listOf("Cây ăn quả hạch", "Tán thoáng giúp giảm bệnh", "Lá ướt lâu dễ bị đốm vi khuẩn"),
                listOf("Bacterial spot", "Lá khỏe")
            ),
            GuideItem(
                "Ớt chuông",
                "Rau",
                listOf("Ưa thời tiết ấm", "Hạn chế tưới trực tiếp lên lá", "Trồng quá dày dễ lây bệnh lá"),
                listOf("Bacterial spot", "Lá khỏe")
            ),
            GuideItem(
                "Khoai tây",
                "Củ",
                listOf("Ưa thời tiết mát", "Đất tơi xốp, thoát nước", "Mưa ẩm dễ làm bệnh lan nhanh"),
                listOf("Early blight", "Late blight", "Lá khỏe")
            ),
            GuideItem(
                "Mâm xôi",
                "Ăn quả",
                listOf("Dạng bụi, có cành mang quả", "Sau thu hoạch nên bỏ cành già", "Tán thoáng giúp giảm bệnh"),
                listOf("Lá khỏe")
            ),
            GuideItem(
                "Đậu nành",
                "Họ đậu",
                listOf("Hợp mùa ấm", "Tán rậm và ẩm dễ phát sinh bệnh", "Nên kiểm tra lá tầng dưới và giữa"),
                listOf("Lá khỏe")
            ),
            GuideItem(
                "Bí",
                "Rau",
                listOf("Dạng dây leo", "Cần nắng và không gian bò", "Mảng trắng trên lá thường là dấu hiệu phấn trắng"),
                listOf("Powdery mildew")
            ),
            GuideItem(
                "Dâu tây",
                "Ăn quả",
                listOf("Cây thấp sát mặt đất", "Phủ gốc giúp lá sạch hơn", "Lá già dễ xuất hiện cháy mép hoặc đốm"),
                listOf("Leaf scorch", "Lá khỏe")
            ),
            GuideItem(
                "Cà chua",
                "Rau",
                listOf("Ưa thời tiết ấm", "Cần tán thoáng và tưới đều", "Lá phía dưới thường biểu hiện bệnh trước"),
                listOf("Bacterial spot", "Early blight", "Late blight", "Leaf mold", "Septoria leaf spot", "Target spot", "Spider mites", "Yellow leaf curl virus", "Mosaic virus", "Lá khỏe")
            )
        )
    }

    private fun rounded(fill: Int, stroke: Int, radius: Float): android.graphics.drawable.GradientDrawable {
        return android.graphics.drawable.GradientDrawable().apply {
            setColor(color(fill))
            setStroke(dp(1), color(stroke))
            cornerRadius = dp(radius.toInt()).toFloat()
        }
    }

    private fun color(value: Int): Int {
        return android.graphics.Color.rgb(
            value shr 16 and 0xFF,
            value shr 8 and 0xFF,
            value and 0xFF
        )
    }

    private fun dp(value: Int): Int {
        return (value * resources.displayMetrics.density).toInt()
    }
}

data class GuideItem(
    val name: String,
    val category: String,
    val profile: List<String>,
    val diseases: List<String>
)
