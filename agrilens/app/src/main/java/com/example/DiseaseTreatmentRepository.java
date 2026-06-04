package com.example;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

public class DiseaseTreatmentRepository {

    public static class Treatment {
        public final String diseaseName;
        public final String organicTreatment;
        public final String conventionalTreatment;

        private Treatment(String diseaseName, String organicTreatment, String conventionalTreatment) {
            this.diseaseName = diseaseName;
            this.organicTreatment = organicTreatment;
            this.conventionalTreatment = conventionalTreatment;
        }
    }

    private static class TreatmentData {
        final String nameEn;
        final String nameVi;
        final String organicEn;
        final String organicVi;
        final String conventionalEn;
        final String conventionalVi;

        TreatmentData(
                String nameEn,
                String nameVi,
                String organicEn,
                String organicVi,
                String conventionalEn,
                String conventionalVi
        ) {
            this.nameEn = nameEn;
            this.nameVi = nameVi;
            this.organicEn = organicEn;
            this.organicVi = organicVi;
            this.conventionalEn = conventionalEn;
            this.conventionalVi = conventionalVi;
        }

        Treatment localize(String languageCode) {
            boolean vi = "vi".equalsIgnoreCase(languageCode);
            return new Treatment(
                    vi ? nameVi : nameEn,
                    vi ? organicVi : organicEn,
                    vi ? conventionalVi : conventionalEn
            );
        }
    }

    private static final Map<String, TreatmentData> TREATMENTS = new HashMap<>();

    static {
        add("Apple___Apple_scab", data(
                "Apple Scab",
                "Apple scab trên táo",
                "- Remove fallen leaves and infected fruit.\n- Prune for airflow and avoid wet foliage.\n- Use sulfur or copper sprays early in the season.",
                "- Thu gom lá rụng và quả bị bệnh.\n- Tỉa cành để tán cây thông thoáng, hạn chế lá ẩm lâu.\n- Có thể dùng lưu huỳnh hoặc thuốc gốc đồng đầu vụ.",
                "- Apply labeled fungicides such as captan, mancozeb, or myclobutanil during infection-risk periods.\n- Rotate fungicide groups to reduce resistance.",
                "- Dùng thuốc trừ nấm như captan, mancozeb hoặc myclobutanil khi thời tiết dễ phát bệnh.\n- Luân phiên nhóm thuốc để hạn chế kháng thuốc."
        ));
        add("Apple___Black_rot", data(
                "Apple Black Rot",
                "Black rot trên táo",
                "- Prune cankers and dead branches.\n- Remove mummified fruit and leaf litter.\n- Keep pruning tools disinfected.",
                "- Cắt bỏ cành chết và vết loét bệnh.\n- Thu gom quả khô và lá rụng.\n- Khử trùng dụng cụ tỉa cành.",
                "- Use protective fungicides such as captan or mancozeb from bloom through fruit development.\n- Cover large pruning wounds when needed.",
                "- Dùng thuốc trừ nấm bảo vệ như captan hoặc mancozeb từ lúc ra hoa đến khi quả phát triển.\n- Che phủ vết cắt lớn khi cần."
        ));
        add("Apple___Cedar_apple_rust", data(
                "Cedar Apple Rust",
                "Cedar apple rust trên táo",
                "- Remove nearby juniper galls when possible.\n- Improve airflow around trees.\n- Use sulfur or copper products before wet spring periods.",
                "- Loại bỏ u bệnh trên cây bách/tuyết tùng gần vườn nếu có thể.\n- Tạo tán thông thoáng.\n- Dùng lưu huỳnh hoặc thuốc gốc đồng trước các đợt mưa xuân.",
                "- Apply systemic fungicides such as myclobutanil according to label timing.\n- Treat before symptoms spread widely.",
                "- Dùng thuốc nội hấp như myclobutanil theo đúng hướng dẫn trên nhãn.\n- Xử lý sớm trước khi bệnh lan rộng."
        ));
        addHealthy("Apple___healthy", "Apple", "táo");
        addHealthy("Blueberry___healthy", "Blueberry", "việt quất");
        add("Cherry_(including_sour)___Powdery_mildew", data(
                "Cherry Powdery Mildew",
                "Powdery mildew trên anh đào",
                "- Prune dense growth for sunlight and airflow.\n- Remove infected shoots.\n- Use neem oil, horticultural oil, or potassium bicarbonate.",
                "- Tỉa cành rậm để tăng ánh sáng và lưu thông không khí.\n- Cắt bỏ chồi bị bệnh.\n- Có thể dùng dầu neem, dầu khoáng hoặc kali bicarbonate.",
                "- Apply sulfur or labeled powdery mildew fungicides early.\n- Repeat based on disease pressure and product label.",
                "- Dùng lưu huỳnh hoặc thuốc trị phấn trắng từ sớm.\n- Phun lại theo mức độ bệnh và hướng dẫn trên nhãn."
        ));
        addHealthy("Cherry_(including_sour)___healthy", "Cherry", "anh đào");
        add("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", data(
                "Corn Gray Leaf Spot",
                "Gray leaf spot trên bắp",
                "- Rotate away from corn for 1-2 seasons.\n- Bury or remove infected crop residue.\n- Use resistant hybrids when available.",
                "- Luân canh khỏi cây bắp trong 1-2 vụ.\n- Vùi hoặc loại bỏ tàn dư cây bệnh.\n- Ưu tiên giống kháng nếu có.",
                "- Apply strobilurin or triazole fungicides around tasseling/silking if disease pressure is high.\n- Follow local threshold guidance.",
                "- Dùng thuốc nhóm strobilurin hoặc triazole quanh giai đoạn trổ cờ/phun râu nếu bệnh xuất hiện nhiều.\n- Theo khuyến cáo phòng trừ tại địa phương."
        ));
        add("Corn_(maize)___Common_rust_", data(
                "Corn Common Rust",
                "Common rust trên bắp",
                "- Choose resistant hybrids.\n- Avoid excessive nitrogen.\n- Remove volunteer corn where practical.",
                "- Chọn giống có khả năng kháng.\n- Không bón thừa đạm.\n- Loại bỏ bắp mọc sót nếu có thể.",
                "- Use labeled foliar fungicides when rust appears early and weather favors spread.\n- Protect high-value fields first.",
                "- Dùng thuốc trừ nấm phun lá khi bệnh xuất hiện sớm và thời tiết dễ lây lan.\n- Ưu tiên ruộng có giá trị cao."
        ));
        add("Corn_(maize)___Northern_Leaf_Blight", data(
                "Northern Corn Leaf Blight",
                "Northern leaf blight trên bắp",
                "- Rotate crops and manage corn residue.\n- Plant resistant hybrids.\n- Improve field airflow when possible.",
                "- Luân canh và quản lý tàn dư cây bắp.\n- Dùng giống kháng.\n- Tạo ruộng thông thoáng khi có thể.",
                "- Apply triazole or strobilurin fungicides when lesions appear before or near tasseling.\n- Follow product label intervals.",
                "- Dùng thuốc nhóm triazole hoặc strobilurin khi vết bệnh xuất hiện trước hoặc gần lúc trổ cờ.\n- Phun theo đúng khoảng cách trên nhãn."
        ));
        addHealthy("Corn_(maize)___healthy", "Corn", "bắp");
        add("Grape___Black_rot", data(
                "Grape Black Rot",
                "Black rot trên nho",
                "- Remove mummified berries and infected leaves.\n- Prune canopy for fast drying.\n- Use organic copper before bloom if needed.",
                "- Loại bỏ quả khô và lá nhiễm bệnh.\n- Tỉa tán để lá khô nhanh.\n- Có thể dùng thuốc gốc đồng trước ra hoa nếu cần.",
                "- Apply labeled fungicides such as mancozeb, ziram, or strobilurin products from bud break through early fruit development.\n- Maintain coverage during wet periods.",
                "- Dùng mancozeb, ziram hoặc thuốc nhóm strobilurin từ lúc nảy chồi đến khi quả non phát triển.\n- Giữ lớp thuốc bảo vệ trong thời gian mưa ẩm."
        ));
        add("Grape___Esca_(Black_Measles)", data(
                "Grape Esca / Black Measles",
                "Esca / Black measles trên nho",
                "- Prune out dead wood.\n- Protect pruning wounds in wet periods.\n- Remove severely affected vines.",
                "- Cắt bỏ gỗ chết.\n- Bảo vệ vết cắt khi thời tiết ẩm.\n- Loại bỏ cây bị nặng.",
                "- Chemical cure is limited; use registered wound protectants where available.\n- Focus on sanitation and pruning timing.",
                "- Thuốc thường không chữa khỏi bệnh này; có thể dùng sản phẩm bảo vệ vết cắt nếu có.\n- Ưu tiên vệ sinh vườn và tỉa cành đúng thời điểm."
        ));
        add("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", data(
                "Grape Isariopsis Leaf Blight",
                "Isariopsis leaf spot trên nho",
                "- Collect fallen leaves.\n- Open the canopy for light and airflow.\n- Avoid overhead watering.",
                "- Thu gom lá rụng.\n- Tỉa tán để tăng ánh sáng và thông khí.\n- Tránh tưới phun lên lá.",
                "- Apply protectant fungicides such as captan or mancozeb during humid periods.\n- Repeat according to label.",
                "- Dùng thuốc trừ nấm bảo vệ như captan hoặc mancozeb trong thời kỳ ẩm.\n- Phun lại theo hướng dẫn trên nhãn."
        ));
        addHealthy("Grape___healthy", "Grape", "nho");
        add("Orange___Haunglongbing_(Citrus_greening)", data(
                "Citrus Greening (HLB)",
                "HLB / vàng lá gân xanh trên cam quýt",
                "- Remove heavily infected trees.\n- Control psyllids with monitoring and non-chemical methods where possible.\n- Maintain nutrition to support tree health.",
                "- Loại bỏ cây bị nhiễm nặng.\n- Theo dõi và quản lý rầy chổng cánh bằng biện pháp phù hợp.\n- Duy trì dinh dưỡng để cây khỏe hơn.",
                "- Use registered systemic insecticides to manage Asian citrus psyllid vectors.\n- Coordinate treatment with local citrus programs.",
                "- Dùng thuốc trừ rầy nội hấp để giảm rầy chổng cánh truyền bệnh.\n- Nên làm theo khuyến cáo quản lý HLB tại địa phương."
        ));
        add("Peach___Bacterial_spot", bacterialSpot("Peach Bacterial Spot", "Bacterial spot trên đào"));
        addHealthy("Peach___healthy", "Peach", "đào");
        add("Pepper,_bell___Bacterial_spot", bacterialSpot("Pepper Bacterial Spot", "Bacterial spot trên ớt chuông"));
        addHealthy("Pepper,_bell___healthy", "Bell Pepper", "ớt chuông");
        add("Potato___Early_blight", earlyBlight("Potato Early Blight", "Early blight trên khoai tây"));
        add("Potato___Late_blight", lateBlight("Potato Late Blight", "Late blight trên khoai tây"));
        addHealthy("Potato___healthy", "Potato", "khoai tây");
        addHealthy("Raspberry___healthy", "Raspberry", "mâm xôi");
        addHealthy("Soybean___healthy", "Soybean", "đậu tương");
        add("Squash___Powdery_mildew", data(
                "Squash Powdery Mildew",
                "Powdery mildew trên bí",
                "- Remove badly infected leaves.\n- Improve spacing and sunlight.\n- Use neem oil, sulfur, or potassium bicarbonate products.",
                "- Loại bỏ lá bị nặng.\n- Tăng khoảng cách và ánh sáng.\n- Có thể dùng dầu neem, lưu huỳnh hoặc kali bicarbonate.",
                "- Apply labeled powdery mildew fungicides such as chlorothalonil or systemic products when disease spreads.\n- Rotate modes of action.",
                "- Dùng chlorothalonil hoặc thuốc nội hấp trị phấn trắng khi bệnh lan.\n- Luân phiên nhóm thuốc."
        ));
        add("Strawberry___Leaf_scorch", data(
                "Strawberry Leaf Scorch",
                "Leaf scorch trên dâu tây",
                "- Remove old infected leaves.\n- Use raised, well-drained beds.\n- Avoid overhead irrigation.",
                "- Loại bỏ lá già bị bệnh.\n- Trồng trên luống cao, thoát nước tốt.\n- Tránh tưới phun lên lá.",
                "- Apply labeled fungicides such as captan or pyraclostrobin products when conditions favor disease.\n- Observe harvest intervals.",
                "- Dùng captan hoặc sản phẩm chứa pyraclostrobin khi thời tiết thuận lợi cho bệnh.\n- Tuân thủ thời gian cách ly."
        ));
        addHealthy("Strawberry___healthy", "Strawberry", "dâu tây");
        add("Tomato___Bacterial_spot", bacterialSpot("Tomato Bacterial Spot", "Bacterial spot trên cà chua"));
        add("Tomato___Early_blight", earlyBlight("Tomato Early Blight", "Early blight trên cà chua"));
        add("Tomato___Late_blight", lateBlight("Tomato Late Blight", "Late blight trên cà chua"));
        add("Tomato___Leaf_Mold", data(
                "Tomato Leaf Mold",
                "Leaf mold trên cà chua",
                "- Reduce humidity and improve greenhouse ventilation.\n- Prune lower leaves.\n- Use biological fungicides where available.",
                "- Giảm ẩm độ và tăng thông gió nhà màng.\n- Tỉa lá phía dưới.\n- Có thể dùng chế phẩm sinh học nếu có.",
                "- Apply labeled fungicides such as chlorothalonil or other tomato leaf mold products.\n- Rotate products and avoid repeated single-mode use.",
                "- Dùng chlorothalonil hoặc sản phẩm trị leaf mold trên cà chua.\n- Luân phiên thuốc, tránh dùng lặp một nhóm."
        ));
        add("Tomato___Septoria_leaf_spot", data(
                "Tomato Septoria Leaf Spot",
                "Septoria leaf spot trên cà chua",
                "- Remove infected lower leaves.\n- Mulch soil to prevent splash.\n- Stake plants and avoid wet foliage.",
                "- Loại bỏ lá thấp bị bệnh.\n- Phủ gốc để hạn chế đất bắn lên lá.\n- Cắm giàn và tránh làm ướt tán lá.",
                "- Apply copper, chlorothalonil, or other labeled tomato fungicides at first symptoms.\n- Repeat according to weather and label.",
                "- Dùng thuốc gốc đồng, chlorothalonil hoặc thuốc trừ nấm cho cà chua khi mới có triệu chứng.\n- Phun lại theo thời tiết và hướng dẫn trên nhãn."
        ));
        add("Tomato___Spider_mites Two-spotted_spider_mite", data(
                "Tomato Two-Spotted Spider Mite",
                "Two-spotted spider mite trên cà chua",
                "- Spray undersides of leaves with water to reduce mites.\n- Use insecticidal soap, neem oil, or release predatory mites.\n- Avoid drought stress.",
                "- Xịt mặt dưới lá bằng nước để giảm mật số nhện.\n- Dùng xà phòng trừ sâu, dầu neem hoặc thả thiên địch.\n- Tránh để cây bị khô hạn.",
                "- Use registered miticides such as abamectin where appropriate.\n- Rotate miticide groups and target leaf undersides.",
                "- Dùng thuốc trừ nhện như abamectin khi cần.\n- Luân phiên nhóm thuốc và phun kỹ mặt dưới lá."
        ));
        add("Tomato___Target_Spot", data(
                "Tomato Target Spot",
                "Target spot trên cà chua",
                "- Remove infected debris.\n- Stake plants to keep foliage off wet soil.\n- Improve airflow and avoid overhead watering.",
                "- Dọn tàn dư bệnh.\n- Cắm giàn để lá không chạm đất ẩm.\n- Tăng thông khí và tránh tưới phun.",
                "- Apply labeled fungicides such as chlorothalonil, mancozeb, or systemic mixes when disease pressure rises.\n- Rotate modes of action.",
                "- Dùng chlorothalonil, mancozeb hoặc thuốc nội hấp khi bệnh tăng.\n- Luân phiên nhóm thuốc."
        ));
        add("Tomato___Tomato_Yellow_Leaf_Curl_Virus", data(
                "Tomato Yellow Leaf Curl Virus",
                "Yellow leaf curl virus trên cà chua",
                "- Remove infected plants early.\n- Use insect netting and yellow sticky traps.\n- Control whiteflies with non-chemical methods when possible.",
                "- Nhổ bỏ cây nhiễm sớm.\n- Dùng lưới chắn côn trùng và bẫy dính vàng.\n- Quản lý bọ phấn bằng biện pháp không hóa học khi có thể.",
                "- No chemical cure exists for infected plants.\n- Use registered insecticides to manage whitefly vectors and plant resistant varieties.",
                "- Không có thuốc chữa cây đã nhiễm virus.\n- Giảm bọ phấn truyền bệnh và ưu tiên giống kháng."
        ));
        add("Tomato___Tomato_mosaic_virus", data(
                "Tomato Mosaic Virus",
                "Mosaic virus trên cà chua",
                "- Remove infected plants.\n- Wash hands and disinfect tools.\n- Use certified seed and avoid tobacco contamination.",
                "- Nhổ bỏ cây nhiễm bệnh.\n- Rửa tay và khử trùng dụng cụ.\n- Dùng hạt giống chứng nhận và tránh lây nhiễm từ thuốc lá.",
                "- No chemical cure exists.\n- Focus on sanitation, certified seed, and vector control when insects are present.",
                "- Không có thuốc chữa virus.\n- Tập trung vệ sinh, dùng hạt giống sạch bệnh và quản lý côn trùng truyền bệnh nếu có."
        ));
        addHealthy("Tomato___healthy", "Tomato", "cà chua");
    }

    public static Treatment getTreatment(String labelName) {
        return getTreatment(labelName, "en");
    }

    public static Treatment getTreatment(String labelName, String languageCode) {
        TreatmentData data = TREATMENTS.get(normalize(labelName));
        return data == null ? null : data.localize(languageCode);
    }

    public static String getDisplayName(String labelName, String languageCode) {
        Treatment treatment = getTreatment(labelName, languageCode);
        if (treatment != null) {
            return treatment.diseaseName;
        }

        if (labelName == null) {
            return "Unknown";
        }

        return labelName
                .replace("___", ": ")
                .replace("_", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private static void add(String modelLabel, TreatmentData data) {
        TREATMENTS.put(normalize(modelLabel), data);
    }

    private static void addHealthy(String modelLabel, String cropEn, String cropVi) {
        add(modelLabel, data(
                "Healthy " + cropEn,
                cropVi.substring(0, 1).toUpperCase(new Locale("vi")) + cropVi.substring(1) + " khỏe",
                "- No treatment is needed.\n- Keep watering consistent and monitor new leaves.\n- Maintain airflow and remove dead plant material.",
                "- Chưa cần xử lý bệnh.\n- Tưới đều và theo dõi lá non.\n- Giữ cây thông thoáng, dọn lá/cành khô.",
                "- No chemical treatment is required for a healthy plant.\n- Only treat if pests or disease symptoms appear.",
                "- Không cần dùng thuốc khi cây khỏe.\n- Chỉ xử lý khi có sâu bệnh hoặc triệu chứng bất thường."
        ));
    }

    private static TreatmentData bacterialSpot(String nameEn, String nameVi) {
        return data(
                nameEn,
                nameVi,
                "- Remove infected leaves and avoid touching wet plants.\n- Use drip irrigation instead of overhead watering.\n- Use copper-based organic products early if needed.",
                "- Loại bỏ lá bệnh và tránh chạm cây khi lá ướt.\n- Tưới nhỏ giọt thay vì tưới phun.\n- Có thể dùng thuốc gốc đồng sớm nếu cần.",
                "- Apply labeled copper or copper-mancozeb products during humid periods.\n- Use clean seed or transplants and rotate crops.",
                "- Dùng thuốc gốc đồng hoặc copper-mancozeb trong thời kỳ ẩm.\n- Dùng hạt/cây giống sạch bệnh và luân canh."
        );
    }

    private static TreatmentData earlyBlight(String nameEn, String nameVi) {
        return data(
                nameEn,
                nameVi,
                "- Remove infected lower leaves.\n- Mulch soil and avoid splashing water.\n- Rotate away from related crops for at least 2-3 seasons.",
                "- Loại bỏ lá thấp bị bệnh.\n- Phủ gốc và tránh nước bắn đất lên lá.\n- Luân canh khỏi cây cùng họ trong ít nhất 2-3 vụ.",
                "- Apply labeled protectant fungicides such as chlorothalonil, mancozeb, or copper products.\n- Start early when weather favors disease.",
                "- Dùng chlorothalonil, mancozeb hoặc thuốc gốc đồng.\n- Xử lý sớm khi thời tiết thuận lợi cho bệnh."
        );
    }

    private static TreatmentData lateBlight(String nameEn, String nameVi) {
        return data(
                nameEn,
                nameVi,
                "- Remove and destroy infected foliage immediately.\n- Avoid overhead irrigation and improve airflow.\n- Do not compost infected material.",
                "- Loại bỏ và tiêu hủy lá bệnh ngay.\n- Tránh tưới phun và tăng thông khí.\n- Không ủ phân từ tàn dư bị bệnh.",
                "- Apply labeled late-blight fungicides such as chlorothalonil or systemic products before humid/rainy periods.\n- Follow local disease alerts.",
                "- Dùng chlorothalonil hoặc thuốc nội hấp trước giai đoạn mưa ẩm.\n- Theo dõi cảnh báo bệnh tại địa phương."
        );
    }

    private static TreatmentData data(
            String nameEn,
            String nameVi,
            String organicEn,
            String organicVi,
            String conventionalEn,
            String conventionalVi
    ) {
        return new TreatmentData(nameEn, nameVi, organicEn, organicVi, conventionalEn, conventionalVi);
    }

    private static String normalize(String value) {
        if (value == null) {
            return "";
        }

        return value
                .replace("___", " ")
                .replace("_", " ")
                .toLowerCase(Locale.US)
                .replaceAll("[^a-z0-9]+", " ")
                .trim();
    }
}
