package com.example;

import java.util.HashMap;
import java.util.Map;

public class DiseaseTreatmentRepository {

    public static class Treatment {
        public final String diseaseName;
        public final String organicTreatment;
        public final String conventionalTreatment;

        public Treatment(String diseaseName, String organicTreatment, String conventionalTreatment) {
            this.diseaseName = diseaseName;
            this.organicTreatment = organicTreatment;
            this.conventionalTreatment = conventionalTreatment;
        }
    }

    private static final Map<String, Treatment> TREATMENTS = new HashMap<>();

    static {
        TREATMENTS.put("Apple: Apple scab", new Treatment(
            "Apple Scab",
            "• Remove and destroy fallen leaves in autumn to reduce winter survival of spores.\n• Spray baking soda solutions or sulfur/copper sprays early in the growing season.\n• Use compost tea to enhance beneficial foliar microflora.",
            "• Apply chemical fungicides such as Captan, Mancozeb, or Myclobutanil starting at green-tip stage.\n• Implement systemic fungicides like Flint or Sovran during high-infection risk windows."
        ));

        TREATMENTS.put("Apple: Black rot", new Treatment(
            "Black Rot (Canker / Leaf Spot)",
            "• Prune out dead wood, cankers, and mummified fruit; burn or dispose of them.\n• Apply copper-based organic fungicides at silver tip and bud break stages.",
            "• Apply Captan, Pristine, or sulfur-based fungicides following a tight schedule from bloom through harvest.\n• Paint chemical pruning compounds over large wounds."
        ));

        TREATMENTS.put("Apple: Cedar apple rust", new Treatment(
            "Cedar Apple Rust",
            "• Remove nearby juniper/cedar trees (alternative hosts) within 1-2 miles.\n• Apply organic sulfur or copper sprays when apple buds begin to break in spring.",
            "• Spray systemic chemical fungicides like Myclobutanil (Immunox) or Dithane.\n• Alternate fungicide modes of action during wet spring conditions."
        ));

        TREATMENTS.put("Cherry (including sour): Powdery mildew", new Treatment(
            "Powdery Mildew",
            "• Prune dense clusters to maximize airflow and direct sunlight penetrations.\n• Spray neem oil, horticultural oils, or diluted potassium bicarbonate repeatedly on leaf surfaces.",
            "• Apply sulfur fungicides, Flint, or Rally 40WDF.\n• Treat early in the season before visible symptoms spread."
        ));

        TREATMENTS.put("Corn (maize): Cercospora leaf spot Gray leaf spot", new Treatment(
            "Gray Leaf Spot (Cercospora)",
            "• Rotate crops to non-host species for 1-2 seasons.\n• Deep till to bury crop residues where fungal pathogens winter.\n• Use compost-based soil stimulants.",
            "• Apply foliar chemical fungicides containing strobilurins (e.g., Headline) or triazoles (e.g., Tilt).\n• Treat at VT (tasseling) through R1 (silking) stages if thresholds are met."
        ));

        TREATMENTS.put("Corn (maize): Common rust", new Treatment(
            "Common Rust",
            "• Clean equipment thoroughly to prevent spreading spores.\n• Implement balanced nitrogen fertilization to reduce sweet, dense vegetative growth.",
            "• Use fungicides like BASF Headline, Quadris, or Priaxor in high-value corn fields when lesions appear on lower leaves."
        ));

        TREATMENTS.put("Corn (maize): Northern Leaf Blight", new Treatment(
            "Northern Corn Leaf Blight",
            "• Rotate crops with soybeans or alfalfa to break the spore lifecycle.\n• Clean old stalks and till tillage fields deeply in early autumn.",
            "• Treat infected fields with triazole or strobilurin-class chemical mixtures (e.g., Delaro, Trivapro)."
        ));

        TREATMENTS.put("Grape: Black rot", new Treatment(
            "Grape Black Rot",
            "• Prune canopy closely to encourage rapid leaf drying.\n• Wrap and dispose of dried 'mummy' grapes left on the vine.\n• Apply organic copper sprays during pre-bloom.",
            "• Apply chemical fungicides such as Mancozeb, Ziram, or Abound from bud break until 4 weeks post-bloom."
        ));

        TREATMENTS.put("Grape: Esca (Black Measles)", new Treatment(
            "Esca / Wood Canker",
            "• Paint pruning wounds immediately with natural organic wound sealants.\n• Destroy dead wood on vines and replace severely affected trunk sections.",
            "• Spray chemical wound protectors containing carbendazim or flusilazole to pruning wounds before rain events."
        ));

        TREATMENTS.put("Grape: Leaf blight (Isariopsis Leaf Spot)", new Treatment(
            "Isariopsis Leaf Blight",
            "• Rake and burn fallen leaves around grapevines.\n• Increase light penetration by removing shading suckers and leaves.",
            "• Apply chemical protective sprays like Captan or Dithane during mid-summer humid peaks."
        ));

        TREATMENTS.put("Orange: Haunglongbing (Citrus greening)", new Treatment(
            "Citrus Greening (HLB)",
            "• Inspect trees and prune affected limbs in initial stages; remove heavily infected trees completely.\n• Spray foliar organic nutrient sprays (zinc, manganese, iron) to sustain tree production.",
            "• Apply systemic chemical insecticides (imidacloprid, thiamethoxam) to control the transmitting Asian Citrus Psyllids."
        ));

        TREATMENTS.put("Peach: Bacterial spot", new Treatment(
            "Peach Bacterial Spot",
            "• Apply organic copper or sulfur fungicides late in the fall or during pre-bloom.\n• Maintain high orchard hygiene and prune during dry weather.",
            "• Apply chemical oxytetracycline (Mycoshield) or dodine sprays during early active spring leaf flushes."
        ));

        TREATMENTS.put("Pepper, bell: Bacterial spot", new Treatment(
            "Pepper Bacterial Spot",
            "• Apply organic copper sprays once every 7-10 days under humid forecast conditions.\n• Use drip irrigation instead of sprinkler systems to prevent splashing bacteria.",
            "• Spray agricultural streptomycin or copper-mancozeb tank mixtures when initial spots appear on lower stems."
        ));

        TREATMENTS.put("Potato: Early blight", new Treatment(
            "Potato Early Blight (Alternaria)",
            "• Practice 3-year crop rotation and space plants for maximum air.\n• Spray organic sulfur or copper-based liquids around bulb roots or leaves.",
            "• Use preventive chemical fungicides such as chlorothalonil (Daconil), Mancozeb, or Scala."
        ));

        TREATMENTS.put("Potato: Late blight", new Treatment(
            "Potato Late Blight (Phytophthora)",
            "• Ensure perfect soil drainage and avoid overhead watering altogether.\n• Apply organic copper sprays proactively before rainy weather.\n• Harvest tubers during clean, dry weather and discard rot.",
            "• Apply systemic foliar sprays such as Ridomil Gold, Revus Top, or Ranman directly upon local blight status alerts."
        ));

        TREATMENTS.put("Squash: Powdery mildew", new Treatment(
            "Squash Powdery Mildew",
            "• Spray organic milk-water mixtures (40:60 ratio) in bright sunlight.\n• Spray baking soda (potassium bicarbonate) or neem oil thoroughly under leaves.",
            "• Apply protectant fungicides like chlorothalonil or systemic fungicides like Procure, Rally, or Quintec."
        ));

        TREATMENTS.put("Strawberry: Leaf scorch", new Treatment(
            "Strawberry Leaf Scorch",
            "• Plant in well-drained, raised soil beds.\n• Rake out dead leaves from runners during dormant late winter.\n• Apply natural copper sprays at early leaf emerge.",
            "• Apply Captan, Pristine, or Cabrio chemical fungicides before berry flowering wraps."
        ));

        TREATMENTS.put("Tomato: Bacterial spot", new Treatment(
            "Tomato Bacterial Spot",
            "• Avoid touching wet tomato plants to prevent rapid bacterial transport.\n• Spray organic copper formulas early in growing cycles.",
            "• Apply chemical Mancozeb-copper mixes on weekly schedules during persistent rainy conditions."
        ));

        TREATMENTS.put("Tomato: Early blight", new Treatment(
            "Tomato Early Blight",
            "• Trim bottom leaves up to 12 inches off the ground to prevent soil spore splash.\n• Mulch heavily with straw or cardboard below stems.\n• Use sulfur or copper dusts.",
            "• Treat with protectant chemical fungicides like chlorothalonil, Mancozeb, or copper octanoate."
        ));

        TREATMENTS.put("Tomato: Late blight", new Treatment(
            "Tomato Late Blight",
            "• Remove and destroy infected vines immediately; do not compost.\n• Apply copper fungicides before wet morning cycles.\n• Maximize plant-to-plant air ventilation.",
            "• Apply systemic fungicides like Revus Top or Chlorothalonil before humid rainfall spells."
        ));

        TREATMENTS.put("Tomato: Leaf Mold", new Treatment(
            "Tomato Leaf Mold",
            "• Maintain greenhouse humidity below 85% with fans.\n• Prune lower suckers to maximize cross airflow.\n• Spray organic biological fungicides (Bacillus amyloliquefaciens).",
            "• Apply chemical fungicides such as Scala, Quadris, or Bravo Weather Stik."
        ));

        TREATMENTS.put("Tomato: Septoria leaf spot", new Treatment(
            "Tomato Septoria Leaf Spot",
            "• Mulch soil carefully to protect plants from splashback.\n• Control weeds around plots that harbor Septoria pathogens.\n• Use organic copper or serenade sprays.",
            "• Spray chlorothalonil (Daconil) or copper fungicides at the first sign of lower-leaf spots."
        ));

        TREATMENTS.put("Tomato: Spider mites Two-spotted spider mite", new Treatment(
            "Two-Spotted Spider Mites",
            "• Release beneficial predatory mites (Phytoseiulus persimilis) onto plants.\n• Spray with organic insecticidal soap, rosemary oil, or neem oil solutions under leaves.",
            "• Apply chemical miticides like Abamectin (Agri-Mek), Portal, or Oberon."
        ));

        TREATMENTS.put("Tomato: Target Spot", new Treatment(
            "Tomato Target Spot",
            "• Rake debris cleanly at host field endings.\n• Grow on trellises to uplift foliage off wet soil and spray neem oil weekly.",
            "• Treat with systemic or protective fungicides like Inspire Super, Scala, or Delaro."
        ));

        TREATMENTS.put("Tomato: Tomato Yellow Leaf Curl Virus", new Treatment(
            "Tomato Yellow Leaf Curl Virus (TYLCU)",
            "• Cover young plots with insect exclusion mesh screen.\n• Use organic yellow sticky traps to capture vector Whiteflies.\n• Apply insecticidal soaps or horticultural oils.",
            "• Apply systemic nicotinic chemical insecticides (e.g., Dinotefuran, Acetamiprid) to eliminate vector silverleaf whitefly insects."
        ));

        TREATMENTS.put("Tomato: Tomato mosaic virus", new Treatment(
            "Tomato Mosaic Virus (ToMV)",
            "• Destroy infected plants immediately; wash hands/tools in milk or trisodium phosphate.\n• Sterilize stakes and potting soils completely.",
            "• No chemical cures exist. Treat with vector control insecticides if aphid vectors are present, and replace with certified virus-free seed stocks."
        ));

        TREATMENTS.put("Potato: Healthy", new Treatment(
            "Healthy Potato",
            "• No treatment needed. Keep maintaining good soil health and regular watering.\n• Monitor for early signs of pests or diseases.",
            "• No chemical treatment required for healthy plants."
        ));

        TREATMENTS.put("Tomato: Healthy", new Treatment(
            "Healthy Tomato",
            "• No treatment needed. Ensure consistent watering and proper support (staking/caging).\n• Regularly inspect for signs of pests like hornworms or whiteflies.",
            "• No chemical treatment required for healthy plants."
        ));
    }

    public static Treatment getTreatment(String labelName) {
        if (labelName == null) return null;
        
        // Match exact or contains substring
        Treatment t = TREATMENTS.get(labelName);
        if (t != null) return t;

        // Try standardizing matches
        for (Map.Entry<String, Treatment> entry : TREATMENTS.entrySet()) {
            if (labelName.toLowerCase().contains(entry.getKey().toLowerCase()) || 
                entry.getKey().toLowerCase().contains(labelName.toLowerCase())) {
                return entry.getValue();
            }
        }
        return null;
    }
}
