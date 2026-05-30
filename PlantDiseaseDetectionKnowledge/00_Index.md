# PlantDiseaseDetection — Code Logic Entities Map (Obsidian)

This is an Obsidian-first knowledge base. **Pages are Markdown-only** and the map uses `[[wikilinks]]` without code blocks.

---

## Entity pages
- [[01 Inference_TFLite]]
- [[02 Inference_Preprocess]]
- [[03 Delegates]]
- [[04 Training_Loop]]
- [[05 Data_Split_Sampler]]
- [[06 Augmentation_and_Masking]]
- [[07 Preprocess_Export_Confusion]]

---

## Connection map (wikilink-based)

Inference entrypoint
- [[01 Inference_TFLite]] → [[02 Inference_Preprocess]]
- [[01 Inference_TFLite]] → [[03 Delegates]]

Training data pipeline
- [[04 Training_Loop]] ← [[05 Data_Split_Sampler]]
- [[04 Training_Loop]] ← [[06 Augmentation_and_Masking]]

Pipeline consistency / refactor
- [[07 Preprocess_Export_Confusion]] → affects how inference expects labels + metadata
- [[02 Inference_Preprocess]] → should align with training normalization/masking contract

---

## Suggested reading order
1. [[01 Inference_TFLite]]
2. [[02 Inference_Preprocess]]
3. [[03 Delegates]]
4. [[04 Training_Loop]]
5. [[05 Data_Split_Sampler]]
6. [[06 Augmentation_and_Masking]]
7. [[07 Preprocess_Export_Confusion]]

