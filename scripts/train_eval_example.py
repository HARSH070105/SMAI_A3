"""
train_eval_example.py

Shows how to wire together:
  - Hugging Face dataset loading (train + test splits)
  - Monument metadata JSON for enriched text prompts
  - CLIP model with optional LoRA
  - Zero-shot evaluation on the test split
"""

from dataset_utils import load_hf_dataset, load_monument_metadata, build_enriched_prompts, MonumentDataset
from model_utils import load_model, precompute_text_features, run_inference
from torch.utils.data import DataLoader
from tqdm import tqdm


# -----------------------------------------------------------------------
# CONFIG  – edit these
# -----------------------------------------------------------------------
HF_REPO_ID        = "Amisha2905/Monuments_SMAI"  # <-- your HF repo
METADATA_JSON     = "/home/bluebottle/iiit/acad/sem6/smai/assignments/a3/data/monuments_metadata.json"               # <-- your JSON file
LORA_WEIGHTS_PATH = "/home/bluebottle/iiit/acad/sem6/smai/assignments/a3/models/baseline_lora_adapters"                                    # or path to saved LoRA
BASE_MODEL_ID     = "openai/clip-vit-base-patch32"
BATCH_SIZE        = 32
# -----------------------------------------------------------------------


def evaluate(model, processor, text_features, class_names, test_dataset, device):
    """Zero-shot accuracy over the full test split."""
    dataset    = MonumentDataset(test_dataset)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    correct, total = 0, 0

    for images, labels in tqdm(loader, desc="Evaluating"):
        for img, label in zip(images, labels):
            pred_class, _ = run_inference(img, model, processor, text_features, class_names, device)
            pred_idx      = class_names.index(pred_class)
            if pred_idx == label.item():
                correct += 1
            total += 1

    accuracy = correct / total * 100
    print(f"\nTest Accuracy: {accuracy:.2f}%  ({correct}/{total})")
    return accuracy


def main():
    # 1. Load dataset from Hugging Face
    train_split, test_split, class_names = load_hf_dataset(HF_REPO_ID)

    # 2. Load monument metadata JSON
    metadata = load_monument_metadata(METADATA_JSON)

    # 3. Build enriched prompts (uses history, fun_facts, location, category)
    enriched_prompts = build_enriched_prompts(class_names, metadata)

    # 4. Load CLIP model (+ optional LoRA)
    model, processor, device = load_model(
        base_model_id     = BASE_MODEL_ID,
        lora_weights_path = LORA_WEIGHTS_PATH,
    )

    # 5. Pre-compute text prototype vectors using metadata-enriched prompts
    text_features = precompute_text_features(
        model            = model,
        processor        = processor,
        class_names      = class_names,
        device           = device,
        enriched_prompts = enriched_prompts,   # <-- pass the metadata prompts here
    )
    print(f"Text features shape: {text_features.shape}")  # (num_classes, 512)

    # 6. Zero-shot evaluation
    evaluate(model, processor, text_features, class_names, test_split, device)

    # 7. Single-image inference example
    sample_image, sample_label = MonumentDataset(test_split)[0]
    pred_class, probs = run_inference(
        sample_image, model, processor, text_features, class_names, device
    )
    print(f"\nSample Prediction : {pred_class}")
    print(f"True Label        : {class_names[sample_label]}")
    print("Top-5 Probabilities:")
    for name, prob in list(probs.items())[:5]:
        print(f"  {name:<40} {prob*100:.2f}%")


if __name__ == "__main__":
    main()
