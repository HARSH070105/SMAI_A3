import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ==========================================
# 1. Setup Model and Device
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

print(f"Loading {model_id} on {device}...")
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)
model.eval()

# ==========================================
# 2. Define Classes and Encode Text Baseline
# ==========================================
monuments_24 = [
    "Ajanta Caves", "Charar-E- Sharif", "Chhota_Imambara", "Ellora Caves", 
    "Fatehpur Sikri", "Gateway of India", "Hawa mahal", "Humayun_s Tomb", 
    "India_gate", "Khajuraho", "Sun Temple Konark", "alai_darwaza", 
    "alai_minar", "basilica_of_bom_jesus", "charminar", "golden temple", 
    "iron_pillar", "jamali_kamali_tomb", "lotus_temple", "mysore_palace", 
    "qutub_minar", "tajmahal", "tanjavur temple", "victoria memorial"
]

# Create the single baseline prompt for each monument
# Replacing underscores with spaces for a better prompt
baseline_prompts = [f"A photo of the {monument.replace('_', ' ')}." for monument in monuments_24]

print("Encoding baseline text prompts...")
with torch.no_grad():
    # Process all text prompts at once
    text_inputs = processor(text=baseline_prompts, return_tensors="pt", padding=True).to(device)
    text_outputs = model.get_text_features(**text_inputs)
    
    # Handle if transformers returns an object instead of a tensor
    if not isinstance(text_outputs, torch.Tensor):
        if hasattr(text_outputs, "text_embeds"):
            text_features = text_outputs.text_embeds
        elif hasattr(text_outputs, "pooler_output"):
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs[0]
    else:
        text_features = text_outputs
    
    # Normalize the text vectors
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ==========================================
# 3. Run Inference on Test Set
# ==========================================
# Path to the test folder
TEST_DIR = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "test")

y_true = []
y_pred = []

print("\nStarting Zero-Shot Single Prompt Benchmarking...")
with torch.no_grad():
    for class_idx, true_class in enumerate(monuments_24):
        class_folder = os.path.join(TEST_DIR, true_class)
        
        if not os.path.exists(class_folder):
            print(f"Warning: Folder not found for {true_class}")
            continue
            
        for img_name in tqdm(os.listdir(class_folder), desc=f"Evaluating {true_class}"):
            img_path = os.path.join(class_folder, img_name)
            
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue # Skip broken/unreadable images
                
            # Process image and get embedding
            image_inputs = processor(images=image, return_tensors="pt").to(device)
            image_outputs = model.get_image_features(**image_inputs)
            
            # Handle if transformers returns an object instead of a tensor
            if not isinstance(image_outputs, torch.Tensor):
                if hasattr(image_outputs, "image_embeds"):
                    image_features = image_outputs.image_embeds
                elif hasattr(image_outputs, "pooler_output"):
                    image_features = image_outputs.pooler_output
                else:
                    image_features = image_outputs[0]
            else:
                image_features = image_outputs
            
            # Normalize image feature
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate Cosine Similarity against all 24 text vectors
            # text_features is shape (24, 512). We transpose it to (512, 24) for the dot product.
            similarity_scores = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get the index of the highest score
            predicted_idx = similarity_scores.argmax().item()
            predicted_class = monuments_24[predicted_idx]
            
            y_true.append(true_class)
            y_pred.append(predicted_class)

# ==========================================
# 4. Output Results
# ==========================================
accuracy_str = f"Overall Top-1 Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%"
report_str = classification_report(y_true, y_pred)

print("\n" + "="*50)
print("   SINGLE PROMPT BASELINE RESULTS")
print("="*50)
print(accuracy_str + "\n")
print("Detailed Report:")
print(report_str)

# Save to reports folder
os.makedirs(os.path.join(ROOT_DIR, "reports"), exist_ok=True)
report_path = os.path.join(ROOT_DIR, "reports", "benchmark_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 50 + "\n")
    f.write("   SINGLE PROMPT BASELINE RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(accuracy_str + "\n\n")
    f.write("Detailed Report:\n")
    f.write(report_str + "\n")
print(f"\nReport saved to {report_path}")

# ==========================================
# 5. Normalized Confusion Matrix
# ==========================================
labels = sorted(set(y_true))
cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # normalize rows -> sum to 1

fig, ax = plt.subplots(figsize=(18, 16))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    vmin=0.0,
    vmax=1.0,
    ax=ax,
)
ax.set_title(f"Normalized Confusion Matrix — Single Prompt Baseline\n{accuracy_str}", fontsize=14, pad=16)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

cm_path = os.path.join(ROOT_DIR, "reports", "baseline_confusion_matrix.png")
fig.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"Confusion matrix saved to {cm_path}")

