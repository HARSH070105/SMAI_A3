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
# 2. Define Classes & Process Ensemble Prompts
# ==========================================
monuments_24 = [
    "Ajanta Caves", "Charar-E- Sharif", "Chhota_Imambara", "Ellora Caves", 
    "Fatehpur Sikri", "Gateway of India", "Hawa mahal", "Humayun_s Tomb", 
    "India_gate", "Khajuraho", "Sun Temple Konark", "alai_darwaza", 
    "alai_minar", "basilica_of_bom_jesus", "charminar", "golden temple", 
    "iron_pillar", "jamali_kamali_tomb", "lotus_temple", "mysore_palace", 
    "qutub_minar", "tajmahal", "tanjavur temple", "victoria memorial"
]

# 25 Targeted templates designed to maximize CLIP zero-shot accuracy
# Covers: visual detail, material, geography, style, lighting, viewpoint
templates = [
    # Core identity templates
    "A photo of the {}.",
    "The iconic {}.",
    "A historical landmark known as the {}.",
    "A famous Indian monument: the {}.",
    "A UNESCO World Heritage Site: the {}.",

    # Architecture and style
    "An architectural photograph of the {}.",
    "The intricate stone carvings and architecture of the {}.",
    "The detailed exterior facade of the {}.",
    "The ornate structure of the {} in India.",
    "The ancient {} with its distinctive architectural style.",

    # Visual and material descriptors
    "A photo of the {} showing its stonework and design.",
    "The {} built from red sandstone.",
    "The {} built from white marble.",
    "The {} featuring domes and minarets.",
    "The {} featuring Dravidian temple architecture.",

    # Geographic / contextual
    "A photo of the {} located in India.",
    "A tourist photo at the {}.",
    "Visitors at the {}.",
    "A photo of the {} against the sky.",
    "A photo of the {} from the ground looking up.",

    # Viewpoints and lighting
    "A wide-angle shot of the {}.",
    "A close-up detail shot of the {}.",
    "A photo of the {} in bright daylight.",
    "A photo of the {} at golden hour.",
    "A nighttime photo of the {} illuminated by lights.",
]

print("Encoding and averaging ensemble text prompts...")
prototype_vectors = []

with torch.no_grad():
    for monument in tqdm(monuments_24, desc="Processing Text Prototypes"):
        # Clean up the name (e.g., "India_gate" -> "India gate")
        clean_name = monument.replace('_', ' ')
        
        # Generate the 15 sentences for this specific monument
        ensemble_sentences = [template.format(clean_name) for template in templates]
        
        # Process and encode
        text_inputs = processor(text=ensemble_sentences, return_tensors="pt", padding=True).to(device)
        text_outputs = model.get_text_features(**text_inputs)
        
        # Extract the tensor if transformers returned an object
        if not isinstance(text_outputs, torch.Tensor):
            if hasattr(text_outputs, "text_embeds"):
                text_features = text_outputs.text_embeds
            elif hasattr(text_outputs, "pooler_output"):
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs[0]
        else:
            text_features = text_outputs
        
        # Math: 1. Normalize all 15 vectors
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Math: 2. Average them into a single vector (mean across the 15 prompts)
        mean_vector = text_features.mean(dim=0)
        # Math: 3. Normalize the final averaged prototype vector
        prototype_vector = mean_vector / mean_vector.norm(dim=-1, keepdim=True)
        
        prototype_vectors.append(prototype_vector)

# Stack all 24 individual vectors into a single matrix of shape (24, 512)
ensemble_text_matrix = torch.stack(prototype_vectors).to(device)

# ==========================================
# 3. Run Inference on Test Set
# ==========================================
TEST_DIR = os.path.join(ROOT_DIR, "data", "24_monuments", "Indian-monuments", "images", "test")

y_true = []
y_pred = []

print("\nStarting Zero-Shot Ensemble Prompt Benchmarking...")
with torch.no_grad():
    for class_idx, true_class in enumerate(monuments_24):
        class_folder = os.path.join(TEST_DIR, true_class)
        
        if not os.path.exists(class_folder):
            continue
            
        for img_name in tqdm(os.listdir(class_folder), desc=f"Evaluating {true_class}"):
            img_path = os.path.join(class_folder, img_name)
            
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue 
                
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
            
            # Calculate Cosine Similarity against the ENSEMBLE matrix
            similarity_scores = (100.0 * image_features @ ensemble_text_matrix.T).softmax(dim=-1)
            
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
print("   ENSEMBLE PROMPT RESULTS")
print("="*50)
print(accuracy_str + "\n")
print("Detailed Report:")
print(report_str)

# Save to reports folder
os.makedirs(os.path.join(ROOT_DIR, "reports"), exist_ok=True)
report_path = os.path.join(ROOT_DIR, "reports", "ensemble_report.txt")
with open(report_path, "w") as f:
    f.write("=" * 50 + "\n")
    f.write("   ENSEMBLE PROMPT RESULTS\n")
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
ax.set_title(f"Normalized Confusion Matrix — Ensemble Prompts\n{accuracy_str}", fontsize=14, pad=16)
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()

cm_path = os.path.join(ROOT_DIR, "reports", "ensemble_confusion_matrix.png")
fig.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"Confusion matrix saved to {cm_path}")

