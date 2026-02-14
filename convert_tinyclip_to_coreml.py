import torch
import coremltools as ct
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import os

def export_tinyclip_coreml():
    # Model ID
    model_name = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
    print(f"Loading model: {model_name}")
    
    # 1. Load Model and Processor
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()

    # Get Normalization Parameters from Processor
    # Use default CLIP mean/std if not explicitly present in config, 
    # but normally they are in feature_extractor
    image_mean = getattr(processor.feature_extractor, "image_mean", [0.48145466, 0.4578275, 0.40821073])
    image_std = getattr(processor.feature_extractor, "image_std", [0.26862954, 0.26130258, 0.27577711])
    
    # Ensure they are lists
    if image_mean is None: image_mean = [0.48145466, 0.4578275, 0.40821073]
    if image_std is None: image_std = [0.26862954, 0.26130258, 0.27577711]

    print(f"Using Mean: {image_mean}")
    print(f"Using Std: {image_std}")

    # ============================
    # 1. Vision Model Conversion
    # ============================
    
    # Define a wrapper that handles normalization inside the model.
    # We do this because CoreML ImageType `bias` is added BEFORE `scale` if both are used? 
    # Actually CoreML formula is: output = input * scale + bias
    # But usually it's cleaner to handle channel-wise normalization 
    # explicitly in the PyTorch graph if we want to be 100% sure of the math 
    # and support different std per channel easily if CoreML version has quirks.
    # Here input will be [0, 1] (via scale=1/255).
    class VisionModelWrapper(torch.nn.Module):
        def __init__(self, vision_model, projection, mean, std):
            super().__init__()
            self.vision_model = vision_model
            self.visual_projection = projection
            # Register mean and std as buffers for export
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

        def forward(self, image):
            # image is [0, 1] (from CoreML scale=1/255)
            # Normalize: (image - mean) / std
            x = (image - self.mean) / self.std
            
            # Forward pass through CLIP Vision Model
            vision_outputs = self.vision_model(pixel_values=x)
            pooled_output = vision_outputs[1]  # pooled_output
            image_features = self.visual_projection(pooled_output)
            return image_features

    # Create vision wrapper instance
    vision_wrapper = VisionModelWrapper(
        model.vision_model, 
        model.visual_projection, 
        image_mean, 
        image_std
    )
    
    # Trace the vision model
    # Input shape: 1, 3, 224, 224
    dummy_image = torch.rand(1, 3, 224, 224)
    traced_vision = torch.jit.trace(vision_wrapper, dummy_image)
    
    # Convert to CoreML
    # We specify scale=1/255.0 to transform uint8 [0, 255] input to [0, 1] floating point
    vision_mlmodel = ct.convert(
        traced_vision,
        inputs=[
            ct.ImageType(
                name="image", 
                shape=(1, 3, 224, 224), 
                scale=1/255.0,
                color_layout=ct.colorlayout.RGB
            )
        ],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS16, 
        convert_to="mlprogram"
    )
    
    # Set Metadata
    vision_mlmodel.short_description = "TinyCLIP Vision Model"
    vision_mlmodel.input_description["image"] = "Input image (224x224 RGB). resizing should be done before input."
    vision_mlmodel.output_description["embedding"] = "Image features embedding (512)"
    
    vision_output_path = "TinyCLIP_Vision.mlpackage"
    vision_mlmodel.save(vision_output_path)
    print(f"Saved {vision_output_path}")

    # ============================
    # 2. Text Model Conversion
    # ============================

    class TextModelWrapper(torch.nn.Module):
        def __init__(self, text_model, projection, pad_token_id=1):
            super().__init__()
            self.text_model = text_model
            self.text_projection = projection
            self.pad_token_id = pad_token_id

        def forward(self, input_ids):
            # Create attention mask automatically: 1 for real tokens, 0 for padding
            # pad_token_id is 1 for this model config
            attention_mask = (input_ids != self.pad_token_id).long()
            
            text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = text_outputs[1]
            text_features = self.text_projection(pooled_output)
            return text_features

    # pad_token_id=1 based on config.json
    text_wrapper = TextModelWrapper(model.text_model, model.text_projection, pad_token_id=1)
    
    # Trace the text model
    dummy_input_ids = torch.zeros((1, 77), dtype=torch.long)
    # We don't pass attention_mask explicitly anymore
    
    traced_text = torch.jit.trace(text_wrapper, dummy_input_ids)
    
    # Convert to CoreML
    text_mlmodel = ct.convert(
        traced_text,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, 77), dtype=np.int32),
        ],
        outputs=[ct.TensorType(name="embedding")],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram"
    )

    text_mlmodel.short_description = "TinyCLIP Text Model"
    text_mlmodel.input_description["input_ids"] = "Token IDs (Int32). Padding token is 1."
    text_mlmodel.output_description["embedding"] = "Text features embedding (512)"

    text_output_path = "TinyCLIP_Text.mlpackage"
    text_mlmodel.save(text_output_path)
    print(f"Saved {text_output_path}")

    return vision_output_path, text_output_path

def run_inference_example(vision_model_path, text_model_path):
    print("\nRunning Inference Example...")
    from PIL import Image
    
    # Paths
    model_name = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
    image_path = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M/figure/fig1.jpg"
    
    # 1. Load Data
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}, skipping inference example.")
        return

    image = Image.open(image_path).convert("RGB")
    text = ["a photo of a cat", "a photo of a dog"] 
    # The figure fig1.jpg contains a cat and a dog usually, or just a cat. 
    # Let's use a generic caption relevant to standard testing.
    
    # 2. PyTorch Inference
    print("--- PyTorch Inference ---")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    
    print(f"Logits: {logits_per_image}")
    print(f"Probs: {probs}")

    # 3. CoreML Inference
    print("--- CoreML Inference ---")
    # Load models
    vision_model = ct.models.MLModel(vision_model_path)
    text_model = ct.models.MLModel(text_model_path)
    
    # Prepare Image Input
    # Resize to 224x224 (CoreML model expects this, and usually handles it via ImageType if from PIL, 
    # but let's be explicit if needed. ImageType handles resize automatically usually.)
    # Note: Our CoreML model input is "image" (ImageType).
    
    vision_input = {"image": image.resize((224, 224))} 
    
    # Run Vision
    # Output name is "embedding"
    vision_output = vision_model.predict(vision_input)
    image_embedding = vision_output["embedding"] # Shape (1, 512)
    
    # Prepare Text Input
    # We need to tokenize manually or use the processor's tokenizer
    # The CoreML model expects "input_ids" of shape (1, 77)
    
    for t in text:
        text_inputs = processor.tokenizer(t, padding="max_length", max_length=77, return_tensors="np")
        input_ids = text_inputs["input_ids"].astype(np.int32) # (1, 77)
        
        # Run Text
        text_output = text_model.predict({"input_ids": input_ids})
        text_embedding = text_output["embedding"] # Shape (1, 512)
        
        # Normalize embeddings (CLIP behavior)
        image_embedding_norm = image_embedding / np.linalg.norm(image_embedding)
        text_embedding_norm = text_embedding / np.linalg.norm(text_embedding)
        
        # Calculate Cosine Similarity (logit scale is usually applied after)
        # PyTorch CLIP logit_scale is exp(model.logit_scale)
        logit_scale = model.logit_scale.exp().item()
        
        # dot product
        # Flatten to vectors for simple dot product
        dot_product = np.dot(image_embedding_norm.flatten(), text_embedding_norm.flatten())
        score = dot_product * logit_scale
        
        print(f"Text: '{t}'")
        print(f"  Dot Product: {dot_product:.4f}")
        print(f"  Scaled Score: {score:.4f}")


if __name__ == "__main__":
    vision_path, text_path = export_tinyclip_coreml()
    run_inference_example(vision_path, text_path)
