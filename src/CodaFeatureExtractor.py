import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    Mask2FormerForUniversalSegmentation,
    AutoModel
)
from InternProcessor import load_image

class CodaFeatureExtractor(torch.nn.Module):
    def __init__(self, device, num_patches_per_side=16, max_instances=50):
        super().__init__()
        
        self.device = device
        self.num_patches_per_side = num_patches_per_side
        self.num_patches = num_patches_per_side * num_patches_per_side
        self.total_patches_plus_cls = self.num_patches + 1
        self.max_instances = max_instances
        self.num_classes = 19  # Number of Cityscapes classes
        self.image_size = 336
        
        # Initialize Mask2Former for both segmentation and panoptic
        self.seg_processor = AutoImageProcessor.from_pretrained(
            "facebook/mask2former-swin-small-cityscapes-panoptic"
        )
        self.seg_model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-small-cityscapes-panoptic"
        ).to(device)
        self.seg_model.eval()
        
        # Initialize Depth model
        self.depth_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
        )
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf"
        ).to(device)
        self.depth_model.eval()

        # Initialize ViT model
        self.vit_model = AutoModel.from_pretrained(
            'OpenGVLab/Mini-InternVL2-1B-DA-Drivelm',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        )
        del self.vit_model.language_model
        self.vit_model = self.vit_model.to(device)
        self.vit_model.eval()

    def extract_segmentation_and_panoptic(self, images):
        inputs = self.seg_processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.seg_model(**inputs)
            
        # Process panoptic segmentation
        panoptic_results = self.seg_processor.post_process_panoptic_segmentation(
            outputs, label_ids_to_fuse=[], target_sizes=[(self.image_size, self.image_size)] * len(images)
        )
        
        # Initialize batch of segmentation maps
        seg_mask = torch.zeros((len(images), self.num_classes, self.image_size, self.image_size), device=self.device)
        # Pre-allocate instance features tensor
        instance_features = torch.zeros(len(images), self.max_instances, self.num_classes + 2, device=self.device)
        # Initialize attention mask tensor - True means padding (to be ignored)
        attention_mask = torch.ones(len(images), self.max_instances, dtype=torch.bool, device=self.device)
        attention_mask[:, 0] = False
        
        # Process each image in batch
        for batch_idx, result in enumerate(panoptic_results):
            seg_map = result["segmentation"]  # [image_size, image_size]
            instances = result["segments_info"]
            
            # Process each instance
            for idx, instance in enumerate(instances[:self.max_instances]):
                attention_mask[batch_idx, idx] = False  # Mark this instance as valid (not padding)
                # Get instance mask
                instance_mask = (seg_map == instance["id"]).float()
                
                # Add to segmentation map
                label_id = instance["label_id"]
                assert label_id < self.num_classes  # Ensure valid class index
                seg_mask[batch_idx, label_id] += instance_mask
                
                # Calculate centroid if instance exists
                if instance_mask.sum() > 0:
                    y_indices = torch.arange(self.image_size, device=self.device).float().view(1, -1)
                    x_indices = torch.arange(self.image_size, device=self.device).float().view(-1, 1)
                    
                    y_centroid = (instance_mask * y_indices).sum() / instance_mask.sum()
                    x_centroid = (instance_mask * x_indices).sum() / instance_mask.sum()
                    
                    # One-hot encode class and add normalized coordinates
                    instance_features[batch_idx, idx, label_id] = 1.0  # One-hot class encoding
                    instance_features[batch_idx, idx, -2:] = torch.tensor([
                        x_centroid / self.image_size,   # normalized x
                        y_centroid / self.image_size    # normalized y
                    ], device=self.device)
        
        return seg_mask, instance_features, attention_mask

    def extract_depth(self, images):
        inputs = self.depth_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
        depth = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode="bicubic"
        )
        return depth.squeeze(1)  # [batch_size, image_size, image_size]

    def extract_vit_features(self, images):
        batch_features = []
        for img in images:
            pixel_values = load_image(img, max_num=1).to(torch.bfloat16).to(self.device)
            with torch.no_grad():
                # Get features and reshape to match our patch structure
                features = self.vit_model.extract_feature(pixel_values)  # [1, 256, 896]
                # Construct pseudo-CLS token by averaging patch tokens
                pseudo_cls_token = features.mean(dim=1, keepdim=True)  # [1, 1, 896]
                features = torch.cat([pseudo_cls_token, features], dim=1)  # [1, num_patches + 1, 896]
            batch_features.append(features)
        return torch.cat(batch_features, dim=0)  # [batch_size, num_patches + 1, 896]

    def divide_into_patches(self, feature_map, feature_type):
        batch_size = feature_map.shape[0]
        patch_size = self.image_size // self.num_patches_per_side

        if feature_type == 'segmentation':
            # Original shape: [batch_size, num_classes, image_size, image_size]
            patches = feature_map.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            # Average pool each patch
            patches = patches.mean(dim=(-2, -1))  # [batch_size, num_classes, num_patches, num_patches]
            # Reshape to get patches
            patches = patches.permute(0, 2, 3, 1).reshape(batch_size, self.num_patches, self.num_classes)
            # Add extra patch (mean of all patches)
            extra_patch = patches.mean(dim=1, keepdim=True)  # [batch_size, 1, num_classes]
            patches = torch.cat([extra_patch, patches], dim=1)  # [batch_size, num_patches + 1, num_classes]
        
        elif feature_type == 'depth':
            # Original shape: [batch_size, image_size, image_size]
            patches = feature_map.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
            # Average pool each patch
            patches = patches.mean(dim=(-2, -1))  # [batch_size, num_patches, num_patches]
            # Reshape to get patches
            patches = patches.reshape(batch_size, self.num_patches, 1)
            # Add extra patch (mean of all patches)
            extra_patch = patches.mean(dim=1, keepdim=True)  # [batch_size, 1, 1]
            patches = torch.cat([extra_patch, patches], dim=1)  # [batch_size, num_patches + 1, 1]
            
        return patches

    def process_images(self, images):
        # Extract features
        seg_features, instance_features, instance_attention_mask = self.extract_segmentation_and_panoptic(images)
        depth_features = self.extract_depth(images)
        vit_features = self.extract_vit_features(images)
        
        # Process patch-based features
        seg_patches = self.divide_into_patches(seg_features, 'segmentation')
        depth_patches = self.divide_into_patches(depth_features, 'depth')
        
        # Concatenate patch features
        concatenated_patches = torch.cat([
            seg_patches,    # [batch_size, num_patches + 1, num_classes]
            depth_patches,  # [batch_size, num_patches + 1, 1]
            vit_features   # [batch_size, num_patches + 1, 896]
        ], dim=-1)
        
        return {
            'patch_tokens': concatenated_patches,  # [batch_size, num_patches + 1, feature_dim]
            'instance_tokens': instance_features,  # [batch_size, max_instances, num_classes + 2]
            'instance_attention_mask': instance_attention_mask,  # [batch_size, max_instances]
        }