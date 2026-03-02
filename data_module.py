from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
import os
from datasets import load_dataset

class KleinDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images.
    """

    def __init__(
        self,
        size=(512,512),
        split="train",  
        test_samples=10,  # Number of samples for test split
    ):
        self.size = size
        self.split = split
        self.test_samples = test_samples
        self.musar = load_dataset('guozinan/MUSAR')['train']
        print(f"Loaded MUSAR-Gen dataset with {len(self.musar)} samples.")
        if split == "test":
            self.musar = self.musar.select(range(test_samples))  
        elif split == "train":
            self.musar = self.musar.select(range(test_samples, len(self.musar)))
        else:
            raise ValueError("Invalid split")
        self._length = len(self.musar)
        # Setup transformations
        width, height = self.size
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize((height, width), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _convert_to_pil(self, image_data):
        """
        Convert image data to PIL Image if it's in dict format.
        """
        
        if isinstance(image_data, str):
            return Image.open(image_data)
        elif isinstance(image_data,Image.Image):
            return image_data
        else:
            # Assume it's already a PIL Image or a path string
            raise ValueError("Unsupported image data format.")

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        # Load images on demand from the dataset
        sample = self.musar[index]
        source_image1 = sample['cond_img_0']
        source_image2 = sample['cond_img_1']
        target_image = sample['tgt_img']
        prompt = sample['prompt']
        # Convert to PIL Images if they're in dict format
        source_image1 = self._convert_to_pil(source_image1)
        source_image2 = self._convert_to_pil(source_image2)
        target_image = self._convert_to_pil(target_image)
        
        # Convert image modes if needed
        if not source_image1.mode == "RGB":
            source_image1 = source_image1.convert("RGB")
        if not source_image2.mode == "RGB":
            source_image2 = source_image2.convert("RGB")
        if not target_image.mode == "RGB":
            target_image = target_image.convert("RGB")
        
        # Apply transformations
        source_tensor = torch.stack([self.train_transforms(source_image1),self.train_transforms(source_image2)], dim=0) 
        target_tensor = self.train_transforms(target_image)

        # Return the concatenated image, mask, and prompts in a dictionary
        example = {
            "target_image": target_tensor,
            "source_image": source_tensor,
            "prompts": prompt,
        }
        
        return example


def collate_fn(batch):
    source_images = torch.stack([item['source_image'] for item in batch])
    target_images = torch.stack([item['target_image'] for item in batch])
    captions = [item['prompts'] for item in batch]
    
    return {
        "source_image": source_images, 
        "target_image": target_images, 
        "prompts": captions,
    }

if __name__ == "__main__":
    # run python data_module.py to test the dataset and dataloader
    # Create dataset instance with the updated parameters:
    dataset = KleinDataset()
    
    # Print dataset length
    print(f"Dataset size: {len(dataset)}")
    
    # Get first sample
    print("Loading first sample...")
    # shuffle the dataset
    sample = dataset[0]
    
    # Print tensor shapes
    print("\nTensor shape of concatenated image:")
    print(f"Source image: {sample['source_image'].shape}")
    print(f"Target image: {sample['target_image'].shape}")
    
    # Print prompts (truncated if too long)
    prompts = sample["prompts"]
    if len(prompts) > 100:
        prompts = prompts[:100] + "..."
    print(f"\nCaption: {prompts}")

    # Create and test DataLoader with custom collate function
    batch_size = 4
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print("\nTesting batch processing:")
    print(f"Getting first batch of size {batch_size}...")
    batch = next(iter(dataloader))
    
    # Print tensor shapes for batch
    print("Tensor shapes for batch:")
    print(f"Target images: {batch['target_image'].shape}")
    print(f"Sources: {batch['source_image'].shape}")
    print(f"Number of captions: {len(batch['prompts'])}")
    
    # Print first prompts in batch
    first_caption = batch['prompts'][0]
    if len(first_caption) > 100:
        first_caption = first_caption
    print(f"First prompts in batch: {first_caption}")
