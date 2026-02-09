from datasets.freihand_dataset import FreiHANDDataset

dataset = FreiHANDDataset(
    root_dir=r"C:\Users\Soham\Downloads\FreiHAND_pub_v2"
)

img, heatmaps = dataset[0]

print("Image shape:", img.shape)
print("Heatmaps shape:", heatmaps.shape)
