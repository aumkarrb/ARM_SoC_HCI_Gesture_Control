from datasets.hagrid_clean_dataset import HaGRIDCleanDataset
from torch.utils.data import DataLoader
from config import HAGRID_ROOT, BATCH_SIZE, NUM_WORKERS


def main():
    ds = HaGRIDCleanDataset(HAGRID_ROOT, split="train")

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    x, y = next(iter(loader))
    print("Images:", x.shape)
    print("Labels:", y.shape)


if __name__ == "__main__":
    main()
