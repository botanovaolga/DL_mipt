import torch
import pandas as pd
import cv2
from preprocessing import get_vinbigdata_dicts_test, VinbigdataTwoClassDataset, Flags, flags_dict
from model import Classifier, cross_entropy_with_logits, accuracy_with_logits, accuracy
from PIL import Image
from torch.utils.data.dataloader import DataLoader
flags = Flags().update(flags_dict)

imgdir = '/Users/botanovaolga/Desktop/mipt/DL_mipt/dl_project/test_img/002a34c58c5b758217ed1f584ccbcfe9.png'
model = torch.load("/Users/botanovaolga/Desktop/mipt/DL_mipt/dl_project/classifier.pt", map_location=torch.device('cpu'))
img = Image.open(imgdir)
width, height = img.size
image_id = 1
test_meta = pd.DataFrame([image_id, width, height]).transpose().rename(columns={0: "image_id", 1: "dim0", 2: "dim1"})


flags = Flags().update(flags_dict)

# predictor = build_predictor(model_name=flags.model_name, model_mode=flags.model_mode)
# classifier = Classifier(predictor)

dataset_dicts_test = get_vinbigdata_dicts_test(imgdir, test_meta)
test_dataset = VinbigdataTwoClassDataset(dataset_dicts_test, train=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=flags.valid_batchsize,
    num_workers=flags.num_workers,
    shuffle=False,
    pin_memory=True,
)

test_pred = model.predict_proba(test_loader).cpu().numpy()
test_pred_df = pd.DataFrame({
    "image_id": [d["image_id"] for d in dataset_dicts_test],
    "class0": test_pred[:, 0],
    "class1": test_pred[:, 1]
})

print(test_pred_df)


