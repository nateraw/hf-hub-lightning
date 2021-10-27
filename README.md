# hf-hub-lightning

A callback for pushing lightning models to the Hugging Face Hub.

**Note:** I made this package for myself, mostly...if folks seem to be interested in it, we'll move this into `huggingface_hub` directly and archive this repo. Since there are a lot of options/considerations, I decided to just post it separately for now. 

## Setup

```
pip install hf-hub-lightning
```

## Usage

To periodically upload your model ckpt while training, you can do the following...

```python
import pytorch_lightning as pl
from hf_hub_lightning import HuggingFaceHubCallback

train_dataloader = ...
model = YourCoolModel()
trainer = pl.Trainer(callbacks=[HuggingFaceHubCallback('your_username/model_id')])
trainer.fit(model, train_dataloader)
```

To load your model back from the huggingface hub, just do...

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download('your_username/model_id', 'lit_model.ckpt')
model = YourCoolModel.load_from_checkpoint(ckpt_path)
```
