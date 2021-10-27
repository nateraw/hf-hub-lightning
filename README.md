# hf-hub-lightning

A callback for pushing lightning models to the Hugging Face Hub.


## Setup

```
pip install hf-hub-lightning
```

## Usage

To periodically upload your model ckpt while training, you can do the following...

```python
import pytorch_lightning as pl
from hf_hub_lightning import HuggingFaceHubCallback

model = YourCoolModel()
trainer = pl.Trainer(callbacks=[HuggingFaceHubCallback('your_username/model_id')])
trainer.fit(model)
```

To load your model back from the huggingface hub, just do...

```python
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download('your_username/model_id', 'lit_model.ckpt')
model = YourCoolModel.load_from_checkpoint(ckpt_path)
```
