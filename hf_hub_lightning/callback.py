import tempfile
from pathlib import Path
from shutil import copy2

from huggingface_hub import Repository
from huggingface_hub.constants import ENDPOINT
from pytorch_lightning import Callback


class HuggingFaceHubCallback(Callback):
    def __init__(self, repo_name, local_dir=None):
        self.repo_owner, self.repo_name = repo_name.rstrip('/').split('/')[-2:]
        self.repo_namespace = f"{self.repo_owner}/{self.repo_name}"
        self.repo_url = f'{ENDPOINT}/{self.repo_namespace}'
        self.best_checkpoint_name = None
        self.use_auth_token = True
        self.git_user = None
        self.git_email = None
        self.private = False

        self.repo = None

        self.temp_dir = None if local_dir else tempfile.TemporaryDirectory()
        self.local_dir = local_dir or self.temp_dir.name
        self.tb_dir = Path(self.local_dir) / 'runs'

    def on_init_end(self, trainer):
        self.repo = Repository(
            self.local_dir,
            clone_from=self.repo_url,
            use_auth_token=self.use_auth_token,
            git_user=self.git_user,
            git_email=self.git_email,
            revision=None,  # This should ALWAYS be latest?
            private=self.private,
        )
        self.tb_dir.mkdir(exist_ok=True, parents=True)

    def on_train_epoch_end(self, trainer, pl_module):
        with self.repo.commit("Add/Update Model"):
            trainer.save_checkpoint("lit_model.ckpt")
            for logfile in Path(trainer.log_dir).glob("events*"):
                copy2(logfile, self.tb_dir)
