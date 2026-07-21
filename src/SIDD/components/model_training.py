import torch
from torch import optim


class ModelTraining:
    def __init__(
        self,
        config,
        model,
        train_loader,
        val_loader,
        criterion,
    ):

        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs
        )

        self.scaler = torch.cuda.amp.GradScaler()

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(self.train_loader):

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):

                outputs = self.model(images)

                loss = self.criterion(outputs, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(self.train_loader)

        return epoch_loss

    def validate(self):

        self.model.eval()

        running_loss = 0.0

        with torch.no_grad():

            for images, masks in self.val_loader:

                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda"):

                    outputs = self.model(images)

                    loss = self.criterion(outputs, masks)

                running_loss += loss.item()

        epoch_loss = running_loss / len(self.val_loader)

        return epoch_loss

    def train(self):

        best_loss = float("inf")

        for epoch in range(self.config.epochs):

            train_loss = self.train_one_epoch()

            val_loss = self.validate()

            self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch [{epoch + 1}/{self.config.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            if val_loss < best_loss:

                best_loss = val_loss

                torch.save(
                    self.model.state_dict(),
                    self.config.model_path
                )

        print("\nTraining Completed!")