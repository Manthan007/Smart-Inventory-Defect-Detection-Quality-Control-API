import torch
from torch import optim

# from SIDD.entity.config_entity import ModelTrainingConfig


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

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs
        )


    def train_one_epoch(self):
        import time

        self.model.train()

        print(f"\nTraining on device : {self.device}")
        print(f"Model device       : {next(self.model.parameters()).device}")

        running_loss = 0.0

        for batch_idx, (images, masks) in enumerate(self.train_loader):

            batch_start = time.time()

            images = images.to(self.device)
            masks = masks.to(self.device)

            data_end = time.time()

            self.optimizer.zero_grad()

            outputs = self.model(images)

            forward_end = time.time()

            loss = self.criterion(outputs, masks)

            loss.backward()

            self.optimizer.step()

            backward_end = time.time()

            running_loss += loss.item()

            print(
                f"Batch {batch_idx:02d} | "
                f"Data: {data_end - batch_start:.3f}s | "
                f"Forward: {forward_end - data_end:.3f}s | "
                f"Backward: {backward_end - forward_end:.3f}s | "
                f"Loss: {loss.item():.4f}"
            )

            # Debug only
            if batch_idx == 2:
                break

        epoch_loss = running_loss / (batch_idx + 1)

        return epoch_loss

    def validate(self):
        self.model.eval()

        running_loss = 0.0

        with torch.no_grad():

            for images, masks in self.val_loader:

                images = images.to(self.device)
                masks = masks.to(self.device)

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

            print(
                f"Epoch [{epoch+1}/{self.config.epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Val Loss: {val_loss:.4f}"
            )

            if val_loss < best_loss:

                best_loss = val_loss

                torch.save(
                    self.model.state_dict(),
                    self.config.model_path
                )

        print("Training Completed!")
        