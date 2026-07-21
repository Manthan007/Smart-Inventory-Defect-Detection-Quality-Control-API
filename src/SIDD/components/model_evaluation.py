import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt


class ModelEvaluation:
    def __init__(
        self,
        config,
        model,
        val_loader,
        criterion,
        dice_metric,
        iou_metric
    ):

        self.config = config
        self.model = model
        self.val_loader = val_loader
        self.criterion = criterion
        self.dice_metric = dice_metric
        self.iou_metric = iou_metric

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        checkpoint = torch.load(
            self.config.model_path,
            map_location=self.device
        )

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        print(f"\nLoaded checkpoint: {self.config.model_path}")

    def evaluate(self):

        print("\n========== Evaluation Started ==========\n")

        self.model.eval()

        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0

        num_batches = len(self.val_loader)

        print(f"Validation batches: {num_batches}")

        with torch.no_grad():

            num_classes = self.model.outc.conv.out_channels

            dice_per_class = torch.zeros(num_classes)
            iou_per_class = torch.zeros(num_classes)

            for batch_idx, (images, masks) in enumerate(self.val_loader):

                print(f"Processing batch {batch_idx + 1}/{num_batches}")

                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):

                    outputs = self.model(images)

                    print("Forward pass completed")

                    loss = self.criterion(outputs, masks)

                    print(f"Loss: {loss.item():.4f}")

                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()

                running_loss += loss.item()

                preds_flat = preds.flatten(start_dim=2)
                masks_flat = masks.flatten(start_dim=2)

                intersection = (preds_flat * masks_flat).sum(dim=2)

                pred_sum = preds_flat.sum(dim=2)
                mask_sum = masks_flat.sum(dim=2)

                union = pred_sum + mask_sum
                iou_union = union - intersection

                dice = (2 * intersection + 1e-6) / (union + 1e-6)
                iou = (intersection + 1e-6) / (iou_union + 1e-6)

                running_dice += dice.mean().item()
                running_iou += iou.mean().item()

                dice_per_class += dice.mean(dim=0).cpu()
                iou_per_class += iou.mean(dim=0).cpu()


            dice_per_class /= num_batches
            iou_per_class /= num_batches

            print("\nPer-Class Metrics")

            for cls in range(num_classes):

                print(
                    f"Class {cls+1} | "
                    f"Dice: {dice_per_class[cls]:.4f} | "
                    f"IoU: {iou_per_class[cls]:.4f}"
                )

        results = {

            "validation_loss": running_loss / num_batches,

            "dice_score": running_dice / num_batches,

            "iou_score": running_iou / num_batches,

            "dice_per_class": dice_per_class.tolist(),

            "iou_per_class": iou_per_class.tolist()

        }

        print("\n" + "=" * 50)
        print("Evaluation Results")
        print("=" * 50)
        print(f"Validation Loss : {results['validation_loss']:.4f}")
        print(f"Dice Score      : {results['dice_score']:.4f}")
        print(f"IoU Score       : {results['iou_score']:.4f}")
        print(f"Dice Per CLass    : {results['dice_per_class']}")
        print(f"IoU Per Class     : {results['iou_per_class']}")
        print("=" * 50)

        output_path = Path(self.config.root_dir) / "metrics.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"\nMetrics saved to: {output_path}")

        return results
    

    def visualize_predictions(
        self,
        num_images=3
    ):

        self.model.eval()

        images, masks = next(iter(self.val_loader))

        images = images.to(self.device)
        masks = masks.to(self.device)

        with torch.no_grad():

            outputs = self.model(images)

            preds = torch.sigmoid(outputs)

            preds = (preds > 0.5).float()

        num_images = min(num_images, images.size(0))

        num_classes = masks.shape[1]

        save_dir = Path(self.config.root_dir) / "predictions"
        save_dir.mkdir(parents=True, exist_ok=True)

        for idx in range(num_images):

            fig, axes = plt.subplots(
                num_classes,
                3,
                figsize=(12, 4 * num_classes)
            )

            image = images[idx].cpu().permute(1, 2, 0)

            for cls in range(num_classes):

                axes[cls, 0].imshow(image)
                axes[cls, 0].set_title("Input Image")
                axes[cls, 0].axis("off")

                axes[cls, 1].imshow(
                    masks[idx, cls].cpu(),
                    cmap="gray"
                )
                axes[cls, 1].set_title(
                    f"GT Defect {cls + 1}"
                )
                axes[cls, 1].axis("off")

                axes[cls, 2].imshow(
                    preds[idx, cls].cpu(),
                    cmap="gray"
                )
                axes[cls, 2].set_title(
                    f"Prediction Defect {cls + 1}"
                )
                axes[cls, 2].axis("off")

            plt.tight_layout()

            plt.savefig(
                save_dir / f"prediction_{idx}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.show()

            plt.close()


    def evaluate_thresholds(self):

        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        self.model.eval()

        threshold_results = {}

        with torch.no_grad():

            for threshold in thresholds:

                running_dice = 0.0

                for images, masks in self.val_loader:

                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = self.model(images)

                    preds = torch.sigmoid(outputs)
                    preds = (preds > threshold).float()

                    dice = cal_dice_score(
                        preds,
                        masks
                    )

                    running_dice += dice.item()

                avg_dice = running_dice / len(self.val_loader)

                threshold_results[str(threshold)] = avg_dice

                print(
                    f"Threshold {threshold:.1f} | "
                    f"Dice Score: {avg_dice:.4f}"
                )

        output_path = (
            Path(self.config.root_dir)
            / "threshold_results.json"
        )

        with open(output_path, "w") as f:
            json.dump(
                threshold_results,
                f,
                indent=4
            )

        print(f"\nThreshold results saved to: {output_path}")

        return threshold_results


def cal_dice_score(
    preds,
    targets,
    smooth=1e-6
):

    preds = preds.flatten(start_dim=2)
    targets = targets.flatten(start_dim=2)

    intersection = (preds * targets).sum(dim=2)

    union = (
        preds.sum(dim=2)
        + targets.sum(dim=2)
    )

    dice = (
        (2 * intersection + smooth)
        / (union + smooth)
    )

    return dice.mean()