import torch
from trl import RewardTrainer


class CustomRewardTrainer(RewardTrainer):
    def compute_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_chosen = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
            return_dict=True,
        )["logits"]
        logits_rejected = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
            return_dict=True,
        )["logits"]

        probs_chosen = torch.softmax(logits_chosen, dim=-1)
        probs_rejected = torch.softmax(logits_rejected, dim=-1)

        batch_size, num_classes = probs_chosen.shape
        mask = torch.tril(
            torch.ones(num_classes, num_classes, device=probs_chosen.device),
            diagonal=-1,
        )
        prod = probs_chosen.unsqueeze(2) * probs_rejected.unsqueeze(
            1
        )  # prod[., i, j] = chosen[., i] * rejected[., j]
        total_probability = (prod * mask).sum(dim=(1, 2))

        loss = -torch.log(total_probability.clamp(min=1e-8)).mean()

        if self.args.center_rewards_coefficient is not None:
            pass

        if return_outputs:
            return loss, {
                "probs_chosen": probs_chosen,
                "probs_rejected": probs_rejected,
            }
        return loss

    def prediction_step(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()

        probs_chosen = logits_dict["probs_chosen"]
        probs_rejected = logits_dict["probs_rejected"]
        ratings = torch.arange(
            1,
            probs_chosen.size(-1) + 1,
            device=probs_chosen.device,
            dtype=probs_chosen.dtype,
        )  # [1, 2, ... num_labels]
        expected_chosen = (probs_chosen * ratings).sum(dim=-1)
        expected_rejected = (probs_rejected * ratings).sum(dim=-1)
        logits = torch.stack([expected_chosen, expected_rejected], dim=-1)

        labels = torch.ones(
            expected_chosen.size(0), device=expected_chosen.device, dtype=torch.int32
        )

        return loss, logits, labels
