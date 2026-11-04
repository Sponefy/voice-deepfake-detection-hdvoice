"""
Model i trainer Wav2Vec2 do detekcji deepfake'ów audio.
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score


# ============================================================================
# 1. MODEL
# ============================================================================

class Wav2Vec2DeepfakeDetector(nn.Module):
    """
    Model do detekcji deepfake'ów bazujący na Wav2Vec2.

    Args:
        model_name: Nazwa pretrenowanego modelu z Hugging Face
        freeze_encoder: Czy zamrozić encoder Wav2Vec2
        pooling_mode: 'mean', 'max', lub 'attention'
        dropout: Współczynnik dropout w klasyfikatorze
    """

    def __init__(
            self,
            model_name="facebook/wav2vec2-base",
            freeze_encoder=True,
            pooling_mode='mean',
            dropout=0.3
    ):
        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        hidden_size = self.wav2vec2.config.hidden_size  # 768 dla base, 1024 dla large

        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False
            print("✓ Wav2Vec2 encoder zamrożony")

        self.pooling_mode = pooling_mode

        if pooling_mode == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # bonafide (1) vs spoof (0)
        )

    def pool_features(self, hidden_states, feat_mask=None):
        """
        Pooling po wymiarze czasowym, z opcjonalną maską hidden states.

        Args:
            hidden_states: [batch, time, hidden_size]
            feat_mask:     [batch, time] bool/0-1 — maska na poziomie hidden
                           states (już zdownsamplowana z poziomu próbek).
                           Jeśli None — pooling bez maskowania.
        Returns:
            pooled: [batch, hidden_size]
        """
        if feat_mask is None:
            if self.pooling_mode == 'mean':
                return torch.mean(hidden_states, dim=1)
            elif self.pooling_mode == 'max':
                return torch.max(hidden_states, dim=1)[0]
            elif self.pooling_mode == 'attention':
                w = torch.softmax(self.attention(hidden_states), dim=1)
                return torch.sum(hidden_states * w, dim=1)

        # Z maską: padding nie wchodzi do statystyk
        mask_f = feat_mask.unsqueeze(-1).to(hidden_states.dtype)  # [B, T, 1]
        if self.pooling_mode == 'mean':
            # średnia tylko po realnych pozycjach
            denom = mask_f.sum(dim=1).clamp(min=1)
            return (hidden_states * mask_f).sum(dim=1) / denom
        elif self.pooling_mode == 'max':
            hs = hidden_states.masked_fill(mask_f == 0, float('-inf'))
            return hs.max(dim=1)[0]
        elif self.pooling_mode == 'attention':
            logits = self.attention(hidden_states)                # [B, T, 1]
            logits = logits.masked_fill(mask_f == 0, float('-inf'))
            w = torch.softmax(logits, dim=1)
            return torch.sum(hidden_states * w, dim=1)

    def _build_feat_mask(self, attention_mask, hidden_states):
        """
        Z attention_mask na poziomie próbek (16 kHz) buduje maskę na poziomie
        hidden states (~50 Hz dla Wav2Vec2). Wykorzystuje wewnętrzną funkcję
        Wav2Vec2 obliczającą długość wyjścia ekstraktora cech.

        Zakłada padding z prawej strony (taka jest konwencja w datasecie).
        """
        input_lengths = attention_mask.sum(dim=-1).to(torch.long)
        output_lengths = self.wav2vec2._get_feat_extract_output_lengths(input_lengths).to(torch.long)
        bsz, t_hidden = hidden_states.shape[:2]
        rng = torch.arange(t_hidden, device=hidden_states.device).unsqueeze(0)  # [1, T]
        feat_mask = rng < output_lengths.unsqueeze(1)                           # [B, T]
        return feat_mask

    def forward(self, input_values, attention_mask=None):
        """
        Forward pass.

        Args:
            input_values:   [batch, time_samples] — surowe audio przy 16 kHz
            attention_mask: [batch, time_samples] — 1=realne, 0=padding.
                            Jeśli podana, Wav2Vec2 ignoruje padding wewnętrznie
                            i pooling też dzieli tylko po realnych pozycjach.
        Returns:
            logits: [batch, 2]
        """
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        feat_mask = None
        if attention_mask is not None:
            feat_mask = self._build_feat_mask(attention_mask, hidden_states)

        pooled = self.pool_features(hidden_states, feat_mask)
        return self.classifier(pooled)


# ============================================================================
# 2. METRYKI
# ============================================================================

def compute_metrics(predictions, labels):
    """Oblicza metryki klasyfikacji."""
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)

    probs = torch.softmax(torch.FloatTensor(predictions), dim=1).numpy()
    # AUC wymaga obu klas; nowsze sklearn zwracają wtedy NaN zamiast wyjątku
    if len(np.unique(labels)) < 2:
        auc = 0.0
    else:
        auc = roc_auc_score(labels, probs[:, 1])

    bonafide_mask = labels == 1
    spoof_mask = labels == 0
    bonafide_acc = accuracy_score(labels[bonafide_mask], preds[bonafide_mask]) if bonafide_mask.any() else 0.0
    spoof_acc = accuracy_score(labels[spoof_mask], preds[spoof_mask]) if spoof_mask.any() else 0.0

    return {
        'accuracy': accuracy,
        'auc': auc,
        'bonafide_accuracy': bonafide_acc,
        'spoof_accuracy': spoof_acc,
    }


# ============================================================================
# 3. TRAINER
# ============================================================================

class DeepfakeDetectorTrainer:
    """Klasa do trenowania modelu."""

    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            device,
            learning_rate=1e-4,
            use_wandb=False,
            gradient_accumulation_steps=1,
            label_smoothing=0.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        # label_smoothing > 0 przeciwdziała overconfidence: nie pozwala logitom
        # puchnąć do nieskończoności, więc softmax nie nasyca się przy 0/1
        # (lepsza kalibracja, mniej ekstremalnych P na OOD). Nie zmienia AUC.
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=2,
        )
        # GradScaler tylko dla CUDA; MPS używa autocast bez scalera
        self.scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
        self.use_amp = device.type in ('cuda', 'mps')

    def train_epoch(self):
        """Jeden epoch treningu z mixed precision i gradient accumulation."""
        self.model.train()

        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc="Training", file=sys.stdout)
        for i, batch in enumerate(pbar):
            input_values = batch['input_values'].to(self.device)
            labels = batch['labels'].squeeze(1).to(self.device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(input_values, attention_mask=attention_mask)
                loss = self.criterion(logits, labels) / self.gradient_accumulation_steps

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            all_preds.append(logits.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.train_loader)
        return metrics

    def validate(self, loader=None, desc="Validation"):
        """Walidacja modelu. Domyślnie używa val_loader; podaj loader by ocenić inny zbiór."""
        loader = loader if loader is not None else self.val_loader
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, file=sys.stdout):
                input_values = batch['input_values'].to(self.device)
                labels = batch['labels'].squeeze(1).to(self.device)
                attention_mask = batch.get('attention_mask')
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(input_values, attention_mask=attention_mask)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                all_preds.append(logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(loader)
        return metrics

    def train(self, num_epochs, save_path='best_model.pt', select_loader=None, patience=None):
        """Główna pętla treningowa.

        select_loader: jeśli podany, best-model i scheduler idą po AUC TEGO loadera
            (np. val_ood / OOD), a nie po in-domain val. In-domain val nasyca się
            (AUC→1.0 już po ~2 epokach przy pełnym fine-tuningu) i nie koreluje
            z generalizacją OOD — wtedy selekcja "best po VAL AUC" jest ślepa.
            None → selekcja po in-domain val (stare zachowanie).
        patience: early stopping — przerwij, jeśli metryka selekcji nie poprawia
            się przez `patience` epok (None = bez early stoppingu). Pełny
            fine-tuning często osiąga szczyt OOD wcześnie i potem degraduje.

        AUC niezależne od progu i wiarygodniejsze niż accuracy przy
        niezbalansowanych klasach (sampler 50/50 działa tylko na treningu).
        """
        sel_name = "VAL_OOD" if select_loader is not None else "VAL"
        best_auc = 0.0
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'=' * 50}")

            train_metrics = self.train_epoch()
            print(f"\nTrain — Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"AUC: {train_metrics['auc']:.4f}")

            # in-domain val zawsze (monitoring luki in-domain↔OOD)
            val_metrics = self.validate(desc="Val (in-domain)")
            print(f"Val(in-dom) — Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, AUC: {val_metrics['auc']:.4f}  "
                  f"(Bona {val_metrics['bonafide_accuracy']:.3f} / Spoof {val_metrics['spoof_accuracy']:.3f})")

            # metryka selekcji: val_ood jeśli podany, inaczej in-domain val
            if select_loader is not None:
                sel_metrics = self.validate(select_loader, desc="Val OOD (selekcja)")
                print(f"Val(OOD)    — AUC: {sel_metrics['auc']:.4f}  <-- selekcja modelu po tym")
            else:
                sel_metrics = val_metrics
            sel_auc = sel_metrics['auc']

            self.scheduler.step(sel_auc)

            if sel_auc > best_auc:
                best_auc = sel_auc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"✓ Zapisano najlepszy model ({sel_name} AUC={best_auc:.4f})")
            else:
                epochs_no_improve += 1
                lim = patience if patience is not None else "∞"
                print(f"  (brak poprawy {sel_name} AUC: {epochs_no_improve}/{lim})")

            if self.use_wandb:
                import wandb
                wandb.log({
                    'train_loss': train_metrics['loss'],
                    'train_auc': train_metrics['auc'],
                    'val_auc': val_metrics['auc'],
                    'sel_auc': sel_auc,
                    'epoch': epoch,
                })

            if patience is not None and epochs_no_improve >= patience:
                print(f"\nEarly stopping: {sel_name} AUC bez poprawy przez {patience} epok.")
                break

        print(f"\n{'=' * 50}")
        print(f"Trening zakończony! Najlepsza {sel_name} AUC: {best_auc:.4f}")
        print(f"{'=' * 50}")
