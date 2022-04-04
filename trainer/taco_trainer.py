import time

import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Dict, Any

from models.aligner import Aligner
from models.tacotron import Tacotron
from trainer.common import Averager, TTSSession, to_device, np_now
from utils.checkpoints import save_checkpoint
from utils.dataset import get_tts_datasets
from utils.decorators import ignore_exception
from utils.display import stream, simple_table, plot_mel, plot_attention
from utils.dsp import DSP
from utils.files import parse_schedule
from utils.metrics import attention_score
from utils.paths import Paths


class ForwardSumLoss(torch.nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss()
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = torch.nn.functional.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0).to(attn_logprob.device)
            ql = min(int(query_lens[bid]), int(attn_logprob_padded.size(1)))
            kl = min(int(key_lens[bid]), int(attn_logprob_padded.size(2)))
            curr_logprob = attn_logprob_padded[bid][:ql, :key_lens[bid] + 1]
            curr_logprob = curr_logprob.unsqueeze(1)
            curr_logprob = curr_logprob.log_softmax(dim=-1)
            print(curr_logprob.size(), ql, kl)
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=torch.full((1, 1), ql),
                target_lengths=torch.full((1, 1), kl),
            )
            total_loss = total_loss + loss

        total_loss = total_loss / attn_logprob.shape[0]
        return total_loss


class TacoTrainer:

    def __init__(self,
                 paths: Paths,
                 dsp: DSP,
                 config: Dict[str, Any]) -> None:
        self.paths = paths
        self.dsp = dsp
        self.config = config
        self.train_cfg = config['tacotron']['training']
        self.writer = SummaryWriter(log_dir=paths.taco_log, comment='v1')
        self.loss_fn = ForwardSumLoss()

    def train(self,
              model: Aligner,
              optimizer: Optimizer) -> None:
        tts_schedule = self.train_cfg['schedule']
        tts_schedule = parse_schedule(tts_schedule)
        for i, session_params in enumerate(tts_schedule, 1):
            r, lr, max_step, bs = session_params
            if model.get_step() < max_step:
                train_set, val_set = get_tts_datasets(
                    path=self.paths.data, batch_size=bs, r=r, model_type='tacotron',
                    max_mel_len=self.train_cfg['max_mel_len'], filter_attention=False
                )
                session = TTSSession(
                    index=i, r=r, lr=lr, max_step=max_step,
                    bs=bs, train_set=train_set, val_set=val_set)
                self.train_session(model, optimizer, session=session)

    def train_session(self, model: Tacotron,
                      optimizer: Optimizer,
                      session: TTSSession) -> None:
        current_step = model.get_step()
        training_steps = session.max_step - current_step
        total_iters = len(session.train_set)
        epochs = training_steps // total_iters + 1
        model.r = session.r
        simple_table([(f'Steps with r={session.r}', str(training_steps // 1000) + 'k Steps'),
                      ('Batch Size', session.bs),
                      ('Learning Rate', session.lr),
                      ('Outputs/Step (r)', model.r)])
        for g in optimizer.param_groups:
            g['lr'] = session.lr

        loss_avg = Averager()
        duration_avg = Averager()
        device = next(model.parameters()).device  # use same device as model parameters
        for e in range(1, epochs + 1):
            for i, batch in enumerate(session.train_set, 1):
                batch = to_device(batch, device=device)
                start = time.time()
                model.train()
                attn = model(batch['x'].detach(), batch['mel'].detach())

                loss = self.loss_fn(attn, in_lens=batch['x_len'], out_lens=batch['mel_len'])

                optimizer.zero_grad()
                if not torch.isnan(loss) or torch.isinf(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   self.train_cfg['clip_grad_norm'])
                optimizer.step()

                step = model.get_step()
                k = step // 1000

                duration_avg.add(time.time() - start)
                speed = 1. / duration_avg.get()
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {loss.item():#.4} ' \
                      f'| {speed:#.2} steps/s | Step: {k}k | '

                if step % self.train_cfg['checkpoint_every'] == 0:
                    save_checkpoint(model=model, optim=optimizer, config=self.config,
                                    path=self.paths.taco_checkpoints / f'taco_step{k}k.pt')

                if step % self.train_cfg['plot_every'] == 0:
                    self.generate_plots(model, session)

                _, att_score = attention_score(attn.softmax(-1), batch['mel_len'])
                att_score = torch.mean(att_score)
                self.writer.add_scalar('Attention_Score/train', att_score, model.get_step())
                self.writer.add_scalar('Loss/train', loss, model.get_step())
                self.writer.add_scalar('Params/reduction_factor', session.r, model.get_step())
                self.writer.add_scalar('Params/batch_size', session.bs, model.get_step())
                self.writer.add_scalar('Params/learning_rate', session.lr, model.get_step())

                stream(msg)

            save_checkpoint(model=model, optim=optimizer, config=self.config,
                            path=self.paths.taco_checkpoints / 'latest_model.pt')

            loss_avg.reset()
            duration_avg.reset()
            print(' ')

    @ignore_exception
    def generate_plots(self, model: Tacotron, session: TTSSession) -> None:
        model.eval()
        device = next(model.parameters()).device
        batch = session.val_sample
        batch = to_device(batch, device=device)
        att = model(batch['x'], batch['mel']).softmax(-1)
        att = np_now(att)[0]

        att_fig = plot_attention(att)

        self.writer.add_figure('Ground_Truth_Aligned/attention', att_fig, model.step)