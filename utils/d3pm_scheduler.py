import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
import numpy as np
from typing import *

EPS_PROB = 1e-30
LOG_ZERO = -69

class COS_D3PMScheduler(nn.Module):
    def __init__(
            self,
            num_train_timesteps=200,
            prediction_type='x0',
            num_classes=2, 
        ):
        super().__init__()

        self.num_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.num_classes = num_classes+2   # +2 for empty edge and [mask] token

        #schedule
        at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_classes-1) 

        at = torch.tensor(at.astype("float64"))
        bt = torch.tensor(bt.astype("float64"))
        ct = torch.tensor(ct.astype("float64"))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype("float64"))
        btt = torch.tensor(btt.astype("float64"))
        ctt = torch.tensor(ctt.astype("float64"))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)
        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1e-5

        # Convert to float32 and register buffers.
        self.register_buffer("log_at", log_at.float())
        self.register_buffer("log_bt", log_bt.float())
        self.register_buffer("log_ct", log_ct.float())
        self.register_buffer("log_cumprod_at", log_cumprod_at.float())
        self.register_buffer("log_cumprod_bt", log_cumprod_bt.float())
        self.register_buffer("log_cumprod_ct", log_cumprod_ct.float())
        self.register_buffer("log_1_min_ct", log_1_min_ct.float())
        self.register_buffer("log_1_min_cumprod_ct", log_1_min_cumprod_ct.float())

        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))

    def _extract(self, a: Tensor, t: LongTensor, x_shape: Tuple[int, ...]):
        b, *_ = t.shape
        a=a.to(t.device)
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    def multinomial_kl(self, log_prob1: Tensor, log_prob2: Tensor):  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def log_sample_categorical(self, logits: Tensor):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + EPS_PROB) + EPS_PROB)

        sample = (gumbel_noise + logits).argmax(dim=1) #
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def sample_time(self, b: int, device: torch.device, method="uniform"):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # overwrite L0 (i.e., the decoder nll) term with L1
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt
        elif method == "uniform":
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def q_pred_one_timestep(self, log_x_t_1: Tensor, t: LongTensor):  # q(xt|xt_1)
        """
        log(Q_t * exp(log_x_t_1)), diffusion step: q(x_t | x_{t-1})
        """
        # log_x_t_1 (B, C, N)
        log_at = self._extract(self.log_at, t, log_x_t_1.shape)  # at
        log_bt = self._extract(self.log_bt, t, log_x_t_1.shape)  # bt
        log_ct = self._extract(self.log_ct, t, log_x_t_1.shape)  # ct
        log_1_min_ct = self._extract(self.log_1_min_ct, t, log_x_t_1.shape)  # 1-ct

        log_probs = torch.cat([
                log_add_exp(log_x_t_1[:, :-1, :] + log_at, log_bt),   # dropped a small term
                log_add_exp(log_x_t_1[:, -1:, :] + log_1_min_ct, log_ct),
            ], dim=1)

        return log_probs

    def q_pred(self, log_x_start: Tensor, t: LongTensor):  # q(xt|x0)
        """
        log(bar{Q}_t * exp(log_x_start)), diffuse the data to time t: q(x_t | x_0)
        """
        t = (t + (self.num_timesteps + 1)) % (self.num_timesteps + 1)
        log_cumprod_at = self._extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = self._extract(self.log_cumprod_bt, t, log_x_start.shape)  # bt~
        log_cumprod_ct = self._extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = self._extract(
            self.log_1_min_cumprod_ct, t, log_x_start.shape
        )  # 1-ct~

        log_probs = torch.cat([
                log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(
                    log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct
                ),  # simplified
            ], dim=1)

        return log_probs
    

    def q_posterior(self, log_x_start, log_x_t, t):
        """
        log of prosterior probability q(x_{t-1}|x_t,x_0')
        """
        B, C, N = log_x_start.shape
        log_one_vector = torch.zeros(B, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.full((B, 1, N), LOG_ZERO).type_as(log_x_t)
        
        # notice that log_x_t is onehot
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)

        log_qt = self.q_pred(log_x_t, t)  # q(xt|x0)
        # log_qt = torch.cat((log_qt[:,:-1,:], log_zero_vector), dim=1)
        log_qt = log_qt[:, :-1, :]
        log_cumprod_ct = self._extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        # ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat(
            (log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1
        )
        log_ct = self._extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        q = log_x_start[:, :-1, :] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = \
            self.q_pred(q, t - 1) + log_qt_one_timestep + q_log_sum_exp
        
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, LOG_ZERO, 0)


    def q_sample_one_step(self, log_x_t_1, t):
        """
        sample from q(x_t | x_{t-1})
        """
        log_EV_qxt = self.q_pred_one_timestep(log_x_t_1, t)
        log_sample = self.log_sample_categorical(log_EV_qxt)
        return log_sample

    def q_sample(self, log_x_start: Tensor, t: LongTensor):  # diffusion step, q(xt|x0) and sample xt
        """
        sample from q(x_t | x_0)
        """
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        # Gumbel sample
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)
        return log_sample

    @staticmethod
    def log_pred_from_denoise_out(denoise_out):
        """
        convert output of denoising network to log probability over classes and [mask]
        """
        out = denoise_out.permute((0, 2, 1))    # (B, N, C-1) -> (B, C-1, N)
        B, _, N = out.shape

        log_pred = F.log_softmax(out.double(), dim=1).float()
        log_pred = torch.clamp(log_pred, LOG_ZERO, 0)
        log_zero_vector = torch.full((B, 1, N), LOG_ZERO).type_as(log_pred)
        return torch.cat((log_pred, log_zero_vector), dim=1)

    def predict_start(self, denoise_fn, log_x_t: Tensor,  t: LongTensor):  # p(x0|xt)  
        """
        compute denoise_fn(data, t, condition, condition_cross) and convert output to log prob
        """
        x_t = log_onehot_to_index(log_x_t)
        out_x = denoise_fn(x_t, t)
        log_pred = self.log_pred_from_denoise_out(out_x)
        assert log_pred.shape == log_x_t.shape

        return log_pred

    
    def p_pred(self, denoise_fn, log_x_t: Tensor, t: LongTensor):  # if x0, first p(x0|xt), than sum( q(xt-1|xt,x0) * p(x0|xt) )
        """
        log denoising probability, denoising step: p(x_{t-1} | x_t)
        """
        if self.prediction_type == 'x0':
            # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
            log_x_recon = self.predict_start(denoise_fn, log_x_t, t)
            log_model_pred = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_x_t, t=t)
            return log_model_pred, log_x_recon
        elif self.prediction_type == 'x_prev':
            log_model_pred = self.predict_start(denoise_fn, log_x_t, t)
            return log_model_pred, None
        else:
            raise NotImplemented

    @torch.no_grad()
    def p_sample(self, denoise_fn, log_x_t, t):               
        """
        sample x_{t-1} from p(x_{t-1} | x_t)
        """
        model_log_prob, _ = self.p_pred(denoise_fn, log_x_t, t)
        log_sample = self.log_sample_categorical(model_log_prob)
        return log_sample
    
    '''
    loss
    '''
    def compute_kl_loss(self, log_x_start,log_x0_recon, log_x_t, t):
        """compute train loss of each variable"""
        log_q_prob = self.q_posterior(log_x_start, log_x_t, t)
        log_p_prob = self.q_posterior(log_x0_recon, log_x_t, t)
        kl = self.multinomial_kl(log_q_prob, log_p_prob)
        decoder_nll = -log_categorical(log_x_start, log_p_prob)

        t0_mask = (t == 0).unsqueeze(1).repeat(1, log_x_start.shape[-1])
        kl_loss = torch.where(t0_mask, decoder_nll, kl)
        return kl_loss

    def compute_aux_loss(self, log_x_start, log_x0_recon, t):
        """compute auxilary loss regulating predicted x0"""
        aux_loss = self.multinomial_kl(
            log_x_start[:, :-1, :], log_x0_recon[:, :-1, :]
        )

        t0_mask = (t == 0).unsqueeze(1).repeat(1, log_x_start.shape[-1])
        aux_loss = torch.where(t0_mask, torch.zeros_like(aux_loss), aux_loss)
        return aux_loss



#################################################################
## Helper functions

def log_1_min_a(a: Tensor):
    return torch.log(1. - a.exp() + EPS_PROB)

def log_add_exp(a: Tensor, b: Tensor):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def log_categorical(log_x_start: Tensor, log_prob: Tensor):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x: LongTensor, num_classes: int):
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"

    x_onehot: Tensor = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.shape)))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=EPS_PROB))
    return log_x

def log_onehot_to_index(log_x: Tensor):
    return log_x.argmax(dim=1)


def cosine_att_ctt(
    time_step,
    att_1=0.99999,
    att_T=0.000009,
    ctt_1=0.000009,
    ctt_T=0.99999,
    s=0.008,
):
    t = np.arange(time_step)

    # shared cosine prototype
    cos_raw = np.cos((t / (time_step - 1) + s) / (1 + s) * np.pi / 2) ** 2
    cos_raw = cos_raw / cos_raw[0]  # normalize to 1 → 0

    # att: decreasing
    att = att_T + (att_1 - att_T) * cos_raw

    # ctt: increasing (mirror)
    ctt = ctt_1 + (ctt_T - ctt_1) * (1.0 - cos_raw)

    return att, ctt

def alpha_schedule(time_step: int, N: int, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999): 

    att,ctt = cosine_att_ctt(time_step,att_1,att_T,ctt_1,ctt_T)
    att = np.concatenate([[1], att])
    at = att[1:] / att[:-1]

    ctt = np.concatenate([[0], ctt])
    one_minus_ctt = 1. - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1. - one_minus_ct

    bt = (1. - at - ct) / N

    att = np.concatenate([att[1:], [1]])
    ctt = np.concatenate([ctt[1:], [0]])
    btt = (1. - att - ctt) / N

    # 1. Check for non negative probability
    assert np.all(at >= 0), f"at contains negative values: {at[at < 0]}"
    assert np.all(ct >= 0), f"ct contains negative values: {ct[ct < 0]}"
    assert np.all(bt >= 0), f"bt contains negative values. Decrease att_T or ctt_T. Min bt: {np.min(bt)}"

    # 2. Check if the sum of probabilities is 1  (at + ct + N*bt = 1)
    sum_prob = at + ct + N * bt
    assert np.allclose(sum_prob, 1.0, atol=1e-6), "Probabilities do not sum to 1"

    # 3. Check if NaN exists
    assert not np.isnan(at).any(), "NaN detected in at"
    assert not np.isnan(bt).any(), "NaN detected in bt"
    assert not np.isnan(ct).any(), "NaN detected in ct"
    
    # 4. Check if the cumulative probability is valid
    assert np.all(att + ctt <= 1.00001), "Cumulative att + ctt exceeds 1"

    return at, bt, ct, att, btt, ctt
