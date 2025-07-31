"""Acoustic echo cancellation, in numpy, because Windows has no wheels.

MEASURED PROBLEM (2026-08-01): with the speakers at normal volume, the mic hears
Sonara clearly enough that Whisper transcribes it back - 19 of 20 trials. With
barge-in enabled, Sonara would interrupt itself on nearly every reply.

webrtc-audio-processing, webrtc-noise-gain and speexdsp all fail to build on this
machine (no wheels, no C toolchain). So this is a from-scratch NLMS adaptive
filter, which is the classical AEC core anyway.

THE KEY ADVANTAGE over a generic library: Sonara GENERATED the audio it is playing,
so the reference signal is exact and free. A generic AEC has to capture the loopback;
we already have the array.

WHAT THIS IS FOR. Barge-in does not need clean transcribable audio during playback -
it needs a DECISION: "is this the user, or is it me?" So the useful output is the
residual after cancellation, and the useful metric is ERLE (echo return loss
enhancement, in dB): how much quieter the echo got. High ERLE plus a loud residual
means a real interruption; high ERLE and a quiet residual means it was just us.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CancelResult:
    residual: np.ndarray     # mic with the echo removed
    erle_db: float           # how much echo energy was removed
    delay_samples: int       # measured speaker -> mic delay
    residual_rms: float
    mic_rms: float


def estimate_delay(mic: np.ndarray, ref: np.ndarray, max_lag: int = 16_000) -> int:
    """Find the speaker-to-mic delay by cross-correlation.

    NLMS only converges if the echo lands inside the filter's tap window, and on
    Windows the buffer path adds tens of milliseconds on top of the acoustic flight
    time. Measuring the lag first means the taps model the echo, not the silence
    before it.
    """
    n = min(len(mic), len(ref))
    if n < 1024:
        return 0
    m = mic[:n] - mic[:n].mean()
    r = ref[:n] - ref[:n].mean()
    # Correlate envelopes: robust to the speaker's nonlinearity and phase inversion.
    me = np.abs(m)
    re = np.abs(r)
    # FFT cross-correlation over the full lag range. The first version used a
    # 4800-sample clamp and saturated at exactly 4800 on every trial - the true
    # delay here is 5,349 samples (334 ms) of WASAPI shared-mode buffer latency,
    # so the filter was modelling silence and NLMS diverged.
    size = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(me, size) * np.conj(np.fft.rfft(re, size)), size)
    return int(np.argmax(cc[: min(max_lag, size // 2)]))


def nlms_cancel(mic: np.ndarray, ref: np.ndarray, *, taps: int = 512,
                mu: float = 0.35, eps: float = 1e-6,
                delay: int | None = None) -> CancelResult:
    """Normalised least-mean-squares echo cancellation.

    w is an adaptive estimate of the room's impulse response - speaker, air, mic
    case, all of it. Each sample it predicts the echo from the reference history and
    subtracts it, then nudges itself toward whatever was left over.

    mu trades convergence speed against stability. 0.35 converges within roughly a
    second of speech, which matters because Sonara's replies are short: a filter that
    needs ten seconds to adapt has nothing to cancel by the time it is ready.
    """
    mic = np.asarray(mic, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()

    d = delay if delay is not None else estimate_delay(mic, ref)
    # Align: the echo of ref[i] shows up at mic[i + d].
    ref_a = np.concatenate([np.zeros(d), ref])[: len(mic)]
    if len(ref_a) < len(mic):
        ref_a = np.concatenate([ref_a, np.zeros(len(mic) - len(ref_a))])

    w = np.zeros(taps)
    residual = np.empty(len(mic))
    x = np.zeros(taps)

    # Regularisation MUST scale with reference power. With a fixed eps=1e-6 this
    # filter diverged in every configuration tried (ERLE -17 to -31 dB): between
    # words x.x collapses toward zero, so mu*e/norm explodes and the taps blow up.
    # The fix is a delta proportional to the signal it is normalising, plus a gate
    # that simply does not adapt when there is no reference energy to learn from -
    # you cannot estimate an echo path from silence.
    ref_power = float(np.mean(ref_a ** 2)) + 1e-12
    delta = max(1e-2 * taps * ref_power, eps)
    activity = delta  # adapt only when the reference is genuinely playing

    for i in range(len(mic)):
        x[1:] = x[:-1]
        x[0] = ref_a[i]
        y = float(w @ x)               # predicted echo
        e = mic[i] - y                 # what is left: the user, plus error
        residual[i] = e
        xx = float(x @ x)
        if xx > activity:
            w += (mu * e / (xx + delta)) * x   # adapt toward the residual

    mic_rms = float(np.sqrt(np.mean(mic ** 2))) or 1e-9
    res_rms = float(np.sqrt(np.mean(residual ** 2))) or 1e-9
    erle = 20.0 * np.log10(mic_rms / res_rms)
    return CancelResult(residual.astype(np.float32), erle, d, res_rms, mic_rms)


def residual_suppress(residual: np.ndarray, echo_est: np.ndarray, *,
                      n_fft: int = 512, hop: int = 128,
                      over: float = 2.0, floor: float = 0.05) -> np.ndarray:
    """Spectral residual echo suppression — the stage AEC3 has and plain NLMS does not.

    A linear filter can only remove the part of the echo that is a linear function of
    the reference. Laptop speakers clip and resonate, so a real chunk of the echo is
    nonlinear and survives cancellation. That leftover is why 7 dB of NLMS still left
    Whisper able to read us back.

    This is a Wiener-style post-filter: per time-frequency bin, compare what is left
    against what the filter THINKS the echo was, and attenuate the bins the echo still
    dominates. Frequencies carrying only the user's voice pass through; frequencies
    that are mostly our own leftover echo get pushed toward the floor.

    `over` is oversubtraction (higher = more aggressive), `floor` is the residual gain
    floor that stops it sounding gated and destroying a real interruption.
    """
    x = np.asarray(residual, dtype=np.float32).ravel()
    e = np.asarray(echo_est, dtype=np.float32).ravel()
    n = min(len(x), len(e))
    x, e = x[:n], e[:n]
    if n < n_fft:
        return x

    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (n - n_fft) // hop
    out = np.zeros(n, dtype=np.float32)
    wsum = np.zeros(n, dtype=np.float32)

    for i in range(frames):
        s = i * hop
        seg_x = x[s:s + n_fft] * win
        seg_e = e[s:s + n_fft] * win
        X = np.fft.rfft(seg_x)
        E = np.fft.rfft(seg_e)
        mx = np.abs(X)
        me = np.abs(E)
        # Wiener gain: keep the bin in proportion to how much of it is NOT echo.
        gain = np.maximum(floor, 1.0 - over * me / (mx + 1e-9))
        seg = np.fft.irfft(X * gain, n_fft).astype(np.float32)
        out[s:s + n_fft] += seg * win
        wsum[s:s + n_fft] += win ** 2

    # Normalise only where the window sum is meaningful. Dividing by a near-zero
    # wsum at the edges amplifies them enormously - that alone turned 6.3 dB of
    # cancellation into -0.1 dB, i.e. the "suppressor" was making things louder.
    thresh = 0.1 * float(wsum.max() or 1.0)
    nz = wsum > thresh
    out[nz] /= wsum[nz]
    out[~nz] = 0.0
    return out


def cancel_full(mic: np.ndarray, ref: np.ndarray, *, taps: int = 4096, mu: float = 1.0,
                over: float = 2.0, floor: float = 0.05) -> CancelResult:
    """Two-stage cancellation: linear NLMS, then spectral residual suppression.

    This is AEC3's architecture in miniature — the linear filter removes what it can
    model, the post-filter removes what it cannot.
    """
    lin = nlms_cancel(mic, ref, taps=taps, mu=mu)
    mic_a = np.asarray(mic, dtype=np.float32).ravel()
    echo_est = mic_a[: len(lin.residual)] - lin.residual        # what NLMS removed
    out = residual_suppress(lin.residual, echo_est, over=over, floor=floor)

    mic_rms = float(np.sqrt(np.mean(mic_a ** 2))) or 1e-9
    res_rms = float(np.sqrt(np.mean(out ** 2))) or 1e-9
    return CancelResult(out, 20.0 * np.log10(mic_rms / res_rms), lin.delay_samples,
                        res_rms, mic_rms)


class EchoGate:
    """Runtime decision: is this incoming audio the user, or Sonara hearing itself?

    Barge-in does not need a clean signal, only a correct answer to that question.
    A residual well above the noise floor after cancellation is a real interruption;
    anything at or below it is our own voice coming back.
    """

    def __init__(self, *, noise_floor: float = 0.02, margin_db: float = 6.0) -> None:
        self.noise_floor = noise_floor
        self.margin_db = margin_db

    def is_user_speaking(self, mic: np.ndarray, playing_ref: np.ndarray | None) -> tuple[bool, float]:
        """Returns (user_is_speaking, erle_db). erle is 0.0 when nothing is playing."""
        mic = np.asarray(mic, dtype=np.float32).ravel()
        if playing_ref is None or len(playing_ref) == 0:
            return float(np.sqrt(np.mean(mic ** 2))) > self.noise_floor, 0.0

        out = nlms_cancel(mic, playing_ref)
        threshold = self.noise_floor * (10 ** (self.margin_db / 20.0))
        return out.residual_rms > threshold, out.erle_db
