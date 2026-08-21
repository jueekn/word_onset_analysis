# Visualization-only utility. Matplotlib's pyplot stubs annotate plot side-
# effect functions with **kwargs: Unknown, so every plt.* call here trips
# reportUnknownMemberType / reportUnknownArgumentType with no real signal.
# Downgrade those two rules to "warning" at the file level — non-pipeline
# helper, no statistics logic. The other strict-mode rules stay on.
# pyright: reportUnknownMemberType=warning, reportUnknownArgumentType=warning, reportUnknownVariableType=warning, reportCallIssue=warning
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import scipy as sp  # pyright: ignore[reportMissingTypeStubs]
import matplotlib.pyplot as plt

from misc import duration_to_samples

NDArrayAny = npt.NDArray[Any]


class Wavelet:
    """Log-spaced Morlet wavelet bank for visualizing per-frequency power response.

    Used by the convergence/buffer audits to confirm that a chosen wavelet
    configuration (fmin, fmax, fnum, morlet_reps) has the intended spectral
    coverage and does not leak power into notch-filtered bands.

    Note: this is a *visualization* helper. FC compute itself goes through
    mne_connectivity.spectral_connectivity_epochs, not this class.
    """

    def __init__(
        self,
        fmin: float,
        fmax: float,
        fnum: int,
        tmin: float = -4,
        tmax: float = 4,
        sampling_rate: float = 1000,
        morlet_reps: int = 5,
        amplitude: float | None = None,
    ) -> None:
        """Build a Morlet bank.

        Args:
            fmin, fmax: log-spaced frequency endpoints in Hz.
            fnum: number of log-spaced frequencies.
            tmin, tmax: time-domain wavelet support in seconds.
            sampling_rate: samples / second.
            morlet_reps: Morlet wave repetitions (controls time/frequency
                trade-off; larger = narrower in frequency).
            amplitude: optional uniform amplitude multiplier; None keeps the
                envelope's natural (unity-energy-ish) scale.
        """
        self.tmin = tmin
        self.tmax = tmax
        self.sampling_rate = sampling_rate

        self.fmin = fmin
        self.fmax = fmax
        self.fnum = fnum
        self.morlet_reps = morlet_reps
        self.amplitude = amplitude
        self.freqs: NDArrayAny = np.logspace(
            np.log10(self.fmin), np.log10(self.fmax), self.fnum)
        self.tvals: NDArrayAny = np.linspace(
            self.tmin, self.tmax,
            duration_to_samples(self.tmax - self.tmin, self.sampling_rate))
        self.tlen: int = len(self.tvals)

    def get_morlet_width(self, f: float) -> float:
        """Gaussian-envelope sigma in seconds for a Morlet centered at f Hz."""
        return self.morlet_reps / (2 * np.pi * f)

    def get_morlet_envelope(self, t: NDArrayAny, sig: float) -> NDArrayAny:
        """Real-valued Gaussian envelope, normalized to A = 1/sqrt(sig*sqrt(pi)).

        Reference: Ryan Colyer's implementation
        (https://cp.copernicus.org/preprints/cp-2019-105/cp-2019-105-supplement.pdf).
        Optionally scaled by self.amplitude when set.
        """
        amp = 1 / np.sqrt(sig * np.sqrt(np.pi))
        if self.amplitude is not None:
            amp = amp * self.amplitude
        return cast(NDArrayAny, amp * np.exp(-t**2 / (2 * sig**2)))

    def Morlet(self, t: NDArrayAny, morlet_reps: int, f: float) -> NDArrayAny:
        """Complex Morlet at center frequency f (Hz). morlet_reps is unused
        here; the class-level self.morlet_reps drives sigma via
        get_morlet_width.

        TODO(types): the morlet_reps parameter is unused (sigma is computed
        from self.morlet_reps inside get_morlet_width). Likely a dead arg;
        flagging rather than silently removing it.
        """
        sig = self.get_morlet_width(f)
        return self.get_morlet_envelope(t, sig) * np.exp(1j * 2 * np.pi * f * t)

    def PlotF(self, f: float, plot_wavelet: bool = False) -> None:
        """Plot the Morlet's frequency response (or time-domain shape) at f Hz.

        plot_wavelet=False (default): semilog-x of |FT(Morlet)|² vs frequency
        on the active matplotlib axis. plot_wavelet=True: time-domain real /
        imaginary / envelope traces with sigma-multiple guide-lines.
        """
        mvals = np.array([self.Morlet(t, self.morlet_reps, f) for t in self.tvals])

        ftlen = (self.tlen+1)//2
        ftlen = self.tlen
        fvals = sp.fft.fftfreq(self.tlen, self.tvals[1]-self.tvals[0])[0:ftlen]
        ft_morlet = sp.fft.fft(mvals)[0:ftlen]

        # To see the wavelets generated.
        if plot_wavelet:
            sigma = self.get_morlet_width(f)
            envelope = self.get_morlet_envelope(self.tvals, sigma)
            max_morlet = envelope.max()
            envelope /= max_morlet
            mvals /= max_morlet
            plt.plot(self.tvals, mvals.real, label='Real')
            plt.plot(self.tvals, mvals.imag, label='Imaginary')
            plt.plot(self.tvals,
                     envelope,
                     label='Envelope',
                     alpha=0.5)
            plt.plot(self.tvals,
                     envelope ** 2,
                     label='Envelope-Squared',
                     alpha=0.5)
            n_sigma = 3
            plt.vlines(x=[i * sigma for i in range(-n_sigma, n_sigma + 1) if i != 0],
                       ymin=0, ymax=mvals.real.max(), colors='c', alpha=0.3, label='Sigma multiples from t = 0')
            plt.legend(loc=(1.02, 0))
            plt.title(f'Max-Normalized Wavelet: f = {f} Hz', fontsize=25)
            plt.xlabel('Time (s)')
            plt.ylabel('Wavelet')
        else:
            start_f = np.argwhere(fvals >= 1).ravel()[0]
            end_f = np.argwhere(fvals > 300).ravel()[0]
            ft_mor_rel_power = np.abs(ft_morlet)**2
            ft_mor_rel_power /= np.max(ft_mor_rel_power)
            plt.semilogx(fvals[start_f:end_f], ft_mor_rel_power[start_f:end_f],
                label=f'{f:.2f}Hz')


    def PlotButterworth(self, bw_min: float, bw_max: float) -> None:
        """Plot the 4th-order band-stop Butterworth response on the active axis."""
        yvals = np.zeros(self.tlen)
        yvals[self.tlen//2] = 1

        nyq = self.sampling_rate / 2
        b, a = sp.signal.butter(4, [bw_min/nyq, bw_max/nyq], 'stop')
        yvals = sp.signal.filtfilt(b, a, yvals, axis=0)

        ftlen = (self.tlen+1)//2
        ftlen = self.tlen
        fvals = sp.fft.fftfreq(self.tlen, self.tvals[1]-self.tvals[0])[0:ftlen]
        ft_bw = sp.fft.fft(yvals)[0:ftlen]

        start_f = np.argwhere(fvals >= 1).ravel()[0]
        end_f = np.argwhere(fvals > 300).ravel()[0]
        ft_bw_rel_power = np.abs(ft_bw)**2
        ft_bw_rel_power /= np.max(ft_bw_rel_power)
        plt.semilogx(fvals[start_f:end_f], ft_bw_rel_power[start_f:end_f],
            color='darkgrey', label=f'BW {bw_min}-{bw_max}')


    def PlotValidate(self, f: float) -> None:
        '''Generates (slowly) a comparable plot to PlotF but using PTSA and
           measuring the power it produces for sinusoids of each frequency.
           This validates that the analytical approach in PlotF correctly
           shows the power extracted by a Morlet Transform done by PTSA.'''
        from ptsa.data.timeseries import TimeSeries
        from ptsa.data.filters import morlet

        cvals = []
        xfreqs = np.logspace(np.log10(f/10), np.log10(f*10), 200)
        for fx in xfreqs:
            cvals.append(np.cos(2*np.pi*fx*self.tvals))

        cos_timeseries = TimeSeries(cvals, {'samplerate':self.sampling_rate},
                ['testfreq', 'time'])
        wf = morlet.MorletWaveletFilter(timeseries=cos_timeseries,
                freqs=[f], width=self.morlet_reps, output=['power'], complete=True)
        power = wf.filter()

        pow_plot = np.mean(power[0, :, self.tlen//12:-self.tlen//12],
                axis=1)
        pow_plot /= np.max(pow_plot)
        plt.semilogx(xfreqs, pow_plot, label=f'PTSA')


    def MakePlot(self, show: bool = False) -> None:
        """Build the full frequency-response figure: every wavelet in the bank
        plus the three Butterworth notch responses (58–62, 118–122, 178–182 Hz)
        and 60/120/180 Hz reference verticals. `show` is currently ignored
        (always calls plt.show); kept for API stability.

        TODO(types): The `show` parameter is unused — plt.show() is called
        unconditionally at the end. Flagging rather than silently removing.
        """
        self.fig = plt.figure()
        plt.rcParams.update({'font.size': 12})

        for f in self.freqs:
            self.PlotF(f)

        self.PlotButterworth(58, 62)
        # if self.fmax > 130:
        self.PlotButterworth(118, 122)
        # if self.fmax > 190:
        self.PlotButterworth(178, 182)

        #self.PlotValidate(6)

        plt.vlines(60, 0, 1, linestyles='dashdot', colors='black', label='60Hz')
        plt.vlines(120, 0, 1, linestyles='dashdot', colors='black', label='120Hz')
        plt.vlines(180, 0, 1, linestyles='dashdot', colors='black', label='180Hz')
        plt.legend(loc=(1.02, 0))
        plt.ylabel('Relative power')
        plt.xlabel('Frequency (Hz)')
        plt.title(f'Morlet power {self.fnum} freqs {self.fmin}Hz to {self.fmax}Hz, wavenum {self.morlet_reps}')
        basename = f'morlet_power_{self.fnum}_{self.fmin}_{self.fmax}'
        if self.morlet_reps != 5:
            basename += f'_{self.morlet_reps}'
        # plt.savefig(f'{basename}.png')
        # plt.savefig(f'{basename}.pdf')
        # if show:
        plt.show()
