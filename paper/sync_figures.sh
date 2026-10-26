#!/bin/zsh
# Copy the paper's figures from figures/ (gitignored) into paper/img/ (tracked), so
# draft.md renders in the VS Code preview and on GitHub. Rerun after replotting.
cd "${0:A:h}/.."
typeset -A F=(
  fig1_power                 burke_roi_power/roi_power_word_on_sem.png
  fig1b_power_mixed          burke_roi_power/roi_power_word_on_mixed.png
  fig2a_spectrum             burke_roi_power/power_spectrum_word_on_rois.png
  fig2b_occipital            burke_roi_power/power_spectrum_word_on_occipital.png
  fig3_timecourse            burke_roi_power/power_timecourse_word_on_high_gamma.png
  fig4a_latency              latency/hg_latency_word_on.png
  fig4b_latency_fine         latency/hg_latency_word_on_fine_pooled.png
  fig4c_latency_mixed        latency/hg_latency_word_on_mixed.png
  fig5_synchrony             burke_roi_synchrony/roi_synchrony_word_on_sem.png
  fig6a_epochs               burke_roi_synchrony/roi_synchrony_epochs_word_on.png
  fig6c_epochs_alpha         burke_roi_synchrony/roi_synchrony_epochs_word_on_alpha.png
  fig6b_epochs_fine          burke_roi_synchrony/roi_synchrony_epochs_fine_word_on.png
  fig7_recall_time           decoding/recall_auc_time_word_on.png
  fig8_reliability           reliability/reliability_word_on.png
  fig9a_wait_times           latency/wait_times_word_on.png
  fig9_anticipation          latency/hg_anticipation_word_on.png
  figS1_power_subjects       burke_roi_power/roi_power_word_on.png
  figS2_synchrony_subjects   burke_roi_synchrony/roi_synchrony_word_on.png
  figS4_recovery             simulations/recovery.png
)
for k v in ${(kv)F}; do [[ -f figures/$v ]] && cp figures/$v paper/img/$k.png || echo "[missing] figures/$v"; done
echo "[sync] $(ls paper/img | wc -l | tr -d ' ') figures in paper/img"
