# mixed_models.R -- mixed-effects tests on the per-electrode CSVs written by the Python
# plot scripts (word-on contrast).
#
#   Rscript mixed_models.R latency figures/latency/hg_latency_word_on_electrodes.csv
#   Rscript mixed_models.R bands   figures/burke_roi_power/roi_power_word_on_alpha_per_electrode.csv
#                                  (the high-gamma CSV is read from the same place)
#
# Input: one row per electrode (sessions averaged) with its subject, Burke ROI ("L-frontal")
# and anatomical label ("L fusiform gyrus"). lobe = ROI without hemisphere; region = the
# anatomical label without hemisphere (hemispheres pooled, as the fine-region figures);
# electrodes without an anatomical label are dropped.
#   latency  latency_ms ~ lobe + (1|sub) + (1|sub:lobe) + (1|sub:region)
#            -> estimated latency per lobe; all 15 lobe pairs (Holm)
#   bands    cohens_d ~ lobe * band + (1 + band|sub) + (1|sub:lobe) + (1|sub:lobe:band)
#                       + (1|sub:region) + (1|sub:label)
#            band: alpha / high gamma (each electrode has one row per band, hence sub:label;
#            each subject its own alpha-gamma pattern, hence 1 + band|sub)
#            -> per lobe and band: estimated d (test vs 0); per lobe pair: does the alpha vs
#               high-gamma pattern differ (lobe x band interaction contrast, Holm over 15 pairs)
# sub:lobe (and sub:lobe:band): each subject has its own lobe effects, so lobe comparisons are
# tested against their variability across subjects, not across thousands of electrodes.
# Fine regions enter only within subject (sub:region: a subject's electrodes in one region are
# related). Not as (1|region): with 1-11 regions per lobe (hippocampus 1) the model cannot
# separate a lobe effect from its few regions and moves the lobe signal into them.
# Tests are asymptotic (z): with hundreds of subjects the df approximation does not matter.
# Output: <input stem>_mixed_*.csv next to the input.
suppressMessages({library(lme4); library(emmeans); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]; path <- args[2]
emm_options(lmer.df = "asymptotic")
load <- function(p) {
  d <- fread(p)
  d[, lobe := factor(sub(".*-", "", roi))]
  d[, region := sub("^[LR] ", "", fine)]
  d[!is.na(region) & !(region %in% c("", "nan", "None"))]
}

if (mode == "latency") {
  out <- sub("_electrodes\\.csv$", "_mixed", path)
  d <- load(path)
  m <- lmer(latency_ms ~ lobe + (1 | sub) + (1 | sub:lobe) + (1 | sub:region), data = d,
            control = lmerControl(optimizer = "bobyqa"))
  e <- emmeans(m, ~ lobe)
  pr <- pairs(e, adjust = "holm")
  fwrite(as.data.frame(e), paste0(out, "_lobes.csv"))
  fwrite(as.data.frame(pr), paste0(out, "_pairs.csv"))
  print(e); print(pr)
} else if (mode == "bands") {
  out <- sub("_alpha_per_electrode\\.csv$", "_mixed", path)
  d <- rbind(load(path)[, band := "alpha"],
             load(sub("_alpha_", "_high_gamma_", path))[, band := "high_gamma"])
  d[, band := factor(band)]
  m <- lmer(cohens_d ~ lobe * band + (1 + band | sub) + (1 | sub:lobe) + (1 | sub:lobe:band) +
              (1 | sub:region) + (1 | sub:label),
            data = d, control = lmerControl(optimizer = "bobyqa"))
  e <- emmeans(m, ~ lobe | band)
  inter <- contrast(emmeans(m, ~ lobe * band), interaction = c("pairwise", "pairwise"), adjust = "holm")
  fwrite(as.data.frame(test(e)), paste0(out, "_lobes.csv"))
  fwrite(as.data.frame(inter), paste0(out, "_pairs.csv"))
  print(joint_tests(m)); print(test(e)); print(inter)
}
