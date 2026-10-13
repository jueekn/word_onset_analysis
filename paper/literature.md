# Literature for the word-onset paper
Built 2026-10-13 from six searches, each checking authors, year and journal (DOIs via Crossref, PubMed or Europe PMC).
Each paper is listed once, in the section where it is used most; extra uses are given after the dash.
**pre** = preprint, not peer reviewed. **check** = some details not confirmed, verify before citing.

## 0. What this paper builds on directly
- Rao, DeHaan & Kahana 2025, *J Neurosci* 45(16):e1332242025, doi:10.1523/JNEUROSCI.1332-24.2025 — base of the helper code (github.com/adityamrao/RaoEtal25JNeurosci). Also the source for: per-electrode mean change over all partners; hub definition; BKY FDR; TFCE with sign flips. Same FR1/catFR1/pyFR data, 382 patients. Two differences from us: Rao excluded bipolar pairs closer than 20 mm (we exclude pairs sharing a contact), and Rao used PPC, not ciPLV.
- Rao & Kahana 2024, bioRxiv, doi:10.1101/2024.07.12.603286 — preprint of the above; cite the J Neurosci version.
- Solomon et al. 2017, *Nat Commun* 8:1704, doi:10.1038/s41467-017-01763-2 — hub/edge figure style; theta synchrony with high-frequency desynchronization (context for the HG AEC-c null); 200 ms windows with a label-shuffle null.
- Burke et al. 2013, *J Neurosci* 33:292–304, doi:10.1523/JNEUROSCI.2057-12.2013 — the 12 ROIs (README); gamma increases with lower global synchrony (asynchronous HFA). **check** the ROI grouping against the paper or the riley-thesis code.
- Ezzyat et al. 2017, *Curr Biol* 27:1251–1258, doi:10.1016/j.cub.2017.03.028 — L2 logistic encoding classifier, C = 7.2e-4. **check** whether C is stated here or in Ezzyat 2018.
- Long et al. 2020, Research Square **pre**, doi:10.21203/rs.3.rs-95141/v1 — predecessor paper (Mesgarani and Jacobs labs, Kahana co-author; 151 patients, same task). HG 70–150 Hz responsiveness vs the blank screen; RSA feature timing (lingual ~40–80 ms, fusiform ~120, IFG ~240). No journal version found. Our replication found their stats wrong (uncorrected per-timepoint t).
- pennmem/fc_methods_comparison_cml (riley-thesis branch) — reference pipeline; unpublished (no DeHaan paper found). Closest public work: DeHaan et al. 2025 **pre**, bioRxiv, doi:10.1101/2025.07.25.666835.
- Software with no paper: cmlreaders (github.com/pennmem/cmlreaders) and PTSA — cite by URL.

## 1. Data and cohort
- Herrema & Kahana, FR1, OpenNeuro ds004789, doi:10.18112/openneuro.ds004789.v3.1.0
- Herrema & Kahana, catFR1, OpenNeuro ds004809, doi:10.18112/openneuro.ds004809.v2.2.0
- Herrema & Kahana, pyFR, OpenNeuro ds004865, doi:10.18112/openneuro.ds004865.v2.0.1 — cite the version you downloaded. No peer-reviewed paper describing the RAM data release exists.
- Sederberg et al. 2003, *J Neurosci* 23:10809, doi:10.1523/JNEUROSCI.23-34-10809.2003 — origin of pyFR; theta and gamma during encoding predict recall.
- Sederberg et al. 2007, *Cereb Cortex* 17:1190, doi:10.1093/cercor/bhl030 — pyFR; hippocampal and neocortical gamma SME.
- Sederberg et al. 2007, *Psychol Sci* 18:927, doi:10.1111/j.1467-9280.2007.02003.x — pyFR.
- Kahana 2012, *Foundations of Human Memory* (OUP) — free-recall paradigm.
- Kahana et al. 2024, *JEP:LMC* 50:1421, doi:10.1037/xlm0001319 — PEERS (scalp EEG); only for the task lineage.
- Holdgraf et al. 2019, *Sci Data* 6:102, doi:10.1038/s41597-019-0105-7 — iEEG-BIDS.
- Gorgolewski et al. 2016, *Sci Data* 3:160044, doi:10.1038/sdata.2016.44 — BIDS.
- Markiewicz et al. 2021, *eLife* 10:e71774, doi:10.7554/eLife.71774 — OpenNeuro.
- Appelhoff et al. 2019, *JOSS* 4:1896, doi:10.21105/joss.01896 — MNE-BIDS (only if used).
- Rahimzadeh et al. 2023, *Neuron* 111:3710, doi:10.1016/j.neuron.2023.09.029 — optional, on sharing human neurophysiology data.

## 2. Software
- Gramfort et al. 2013, *Front Neurosci* 7:267, doi:10.3389/fnins.2013.00267; Gramfort et al. 2014, *NeuroImage* 86:446, doi:10.1016/j.neuroimage.2013.10.027 — MNE.
- MNE-Connectivity, Zenodo, doi:10.5281/zenodo.10278399 — give the version.
- Abraham et al. 2014, *Front Neuroinform* 8:14, doi:10.3389/fninf.2014.00014 — nilearn.
- Mölder et al. 2021, *F1000Research* 10:33, doi:10.12688/f1000research.29032.2; Köster & Rahmann 2012, *Bioinformatics* 28:2520, doi:10.1093/bioinformatics/bts480 — Snakemake.
- Pedregosa et al. 2011, *JMLR* 12:2825 (no DOI) — scikit-learn.
- Virtanen et al. 2020, *Nat Methods* 17:261, doi:10.1038/s41592-019-0686-2 — SciPy.
- Harris et al. 2020, *Nature* 585:357, doi:10.1038/s41586-020-2649-2 — NumPy.
- Seabold & Perktold 2010, SciPy Proc. 92–96, doi:10.25080/Majora-92bf1922-011 — statsmodels (BH, BKY).

## 3. Preprocessing and spectral estimation
- Thomson 1982, *Proc IEEE* 70:1055, doi:10.1109/PROC.1982.12433 — multitaper.
- Slepian 1978, *Bell Syst Tech J* 57:1371, doi:10.1002/j.1538-7305.1978.tb02104.x — DPSS tapers.
- Percival & Walden 1993, CUP, doi:10.1017/CBO9780511622762 — multitaper textbook.
- Mitra & Pesaran 1999, *Biophys J* 76:691, doi:10.1016/S0006-3495(99)77236-X — moving-window multitaper.
- Prerau et al. 2017, *Physiology* 32:60, doi:10.1152/physiol.00062.2015 — choosing window and bandwidth.
- Babadi & Brown 2014, *IEEE TBME* 61:1555, doi:10.1109/TBME.2014.2311996 — multitaper review.
- Bruns 2004, *J Neurosci Methods* 137:321, doi:10.1016/j.jneumeth.2004.03.002 — Fourier, Hilbert and wavelet are equivalent in principle.
- Le Van Quyen et al. 2001, *J Neurosci Methods* 111:83, doi:10.1016/S0165-0270(01)00372-7 — Hilbert vs wavelet.
- Edwards et al. 2009, *J Neurophysiol* 102:377, doi:10.1152/jn.90954.2008 — HG sub-bands, each normalised, then averaged. **check** that it is the origin of this method.
- de Cheveigné & Nelken 2019, *Neuron* 102:280, doi:10.1016/j.neuron.2019.02.039 — filter ringing (notch, 500 ms buffer).
- Widmann et al. 2015, *J Neurosci Methods* 250:34, doi:10.1016/j.jneumeth.2014.08.002 — IIR/Butterworth design, edge effects.
- VanRullen 2011, *Front Psychol* 2:365, doi:10.3389/fpsyg.2011.00365 — filters smear onsets in time (latency).
- Cohen 2014, *Analyzing Neural Time Series Data* (MIT Press), doi:10.7551/mitpress/9609.001.0001 — buffers, baselines, phase-locked vs non-phase-locked power.
- Bedrosian 1963, *Proc IEEE* 51:868, doi:10.1109/PROC.1963.2308 — why the AEC-c band is 70–110 Hz.

## 4. Electrode localization and referencing
- Desikan et al. 2006, *NeuroImage* 31:968, doi:10.1016/j.neuroimage.2006.01.021 — DK atlas.
- Fischl et al. 2002, *Neuron* 33:341, doi:10.1016/S0896-6273(02)00569-X — volumetric labels; Fischl et al. 2004, *Cereb Cortex* 14:11, doi:10.1093/cercor/bhg087 — parcellation; Fischl 2012, *NeuroImage* 62:774, doi:10.1016/j.neuroimage.2012.01.021 — FreeSurfer.
- Avants et al. 2008, *Med Image Anal* 12:26, doi:10.1016/j.media.2007.06.004 — ANTs registration (CT to MRI, MNI).
- Wang et al. 2013, *IEEE TPAMI* 35:611, doi:10.1109/TPAMI.2012.143 — joint label fusion (Penn MTL labels).
- Yushkevich et al. 2015, *Hum Brain Mapp* 36:258, doi:10.1002/hbm.22627 — ASHS hippocampal/MTL subregions.
- Dykstra et al. 2012, *NeuroImage* 59:3563, doi:10.1016/j.neuroimage.2011.11.046 — coordinates to labels (cited by Rao).
- Azarion et al. 2014, *Epilepsia* 55:2028, doi:10.1111/epi.12827 — Penn grid/strip localization. **check** that RAM papers cite it.
- Stolk et al. 2018, *Nat Protoc* 13:1699, doi:10.1038/s41596-018-0009-6; Groppe et al. 2017, *J Neurosci Methods* 281:40, doi:10.1016/j.jneumeth.2017.01.022; Hamilton et al. 2017, *Front Neuroinform* 11:62, doi:10.3389/fninf.2017.00062 — localization tools, MNI warping (relevant to neighbour-transferred labels).
- Mercier et al. 2022, *NeuroImage* 260:119438, doi:10.1016/j.neuroimage.2022.119438 — iEEG good-practice guidelines (Kahana co-author).
- Mercier et al. 2017, *NeuroImage* 147:219, doi:10.1016/j.neuroimage.2016.08.037 — signal spreads across sEEG contacts (supports excluding pairs that share a contact).
- Li et al. 2018, *NeuroImage* 183:327, doi:10.1016/j.neuroimage.2018.08.020 — sEEG referencing.
- Arnulfo et al. 2015, *NeuroImage* 112:114, doi:10.1016/j.neuroimage.2015.02.031 — bipolar vs closest-white-matter reference; PLV and amplitude correlation in sEEG.
- Shirhatti et al. 2016, *Neural Comput* 28:882, doi:10.1162/NECO_a_00827 — reference choice changes coherence and phase.
- Zaveri et al. 2006, *Clin Neurophysiol* 117:2102, doi:10.1016/j.clinph.2006.05.032 — caveats of bipolar montages.
- Trongnetrpunya et al. 2016, *Front Syst Neurosci* 9:189, doi:10.3389/fnsys.2015.00189 — bipolar derivation removes common signals.
- Shi et al. 2024, *Brain Commun* fcae165, doi:10.1093/braincomms/fcae165 — Penn group; 24 connectivity measures × referencing. **check** (details from a search snippet).

## 5. Power at word onset (Results 1)
### HFA / broadband as a marker of local activity
- Manning et al. 2009, *J Neurosci* 29:13613, doi:10.1523/JNEUROSCI.2041-09.2009 — broadband power tracks spiking (Kahana lab).
- Ray & Maunsell 2011, *PLoS Biol* 9:e1000610, doi:10.1371/journal.pbio.1000610 — HG is broadband/spiking, not a rhythm.
- Mukamel et al. 2005, *Science* 309:951, doi:10.1126/science.1110913; Nir et al. 2007, *Curr Biol* 17:1275, doi:10.1016/j.cub.2007.06.066 — firing, gamma LFP and BOLD.
- Miller, Sorensen et al. 2009, *PLoS Comput Biol* 5:e1000609, doi:10.1371/journal.pcbi.1000609 — power-law spectrum.
- Miller, Zanos et al. 2009, *J Neurosci* 29:3132, doi:10.1523/JNEUROSCI.5506-08.2009; Miller et al. 2014, *NeuroImage* 85:711, doi:10.1016/j.neuroimage.2013.08.070 — broadband activation.
- Crone et al. 2006, *Prog Brain Res* 159:275, doi:10.1016/S0079-6123(06)59019-3; Crone et al. 2011, *Int J Psychophysiol* 79:9, doi:10.1016/j.ijpsycho.2010.10.013 — HFA reviews.
- Crone et al. 1998a, *Brain* 121:2271, doi:10.1093/brain/121.12.2271 (alpha/beta ERD); Crone et al. 1998b, *Brain* 121:2301, doi:10.1093/brain/121.12.2301 (gamma ERS) — classic alpha/beta decrease with gamma increase.
- Lachaux et al. 2012, *Prog Neurobiol* 98:279, doi:10.1016/j.pneurobio.2012.06.008 — HFA in iEEG review.
- Jerbi et al. 2009, *Hum Brain Mapp* 30:1758, doi:10.1002/hbm.20750 — task-related gamma in iEEG.
- Burke, Ramayya & Kahana 2015, *Curr Opin Neurobiol* 31:104, doi:10.1016/j.conb.2014.09.003 — HFA: oscillation or volatility.
- Phan et al. 2019, *eLife* 8:e42950, doi:10.7554/eLife.42950 — HFA as stochastic volatility. **check** author list.
- Winawer et al. 2013, *Curr Biol* 23:1145, doi:10.1016/j.cub.2013.05.001 — asynchronous broadband activity drives BOLD in visual cortex.
- Leszczyński et al. 2020, *Sci Adv* 6:eabb0977, doi:10.1126/sciadv.abb0977 — caveat: HFA can reflect synaptic input rather than spiking.
- Ojemann et al. 2013, *Front Hum Neurosci* 7:34, doi:10.3389/fnhum.2013.00034 — relating fMRI, LFP and units.
- Jacobs, Kahana, Ekstrom & Fried 2007, *J Neurosci* 27:3839, doi:10.1523/JNEUROSCI.4636-06.2007 — oscillations time single neurons.

### Alpha/beta decrease (event-related desynchronization)
- Pfurtscheller & Aranibar 1977, *EEG Clin Neurophysiol* 42:817, doi:10.1016/0013-4694(77)90235-8; Pfurtscheller & Lopes da Silva 1999, *Clin Neurophysiol* 110:1842, doi:10.1016/S1388-2457(99)00141-8 — ERD/ERS.
- Klimesch 1999, *Brain Res Rev* 29:169, doi:10.1016/S0165-0173(98)00056-3; Klimesch et al. 2007, *Brain Res Rev* 53:63, doi:10.1016/j.brainresrev.2006.06.003; Klimesch 2012, *TICS* 16:606, doi:10.1016/j.tics.2012.10.007 — alpha reviews (inhibition-timing).
- Jensen & Mazaheri 2010, *Front Hum Neurosci* 4:186, doi:10.3389/fnhum.2010.00186 — alpha gating by inhibition; core reading of occipital alpha↓ + HG↑.
- Haegens et al. 2011, *PNAS* 108:19377, doi:10.1073/pnas.1117190108 — alpha inversely related to spiking.
- Haegens et al. 2014, *NeuroImage* 92:46, doi:10.1016/j.neuroimage.2014.01.049 — alpha peak frequency varies (band definition).
- Halgren M et al. 2019, *PNAS* 116:23772, doi:10.1073/pnas.1913092116 — alpha generators; alpha propagates from higher to lower areas.
- Haegens et al. 2015, *J Neurosci* 35:14341, doi:10.1523/JNEUROSCI.0600-15.2015 — laminar alpha in sensory cortices.
- Engel & Fries 2010, *Curr Opin Neurobiol* 20:156, doi:10.1016/j.conb.2010.02.015; Spitzer & Haegens 2017, *eNeuro* 4:ENEURO.0170-17.2017, doi:10.1523/ENEURO.0170-17.2017 — frontal beta decrease.
- Geller et al. 2014, *Clin Neurophysiol* 125:1764, doi:10.1016/j.clinph.2014.01.021 — Penn iEEG: eye closure raises low frequencies and lowers occipital gamma (occipital reactivity).
- Frauscher et al. 2018, *Brain* 141:1130, doi:10.1093/brain/awy035 — normal iEEG atlas (regional resting alpha).
- Groppe et al. 2013, *NeuroImage* 79:223, doi:10.1016/j.neuroimage.2013.04.044 — dominant resting ECoG frequencies.

### Visual-onset alpha↓ + gamma↑ in iEEG (occipital, fusiform)
- Vidal et al. 2010, *Front Hum Neurosci* 4:195, doi:10.3389/fnhum.2010.00195 — closest analogue: gamma↑, alpha/beta↓ to visual stimuli including pseudowords; less than 10% spatial overlap between band selectivities.
- Lachaux et al. 2005, *NeuroImage* 25:491, doi:10.1016/j.neuroimage.2004.11.052 — visual gamma in many forms.
- Tallon-Baudry et al. 2005, *Cereb Cortex* 15:654, doi:10.1093/cercor/bhh167 — gamma in lateral occipital vs fusiform.
- Hermes et al. 2015, *Cereb Cortex* 25:2951, doi:10.1093/cercor/bhu091 — narrowband vs broadband gamma (words should give broadband).
- Bartoli et al. 2019, *Curr Biol* 29:3345, doi:10.1016/j.cub.2019.08.004 — distinct gamma-range activities in visual cortex.
- Jacobs & Kahana 2009, *J Neurosci* 29:10203, doi:10.1523/JNEUROSCI.2187-09.2009 — Penn: gamma codes letter identity in occipitotemporal sites.
- Osipova et al. 2008, *PLoS One* 3:e3990, doi:10.1371/journal.pone.0003990 — gamma phase-locked to posterior alpha.
- Fisch et al. 2009, *Neuron* 64:562, doi:10.1016/j.neuron.2009.11.001; Noy et al. 2015, *Conscious Cogn* 35:206, doi:10.1016/j.concog.2015.03.006; Gaillard et al. 2009, *PLoS Biol* 7:e61, doi:10.1371/journal.pbio.1000061 — ventral-stream "ignition"; late widespread gamma plus long-range synchrony.
- Golan et al. 2016, *eLife* 5:e17243, doi:10.7554/eLife.17243 — visual transients in iEEG.
- Kucyi et al. 2020, *Nat Commun* 11:325, doi:10.1038/s41467-019-14166-2; Ossandón et al. 2011, *J Neurosci* 31:14521, doi:10.1523/JNEUROSCI.2483-11.2011 — default-mode HG suppression (non-visual decreases, e.g., R frontal).

### Memory-encoding spectral signature (Kahana lab and RAM)
- Long, Burke & Kahana 2014, *NeuroImage* 84:488, doi:10.1016/j.neuroimage.2013.08.052 — low-frequency↓ / HFA↑ SME.
- Burke et al. 2014, *J Neurosci* 34:11355, doi:10.1523/JNEUROSCI.2654-13.2014 — same pattern at recall.
- Greenberg et al. 2015, *NeuroImage* 114:257, doi:10.1016/j.neuroimage.2015.03.077 — theta↓ / HFA↑ for associative encoding.
- Serruya, Sederberg & Kahana 2014, *Cereb Cortex* 24:403, doi:10.1093/cercor/bhs318 — power shifts with serial position (possible confound).
- Sederberg et al. 2006, *NeuroImage* 32:1422, doi:10.1016/j.neuroimage.2006.04.223 — oscillatory correlates of primacy.
- Long & Kahana 2015, *NeuroImage* 119:332, doi:10.1016/j.neuroimage.2015.06.073 — core memory network.
- Rubinstein et al. 2023, *Cereb Cortex* 33:6891, doi:10.1093/cercor/bhad008 — RAM: low-frequency↓ / high-frequency↑; causal SME.
- Rudoler, Herweg & Kahana 2023, *J Neurosci* 43:613, doi:10.1523/JNEUROSCI.1045-22.2022 — broadband tilt masks narrowband theta (supports the planned specparam check).
- Herweg, Solomon & Kahana 2020, *TICS* 24:208, doi:10.1016/j.tics.2019.12.006 — theta review; tilt vs oscillation.
- Katerman et al. 2022, *NeuroImage* 246:118748, doi:10.1016/j.neuroimage.2021.118748 — scalp: theta↑, alpha↓, gamma↑.
- Healey & Kahana 2020, *PLoS One* 15:e0227274, doi:10.1371/journal.pone.0227274 — time course of encoding power (scalp).
- Broitman & Kahana 2026 **pre**, PsyArXiv, doi:10.31234/osf.io/4xw56_v1 — "TAG" signature (theta↑, alpha↓, gamma↑); frames the occipital/limbic results.
- Randazzo, Ezzyat & Kahana 2019 **pre**, bioRxiv, doi:10.1101/601880 — spectral tilt in math problem solving (RAM).
- Falkenburg, Broitman & Kahana 2026 **pre**, doi:10.64898/2026.07.29.741612 — spectral SME separates good from poor learners.
- Broitman, Healey & Kahana 2025, *Psychol Aging* 40:265, doi:10.1037/pag0000876 — scalp; aging.
- Kahana 2006, *J Neurosci* 26:1669, doi:10.1523/JNEUROSCI.3737-05c.2006; Kahana, Seelig & Madsen 2001, *Curr Opin Neurobiol* 11:739, doi:10.1016/S0959-4388(01)00278-1; Jacobs & Kahana 2010, *TICS* 14:162, doi:10.1016/j.tics.2010.01.005 — background reviews.
- Lohnas, Davachi & Kahana 2020, *Neuropsychologia* 143:107471, doi:10.1016/j.neuropsychologia.2020.107471 — repetition/fatigue in hippocampal encoding (RAM).
- Kucewicz et al. 2014, *Brain* 137:2231, doi:10.1093/brain/awu149 — HFOs induced by word presentation.
- Kucewicz et al. 2017, *Brain* 140:1337, doi:10.1093/brain/awx043 — RAM, 101 patients: narrowband vs broadband gamma to words.
- Kucewicz et al. 2018, *eNeuro* 5:ENEURO.0369-17.2018, doi:10.1523/ENEURO.0369-17.2018 — HG 62–118 Hz response to word presentation.
- Marks et al. 2021, *NeuroImage* 245:118637, doi:10.1016/j.neuroimage.2021.118637 — RAM: theta, alpha/beta and gamma spread on independent timelines across visual, temporal and prefrontal cortex.
- Topçu et al. 2022, *EBioMedicine* 82:104135, doi:10.1016/j.ebiom.2022.104135 — RAM: left anterior PFC encoding hotspot.
- Rubinstein et al. 2021, *NeuroImage* 225:117514, doi:10.1016/j.neuroimage.2020.117514 — RAM: supramarginal/angular HFA during encoding.
- Weiss et al. 2025, *Front Neurol* 16:1627528, doi:10.3389/fneur.2025.1627528 — RAM: supramarginal HG and beta bursts.
- Lega, Jacobs & Kahana 2012, *Hippocampus* 22:748, doi:10.1002/hipo.20937; Lin et al. 2017, *Hippocampus* 27:1040, doi:10.1002/hipo.22751; Lin et al. 2019, *Hippocampus* 29:68, doi:10.1002/hipo.23016; Goyal et al. 2020, *Nat Commun* 11:2469, doi:10.1038/s41467-020-15670-6; Kota, Rugg & Lega 2020, *J Neurosci* 40:9507, doi:10.1523/JNEUROSCI.0767-20.2020 — hippocampal theta/gamma during encoding (context for limbic/hippocampal effects).
- Lega et al. 2016, *Cereb Cortex* 26:268, doi:10.1093/cercor/bhu232 — hippocampal PAC (only if PAC returns).

### Memory power theory and other groups
- Hanslmayr, Staudigl & Fellner 2012, *Front Hum Neurosci* 6:74, doi:10.3389/fnhum.2012.00074 — information via desynchronization.
- Hanslmayr, Staresina & Bowman 2016, *Trends Neurosci* 39:16, doi:10.1016/j.tins.2015.11.004 — cortical alpha/beta↓ vs hippocampal theta/gamma↑; alpha power↓ alongside synchrony↑ (also synchrony).
- Hanslmayr & Staudigl 2014, *NeuroImage* 85:648, doi:10.1016/j.neuroimage.2013.05.121; Hanslmayr, Spitzer & Bäuml 2009, *Cereb Cortex* 19:1631, doi:10.1093/cercor/bhn197; Hanslmayr et al. 2011, *J Neurosci* 31:15674, doi:10.1523/JNEUROSCI.3140-11.2011 — alpha/beta SMEs; left inferior frontal beta.
- Fellner et al. 2013, *NeuroImage* 79:361, doi:10.1016/j.neuroimage.2013.04.121.
- Fellner et al. 2019, *PLoS Biol* 17:e3000403, doi:10.1371/journal.pbio.3000403 — argues against a single tilt; gamma↑ comes before theta↓ (frontal-tilt interpretation).
- Griffiths et al. 2019, *PNAS* 116:21834, doi:10.1073/pnas.1914180116; Griffiths et al. 2021, *NeuroImage* 242:118454, doi:10.1016/j.neuroimage.2021.118454 — neocortical alpha/beta↓ with hippocampal gamma↑.
- Staresina et al. 2016, *eLife* 5:e17397, doi:10.7554/eLife.17397 — hippocampal gamma↑ / alpha↓.
- Nyhus & Curran 2010, *Neurosci Biobehav Rev* 34:1023, doi:10.1016/j.neubiorev.2009.12.014 — review.
- Mölle et al. 2002, *Eur J Neurosci* 15:923, doi:10.1046/j.1460-9568.2002.01921.x — theta sync + alpha desync during intentional encoding.
- Johnson & Knight 2015, *Curr Opin Neurobiol* 31:18, doi:10.1016/j.conb.2014.07.021; Johnson et al. 2020, *J Neural Eng* 17:051001, doi:10.1088/1741-2552/abb7a5; Parvizi & Kastner 2018, *Nat Neurosci* 21:474, doi:10.1038/s41593-018-0108-2 — iEEG reviews.

### Frontal theta / low-alpha increase
- Raghavachari et al. 2001, *J Neurosci* 21:3175, doi:10.1523/JNEUROSCI.21-09-03175.2001; Raghavachari et al. 2006, *J Neurophysiol* 95:1630, doi:10.1152/jn.00409.2005 — task-gated cortical theta from local generators.
- Rizzuto et al. 2006, *NeuroImage* 31:1352, doi:10.1016/j.neuroimage.2006.01.009 — stimulus-locked theta phase reset.
- Mormann et al. 2005, *Hippocampus* 15:890, doi:10.1002/hipo.20117 — MTL phase/amplitude reset to words.
- Cavanagh & Frank 2014, *TICS* 18:414, doi:10.1016/j.tics.2014.04.012 — frontal theta as cognitive control (alternative reading).
- Hasselmo et al. 2002, *Neural Comput* 14:793, doi:10.1162/089976602317318965 — theory.

### Spectral tilt / aperiodic activity (spectrum figure, planned specparam)
- Podvalny et al. 2015, *J Neurophysiol* 114:505, doi:10.1152/jn.00943.2014 — activation rotates the spectrum; closest to the frontal tilt.
- Donoghue et al. 2020, *Nat Neurosci* 23:1655, doi:10.1038/s41593-020-00744-x — specparam.
- Gao, Peterson & Voytek 2017, *NeuroImage* 158:70, doi:10.1016/j.neuroimage.2017.06.078 — aperiodic exponent and E/I balance.
- Gerster et al. 2022, *Neuroinformatics* 20:991, doi:10.1007/s12021-022-09581-8 — fitting recommendations.
- Wen & Liu 2016, *Brain Topogr* 29:13, doi:10.1007/s10548-015-0448-0 — IRASA (robustness alternative).
- Gyurkovics et al. 2022, *J Neurosci* 42:7144, doi:10.1523/JNEUROSCI.0414-22.2022 — the aperiodic background changes at stimulus onset (our contrast).
- Waschke et al. 2021, *eLife* 10:e70068, doi:10.7554/eLife.70068 — the spectral exponent tracks attention.
- Cole & Voytek 2017, *TICS* 21:137, doi:10.1016/j.tics.2016.12.008 — waveform shape.

## 6. Order of activation / HG latency (Results 2)
### RAM / Penn precedents
- Kucewicz et al. 2019, *eNeuro* 6:ENEURO.0214-18.2018, doi:10.1523/ENEURO.0214-18.2018 — RAM: HG after word onset runs visual → MTL → prefrontal. Closest precedent.
- Burke, Long et al. 2014, *NeuroImage* 85:834, doi:10.1016/j.neuroimage.2013.06.067 — pyFR, 98 patients: early HFA (0–500 ms) in visual/fusiform/MTL, late (400–900 ms) in IFG and parietal. Uses Talairach/Brodmann ROIs, not the 12 ROIs.
- Haque et al. 2015, *J Neurosci* 35:13577, doi:10.1523/JNEUROSCI.0687-15.2015 — posterior-to-anterior progression before successful encoding.
- Mohan et al. 2024, *Nat Hum Behav* 8:1124, doi:10.1038/s41562-024-01838-3 — RAM-era: alpha/theta travelling waves posterior → anterior at encoding (also alpha synchrony).
- Zhang et al. 2018, *Neuron* 98:1269, doi:10.1016/j.neuron.2018.05.019 — theta/alpha travelling waves.
- Alexander & Dugué 2026, *eLife* 13:RP100674, doi:10.7554/eLife.100674 — RAM public data: large-scale phase gradients.
- Das, Zabeh, Ermentrout & Jacobs 2026, *Nat Commun* 17:5143, doi:10.1038/s41467-026-71386-z — RAM travelling waves.

### Ventral occipitotemporal / word-reading timing (compare with our lateral occipital 122, lingual 150, fusiform 157 ms)
- Woolnough et al. 2021, *Nat Hum Behav* 5:389, doi:10.1038/s41562-020-00982-w — word-length effect starts at the occipital pole at 75 ms and moves forward; frequency effect at 150 ms in mid-fusiform; feedback ~250–500 ms (also word-length split).
- Woolnough et al. 2022, *J Neurosci* 42:5438, doi:10.1523/JNEUROSCI.2324-21.2022 — fusiform → IFG/IPS when reading aloud.
- Woolnough et al. 2023, *PNAS* 120:e2300252120, doi:10.1073/pnas.2300252120 — frontotemporal networks in sentence reading.
- Kadipasaoglu et al. 2016, *PLoS One* 11:e0157109, doi:10.1371/journal.pone.0157109 — word-selective onsets: L ventral temporal 153 ms, L lateral occipital 194 ms; peaks 350–400 ms.
- Mainy et al. 2008, *Hum Brain Mapp* 29:1215, doi:10.1002/hbm.20457 — gamma in basal temporal at 150 ms, VWFA at 200 ms; STG/Broca's after 400 ms.
- Thesen et al. 2012, *Nat Commun* 3:1284, doi:10.1038/ncomms2220 — letter responses ~60 ms before word responses; sustained HG.
- Nobre, Allison & McCarthy 1994, *Nature* 372:260, doi:10.1038/372260a0 — posterior → anterior fusiform word responses.
- Allison et al. 1994, *Cereb Cortex* 4:544, doi:10.1093/cercor/4.5.544; Allison et al. 1999, *Cereb Cortex* 9:415, doi:10.1093/cercor/9.5.415 — ventral occipitotemporal N200 at ~200 ms.
- Hirshorn et al. 2016, *PNAS* 113:8162, doi:10.1073/pnas.1604126113 — mid-fusiform stages.
- Gaillard et al. 2006, *Neuron* 50:191, doi:10.1016/j.neuron.2006.03.031 — causal role of left inferotemporal cortex in reading.
- Hamamé et al. 2013, *Neurology* 80:602, doi:10.1212/WNL.0b013e31828154d9 — letter-string-selective HG.
- Lochy et al. 2018, *PNAS* 115:E7595, doi:10.1073/pnas.1718987115 — letter/word selectivity in L fusiform.
- Forseth et al. 2018, *Brain* 141:2112, doi:10.1093/brain/awy120 — mid-fusiform semantic hub (late fusiform → temporal network).
- Jacques et al. 2016, *Neuropsychologia* 83:14, doi:10.1016/j.neuropsychologia.2015.07.024 — VTC signals from ~100 ms.
- Ghuman et al. 2014, *Nat Commun* 5:5672, doi:10.1038/ncomms6672 — sharp early then sustained fusiform response.
- Bastin et al. 2013, *J Neurosci* 33:10123, doi:10.1523/JNEUROSCI.4646-12.2013 — PPA at ~80 ms (compare our parahippocampal 167 ms).
- Groen et al. 2022, *J Neurosci* 42:7562, doi:10.1523/JNEUROSCI.1812-21.2022 — response dynamics across the visual hierarchy.
- Halgren et al. 1994, *J Physiol Paris* 88:1, doi:10.1016/0928-4257(94)90092-2 — depth-recorded stages of word processing.
- Perrone-Bertolotti et al. 2012, *J Neurosci* 32:17554, doi:10.1523/JNEUROSCI.2982-12.2012 — temporal voice areas during silent reading (STG ~190 ms).
- Vidal et al. 2012, *J Neurosci* 32:6421, doi:10.1523/JNEUROSCI.4363-11.2012 — HG amplitude correlations in the reading network (also synchrony).
- Riès et al. 2017, *PNAS* 114:E4530, doi:10.1073/pnas.1620669114 — HFA timing in word retrieval.
- Conner et al. 2011, *J Neurosci* 31:12855, doi:10.1523/JNEUROSCI.1457-11.2011; Kadipasaoglu et al. 2014, *NeuroImage* 101:215, doi:10.1016/j.neuroimage.2014.07.006 — grouped iEEG methods.
- Schrouff et al. 2020, *Nat Commun* 11:656, doi:10.1038/s41467-020-14432-8 — fast face-processing dynamics in temporal cortex.

### Reading theory
- Cohen et al. 2000, *Brain* 123:291, doi:10.1093/brain/123.2.291; Dehaene & Cohen 2011, *TICS* 15:254, doi:10.1016/j.tics.2011.04.003 — VWFA (feedforward account).
- Price & Devlin 2011, *TICS* 15:246, doi:10.1016/j.tics.2011.04.001 — interactive account (feedback; late network).
- Carreiras et al. 2014, *TICS* 18:90, doi:10.1016/j.tics.2013.11.005; Grainger & Holcomb 2009, *Lang Linguist Compass* 3:128, doi:10.1111/j.1749-818X.2008.00121.x — time course of word recognition.
- Grill-Spector & Weiner 2014, *Nat Rev Neurosci* 15:536, doi:10.1038/nrn3747; Lerner et al. 2001, *Cereb Cortex* 11:287, doi:10.1093/cercor/11.4.287 — ventral hierarchy.

### Temporal, frontal and MTL timing (compare our temporal ~190, frontal 215–237, OFC 300 ms; late peaks 350–550 ms)
- Halgren et al. 2002, *NeuroImage* 17:1101, doi:10.1006/nimg.2002.1268 — MEG: Wernicke's ~250 → anterior temporal → Broca's ~300 → DLPFC → orbital/frontopolar ~370 ms (same order, OFC last).
- Marinkovic et al. 2003, *Neuron* 38:487, doi:10.1016/S0896-6273(03)00197-1 — activity converges in anterior temporal and inferior prefrontal cortex at ~400 ms.
- Halgren et al. 2006, *NeuroImage* 30:1401, doi:10.1016/j.neuroimage.2005.10.053 — laminar recordings: feedforward ~120–170 ms, feedback after ~550 ms (mechanism for sharp-early vs broad-late peaks).
- Chan et al. 2011, *J Neurosci* 31:18119, doi:10.1523/JNEUROSCI.3122-11.2011 — semantic category selectivity in anteroventral temporal lobe at 130 ms.
- McCarthy et al. 1995, *J Neurosci* 15:1080, doi:10.1523/JNEUROSCI.15-02-01080.1995 — anterior medial temporal N400.
- Sahin et al. 2009, *Science* 326:445, doi:10.1126/science.1174481 — Broca's area: lexical ~200, grammatical ~320, phonological ~450 ms.
- Flinker et al. 2015, *PNAS* 112:2871, doi:10.1073/pnas.1414491112 — temporal → Broca's → motor.
- Kawasaki et al. 2001, *Nat Neurosci* 4:15, doi:10.1038/82850 — ventral PFC units at 120–160 ms to emotional pictures (contrast).
- Kirchner et al. 2009, *J Neurosci* 29:7599, doi:10.1523/JNEUROSCI.1233-09.2009 — frontal eye field at 45 ms (contrast).
- Halgren et al. 2015, *Neuropsychologia* 76:108, doi:10.1016/j.neuropsychologia.2015.03.021 — laminar theta during word integration.
- Mormann et al. 2008, *J Neurosci* 28:8865, doi:10.1523/JNEUROSCI.1640-08.2008 — units: parahippocampal 271, hippocampus 394, amygdala 397 ms (same order as ours, later than population HG).
- Barbeau et al. 2008, *Cereb Cortex* 18:997, doi:10.1093/cercor/bhm140 — hippocampus last at 480 ms (our broad hippocampal peak).
- Méndez-Bértolo et al. 2016, *Nat Neurosci* 19:1041, doi:10.1038/nn.4324 — fast amygdala response to fear, 74 ms (contrast).
- Oya et al. 2002, *J Neurosci* 22:9502, doi:10.1523/JNEUROSCI.22-21-09502.2002 — amygdala gamma.
- Halgren et al. 1980, *Science* 210:803, doi:10.1126/science.7434000 — late MTL potentials.
- Quiroga et al. 2005, *Nature* 435:1102, doi:10.1038/nature03687; Quian Quiroga et al. 2008, *PNAS* 105:3599, doi:10.1073/pnas.0707043105; Kreiman et al. 2000, *Nat Neurosci* 3:946, doi:10.1038/78868 — MTL single units.
- Ison et al. 2015, *Neuron* 87:220, doi:10.1016/j.neuron.2015.06.016; Rutishauser et al. 2006, *Neuron* 49:805, doi:10.1016/j.neuron.2006.02.015; Rutishauser et al. 2010, *Nature* 464:903, doi:10.1038/nature08860; Urgolites et al. 2020, *PNAS* 117:13767, doi:10.1073/pnas.2001338117; Jang et al. 2017, *Curr Biol* 27:1700, doi:10.1016/j.cub.2017.05.014 — single units during encoding.

### Speed of visual recognition
- Thorpe et al. 1996, *Nature* 381:520, doi:10.1038/381520a0 — ~150 ms (our early peak).
- Liu et al. 2009, *Neuron* 62:281, doi:10.1016/j.neuron.2009.02.025 — decoding object category from 100 ms.
- Schmolesky et al. 1998, *J Neurophysiol* 79:3272, doi:10.1152/jn.1998.79.6.3272 — macaque latencies across visual areas.
- Bullier 2001, *Brain Res Rev* 36:96, doi:10.1016/S0165-0173(01)00085-6; Yoshor et al. 2007, *Cereb Cortex* 17:2293, doi:10.1093/cercor/bhl138.
- Lemaréchal et al. 2022, *Brain* 145:1653, doi:10.1093/brain/awab362 — atlas of conduction delays (compare inter-regional latency differences).
- Trebaul et al. 2018, *NeuroImage* 181:414, doi:10.1016/j.neuroimage.2018.07.039 — F-TRACT.

### Latency method (half-max) and pairwise order tests
- Kiesel et al. 2008, *Psychophysiology* 45:250, doi:10.1111/j.1469-8986.2007.00618.x — percent-of-peak criterion + jackknife is the most accurate.
- Miller, Patterson & Ulrich 1998, *Psychophysiology* 35:99, doi:10.1111/1469-8986.3510099; Ulrich & Miller 2001, *Psychophysiology* 38:816, doi:10.1111/1469-8986.3850816 — jackknife.
- Liesefeld 2018, *Front Neurosci* 12:765, doi:10.3389/fnins.2018.00765 — latency-measure primer.
- Hansen & Hillyard 1980, *EEG Clin Neurophysiol* 49:277, doi:10.1016/0013-4694(80)90222-9 — fractional-area latency.
- Luck 2014, *An Introduction to the ERP Technique*, 2nd ed., MIT Press.

### Word length (median split)
- Hauk & Pulvermüller 2004, *Clin Neurophysiol* 115:1090, doi:10.1016/j.clinph.2003.12.020 — length changes response size, not peak latency.
- Sereno et al. 1998, *NeuroReport* 9:2195, doi:10.1097/00001756-199807130-00009 — lexical effects at ~130–150 ms (**check**).
- (Woolnough 2021 above.)

## 7. Synchrony (Results 3)
### Phase metrics (alpha ciPLV)
- Bruña, Maestú & Pereda 2018, *J Neural Eng* 15:056011, doi:10.1088/1741-2552/aacfe4 — ciPLV.
- Lachaux et al. 1999, *Hum Brain Mapp* 8:194, doi:10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C — PLV.
- Nolte et al. 2004, *Clin Neurophysiol* 115:2292, doi:10.1016/j.clinph.2004.04.029 — imaginary coherency.
- Stam et al. 2007, *Hum Brain Mapp* 28:1178, doi:10.1002/hbm.20346 — PLI.
- Vinck et al. 2010, *NeuroImage* 51:112, doi:10.1016/j.neuroimage.2010.01.073 — PPC; PLV trial-count bias.
- Vinck, Lima et al. 2010, *J Neurosci* 30:1250, doi:10.1523/JNEUROSCI.1623-09.2010 — the paper Rao cites for PPC.
- Vinck et al. 2011, *NeuroImage* 55:1548, doi:10.1016/j.neuroimage.2011.01.055 — wPLI (robustness option).
- Aydore et al. 2013, *NeuroImage* 74:231, doi:10.1016/j.neuroimage.2013.02.008 — PLV properties.
- Lowet et al. 2016, *PLoS One* 11:e0146443, doi:10.1371/journal.pone.0146443 — coherence vs PLV.
- Cohen 2015, *J Neurosci Methods* 250:137, doi:10.1016/j.jneumeth.2014.09.005 — effects of lag and frequency mismatch.
- Yoshinaga et al. 2020, *Front Neurosci* 14:648, doi:10.3389/fnins.2020.00648 — lag-based metrics for stimulus-induced coupling.
- Lachaux et al. 2000, *Int J Bifurcat Chaos* 10:2429, doi:10.1142/S0218127400001560 — single-trial PLV.
- Daffertshofer & van Wijk 2011, *Front Neuroinform* 5:6, doi:10.3389/fninf.2011.00006 — amplitude changes affect phase coupling (report power alongside ciPLV).

### Envelope coupling (HG AEC-c)
- Hipp et al. 2012, *Nat Neurosci* 15:884, doi:10.1038/nn.3101 — pairwise-orthogonalised AEC (mne envelope_correlation).
- Bruns et al. 2000, *NeuroReport* 11:1509, doi:10.1097/00001756-200005150-00029 — original AEC (subdural).
- Brookes et al. 2012, *NeuroImage* 63:910, doi:10.1016/j.neuroimage.2012.03.048; Colclough et al. 2015, *NeuroImage* 117:439, doi:10.1016/j.neuroimage.2015.03.071 — multivariate/symmetric leakage correction (pairwise orthogonalisation is asymmetric: say how it is symmetrised).
- O'Neill et al. 2015, *Phys Med Biol* 60:R271, doi:10.1088/0031-9155/60/21/R271; O'Neill et al. 2017, *NeuroImage* 146:667, doi:10.1016/j.neuroimage.2016.08.061; O'Neill et al. 2018, *NeuroImage* 180:559, doi:10.1016/j.neuroimage.2017.10.003 — AEC review; task-locked dynamic AEC; short-window trade-offs.
- Siems & Siegel 2020, *NeuroImage* 209:116538, doi:10.1016/j.neuroimage.2020.116538 — phase and amplitude coupling dissociate.
- Engel et al. 2013, *Neuron* 80:867, doi:10.1016/j.neuron.2013.09.038 — intrinsic coupling modes (different metric per band).
- Foster et al. 2015, *Neuron* 86:578, doi:10.1016/j.neuron.2015.03.018 — HG envelope coupling is task dependent (contrast with our null).
- Prathapagiri et al. 2026, *Nat Commun* 17:3996, doi:10.1038/s41467-026-70633-7 — global co-bursts of HFOs during word encoding (contrast).

### Pitfalls (volume conduction, leakage, transients)
- Bastos & Schoffelen 2016, *Front Syst Neurosci* 9:175, doi:10.3389/fnsys.2015.00175 — connectivity pitfalls tutorial.
- Schoffelen & Gross 2009, *Hum Brain Mapp* 30:1857, doi:10.1002/hbm.20745.
- Haufe et al. 2013, *NeuroImage* 67:318, doi:10.1016/j.neuroimage.2012.09.036 — simulation; mixing-robust measures win.
- Palva et al. 2018, *NeuroImage* 173:632, doi:10.1016/j.neuroimage.2018.02.032 — "ghost" edges next to real ones (caveat for the top-5 edges).
- Ewald et al. 2012, *NeuroImage* 60:476, doi:10.1016/j.neuroimage.2011.11.084; Nolte et al. 2008, *Phys Rev Lett* 100:234101, doi:10.1103/PhysRevLett.100.234101 — mixing-invariant measures; phase-slope index.
- Brunner et al. 2016, *Front Comput Neurosci* 10:121, doi:10.3389/fncom.2016.00121; Pellegrini et al. 2023, *NeuroImage* 277:120218, doi:10.1016/j.neuroimage.2023.120218.
- Kramer, Tort & Kopell 2008, *J Neurosci Methods* 170:352, doi:10.1016/j.jneumeth.2008.01.020 — sharp transients (evoked deflections) create spurious comodulation.
- Lozano-Soldevilla et al. 2016, *Front Comput Neurosci* 10:87, doi:10.3389/fncom.2016.00087 — waveform shape artefacts.
- Aru et al. 2015, *Curr Opin Neurobiol* 31:51, doi:10.1016/j.conb.2014.08.002 — safeguards for coupling metrics.
- Mostame et al. 2019, *Brain Topogr* 32:882, doi:10.1007/s10548-019-00718-8 — ECoG PLV significance with background couplings; bootstrap works best.
- Kramer et al. 2009, *Phys Rev E* 79:061916, doi:10.1103/PhysRevE.79.061916 — edge inference with FDR.
- Betzel et al. 2019, *Nat Biomed Eng* 3:902, doi:10.1038/s41551-019-0404-5 — RAM: distance shapes iEEG connectivity.

### Stimulus-locked (evoked) activity in connectivity — for the on-hold TODO item
Remedies recurring across these papers:
1. Matched condition contrast with a label-shuffle null (e.g., recalled vs forgotten).
2. Trial-shuffled (shift-predictor) surrogates.
3. Removing the evoked response per trial or with a flexible fit, not a plain average.
4. Lag-based metrics (these do not remove genuinely lagged evoked responses).
5. Reporting power and inter-trial phase consistency alongside connectivity.

- Truccolo et al. 2002, *Clin Neurophysiol* 113:206, doi:10.1016/S1388-2457(01)00739-8 — trial-varying ERP amplitude and latency leave a residual after average subtraction. Remedy: estimate the ERP per trial.
- Wang, Chen & Ding 2008, *NeuroImage* 42:751, doi:10.1016/j.neuroimage.2008.03.025 — same problem for coherence and Granger causality. Remedy: single-trial-adjusted removal.
- Knuth et al. 2006, *J Neurophysiol* 95:3257, doi:10.1152/jn.00663.2005 — dVCA: several evoked components with trial-specific amplitude and latency.
- Ding et al. 2000, *Biol Cybern* 83:35, doi:10.1007/s004229900137 — remove the mean, normalise per time point, short-window MVAR.
- Brovelli et al. 2004, *PNAS* 101:9849, doi:10.1073/pnas.0308538101 — applied example of the above.
- Kalcher & Pfurtscheller 1995, *EEG Clin Neurophysiol* 94:381, doi:10.1016/0013-4694(95)00040-6 — inter-trial variance method (canonical "remove the ERP").
- Cole et al. 2019, *NeuroImage* 189:1, doi:10.1016/j.neuroimage.2018.12.054 — task activations inflate connectivity; remove with a flexible (FIR) fit.
- Ito et al. 2020, *PLoS Comput Biol* 16:e1007983, doi:10.1371/journal.pcbi.1007983 — once evoked activity is removed, tasks usually reduce inter-areal correlations (fits the HG AEC-c null).
- van Diepen & Mazaheri 2018, *Sci Rep* 8:2990, doi:10.1038/s41598-018-20423-z — power or ERP changes alone can produce apparent phase-locking.
- David, Kilner & Friston 2006, *NeuroImage* 31:1580, doi:10.1016/j.neuroimage.2006.02.034 — evoked and induced cannot be cleanly separated (supports the PI's view).
- Shah et al. 2004, *Cereb Cortex* 14:476, doi:10.1093/cercor/bhh009 — ERPs reflect both added evoked activity and phase reset; test, don't assume.
- Makeig et al. 2002, *Science* 295:690, doi:10.1126/science.1066168 — alpha/theta phase reset in visual ERPs.
- Mazaheri & Jensen 2006, *PNAS* 103:2948, doi:10.1073/pnas.0505785103 — posterior alpha is NOT phase-reset (counterpoint; occipital ciPLV).
- Hanslmayr et al. 2007, *Cereb Cortex* 17:1, doi:10.1093/cercor/bhj129 — alpha phase reset contributes to ERPs.
- Sauseng et al. 2007, *Neuroscience* 146:1435, doi:10.1016/j.neuroscience.2007.03.014; Yeung et al. 2004, *Psychophysiology* 41:822, doi:10.1111/j.1469-8986.2004.00239.x — phase reset vs added evoked activity is hard to tell apart.
- Tallon-Baudry et al. 1996, *J Neurosci* 16:4240, doi:10.1523/JNEUROSCI.16-13-04240.1996 — evoked vs induced 40 Hz.
- Infantosi & Miranda de Sá 2006, *J Neurosci Methods* 156:267, doi:10.1016/j.jneumeth.2006.01.012; Benhamou et al. 2023, *IEEE TBME* 70:1324, doi:10.1109/TBME.2022.3222943 — split phase-locked from non-phase-locked power; link evoked power to inter-trial coherence.
- Perkel et al. 1967, *Biophys J* 7:419, doi:10.1016/S0006-3495(67)86597-4; Aertsen et al. 1989, *J Neurophysiol* 61:900, doi:10.1152/jn.1989.61.5.900 — shift predictor (origin of trial shuffling).
- Brody 1999, *Neural Comput* 11:1537, doi:10.1162/089976699300016133 (+ companion doi:10.1162/089976699300016124) — latency/excitability covariation survives shift-predictor correction.
- Ventura et al. 2005, *J Neurophysiol* 94:2928, doi:10.1152/jn.00644.2004 — model trial-specific gain and latency, then test the residual.
- Maris, Schoffelen & Fries 2007, *J Neurosci Methods* 163:161, doi:10.1016/j.jneumeth.2007.02.011 — permutation tests of coherence differences between conditions.
- Lepage et al. 2013, *J Comput Neurosci* 34:303, doi:10.1007/s10827-012-0422-8 — time-locked activity biases passive connectivity estimates (background).
- Dede et al. 2025, *PLoS Biol* 23:e3003481, doi:10.1371/journal.pbio.3003481 — connectivity aligned to stimulus vs internal HFA peaks (alternative to ERP subtraction).
- Colyer & Kahana 2025, *J Neurosci* 45:e2077242025, doi:10.1523/JNEUROSCI.2077-24.2025 — RAM: phase consistency during encoding (same data; read alongside this issue).

### Synchrony findings and theory
- Solomon et al. 2019, *Curr Biol* 29:1100, doi:10.1016/j.cub.2019.02.020 — dynamic MTL theta networks (RAM).
- Solomon et al. 2019, *PNAS* 116:24343, doi:10.1073/pnas.1906729116 — hippocampal theta and semantic/temporal distance (RAM).
- Solomon et al. 2018, *Nat Commun* 9:4437, doi:10.1038/s41467-018-06876-w — MTL connectivity predicts stimulation-induced theta.
- Adamovich-Zeitlin et al. 2021, *Brain Commun* 3:fcaa202, doi:10.1093/braincomms/fcaa202 — RAM encoding theta-synchrony window.
- Ezzyat et al. 2024, *Cereb Cortex* 34:bhad427, doi:10.1093/cercor/bhad427 — RAM connectivity predicts stimulation effects.
- Watrous et al. 2013, *Nat Neurosci* 16:349, doi:10.1038/nn.3315 — iEEG frequency-specific network increases; MTL hub.
- Ekstrom & Watrous 2014, *NeuroImage* 85:667, doi:10.1016/j.neuroimage.2013.06.049.
- Fell et al. 2001, *Nat Neurosci* 4:1259, doi:10.1038/nn759; Fell et al. 2003, *Eur J Neurosci* 17:1082, doi:10.1046/j.1460-9568.2003.02522.x; Fell et al. 2008, *NeuroImage* 43:410, doi:10.1016/j.neuroimage.2008.07.021; Fell & Axmacher 2011, *Nat Rev Neurosci* 12:105, doi:10.1038/nrn2979 — MTL phase synchrony in memory.
- Weiss & Rappelsberger 2000, *Cogn Brain Res* 9:299, doi:10.1016/S0926-6410(00)00011-2 — long-range synchrony during word encoding (scalp).
- Palva et al. 2010, *PNAS* 107:7580, doi:10.1073/pnas.0913113107; Palva & Palva 2007, *Trends Neurosci* 30:150, doi:10.1016/j.tins.2007.02.001; Palva & Palva 2011, *Front Psychol* 2:204, doi:10.3389/fpsyg.2011.00204 — alpha phase synchrony as active long-range coordination.
- Doesburg et al. 2009, *Brain Res* 1303:97, doi:10.1016/j.brainres.2009.09.069 — long-range alpha synchrony rises as local alpha changes (occipital ciPLV).
- Hanslmayr et al. 2007, *NeuroImage* 37:1465, doi:10.1016/j.neuroimage.2007.07.011 — pre-stimulus alpha phase coupling and perception.
- Zumer et al. 2014, *PLoS Biol* 12:e1001965, doi:10.1371/journal.pbio.1001965 — occipital alpha gates flow to object cortex.
- Bonnefond et al. 2017, *eNeuro* 4:ENEURO.0153-16.2017, doi:10.1523/ENEURO.0153-16.2017 — alpha as a communication channel.
- van Kerkoerle et al. 2014, *PNAS* 111:14332, doi:10.1073/pnas.1402773111; Bastos et al. 2015, *Neuron* 85:390, doi:10.1016/j.neuron.2014.12.018; Michalareas et al. 2016, *Neuron* 89:384, doi:10.1016/j.neuron.2015.12.018; Mejias et al. 2016, *Sci Adv* 2:e1601335, doi:10.1126/sciadv.1601335 — alpha/beta feedback, gamma feedforward (occipital alpha hubs 0–400 ms).
- Bosman et al. 2012, *Neuron* 75:875, doi:10.1016/j.neuron.2012.06.037 — selective inter-areal synchronization.
- Fries 2005, *TICS* 9:474, doi:10.1016/j.tics.2005.08.011; Fries 2015, *Neuron* 88:220, doi:10.1016/j.neuron.2015.09.034 — communication through coherence.
- Siegel, Donner & Engel 2012, *Nat Rev Neurosci* 13:121, doi:10.1038/nrn3137; Hipp, Engel & Siegel 2011, *Neuron* 69:387, doi:10.1016/j.neuron.2010.12.027 — spectral fingerprints; task synchrony networks.
- Engel, Fries & Singer 2001, *Nat Rev Neurosci* 2:704, doi:10.1038/35094565; Varela et al. 2001, *Nat Rev Neurosci* 2:229, doi:10.1038/35067550 — synchrony theory.
- Rodriguez et al. 1999, *Nature* 397:430, doi:10.1038/17120 — long-range synchrony in visual perception.
- Lachaux et al. 2000, *Eur J Neurosci* 12:2608, doi:10.1046/j.1460-9568.2000.00163.x — iEEG visual gamma, evoked vs induced.
- Bressler & Menon 2010, *TICS* 14:277, doi:10.1016/j.tics.2010.04.004; Helfrich & Knight 2016, *TICS* 20:916, doi:10.1016/j.tics.2016.09.007.
- Keller et al. 2011, *PNAS* 108:10308, doi:10.1073/pnas.1019750108; Keller et al. 2014, *Phil Trans R Soc B* 369:20130528, doi:10.1098/rstb.2013.0528; Keller et al. 2014, *J Neurosci* 34:9152, doi:10.1523/JNEUROSCI.4289-13.2014 — iEEG connectivity vs stimulation-evoked responses.

## 8. Time-resolved network and hubs
- Rubinov & Sporns 2010, *NeuroImage* 52:1059, doi:10.1016/j.neuroimage.2009.10.003 — node strength.
- Sporns 2014, *Nat Neurosci* 17:652, doi:10.1038/nn.3690 — hub definition (cited by Rao).
- Xia et al. 2013, *PLoS One* 8:e68910, doi:10.1371/journal.pone.0068910 — BrainNet Viewer (only if used).
- Das & Menon 2020, *Cereb Cortex* 30:5309, doi:10.1093/cercor/bhaa111 (**check** volume/pages); 2021, *J Neurosci* 41:8427, doi:10.1523/JNEUROSCI.0802-21.2021 (**check** volume/pages); 2022, *Cereb Cortex* 32:5343, doi:10.1093/cercor/bhac018; 2022, *Cortex* 147:24, doi:10.1016/j.cortex.2021.11.012; Das, de los Angeles & Menon 2022, *NeuroImage* 250:118927, doi:10.1016/j.neuroimage.2022.118927; 2023, *J Neurosci* 43:3159, doi:10.1523/JNEUROSCI.1728-22.2023; 2024, *Cereb Cortex* 34:bhae287, doi:10.1093/cercor/bhae287; 2024, *eLife* 13:RP99018, doi:10.7554/eLife.99018 — Stanford group on RAM data: directed connectivity and network dynamics (eLife 2024: 177 patients).
- Khambhati et al. 2019, *Netw Neurosci* 3:848, doi:10.1162/netn_a_00089; Stiso et al. 2019, *Cell Rep* 28:2554, doi:10.1016/j.celrep.2019.08.008 — Bassett lab RAM networks.
- Chaitanya et al. 2020, *Neuroscience* 425:194, doi:10.1016/j.neuroscience.2019.11.001 — HG-defined encoding network and hubs (Jefferson).
- Tan, Rugg & Lega 2020, *Neuropsychologia* 147:107595, doi:10.1016/j.neuropsychologia.2020.107595; Choi et al. 2020, *Cereb Cortex Commun* 1:tgaa010, doi:10.1093/texcom/tgaa010; Wang et al. 2021, *Hippocampus* 31:481, doi:10.1002/hipo.23309 — Lega lab networks (may be RAM data).
- Owen et al. 2020, *Cereb Cortex* 30:5333, doi:10.1093/cercor/bhaa115 (**check** volume/pages) — Gaussian-process model on Kahana data.
- Nozari et al. 2024, *Nat Biomed Eng* 8:68, doi:10.1038/s41551-023-01117-y — RAM, task-free segments.
- Gattas et al. 2023, *Nat Commun* 14:8505, doi:10.1038/s41467-023-44011-6 (**check** article number) — theta hippocampal-neocortical dynamics (UC Irvine).
- Gonzalez et al. 2015, *PNAS* 112:11066, doi:10.1073/pnas.1510749112 — parietal ECoG dynamics in recognition memory.

## 9. Statistics
- Benjamini & Hochberg 1995, *JRSS B* 57:289, doi:10.1111/j.2517-6161.1995.tb02031.x — BH FDR.
- Benjamini, Krieger & Yekutieli 2006, *Biometrika* 93:491, doi:10.1093/biomet/93.3.491 — two-stage FDR.
- Holm 1979, *Scand J Stat* 6:65 (no DOI) — Holm correction (used by Rao).
- Smith & Nichols 2009, *NeuroImage* 44:83, doi:10.1016/j.neuroimage.2008.03.061 — TFCE.
- Mensen & Khatami 2013, *NeuroImage* 67:111, doi:10.1016/j.neuroimage.2012.10.027 — TFCE for EEG.
- Nichols & Holmes 2002, *Hum Brain Mapp* 15:1, doi:10.1002/hbm.1058 — sign-flip permutation.
- Maris & Oostenveld 2007, *J Neurosci Methods* 164:177, doi:10.1016/j.jneumeth.2007.03.024 — cluster permutation.
- Groppe et al. 2011, *Psychophysiology* 48:1711, doi:10.1111/j.1469-8986.2011.01273.x — mass-univariate review.
- Holmes & Friston 1998, *NeuroImage* 7:S754, doi:10.1016/S1053-8119(18)31587-8 — subjects as the random effect.
- Welch 1947, *Biometrika* 34:28, doi:10.1093/biomet/34.1-2.28 — Welch t test.
- Cohen 1988, *Statistical Power Analysis* (reprint doi:10.4324/9780203771587); Lakens 2013, *Front Psychol* 4:863, doi:10.3389/fpsyg.2013.00863; Hedges 1981, *J Educ Stat* 6:107, doi:10.3102/10769986006002107 — Cohen's d and its small-sample correction.
- Rouder et al. 2009, *Psychon Bull Rev* 16:225, doi:10.3758/PBR.16.2.225 — JZS Bayes factors (to support the HG AEC-c null; Rao used them).
- Kriegeskorte et al. 2009, *Nat Neurosci* 12:535, doi:10.1038/nn.2303 — circular analysis: responsive electrodes are selected and then their latency is measured on the same data (say why this is not circular, or split the data).
- Button et al. 2013, *Nat Rev Neurosci* 14:365, doi:10.1038/nrn3475; Marek et al. 2022, *Nature* 603:654, doi:10.1038/s41586-022-04492-9 — sample size (value of N = 362).

## 10. Validation by simulation
- No canonical citation for pipelines validated with planted effects. Haufe 2013 and Mostame 2019 (simulated connectivity), Maris & Oostenveld 2007 and Smith & Nichols 2009 all validate methods with simulated effects.

## 11. Single-trial recall prediction (SME decoding)
- Ezzyat et al. 2018, *Nat Commun* 9:365, doi:10.1038/s41467-017-02753-0 — closed-loop classifier; benchmark AUCs.
- Kragel et al. 2017, *NeuroImage* 155:60, doi:10.1016/j.neuroimage.2017.03.042 — classifiers across encoding and retrieval.
- Arora et al. 2018, *J Neural Eng* 15:066028, doi:10.1088/1741-2552/aae131 — logistic vs SVM vs deep learning (justifies logistic regression).
- Meisler, Kahana & Ezzyat 2019, *J Neurosci Methods* 328:108421, doi:10.1016/j.jneumeth.2019.108421 — does data cleaning help classification (**check** author list).
- Weidemann & Kahana 2021, *JEP:LMC* 47:641, doi:10.1037/xlm0000966 — SME classifiers track endogenous state (key caveat).
- Weidemann et al. 2019, *JEP:General* 148:1, doi:10.1037/xge0000480 (correction doi:10.1037/xge0000604) — FR1 + catFR1 classifiers.
- Kahana, Aggarwal & Phan 2018, *JEP:LMC* 44:1857, doi:10.1037/xlm0000553 — variability limits prediction (AUC ceiling).
- Aka, Phan & Kahana 2021, *JEP:LMC* 47:765, doi:10.1037/xlm0000964 — behavioural ceiling; word/list covariates.
- Rubinstein et al. 2023 (above) — causal SME.
- Burke et al. 2015, *Front Hum Neurosci* 8:1055, doi:10.3389/fnhum.2014.01055 — early iEEG recall classifier/BCI.
- Kahana et al. 2023, *Brain Stimul* 16:1086, doi:10.1016/j.brs.2023.07.002 — classifier-guided stimulation in TBI.
- Ezzyat & Rizzuto 2018, *Curr Opin Biomed Eng* 8:78, doi:10.1016/j.cobme.2018.11.004 — review.
- Rudoler et al. 2024, *J Neurosci Methods* 410:110220, doi:10.1016/j.jneumeth.2024.110220; Li, Pazdera & Kahana 2024, *Nat Commun* 15:2981, doi:10.1038/s41467-024-46926-0; Dougherty et al. 2024, *JEP:LMC* 50:1404, doi:10.1037/xlm0001341 — scalp decoders.
- Halpern et al. 2025, *Nat Neurosci* 28:883, doi:10.1038/s41593-025-01884-8 — study-phase reinstatement predicts recall (RAM).
- Halpern et al. 2023, *PNAS* 120:e2120288120, doi:10.1073/pnas.2120288120 — identifying causal SMEs (behaviour).
- Sheehan et al. 2018, *J Neurosci* 38:1744, doi:10.1523/JNEUROSCI.2389-17.2017 — signal complexity as an SME feature.
- Wang et al. 2023, *Cereb Cortex* 33:8150, doi:10.1093/cercor/bhad105 — classifiers for electrode selection.
- Saboo et al. 2019, *Sci Rep* 9:17390, doi:10.1038/s41598-019-53925-5 — unsupervised classification of responsive electrodes (RAM).
- Varoquaux et al. 2017, *NeuroImage* 145:166, doi:10.1016/j.neuroimage.2016.10.038 — cross-validation practice (nested CV for C, TODO).

## 12. Anticipation / pre-word blank (plot_anticipation; baseline limitation)
No iEEG study found showing that a longer variable blank increases ventral-stream HG during verbal encoding, so this result may be new. Closest: Lima 2011 (monkey V1), Besle 2011 and Stefanics 2010 (human).
- Nobre & van Ede 2018, *Nat Rev Neurosci* 19:34, doi:10.1038/nrn.2017.141; Nobre, Correa & Coull 2007, *Curr Opin Neurobiol* 17:465, doi:10.1016/j.conb.2007.07.006 — temporal expectation; hazard rate.
- Niemi & Näätänen 1981, *Psychol Bull* 89:133, doi:10.1037/0033-2909.89.1.133 — variable-foreperiod effect.
- Janssen & Shadlen 2005, *Nat Neurosci* 8:234, doi:10.1038/nn1386; Ghose & Maunsell 2002, *Nature* 419:616, doi:10.1038/nature01057 — neural hazard coding.
- Lima, Singer & Neuenschwander 2011, *J Neurosci* 31:15919, doi:10.1523/JNEUROSCI.0957-11.2011 — V1 gamma↑ / alpha↓ when an event is temporally expected (closest analogue).
- Coull & Nobre 1998, *J Neurosci* 18:7426, doi:10.1523/JNEUROSCI.18-18-07426.1998; Coull & Nobre 2008, *Curr Opin Neurobiol* 18:137, doi:10.1016/j.conb.2008.07.011; Coull et al. 2011, *Neuropsychopharmacology* 36:3, doi:10.1038/npp.2010.113.
- Doherty et al. 2005, *J Neurosci* 25:8259, doi:10.1523/JNEUROSCI.1821-05.2005; Vangkilde et al. 2012, *JEP:HPP* 38:1183, doi:10.1037/a0026343.
- Rohenkohl & Nobre 2011, *J Neurosci* 31:14076, doi:10.1523/JNEUROSCI.3387-11.2011; Cravo et al. 2011, *J Neurophysiol* 106:2964, doi:10.1152/jn.00157.2011; Cravo et al. 2013, *J Neurosci* 33:4002, doi:10.1523/JNEUROSCI.4675-12.2013 — alpha/low-frequency activity follows temporal expectation.
- Besle et al. 2011, *J Neurosci* 31:3176, doi:10.1523/JNEUROSCI.4518-10.2011; Stefanics et al. 2010, *J Neurosci* 30:13578, doi:10.1523/JNEUROSCI.0703-10.2010; Lakatos et al. 2008, *Science* 320:110, doi:10.1126/science.1154735 — human ECoG/EEG expectation; entrainment.
- van Dijk et al. 2008, *J Neurosci* 28:1816, doi:10.1523/JNEUROSCI.1853-07.2008; Thut et al. 2006, *J Neurosci* 26:9494, doi:10.1523/JNEUROSCI.0875-06.2006; Ergenoglu et al. 2004, *Cogn Brain Res* 20:376, doi:10.1016/j.cogbrainres.2004.03.009; Busch et al. 2009, *J Neurosci* 29:7869, doi:10.1523/JNEUROSCI.0113-09.2009; Mathewson et al. 2009, *J Neurosci* 29:2725, doi:10.1523/JNEUROSCI.3963-08.2009; Iemi et al. 2017, *J Neurosci* 37:807, doi:10.1523/JNEUROSCI.1432-16.2016; Samaha et al. 2020, *TICS* 24:639, doi:10.1016/j.tics.2020.05.004 — pre-stimulus alpha and perception.
- Fiebelkorn et al. 2013, *Curr Biol* 23:2553, doi:10.1016/j.cub.2013.10.063; Fiebelkorn et al. 2018, *Neuron* 99:842, doi:10.1016/j.neuron.2018.07.038; Fiebelkorn & Kastner 2019, *TICS* 23:87, doi:10.1016/j.tics.2018.11.009; Fiebelkorn et al. 2011, *J Neurosci* 31:9971, doi:10.1523/JNEUROSCI.1338-11.2011 — rhythmic attention; phase reset.
- Jones & Ward 2019, *J Cogn Neurosci* 31:1549, doi:10.1162/jocn_a_01431 — rhythmic temporal structure helps encoding.

### Pre-stimulus activity predicting memory (pre-word decoder features)
- Merkow et al. 2014, *Hippocampus* 24:1562, doi:10.1002/hipo.22335 — Kahana lab: pre-stimulus hippocampal theta predicts recognition, not recall.
- Fell et al. 2011, *J Neurosci* 31:5392, doi:10.1523/JNEUROSCI.3668-10.2011 — iEEG MTL theta/alpha before successful encoding.
- Haque et al. 2015 (above) — cortical low-frequency power before encoding.
- Otten et al. 2006, *Nat Neurosci* 9:489, doi:10.1038/nn1663; Guderian et al. 2009, *PNAS* 106:5365, doi:10.1073/pnas.0900289106; Park & Rugg 2010, *Hippocampus* 20:24, doi:10.1002/hipo.20663; Addante et al. 2011, *PNAS* 108:10702, doi:10.1073/pnas.1014528108; Gruber et al. 2013, *NeuroImage* 64:68, doi:10.1016/j.neuroimage.2012.07.064; Salari & Rose 2016, *NeuroImage* 125:1013, doi:10.1016/j.neuroimage.2015.10.037; Schneider & Rose 2016, *NeuroImage* 125:978, doi:10.1016/j.neuroimage.2015.11.024 — non-iEEG pre-stimulus SMEs.

## 13. Reliability and multiverse (next analysis)
- Finn et al. 2015, *Nat Neurosci* 18:1664, doi:10.1038/nn.4135 — fingerprinting / identifiability.
- da Silva Castanheira et al. 2021, *Nat Commun* 12:5713, doi:10.1038/s41467-021-25895-8 — MEG spectral fingerprints.
- Wiesman et al. 2022, *NeuroImage* 247:118823, doi:10.1016/j.neuroimage.2021.118823 — data needed for stable spectra.
- Colclough et al. 2016, *NeuroImage* 138:284, doi:10.1016/j.neuroimage.2016.05.070 — orthogonalised AEC most reliable; lag-based phase metrics less so (ciPLV vs AEC-c).
- Colclough et al. 2017, *eLife* 6:e20178, doi:10.7554/eLife.20178 — heritability / subject specificity of connectivity.
- Martín-Buro et al. 2016, *Hum Brain Mapp* 37:179, doi:10.1002/hbm.23027 — test-retest of MEG power.
- Demuru & Fraschini 2020, *Comput Biol Med* 120:103748, doi:10.1016/j.compbiomed.2020.103748 — aperiodic fingerprint.
- Lopez et al. 2023, *NeuroImage* 275:120116, doi:10.1016/j.neuroimage.2023.120116 — EEG reliability review.
- Noble et al. 2019, *NeuroImage* 203:116157, doi:10.1016/j.neuroimage.2019.116157; Elliott et al. 2020, *Psychol Sci* 31:792, doi:10.1177/0956797620916786 — fMRI reliability (task effects are poorly reliable).
- Koo & Li 2016, *J Chiropr Med* 15:155, doi:10.1016/j.jcm.2016.02.012; Shrout & Fleiss 1979, *Psychol Bull* 86:420, doi:10.1037/0033-2909.86.2.420 — ICC.
- Kucyi et al. 2018, *J Neurosci* 38:4230, doi:10.1523/JNEUROSCI.0217-18.2018 — reproducible iEEG connectivity across sessions.
- Wang et al. 2023, *Epilepsia* 64:2070, doi:10.1111/epi.17663 — iEEG band-power maps stable across days.
- Papasavvas et al. 2020, *J Neural Eng* 17:054001, doi:10.1088/1741-2552/abbecf — RAM: cross-session consistency of stimulation effects on band power.
- Taylor et al. 2022, *Brain* 145:939, doi:10.1093/brain/awab380; Bernabei et al. 2022, *Brain* 145:1949, doi:10.1093/brain/awab480 (**check** volume/pages) — normative iEEG maps (RAM used as a normative baseline in Taylor).
- Steegen et al. 2016, *Perspect Psychol Sci* 11:702, doi:10.1177/1745691616658637 — multiverse.
- Simonsohn et al. 2020, *Nat Hum Behav* 4:1208, doi:10.1038/s41562-020-0912-z — specification curve.
- Botvinik-Nezer et al. 2020, *Nature* 582:84, doi:10.1038/s41586-020-2314-9 — NARPS.
- Clayson et al. 2021, *NeuroImage* 245:118712, doi:10.1016/j.neuroimage.2021.118712 — ERP multiverse.
- Carp 2012, *Front Neurosci* 6:149, doi:10.3389/fnins.2012.00149 — analytic flexibility.

## 14. Epilepsy and data quality (likely reviewer questions)
- Horak et al. 2017, *Epilepsia* 58:373, doi:10.1111/epi.13633 — RAM: interictal discharges impair recall.
- Ung et al. 2017, *Brain* 140:2157, doi:10.1093/brain/awx143 — interictal activity outside the seizure onset zone.
- Camarillo-Rodriguez et al. 2022, *Epilepsia* 63:2325, doi:10.1111/epi.17334 — RAM: temporal spikes disrupt encoding.
- Meisenhelter et al. 2021, *Cereb Cortex Commun* 2:tgab019, doi:10.1093/texcom/tgab019; Quon et al. 2021, *Epilepsia* 62:481, doi:10.1111/epi.16792; Quon et al. 2021, *Epilepsia* 62:2615, doi:10.1111/epi.17060 — RAM interictal discharges.
- Waldman et al. 2018, *Epilepsy Behav* 88:33, doi:10.1016/j.yebeh.2018.08.018 — RAM: neocortical ripples and encoding.
- Saboo et al. 2021, *Epilepsia* 62:2627, doi:10.1111/epi.17067 — RAM: encoding features differ in the seizure onset zone.
- Goldstein et al. 2019, *J Neural Eng* 16:066039, doi:10.1088/1741-2552/ab4365 (**check** volume/pages) — RAM stimulation safety.

## 15. Other RAM / Penn papers (dataset context, lower relevance)
- Retrieval and reinstatement: Kragel et al. 2021, *Nat Commun* 12:4444, doi:10.1038/s41467-021-24393-1; Herweg et al. 2020, *J Neurosci* 40:2119, doi:10.1523/JNEUROSCI.1640-19.2019; Long et al. 2017, *Curr Biol* 27:1074, doi:10.1016/j.cub.2017.02.054; Manning et al. 2011, *PNAS* 108:12893, doi:10.1073/pnas.1015174108; Manning et al. 2012, *J Neurosci* 32:8871, doi:10.1523/JNEUROSCI.5321-11.2012; Morton et al. 2013, *Cereb Cortex* 23:2407, doi:10.1093/cercor/bhs229; Herz et al. 2023, *PNAS* 120:e2305292120, doi:10.1073/pnas.2305292120; Broitman & Kahana 2026, *Psychol Sci* 37:318, doi:10.1177/09567976261417325.
- Ripples, single units and MTL: Sakon & Kahana 2022, *PNAS* 119:e2201657119, doi:10.1073/pnas.2201657119; Sakon et al. 2024, *J Neurosci* 44:e0111232023, doi:10.1523/JNEUROSCI.0111-23.2023; Schonhaut et al. 2024, *eLife* 13:e85753, doi:10.7554/eLife.85753; Herweg et al. 2023, *J Neurosci* 43:3538, doi:10.1523/JNEUROSCI.0181-22.2023; Umbach et al. 2020, *PNAS* 117:28463, doi:10.1073/pnas.2013250117; Umbach et al. 2022, *Nat Commun* 13:6162, doi:10.1038/s41467-022-33587-0; Qasim et al. 2023, *Nat Hum Behav* 7:754, doi:10.1038/s41562-022-01502-8; Gedankien et al. 2023, *Nat Commun* 14:5283, doi:10.1038/s41467-023-41025-y; Hill et al. 2021, *J Neurosci* 41:6343, doi:10.1523/JNEUROSCI.0278-21.2021; Miller et al. 2018, *Nat Commun* 9:2423, doi:10.1038/s41467-018-04847-9.
- Stimulation: Jacobs et al. 2016, *Neuron* 92:983, doi:10.1016/j.neuron.2016.10.062; Kucewicz et al. 2018, *Brain* 141:971, doi:10.1093/brain/awx373; Solomon et al. 2021, *Brain Stimul* 14:1271, doi:10.1016/j.brs.2021.08.014; Mohan et al. 2020, *Brain Stimul* 13:1183, doi:10.1016/j.brs.2020.05.009; Goyal et al. 2018, *J Neurosci* 38:4471, doi:10.1523/JNEUROSCI.3049-17.2018; Lech et al. 2021, *IEEE TBME* 68:3701, doi:10.1109/TBME.2021.3082320; Acharya et al. 2024, *Commun Biol* 7:1210, doi:10.1038/s42003-024-06859-2; Kucewicz et al. 2023, *Brain* 146:2214, doi:10.1093/brain/awac435; Kucewicz et al. 2024, *Brain* 147:2966, doi:10.1093/brain/awae159; Cole et al. 2026 **pre**, medRxiv, doi:10.64898/2026.01.10.26343861.
- Theory: Howard & Kahana 1999, *JEP:LMC* 25:923, doi:10.1037/0278-7393.25.4.923; Kahana 2020, *Annu Rev Psychol* 71:107, doi:10.1146/annurev-psych-010418-103358.

## 16. Open datasets (cross-cohort replication option)
- Cimbalnik et al. 2025, *Sci Data* 12:899, doi:10.1038/s41597-025-05222-2 — open iEEG battery including free recall (best replication candidate).
- Boran et al. 2020, *Sci Data* 7:30, doi:10.1038/s41597-020-0364-3; Boran et al. 2019, *Sci Adv* 5:eaav3687, doi:10.1126/sciadv.aav3687 — Sternberg iEEG and single units.
- Faraut et al. 2018, *Sci Data* 5:180010, doi:10.1038/sdata.2018.10; Kyzar et al. 2024, *Sci Data* 11:89, doi:10.1038/s41597-024-02943-8 — MTL single units.
- Miller 2019, *Nat Hum Behav* 3:1225, doi:10.1038/s41562-019-0678-3 — ECoG library.
- Berezutskaya et al. 2022, *Sci Data* 9:91, doi:10.1038/s41597-022-01173-0 — iEEG + fMRI film dataset.

## Corrections to names I gave the searches (and items not found)
- "Weidemann 2019 PNAS" does not exist → Weidemann & Kahana 2021 *JEP:LMC*, or Weidemann et al. 2019 *JEP:General*.
- Thesen 2012 is *Nat Commun*, not PNAS. "Gaillard 2009 reading" is *Neuron* 2006 (Gaillard 2009 *PLoS Biol* is the conscious-access paper). Kadipasaoglu 2016 is *PLoS One*. Zhang 2018 travelling waves is *Neuron*. Babadi & Brown 2014 is *IEEE TBME*. Colclough heritability is 2017.
- Not found: Kahana 2023 *Nat Hum Behav*; "Solomon 2021 causal connectivity"; RAM electrode-localization paper (Davis/Geller); Ezzyat 2022; a Fallon/Kahana iEEG reliability paper; a Betzel 2019 iEEG reliability paper (that one is fMRI); a Mehrkanoon 2014 evoked paper; "Daffertshofer 2018"; a Jacobs/Kahana 2012 single-unit word paper.
