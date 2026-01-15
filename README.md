# Market-Regime-Classifier
## Introduction
This project investigates market regime classification using statistical and machine learning methods from LSEG's Blueprint Pipeline [1], utilising the Refinitiv API for data access on the ESc1 futures contract. Focused on Hidden Markov Models (HMMs) and Gaussian Mixture Models (GMMs), this endeavour introduced an *entropy-gated* post-model diagnostic layer to stabilise market regime classification transitions by suppressing spurious switching in low-volatility conditions (based on model classification certainties), while attempting to preserve responsiveness during genuine market regime shifts.  

The Shannon-Entropy-based gating mechanism is evaluated with custom metrics that quantify reductions in noise-driven regime changes and the sensitivity of the gated models relative to fully trained baselines. The aim is to address a core failure mode of regime models: instability in structurally quiet markets without sacrificing adaptability in turbulent market conditions. 

## Background
A market regime represents a distinct period in which the market exhibits a relatively stable pattern of behaviour, typically characterised by a combination of trend direction and volatility level. Within a given regime, tailored trading or allocation strategies tend to yield more consistent statistical characteristics; however, their efficacy does not necessarily translate across other regimes. 
Since different strategies operate more effectively in particular market regimes, greater confidence in the inferred classification of the market's behaviour (and, thus, fewer unnecessary label flips) provides more robust signals to regime- or state-dependent decision systems. 

Consequently, mitigating over-reaction to noise can reduce unnecessary reallocations of capital and hedging activity triggered by transient uncertainties about how the current market environment, which helps lower trading friction and losses.

Furthermore, in the context of regime identification, HMMs and GMMS are commonly applied to market regime detection and provide complementary perspectives - where HMMs are used to model latent market states, while GMMs cluster observations purely based on statistical distribution.

## Methodology
This model retained the feature engineering component of LSEG's Blueprint Pipeline; therefore, the models were trained on the log returns of the ESc1 futures contract. 

Although introducing look-ahead bias, baseline models were obtained by training an HMM and a GMM on the entire dataset of the historical price data of the instrument; these models serve as the benchmark for each model's ideal performance.

To simulate the forecast performance of both models, a Feed-Forward-Training (FFT) derivative of these models was implemented. The Shannon-Entropy of both the benchmark and FFT models was measured, enabling the identification of the baseline confidence values of the models.

Subsequently, the post-model diagnostic layer was implemented onto the FFT variants of the HMM and GMM, suppressing regime switches when the model's entropy exceeded a specified uncertainty threshold for a given prediction.

Custom metrics were defined to (1) count whipsaw classification errors and (2) measure the latency of regime transitions relative to objectively identified market condition shifts, to quantify the reduction in noise-driven decisions and the model's retained sensitivity to genuine market changes between the FFT and FFT-entropy gated versions of the models.

A Pareto compromise enabled the identification of a more robust entropy-threshold value for the entropy-gating mechanism. This was achieved by iterating through a range of potential entropy threshold values and assessing the subsequent models in accordance with the aforementioned custom metrics. An identified threshold value provided a balance between latency and effectiveness in reducing spurious classification switches overall. 

## Results (summary)
The HMM observed higher confidence in market regime classification than the GMM, and the FFT variant of the GMM showed a larger increase in classification uncertainty relative to its baseline than the corresponding HMM variant.

As a consequence of this greater classification uncertainty, the entropy-gating layer delivered more value for the GMM than for the already more confident HMM, improving the stability of the GMM's regime labels.

Using an entropy threshold selected via a Pareto-style compromise between whipsaw reduction and detection speed, the gated GMM variant eliminated all identified whipsaw classification error at the expense of (approximately) doubling the classification latency relative to its non-gated counterpart.

For more information, please refer to the Market-Regime-Classification Notebook.

## References
[1] https://github.com/LSEG-API-Samples/Article.RD.Python.MarketRegimeDetectionUsingStatisticalAndMLBasedApproaches/blob/main/Market%20regime%20detection.ipynb
