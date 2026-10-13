# Financial Time Series: Scaling and Predictability

**MSc thesis, literature review, empirical notebooks, and synthetic experiments connecting econophysics, information theory, statistical modeling, and machine learning.**

[![Thesis](https://img.shields.io/badge/MSc-Statistical_Physics_%26_Complex_Systems-16324F)](#thesis-and-research-scope)
[![Research](https://img.shields.io/badge/Research-Scaling_%26_Predictability-176B87)](#main-research-synthetic-dynamics-information-and-predictability)
[![Python](https://img.shields.io/badge/Code-Python_Notebooks-3776AB?logo=python&logoColor=white)](#repository-guide)
[![Reports](https://img.shields.io/badge/Reports-English_%26_Persian-8B5E3C)](#reports-and-presentations)

**Author:** [Ali Khabazian](https://github.com/Alikhabazian0)  
**University:** Shahid Beheshti University, Faculty of Physics  
**Field:** Statistical Physics and Complex Systems  
**Supervisor:** Dr. Ali Hosseiny  
**Thesis:** *The Role of Scaling Behavior in Time-Series Analysis and Predictability*

[Read the English thesis](Reports/Final%20Thesis/English_Translated_Report.pdf) · [Read the Persian thesis](Reports/Final%20Thesis/Persian_Report.pdf) · [View the defense presentation](Reports/Final%20Thesis/Thesis%20Presentation.pdf)

## Contents

- [Thesis and research scope](#thesis-and-research-scope)
- [Literature review: stylized facts in finance and crypto](#literature-review-stylized-facts-in-finance-and-crypto)
- [Scaling and multiscaling: the Di Matteo framework](#scaling-and-multiscaling-the-di-matteo-framework)
- [Traditional time-series modeling and tail risk](#traditional-time-series-modeling-and-tail-risk)
- [Main research: synthetic dynamics, information, and predictability](#main-research-synthetic-dynamics-information-and-predictability)
- [Repository guide](#repository-guide)
- [Reports and presentations](#reports-and-presentations)
- [Working with the notebooks](#working-with-the-notebooks)
- [Selected references](#selected-references)

## Thesis and research scope

This work investigates how the **scaling structure**, **statistical complexity**, and **temporal information** of a time series relate to its predictability. Financial markets and cryptocurrencies provide the empirical setting, while controlled synthetic processes allow the roles of noise and temporal dependence to be studied separately.

The thesis develops three connected strands:

| Strand | Main questions | Methods |
|---|---|---|
| Empirical econophysics | Which statistical regularities and scaling behaviors appear in financial and cryptocurrency data? | Stylized facts, generalized Hurst exponents, moment scaling, spectra, and rank-size analysis. |
| Traditional modeling | How can dependence, volatility, and extreme losses be modeled? | AR/ARIMA concepts, GARCH-family models, Student-t innovations, Monte Carlo simulation, and EVT. |
| Main simulation research | How do increasing noise and structured temporal dependence change the information available for prediction? | Synthetic stochastic processes, Shannon and block entropy, regression, classification, and neural forecasting models. |

The central distinction is between **uncertainty in individual observations** and **information carried by temporal structure**. A series can have high single-symbol entropy while retaining predictable dependencies across longer sequences.

## Literature review: stylized facts in finance and crypto

Chapter 2 combines a literature review with empirical illustrations for conventional financial assets and cryptocurrencies. Its notebooks examine properties that simple independent Gaussian-return assumptions do not fully describe:

- **Heavy-tailed returns:** distributional comparisons, excess kurtosis, and tail/exceedance plots.
- **Aggregational Gaussianity:** changes in return distributions as the aggregation horizon increases.
- **Weak linear return autocorrelation:** correlation diagnostics for signed returns.
- **Volatility clustering:** persistent dependence in absolute or squared returns and conditional heteroskedasticity.
- **Differences across assets and sampling choices:** comparisons between stock-market examples and cryptocurrency returns.

The analysis uses log returns,

$$
r_t=\log P_t-\log P_{t-1},
$$

and aggregated returns over a horizon $\tau$,

$$
r_t^{(\tau)}=\log P_{t+\tau}-\log P_t.
$$

The notebooks combine visual diagnostics with tests such as Jarque-Bera, Ljung-Box, ARCH-LM, and unit-root checks where appropriate. Data periods and assets vary across notebook sections rather than forming a single universal sample.

### Return aggregation

![Apple return aggregation compared with Gaussian references](Chapter%202%20-%20Stylized%20Facts/Results/s2p7.png)

*Apple return-exceedance distributions at 1-, 5-, 20-, and 60-day horizons are compared with Gaussian reference curves to examine aggregational Gaussianity.*

## Scaling and multiscaling: the Di Matteo framework

The literature review **primarily follows the work of Professor Tiziana Di Matteo**, Professor of Econophysics in the Department of Mathematics at **King's College London**, and her collaborators. Her work on generalized Hurst exponents, market-dependent scaling, and comparisons between time- and frequency-domain behavior provides the main methodological foundation for this part of the thesis.

The review particularly draws on *Multi-scaling in finance*, the earlier scaling analyses with Tomaso Aste and Michel M. Dacorogna, and the work with Riccardo Junior Buonocore on measuring multiscaling. These are cited below. The repository contains my literature-review implementations and empirical applications, including extensions of the discussion to cryptocurrencies.

### Generalized Hurst exponents

For a selected series $X(t)$, the generalized moment ratio is written as:

$$
K_q(\tau)=
\frac{\left\langle|X(t+\tau)-X(t)|^q\right\rangle}
{\left\langle|X(t)|^q\right\rangle}
\propto\tau^{qH(q)}.
$$

A log-log fit estimates the moment-scaling exponent:

$$
\zeta(q)=qH(q).
$$

The notebooks examine how $H(q)$ changes with moment order, compare estimated slopes across markets, and evaluate the sensitivity to scale ranges and preprocessing. A nonlinear dependence of $\zeta(q)$ on $q$ motivates multiscaling analysis, while finite samples, heavy tails, and aggregation choices must be considered when interpreting the result.

The included `di_matteo_data/` files support historical illustrations involving the Nikkei 225, Bangkok SET, and JPY/USD. Other notebook sections use downloaded equity and cryptocurrency data.

![Generalized Hurst moment scaling for Nikkei 225 and SET](Chapter%203%20-%20Scaling%20and%20Multiscaling/Results/GHE%20Plot.png)

*Log-log generalized-moment scaling for the Nikkei 225 and Thailand's SET index compares the fitted slopes for orders $q=1,2,3$ over the illustrated 1990-2001 sample.*

### Frequency-domain comparison

Spectral analysis provides a complementary description through:

$$
S(f)\propto f^{-\beta}.
$$

The notebooks compare empirical spectra with GHE-based reference slopes. Relations between $\beta$ and $H(2)$ depend on the process and whether levels or increments are analyzed; the level-process reference $\beta=2H(2)+1$ is not a universal relation for every transformed series. Rank-size/Zipf plots provide another empirical perspective on scaling and tails.

![Nikkei 225 power spectrum and a GHE-based power-law reference](Chapter%203%20-%20Scaling%20and%20Multiscaling/Results/Power%20Spectrum%20Plot.png)

*The Nikkei 225 power spectrum is compared with a GHE-based power-law reference to examine consistency between time-domain and frequency-domain scaling descriptions.*

## Traditional time-series modeling and tail risk

Chapter 4 develops a statistical modeling workflow for **Gold and Bitcoin**, aligning Bitcoin observations with Gold trading dates for comparison. The workflow combines dependence diagnostics, conditional-volatility modeling, simulation, and tail-risk estimation.

### Modeling workflow

1. Prepare prices and log returns and inspect their distributions and stationarity.
2. Examine mean dependence using ACF/PACF and Ljung-Box diagnostics.
3. Examine volatility dependence using squared-return correlations and ARCH-LM tests.
4. Compare candidate mean, volatility, and innovation specifications using fit criteria and residual diagnostics.
5. Simulate future return and price paths and estimate downside risk.
6. Fit extreme-loss exceedances using a peaks-over-threshold approach.

The notebook explores GARCH-family specifications and Student-t innovations. Its final simulation specifications include **AR(1)-GJR-GARCH(1,1)-t for Gold** and **AR(3)-EGARCH(1,1)-t for Bitcoin**. These choices describe the stored analysis and should be reassessed for a new sample.

The simulation section uses **5,000 paths over a 30-step horizon**, reporting lower-tail return quantiles and expected shortfall at selected confidence levels. Monte Carlo cumulative-return risk and daily-loss EVT calculations refer to different horizons and should retain their original units and definitions.

### Extreme-value analysis

For loss $L$ above threshold $u$, the excess $Y=L-u$ is modeled with a generalized Pareto distribution:

$$
G(y)=1-\left(1+\xi\frac{y}{\beta}\right)^{-1/\xi},
$$

on its valid support, with the exponential limit at $\xi=0$. The notebook uses a 95th-percentile threshold by default and compares empirical exceedances with fitted GPD tails.

![Gold and Bitcoin extreme-loss exceedances and GPD fits](Chapter%204%20-%20Statistical%20Models/Results/EVT%20Exceedances%20and%20GPD%20Fit.png)

*Gold and Bitcoin loss exceedances are compared with fitted generalized Pareto distributions to characterize extreme downside tails.*

![Gold and Bitcoin forecasts compared with realized prices](Chapter%204%20-%20Statistical%20Models/Results/Forecast%20vs%20Realized%20Plot.png)

*Simulated Gold and Bitcoin price paths, mean forecasts, and 95% VaR reference scenarios are displayed alongside realized prices after the forecast origin.*

## Main research: synthetic dynamics, information, and predictability

**Chapter 6 is the main simulation contribution of the thesis.** It uses processes with controlled noise or explicitly constructed temporal dependencies to examine what forecasting models can learn and how that learning relates to information-theoretic quantities.

### Models and evaluation

| Model family | Examples examined in the thesis | Purpose |
|---|---|---|
| Traditional statistical models | ARIMA and simple stochastic baselines | Reference forecasts and temporal-dependence modeling. |
| Machine learning | XGBoost | A nonlinear tree-based comparison. |
| Neural network models | LSTM, GRU, and TCN | Recurrent and convolutional sequence learning. |
| Transformer family | Causal encoder, residual causal Transformer, and PatchTST | Attention-based context modeling and alternative representations of temporal structure. |

The study uses regression and binary direction classification, evaluating **RMSE**, **NRMSE**, **directional accuracy**, confusion matrices, reconstructed paths, and probability calibration as appropriate to each experiment. Architectural differences are considered together with the information supplied by the generating process.

### Experiment 1: increasing noise relative to drift

The first experiment uses a multiplicative stochastic process:

$$
dx_t=\mu x_t\,dt+g x_t\,dW_t,
$$

with an Euler-style relative-return approximation:

$$
r_{t+1}\approx\mu\Delta t+g\sqrt{\Delta t}\,\epsilon_t.
$$

The thesis examines **19 noise-to-drift settings**, with $g/\mu$ from 0 to 9 in increments of 0.5. Models are trained and evaluated on these controlled series to compare increasing uncertainty with forecast error and directional performance.

Shannon entropy is used to describe the uncertainty of the return distribution:

$$
H=-\sum_i p_i\log_2 p_i.
$$

The reported experiment shows increasing entropy and forecast error as noise grows, while raw directional accuracy approaches the chance benchmark of 0.5. Forecasted path reconstruction additionally shows how return errors accumulate through compounding.

![Normalized entropy and normalized directional accuracy under increasing noise](Chapter%206%20-%20Simulations%20and%20Results/Some%20Results/Fig.%2027%20-%20Normalized%20Shannon%20Entropy%20and%20Directional%20Accuracy%20vs%20Noise-to-Drift%20Ratio.png)

*Figure 27: Normalized Shannon entropy rises while normalized directional accuracy declines across forecasting models as the noise-to-drift ratio increases.*

Both entropy and directional accuracy in this figure are min-max normalized over the displayed sweep. Consequently, a low plotted accuracy value does not mean that raw directional accuracy is near zero, and curve intersections are descriptive comparisons rather than universal predictability thresholds.

### Experiment 2: structured symbolic sequences and temporal redundancy

The second experiment constructs eight-symbol blocks from three independent binary variables:

$$
B_k=[U_k,V_k,W_k,V_k,U_k,V_k,W_k,W_k],
\qquad U_k,V_k,W_k\in\{-1,+1\}.
$$

Eight block types are selected independently and uniformly. A sequence of 1,000 blocks produces 8,000 symbols, which generate a multiplicative trajectory:

$$
x_{t+1}=x_t(1+\mu s_t),
\qquad \mu=0.01.
$$

The relative returns are $\pm0.01$. The reported modeling protocol uses a **70% training / 15% validation / 15% test split**, providing return histories without engineered block identities or positions.

This construction introduces three fresh random bits per eight symbols. Its aligned-block entropy is therefore three bits, and its asymptotic entropy rate is:

$$
h_\infty=\frac{3}{8}=0.375\ \text{bits per symbol}.
$$

Single-symbol entropy is approximately one bit, so the asymptotic relative redundancy is approximately $1-0.375=0.625$. This provides a controlled example of substantial temporal information despite high marginal uncertainty.

Sliding-word block entropy measures the uncertainty of consecutive words:

$$
H(L)=-\sum_w p_L(w)\log_2 p_L(w),
\qquad
h(L)=H(L)-H(L-1).
$$

Sliding words can cross the generating block boundaries, so their finite-length entropy differs from the entropy of aligned eight-symbol blocks. The experiment compares theoretical and empirical estimates and identifies downward finite-sample bias when long words become sparsely observed.

![Theoretical and empirical sliding-block entropy per symbol](Chapter%206%20-%20Simulations%20and%20Results/Some%20Results/Fig.%2030%20%E2%80%93%20Convergence%20of%20Sliding%20Block%20Entropy%20per%20Symbol.png)

*Figure 30: Theoretical and empirical sliding-block entropy per symbol are compared with the asymptotic rate $3/8$, highlighting sparse-word estimation bias beyond the illustrated $L=12$ region.*

The same process is studied through continuous-return regression and binary direction classification. In the reported classification experiment, PatchTST reaches approximately 82% directional accuracy, with GRU and LSTM near 80%; those results refer to this constructed process and configuration.

### Probability calibration

Classification accuracy and probability reliability answer different questions. The thesis compares observed positive-class frequencies with predicted probabilities and examines the distribution of each model's confidence scores.

![Reliability diagrams and probability distributions for four classifiers](Chapter%206%20-%20Simulations%20and%20Results/Some%20Results/Fig.%2032%20%E2%80%93%20Reliability%20diagram%20of%20classification%20task.png)

*Figure 32: Reliability curves and predicted-probability histograms compare GRU, PatchTST, TCN, and XGBoost, separating classification confidence from empirical calibration.*

In the illustrated comparison, XGBoost tracks the calibration diagonal more closely across much of its probability range, while GRU and PatchTST frequently produce extreme probabilities. Thus, a model's directional accuracy should be considered together with the reliability of its confidence estimates.

### Experiment 3: RSI-guided Boltzmann block selection

The third experiment replaces uniform block selection with history-dependent probabilities inspired by statistical physics. A count-based RSI is computed from the preceding 14 binary symbols. For candidate block $b$, define its compounded return:

$$
R_b=\prod_{j=1}^{8}(1+\mu s_{b,j})-1,
$$

and an energy-like quantity:

$$
E_b(t)=(\mathrm{RSI}_t-50)R_b.
$$

The next block is selected according to:

$$
p_b(t)=\frac{\exp[-E_b(t)/(KT)]}
{\sum_c\exp[-E_c(t)/(KT)]}.
$$

The temperature-like parameter controls how sharply recent directional imbalance affects block selection. The thesis examines $KT\in\{0.7,1,2,2.8\}$, comparing block frequencies, entropy estimates, and classifier performance.

This mechanism introduces a tunable temporal dependence: blocks opposing a recent directional imbalance receive larger probabilities under the energy definition. The reported experiments show that the additional structure can be learned by sequence models, with the effect depending on both temperature and architecture. These are results for the specified synthetic mechanism, rather than a fitted thermodynamic model of an actual market.

### Main scientific perspective

Taken together, the experiments connect three aspects of predictability: noise can reduce directional information, temporal redundancy can make high-marginal-entropy sequences learnable, and history-dependent generation can change both entropy and model performance. Scaling analysis, information measures, forecast accuracy, and calibration provide complementary descriptions rather than interchangeable indicators.

## Repository guide

| Location | Contents |
|---|---|
| [`Chapter 2 - Stylized Facts/`](Chapter%202%20-%20Stylized%20Facts/) | Empirical financial/crypto notebooks and stylized-fact figures. |
| [`Chapter 3 - Scaling and Multiscaling/`](Chapter%203%20-%20Scaling%20and%20Multiscaling/) | GHE, moment scaling, power-spectrum and Zipf notebooks, historical data, and figures. |
| [`Chapter 4 - Statistical Models/`](Chapter%204%20-%20Statistical%20Models/) | Gold/Bitcoin statistical modeling, processed data, Monte Carlo and EVT results. |
| [`Chapter 6 - Simulations and Results/`](Chapter%206%20-%20Simulations%20and%20Results/) | Selected thesis simulation figures and the current implementation-status note. |
| [`Reports/Final Thesis/`](Reports/Final%20Thesis/) | English and Persian thesis reports, defense presentation, and thesis poster. |
| [`Reports/Literature Review/`](Reports/Literature%20Review/) | Literature-review report on financial stylized facts and multiscaling. |
| [`Reports/Talks and Presentations/`](Reports/Talks%20and%20Presentations/) | Transformer-in-finance presentation and poster. |

**Current archive status:** the repository contains executable notebooks for Chapters 2-4 and selected Chapter 6 figures. The Chapter 6 folder still states that its simulation code will be added later; the complete experimental descriptions and results are available in the thesis PDFs.

## Reports and presentations

- [English thesis](Reports/Final%20Thesis/English_Translated_Report.pdf)
- [Persian thesis](Reports/Final%20Thesis/Persian_Report.pdf)
- [Defense presentation](Reports/Final%20Thesis/Thesis%20Presentation.pdf)
- [Thesis poster](Reports/Final%20Thesis/Thesis%20Poster.jpg)
- [Stylized facts and multiscaling literature review](Reports/Literature%20Review/Stylized%20Facts%20and%20Multiscaling%20in%20Finance%20-%20A%20Literature%20Review.pdf)
- [Transformers in finance presentation](Reports/Talks%20and%20Presentations/Transformers%20in%20Finance%20Leveraging%20Attention%20Mechanisms%20for%20Time%20Series%20Forecasting.pdf)

## Working with the notebooks

```bash
git clone https://github.com/Alikhabazian0/Financial-Time-Series-Scaling-and-Predictability.git financial-scaling-thesis
cd financial-scaling-thesis
python -m pip install numpy pandas scipy matplotlib statsmodels arch yfinance openpyxl jupyterlab
jupyter lab
```

Open the notebook for the desired chapter and run cells in order from a fresh kernel. The notebooks include exploratory sections and repeated function definitions, so execution order matters.

Before reproducing an analysis, update machine-specific data paths, check the date range and asset symbols, and select the intended series transformation and scaling interval. Downloaded market data can change with provider revisions; retain the data snapshot and settings used for a reported figure. The historical CSVs and Chapter 4 data are included for the corresponding examples.

The commands above install the dependencies used by the currently uploaded empirical notebooks. They do not execute the Chapter 6 model-training experiments, whose implementation files are pending in this archive.

## Selected references

### Di Matteo literature underlying the scaling review

1. **Di Matteo, T. (2007).** Multi-scaling in finance. *Quantitative Finance, 7*(1), 21-36. [DOI: 10.1080/14697680600969727](https://doi.org/10.1080/14697680600969727).
2. **Di Matteo, T., Aste, T., & Dacorogna, M. M. (2003).** Using the Scaling Analysis to Characterize Financial Markets. [arXiv:cond-mat/0302434](https://arxiv.org/abs/cond-mat/0302434).
3. **Di Matteo, T., Aste, T., & Dacorogna, M. M. (2004, preprint).** Long term memories of developed and emerging markets: using the scaling analysis to characterize their stage of development. [arXiv:cond-mat/0403681](https://arxiv.org/abs/cond-mat/0403681).
4. **Buonocore, R. J., Aste, T., & Di Matteo, T. (2015, preprint).** Measuring multiscaling in financial time-series. [arXiv:1509.05471](https://arxiv.org/abs/1509.05471).

[Professor Tiziana Di Matteo's King's College London profile](https://www.kcl.ac.uk/people/tiziana-di-matteo).

### Information theory and neural sequence modeling

5. **Shannon, C. E. (1948).** A mathematical theory of communication. *The Bell System Technical Journal, 27*(3), 379-423. [DOI](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x).
6. **Bandt, C., & Pompe, B. (2002).** Permutation entropy: A natural complexity measure for time series. *Physical Review Letters, 88*, 174102. [DOI](https://doi.org/10.1103/PhysRevLett.88.174102).
7. **Vaswani, A., et al. (2017).** Attention Is All You Need. *Advances in Neural Information Processing Systems, 30*. [Paper](https://arxiv.org/abs/1706.03762).
8. **Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).** A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. *International Conference on Learning Representations*. [Paper](https://arxiv.org/abs/2211.14730).

The thesis PDFs contain the full bibliography. When using this work, cite the thesis or acknowledge this repository and record the specific commit, notebook configuration, data sample, and experiment being used.
