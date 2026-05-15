# DO-like Asian monsoon transition timing

这个仓库保存的是一篇古气候论文的分析代码。项目的核心问题很简单：

> 中国石笋记录中识别出的 DO-like Asian monsoon transitions，是否只是随机出现，还是受到冰期背景态和岁差相位共同组织？

这里的主事件表来自 Rousseau et al. (2023)。他们在 Cheng et al. (2016) 的
0--640 kyr 中国石笋合成记录中，用 KS-window 方法识别了 weak monsoon starts
和 strong monsoon starts。本文把这些 transition timing 当作事件序列，而不是
连续时间序列，重点检验：

- 事件是否集中在特定的 precession / obliquity phase；
- LR04 和 CO2 代表的慢变冰期背景态，是否提高事件率预测；
- 在控制事件历史、采样分辨率和背景态之后，precession phase 是否仍然提供额外信息；
- 这些结论对 bin width、年龄不确定性、KS-window、额外 forcing 和独立事件表是否稳健。

## 目录

```text
data/
  raw/          原始数据。脚本默认只读取，不在这里写输出。
  processed/    每个脚本生成的 CSV 表格。

figures/        每个脚本生成的图，按脚本名分文件夹。

monsoon_paper/
  main.tex      正文。
  SI.tex        补充材料。
  reference.bib 论文引用库。
  figures/      上传 Overleaf 用的扁平化图片目录，文件名为 Fig01.pdf、FigS01.pdf 等。

toolbox/        复用的小工具，例如 Poisson likelihood、AICc、LR test、bits/event。

run_logs/       runner 和验证脚本的日志。这个目录不进 git。
```

约定是：`data/raw/` 放输入，`data/processed/` 和 `figures/` 放程序生成的结果。
如果要删掉所有输出重新跑，原则上可以只清理 `data/processed/` 和 `figures/`。

## 最常用的运行方式

只刷新正文主线，不跑任何敏感性实验：

```bash
python run_main_paper_workflow.py
```

这个 runner 会依次运行：

1. `Predictive_information_model.py`
2. `Orbital_phase_rayleigh.py`
3. `Lagged_predictive_information.py`
4. `paper_figure_export.py`

其中前三个脚本生成正文核心结果，最后一个脚本把论文中实际引用的 PDF 图复制到
`monsoon_paper/figures/`，并改名为 `Fig01.pdf`、`Fig02.pdf`、`FigS01.pdf` 这类
Overleaf 友好的文件名。

想先看它会跑什么：

```bash
python run_main_paper_workflow.py --dry-run
```

只想刷新分析结果、不更新 `monsoon_paper/figures/`：

```bash
python run_main_paper_workflow.py --skip-export
```

## 论文里几个模型名

代码和论文里反复出现三个模型名：

- **event-process baseline**  
  同类型事件历史 + Cheng composite 的局部采样分辨率。它不是气候模型，而是用来控制
  event clustering 和数据分辨率差异的基线。

- **climate-state model**  
  event-process baseline + LR04 + CO2。这里 LR04 和 CO2 被当作慢变冰期背景态的代表。

- **full predictive model**  
  climate-state model + precession phase。precession phase 用
  `sin(phase)` 和 `cos(phase)` 两个项表示，因为 phase 是圆周变量。

另外，**extended baseline** 只在额外 forcing 敏感性实验里使用，意思是：
在测试 Antarctic temperature、obliquity、eccentricity 和 65N insolation 之前，
先把 full predictive model 当作参考模型。

## 主线脚本

| 脚本 | 做什么 | 主要输出 |
|---|---|---|
| `Predictive_information_model.py` | 主模型。拟合 event-process baseline、climate-state model 和 full predictive model，并输出模型比较、拟合事件率和 history-window sensitivity。 | `data/processed/Predictive_information_model/`；`figures/Predictive_information_model/`；正文 Fig01、Fig03 |
| `Orbital_phase_rayleigh.py` | 把 precession index 和 obliquity 转换成 phase，做 Rayleigh phase clustering test。 | `data/processed/Orbital_phase_rayleigh/`；`figures/Orbital_phase_rayleigh/`；正文 Fig02，补充 FigS01、FigS02 |
| `Lagged_predictive_information.py` | 做 lagged predictive-information scan。LR04 和 CO2 分别加到 event-process baseline 上；lagged precession phase 加到 climate-state model 上。 | `data/processed/Lagged_predictive_information/`；`figures/Lagged_predictive_information/`；正文 Fig04 |
| `Bin_hazard_phase_poisson.py` | 早期版本的 binned Poisson hazard model，不含 event-process baseline。现在主要保留给补充材料中的一个直观对照图：LR04+CO2 vs LR04+CO2+precession phase。 | `data/processed/Bin_hazard_phase_poisson/`；`figures/Bin_hazard_phase_poisson/`；补充 FigS03 |
| `paper_figure_export.py` | 只复制论文正文和 SI 中实际用到的 PDF 图到 `monsoon_paper/figures/`。 | `monsoon_paper/figures/Fig*.pdf` |

## 敏感性和诊断脚本

这些脚本默认不在 `run_main_paper_workflow.py` 里运行，需要时单独跑。

| 脚本 | 用途 |
|---|---|
| `Bin_hazard_phase_poisson_sensitivity.py` | 测试 Antarctic temperature、obliquity、eccentricity 和 65N summer-solstice insolation 是否在 full predictive model 之外还有额外贡献。 |
| `Bin_hazard_phase_poisson_binwidth_sensitivity.py` | 把 bin width 从 0.2 kyr 改到 1.0 kyr，检查主结论是否依赖 0.2 kyr 分箱。 |
| `Composite_age_uncertainty_core_sensitivity.py` | 对事件年龄做 rank-preserving perturbation，检验 Rayleigh 和 predictive-information 结果对年龄不确定性的稳健性。 |
| `KS_window_core_experiment_sensitivity.py` | 比较 Rousseau 0.4--4 kyr 和 0.6--4 kyr KS-window 事件表。 |
| `Predictive_information_bootstrap_diagnostics.py` | 对两个核心 LR test 做 reduced-model parametric bootstrap。这个脚本比较慢，默认 2000 次 bootstrap。 |
| `Barker2011_do_predictive_information.py` | 用 Barker et al. (2011) 的 variable-threshold D-O warming catalogue 做独立事件表检查。 |

## `toolbox/`

`toolbox/` 只放最通用、最不容易手调的部分：

- `toolbox/poisson.py`  
  binned Poisson log-likelihood、negative log-likelihood、design matrix、fitted rate。

- `toolbox/model_stats.py`  
  AIC/AICc/BIC、likelihood-ratio p value、log-likelihood gain、bits/event。

绘图代码仍然留在各自脚本里。原因很现实：论文图有很多手调过的小参数，
而且 Fig01、FigS09 这类图看起来相似但并不完全相同，强行抽统一 helper 反而容易把图改乱。

## 依赖

主要依赖是常见科学计算包：

```text
numpy
pandas
scipy
matplotlib
xarray
openpyxl
xlrd
```

运行脚本前请确认 `data/raw/` 中的数据文件已经放好。部分 Excel 文件在读取时会触发
`openpyxl` 的 “Unknown extension is not supported” warning，这是 Excel metadata 的问题，
不影响脚本结果。

## 主要数据和文献来源

下面这些条目来自 `monsoon_paper/reference.bib`，列在这里是为了让代码使用者不用打开论文也能知道
主要数据从哪里来。

- Barker, S., Knorr, G., Edwards, R. L., Parrenin, F., Putnam, A. E.,
  Skinner, L. C., Wolff, E., & Ziegler, M. (2011).
  800,000 years of abrupt climate variability. *Science*, 334(6054), 347--351.

- Bereiter, B., Eggleston, S., Schmitt, J., Nehrbass-Ahles, C., Stocker, T. F.,
  Fischer, H., Kipfstuhl, S., & Chappellaz, J. (2015).
  Revision of the EPICA Dome C CO2 record from 800 to 600 kyr before present.
  *Geophysical Research Letters*, 42(2), 542--549.

- Cheng, H., Edwards, R. L., Sinha, A., Spötl, C., Yi, L., Chen, S.,
  Kelly, M., Kathayat, G., Wang, X., Li, X., et al. (2016).
  The Asian monsoon over the past 640,000 years and ice age terminations.
  *Nature*, 534(7609), 640--646.

- Corrick, E. C., Drysdale, R. N., Hellstrom, J. C., Capron, E.,
  Rasmussen, S. O., Zhang, X., Fleitmann, D., Couchoud, I., & Wolff, E. (2020).
  Synchronous timing of abrupt climate changes during the last glacial period.
  *Science*, 369(6506), 963--969.

- Laskar, J., Robutel, P., Joutel, F., Gastineau, M., Correia, A. C. M.,
  & Levrard, B. (2004).
  A long-term numerical solution for the insolation quantities of the Earth.
  *Astronomy & Astrophysics*, 428(1), 261--285.

- Lisiecki, L. E., & Raymo, M. E. (2005).
  A Pliocene-Pleistocene stack of 57 globally distributed benthic δ18O records.
  *Paleoceanography*, 20(1).

- Lohmann, J., & Ditlevsen, P. D. (2018).
  Random and externally controlled occurrence of Dansgaard-Oeschger events.
  *Climate of the Past*, 14(5), 609--617. https://doi.org/10.5194/cp-14-609-2018

- Rousseau, D.-D., Bagniewski, W., & Cheng, H. (2023).
  A reliable benchmark of the last 640,000 years millennial climate variability.
  *Scientific Reports*, 13(1), 22851.
