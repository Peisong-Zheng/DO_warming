# DO_warming

这个项目现在按“原始数据”和“程序生成产物”分开组织，方便后续继续做中国石笋 GI/DO 事件筛选与 MCV 训练数据准备。

## 目录结构

```text
data/
  raw/
    Corrick2020_China_ASM_extracted_checked.xlsx
    split_by_cave/
      Corrick2020_S7_China_ASM_*.xlsx
      Corrick2020_S7_China_ASM_split_manifest.(xlsx|csv)
    samples_csv/
      *.csv
      Corrick2020_S7_China_ASM_samples_csv_manifest.(xlsx|csv)
  processed/
    gi_event_windows_L300R300/
    gi_screening_L300R300/
    age_ensembles_L300R300_monotone_gibbs/

figures/
  gi_event_windows_L300R300/
  gi_screening_L300R300/
  age_ensembles_L300R300_monotone_gibbs/
  age_ensembles_QC_L300R300_monotone_gibbs/
```

## 当前约定

- `data/raw/` 只放基础数据，不再混入脚本生成表格。
- `data/processed/` 只放程序生成的表格结果。
- `figures/` 只放图件，不再把 CSV/XLSX 混进来。
- 生成表格尽量只保留一份：
  - 只有 `Corrick2020_China_ASM_events_with_window_metrics_*.xlsx` 保留为 Excel，因为它需要保留原始工作簿里的 `QC_notes` 和 `Source` sheet。
  - 其他生成表全部只保留 CSV，避免同一份信息重复存成 `xlsx + csv`。

## 脚本

- `Filter_GI.py`
  - 读取原始事件表和 sample CSV。
  - 输出带窗口指标的事件工作簿、GI 事件图清单、逐事件图和总 PDF。
- `GI_cropp_win.py`
  - 生成筛选汇总表和相对年龄坐标的总览图。
- `training_data_augmentation.py`
  - 生成通过筛选后的事件表、年龄扰动集合图清单和对应图件。
  - 同时为每条 accepted 记录输出 4-panel QC 图，用于目视检查 valid segment、裁剪窗口、误差解释和最终 ensemble。

## 运行

在项目根目录执行：

```bash
python Filter_GI.py
python GI_cropp_win.py
python training_data_augmentation.py
```

脚本会自动把结果写到对应的 `data/processed/...` 和 `figures/...` 子目录。
