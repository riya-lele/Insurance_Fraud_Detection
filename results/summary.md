# Results summary

Tables only. Costs are hypothetical (c = 1).

## Imbalance arms (mean over folds)

| arm | n_folds | pr_auc_mean | roc_auc_mean | precision_at_1pct_mean | precision_at_2pct_mean | precision_at_5pct_mean | precision_at_10pct_mean | recall_at_1pct_mean | recall_at_2pct_mean | recall_at_5pct_mean | recall_at_10pct_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G0 | 15 | 0.2104 | 0.8107 | 0.3844 | 0.2831 | 0.2195 | 0.1887 | 0.0625 | 0.0935 | 0.1831 | 0.3149 |
| G1 | 15 | 0.1974 | 0.8057 | 0.3467 | 0.2699 | 0.2139 | 0.1827 | 0.0563 | 0.0892 | 0.1784 | 0.3048 |
| G2 | 15 | 0.2076 | 0.8081 | 0.4067 | 0.2951 | 0.2048 | 0.1790 | 0.0661 | 0.0975 | 0.1708 | 0.2987 |
| L0 | 15 | 0.1581 | 0.7993 | 0.1800 | 0.1945 | 0.1835 | 0.1721 | 0.0292 | 0.0643 | 0.1531 | 0.2871 |

## Paired differences vs G0 (imbalance)

| metric | arm | baseline | mean_diff | ci_low | ci_high | n_folds | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pr_auc | G1 | G0 | -0.0130 | -0.0180 | -0.0075 | 15 | folds are correlated; CI is approximate |
| pr_auc | G2 | G0 | -0.0027 | -0.0089 | 0.0040 | 15 | folds are correlated; CI is approximate |
| pr_auc | L0 | G0 | -0.0522 | -0.0645 | -0.0394 | 15 | folds are correlated; CI is approximate |

_Note from CSV: folds are correlated; CI is approximate._

## Anomaly / calibrated methods (mean over folds)

| arm | n_folds | pr_auc_mean | roc_auc_mean | precision_at_5pct_mean | recall_at_5pct_mean |
| --- | --- | --- | --- | --- | --- |
| G0 | 15 | 0.2104 | 0.8107 | 0.2195 | 0.1831 |
| G3 | 15 | 0.1904 | 0.8085 | 0.2199 | 0.1835 |
| IF_all | 15 | 0.0689 | 0.5119 | 0.0762 | 0.0636 |
| IF_normal | 15 | 0.0715 | 0.5232 | 0.0788 | 0.0657 |
| rankavg_IF_all | 15 | 0.1662 | 0.7929 | 0.2165 | 0.1806 |
| rankavg_IF_normal | 15 | 0.1680 | 0.7940 | 0.2273 | 0.1896 |
| stack_IF_all | 15 | 0.1870 | 0.8081 | 0.2182 | 0.1820 |
| stack_IF_normal | 15 | 0.1860 | 0.8067 | 0.2212 | 0.1845 |

## Paired differences vs G0 (anomaly)

| metric | arm | baseline | mean_diff | ci_low | ci_high | n_folds | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pr_auc | G3 | G0 | -0.0199 | -0.0227 | -0.0166 | 15 | folds are correlated; CI is approximate |
| pr_auc | IF_all | G0 | -0.1415 | -0.1521 | -0.1297 | 15 | folds are correlated; CI is approximate |
| pr_auc | IF_normal | G0 | -0.1389 | -0.1494 | -0.1270 | 15 | folds are correlated; CI is approximate |
| pr_auc | rankavg_IF_all | G0 | -0.0442 | -0.0542 | -0.0347 | 15 | folds are correlated; CI is approximate |
| pr_auc | rankavg_IF_normal | G0 | -0.0423 | -0.0514 | -0.0333 | 15 | folds are correlated; CI is approximate |
| pr_auc | stack_IF_all | G0 | -0.0234 | -0.0312 | -0.0144 | 15 | folds are correlated; CI is approximate |
| pr_auc | stack_IF_normal | G0 | -0.0244 | -0.0301 | -0.0181 | 15 | folds are correlated; CI is approximate |

## Hypothetical expected cost per claim (mean over folds)

| method | rule | L=5 | L=10 | L=20 | L=50 |
| --- | --- | --- | --- | --- | --- |
| G0_cal | cost_bayes | 0.2946 | 0.4717 | 0.5280 | 0.6029 |
| G0_cal | cost_optimal_val | 0.2899 | 0.4688 | 0.5293 | 0.6009 |
| G0_cal | f1_optimal | 0.3359 | 0.4813 | 0.7721 | 1.6443 |
| G0_cal | investigate_all | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| G0_cal | investigate_none | 0.2993 | 0.5986 | 1.1971 | 2.9929 |
| G0_cal | threshold_0.5 | 0.2923 | 0.5820 | 1.1613 | 2.8993 |
| G3 | cost_bayes | 0.2946 | 0.4717 | 0.5280 | 0.6029 |
| G3 | cost_optimal_val | 0.2899 | 0.4688 | 0.5293 | 0.6009 |
| G3 | f1_optimal | 0.3359 | 0.4813 | 0.7721 | 1.6443 |
| G3 | investigate_all | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| G3 | investigate_none | 0.2993 | 0.5986 | 1.1971 | 2.9929 |
| G3 | threshold_0.5 | 0.2923 | 0.5820 | 1.1613 | 2.8993 |
| rankavg_IF_all | cost_bayes | 0.2969 | 0.4889 | 0.5953 | 0.6798 |
| rankavg_IF_all | cost_optimal_val | 0.2923 | 0.4810 | 0.5998 | 0.6797 |
| rankavg_IF_all | f1_optimal | 0.3179 | 0.5037 | 0.8752 | 1.9900 |
| rankavg_IF_all | investigate_all | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| rankavg_IF_all | investigate_none | 0.2993 | 0.5986 | 1.1971 | 2.9929 |
| rankavg_IF_all | threshold_0.5 | 0.2974 | 0.5927 | 1.1832 | 2.9550 |
| rankavg_IF_normal | cost_bayes | 0.2951 | 0.4866 | 0.5915 | 0.6868 |
| rankavg_IF_normal | cost_optimal_val | 0.2910 | 0.4791 | 0.5935 | 0.6876 |
| rankavg_IF_normal | f1_optimal | 0.3205 | 0.5039 | 0.8708 | 1.9713 |
| rankavg_IF_normal | investigate_all | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| rankavg_IF_normal | investigate_none | 0.2993 | 0.5986 | 1.1971 | 2.9929 |
| rankavg_IF_normal | threshold_0.5 | 0.2974 | 0.5925 | 1.1826 | 2.9530 |
| stack_IF_all | cost_bayes | 0.2970 | 0.4748 | 0.5233 | 0.6016 |
| stack_IF_all | cost_optimal_val | 0.2896 | 0.4735 | 0.5241 | 0.6016 |
| stack_IF_all | f1_optimal | 0.3432 | 0.4848 | 0.7680 | 1.6176 |
| stack_IF_all | investigate_all | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stack_IF_all | investigate_none | 0.2993 | 0.5986 | 1.1971 | 2.9929 |
| stack_IF_all | threshold_0.5 | 0.2931 | 0.5836 | 1.1647 | 2.9079 |
| stack_IF_normal | cost_bayes | 0.2970 | 0.4724 | 0.5327 | 0.6043 |
| stack_IF_normal | cost_optimal_val | 0.2916 | 0.4705 | 0.5332 | 0.6041 |
| stack_IF_normal | f1_optimal | 0.3351 | 0.4864 | 0.7890 | 1.6970 |
| stack_IF_normal | investigate_all | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stack_IF_normal | investigate_none | 0.2993 | 0.5986 | 1.1971 | 2.9929 |
| stack_IF_normal | threshold_0.5 | 0.2920 | 0.5813 | 1.1597 | 2.8952 |

## Top-K capacity curve (mean over folds)

| method | k_pct | precision_mean | recall_mean |
| --- | --- | --- | --- |
| G0_cal | 1 | 0.3800 | 0.0618 |
| G0_cal | 2 | 0.2951 | 0.0975 |
| G0_cal | 5 | 0.2199 | 0.1835 |
| G0_cal | 10 | 0.1879 | 0.3135 |
| G3 | 1 | 0.3800 | 0.0618 |
| G3 | 2 | 0.2951 | 0.0975 |
| G3 | 5 | 0.2199 | 0.1835 |
| G3 | 10 | 0.1879 | 0.3135 |
| rankavg_IF_all | 1 | 0.3067 | 0.0498 |
| rankavg_IF_all | 2 | 0.2787 | 0.0921 |
| rankavg_IF_all | 5 | 0.2165 | 0.1806 |
| rankavg_IF_all | 10 | 0.1848 | 0.3084 |
| rankavg_IF_normal | 1 | 0.3067 | 0.0498 |
| rankavg_IF_normal | 2 | 0.2842 | 0.0939 |
| rankavg_IF_normal | 5 | 0.2273 | 0.1896 |
| rankavg_IF_normal | 10 | 0.1868 | 0.3116 |
| stack_IF_all | 1 | 0.3844 | 0.0625 |
| stack_IF_all | 2 | 0.2809 | 0.0928 |
| stack_IF_all | 5 | 0.2182 | 0.1820 |
| stack_IF_all | 10 | 0.1877 | 0.3131 |
| stack_IF_normal | 1 | 0.3422 | 0.0556 |
| stack_IF_normal | 2 | 0.2787 | 0.0921 |
| stack_IF_normal | 5 | 0.2212 | 0.1845 |
| stack_IF_normal | 10 | 0.1859 | 0.3102 |

## Figures

- `results/figures/pr_curves.png`
- `results/figures/precision_at_k.png`
- `results/figures/cost_vs_ratio.png`
