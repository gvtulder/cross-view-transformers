python -u tabulate.py --runs log-from-cluster/log-ddsm-20210218-5cv-ds0.125/20210218-ds0.125-model-*-data-ddsm-view-*-run?-fold?-augcoflip-weight-cosine0.0001-warmup30-swa --direction last --tag-to-optimize roc_auc/val --tag-to-measure roc_auc/test roc_auc/val --save-csv t-ds0.125-last.csv
python -u tabulate-chexpert.py --runs log-from-cluster/log-chexpert-20210218/*-aug-cosine0.00001-epochs60 --save-csv tabulate-paper-chexpert-roc-auc-task-max-v1.csv --direction max