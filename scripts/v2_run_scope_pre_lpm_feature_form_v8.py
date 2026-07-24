#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.econometrics_eda_v2.pre_lpm_feature_form_v8 import run_feature_form_v8
BASE=ROOT/'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
def main()->int:
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=BASE/'tables/general_page_taxonomy/scope_condo_lpm_ready_with_general_page_taxonomy.csv');p.add_argument('--eda',type=Path,default=BASE/'tables/scope_condo_eda_ready_post_scrape.csv');p.add_argument('--table-dir',type=Path,default=BASE/'tables/pre_lpm_feature_form');p.add_argument('--figure-dir',type=Path,default=BASE/'figures/pre_lpm_feature_form');a=p.parse_args();print(json.dumps(run_feature_form_v8(a.input,a.eda if a.eda.exists() else None,a.table_dir,a.figure_dir),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
