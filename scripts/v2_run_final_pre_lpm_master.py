#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.econometrics_eda_v2.final_pre_lpm_master import run_final_master
BASE=ROOT/'outputs/econometrics_eda_v2/topic_sensitivity/scope_condo_nonbranded'
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=BASE/'tables/pre_lpm_feature_form/scope_condo_lpm_ready_general_taxonomy_feature_form.csv');p.add_argument('--table-dir',type=Path,default=BASE/'tables/final_pre_lpm_master');p.add_argument('--figure-dir',type=Path,default=BASE/'figures/final_pre_lpm_master');a=p.parse_args();print(json.dumps(run_final_master(a.input,BASE,a.table_dir,a.figure_dir),indent=2))
if __name__=='__main__':main()
