#!/usr/bin/env python3
"""Merge successful tracking-clean retries into the Bright Data pilot table."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.v2_run_area_condo_brightdata_content_pilot import _content_strength, _plot, _save_json
from src.econometrics_eda_v2.scrape_providers.brightdata_provider import normalize_brightdata_response

def main() -> int:
 p=argparse.ArgumentParser();p.add_argument('--detail',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--figure-dir',type=Path,required=True);a=p.parse_args()
 detail=pd.read_csv(a.detail,low_memory=False);audit=pd.read_csv(a.audit,low_memory=False);merged=0
 for column in ['scrape_success','final_url','status_code','page_title','meta_description','content_quality_flag','content_chars','word_count','heading_count','table_count','link_count','scrape_error','content_strength','page_text_excerpt']:
  detail[column]=detail[column].astype(object)
 detail['final_request_url']=detail.get('final_request_url',detail['source_url']);detail['tracking_clean_retry_used']=False
 for _,retry in audit[audit['retry_success'].fillna(False)].iterrows():
  hits=detail.index[detail['source_url'].eq(retry['source_url'])]
  if not len(hits): continue
  idx=hits[0];raw_path=a.raw_dir/f'{idx:03d}.json'
  if not raw_path.exists(): continue
  payload=json.loads(raw_path.read_text(encoding='utf-8'));mode='unlocker_api' if str(detail.at[idx,'final_provider_mode']).endswith('unlocker_api') else 'browser_api'
  final=normalize_brightdata_response(payload.get('raw_response'),str(retry['tracking_clean_url']),mode=mode)
  final=final.to_dict();detail.at[idx,'original_scrape_success']=detail.at[idx,'scrape_success'];detail.at[idx,'original_scrape_error']=detail.at[idx,'scrape_error'];detail.at[idx,'final_request_url']=retry['tracking_clean_url'];detail.at[idx,'tracking_clean_retry_used']=True
  for col,key in [('scrape_success','success'),('final_url','final_url'),('status_code','status_code'),('page_title','title'),('meta_description','meta_description'),('content_quality_flag','content_quality_flag'),('content_chars','text_char_count'),('word_count','word_count'),('heading_count','heading_count'),('table_count','table_count'),('link_count','link_count'),('scrape_error','error')]: detail.at[idx,col]=final.get(key)
  detail.at[idx,'content_strength']=_content_strength(final);detail.at[idx,'page_text_excerpt']=str(final.get('text') or '')[:1200];merged+=1
 detail.to_csv(a.detail,index=False)
 quality=detail.groupby(['final_provider_mode','content_strength'],dropna=False).agg(urls=('normalized_url','size'),scrape_success=('scrape_success','sum'),median_word_count=('word_count','median')).reset_index();quality.to_csv(a.detail.parent/'brightdata_content_pilot_quality_summary.csv',index=False)
 cited=detail.groupby('cited').agg(urls=('normalized_url','size'),scrape_success_rate=('scrape_success','mean'),strong_content_rate=('content_strength',lambda x:x.eq('strong').mean()),median_word_count=('word_count','median'),median_heading_count=('heading_count','median'),median_link_count=('link_count','median')).reset_index();cited.to_csv(a.detail.parent/'brightdata_content_pilot_cited_comparison.csv',index=False)
 _plot(detail,a.figure_dir);summary={'pilot_urls':int(len(detail)),'final_scrape_success_rate':float(detail.scrape_success.mean()),'strong_content_rate':float(detail.content_strength.eq('strong').mean()),'tracking_clean_retry_successes':merged};_save_json(a.detail.parent/'brightdata_content_pilot_summary.json',summary);print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':
 import argparse
 raise SystemExit(main())
