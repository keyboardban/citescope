#!/usr/bin/env python3
"""Retry tracking-clean URLs for non-policy Bright Data failures."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.econometrics_eda_v2.brightdata_config import require_live_brightdata_config
from src.econometrics_eda_v2.scrape_providers.brightdata_provider import scrape_url_brightdata
from src.url_utils import strip_tracking_params

POLICY_TERMS = ("robots.txt", "compliance policies", "kyc")
def main() -> int:
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--raw-dir',type=Path,required=True);p.add_argument('--execute-live-brightdata',action='store_true');a=p.parse_args()
    if not a.execute_live_brightdata: raise SystemExit('Live Bright Data calls require --execute-live-brightdata.')
    settings=require_live_brightdata_config();df=pd.read_csv(a.input,low_memory=False);a.raw_dir.mkdir(parents=True,exist_ok=True);rows=[]
    for i,r in df.iterrows():
        original=str(r.source_url);clean=strip_tracking_params(original);error=str(r.scrape_error or '')
        eligible=not bool(r.scrape_success) and clean!=original and not any(x in error.casefold() for x in POLICY_TERMS)
        row={'source_url':original,'tracking_clean_url':clean,'tracking_params_removed':clean!=original,'prior_provider_mode':r.final_provider_mode,'prior_scrape_success':bool(r.scrape_success),'prior_error':error,'eligible_for_clean_retry':eligible,'retry_attempted':False,'retry_success':False,'retry_error':'','retry_word_count':0,'retry_content_quality_flag':''}
        if eligible:
            mode='unlocker_api' if str(r.final_provider_mode).endswith('unlocker_api') else 'browser_api'
            result=scrape_url_brightdata(clean,mode,settings,live=True,raw_response_path='');normal=result.get('normalized_result') or {}
            (a.raw_dir/f'{i:03d}.json').write_text(json.dumps({'request_url':clean,'mode':mode,'raw_response':result.get('raw_response'),'response_headers':result.get('response_headers')},ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            row.update(retry_attempted=True,retry_success=bool(normal.get('success')),retry_error=str(normal.get('error') or ''),retry_word_count=int(normal.get('word_count') or 0),retry_content_quality_flag=str(normal.get('content_quality_flag') or ''))
        rows.append(row)
    out=pd.DataFrame(rows);a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False)
    print(json.dumps({'rows':len(out),'eligible':int(out.eligible_for_clean_retry.sum()),'attempted':int(out.retry_attempted.sum()),'successes':int(out.retry_success.sum()),'output':str(a.output)},indent=2));return 0
if __name__=='__main__': raise SystemExit(main())
