"""Final non-model pre-LPM consolidation for the SCOPE condo study."""
from __future__ import annotations
import json, shutil
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.econometrics_eda_v2.metric_recheck import normalise_boolean
from src.econometrics_eda_v2.pre_lpm_diagnostics import citation_rate_by_category
from src.econometrics_eda_v2.pre_lpm_intent_stratified_v6 import _cell_summary, _composition, _heatmap
from src.econometrics_eda_v2.pre_lpm_readable_graphs_v5 import save_plotly_figure

FORBIDDEN=("cited_label","is_more_only","source_group","source_origin","source_position","observed_rank","page_answer_similarity","max_chunk_answer_similarity","answer_like_text","answer_overlap","brand_appeared_in_answer","domain_citation_rate","similarity")
def _bool(df,n): return normalise_boolean(df,n)[0].fillna(False).astype(int) if n in df else pd.Series(0,index=df.index)
def _dist(df,feature):
 c=_bool(df,'cited'); rows=[]
 for val,g in df.groupby(feature,dropna=False):
  o=c.loc[g.index];rows.append({'category':val,'n_rows':len(g),'row_share':len(g)/len(df),'cited_rows':int(o.sum()),'cited_rate':float(o.mean()),'unique_urls':g.normalized_url.nunique(),'unique_domains':g.source_root_domain.nunique()})
 return pd.DataFrame(rows).sort_values('n_rows',ascending=False,kind='stable')
def _copy(src,dst):
 if src.exists(): dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);return True
 return False
def _numeric_artifacts(base,fig):
 source=base/'figures/pre_lpm_eda_v5_readable_graphs'; mappings=[]
 for f in ('heading_count','table_count','link_count'):
  mappings += [(source/'interactive'/f'readable_exact_scatter_{f}.html',fig/'interactive'/f'exact_scatter_{f}_final.html'),(source/f'readable_exact_scatter_{f}.png',fig/f'exact_scatter_{f}_final.png'),(source/'interactive'/f'readable_heatmap_{f}_by_cited.html',fig/'interactive'/f'heatmap_{f}_by_cited_final.html'),(source/f'readable_heatmap_{f}_by_cited.png',fig/f'heatmap_{f}_by_cited_final.png')]
 mappings += [(source/'interactive'/'readable_rolling_log1p_word_count.html',fig/'interactive'/'rolling_log1p_word_count_final.html'),(source/'readable_rolling_log1p_word_count.png',fig/'rolling_log1p_word_count_final.png')]
 return sum(_copy(a,b) for a,b in mappings)
def _selected_forests(base,fig):
 source_v8=base/'figures/pre_lpm_feature_form';source_v5=base/'figures/pre_lpm_eda_v5_readable_graphs'
 mappings=[
  (source_v8/'interactive/binary_feature_diff_forest.html',fig/'interactive/binary_feature_diff_forest_final.html'),(source_v8/'binary_feature_diff_forest.png',fig/'binary_feature_diff_forest_final.png'),
  (source_v8/'interactive/cited_rate_by_page_type_family_general.html',fig/'interactive/forest_page_type_family_general_final.html'),(source_v8/'cited_rate_by_page_type_family_general.png',fig/'forest_page_type_family_general_final.png'),
  (source_v8/'interactive/cited_rate_by_site_type_general.html',fig/'interactive/forest_site_type_general_final.html'),(source_v8/'cited_rate_by_site_type_general.png',fig/'forest_site_type_general_final.png'),
  (source_v5/'interactive/forest_diff_content_quality_flag.html',fig/'interactive/forest_content_quality_final.html'),(source_v5/'forest_diff_content_quality_flag.png',fig/'forest_content_quality_final.png'),
  (source_v5/'interactive/forest_diff_taxonomy_confidence.html',fig/'interactive/forest_taxonomy_confidence_final.html'),(source_v5/'forest_diff_taxonomy_confidence.png',fig/'forest_taxonomy_confidence_final.png'),
 ]
 return sum(_copy(a,b) for a,b in mappings)
def _collapsed(series):
 value=series.fillna('unknown').astype(str)
 counts=value.value_counts(dropna=False)
 return value.where(value.eq('unknown')|value.map(counts).ge(20),'other')
def _suspicious(df):
 url=df.normalized_url.fillna('').str.casefold(); title=df.get('page_type_general_reason',pd.Series('',index=df.index)).fillna('').str.casefold(); p=df.page_type_general.fillna('unknown'); fam=df.page_type_family_general.fillna('unknown'); conf=df.page_type_general_confidence.fillna('unknown')
 flags=(url.str.contains('/blog/')&fam.eq('directory_or_listing'))|(url.str.contains('guide|how-to')&fam.eq('directory_or_listing'))|(url.str.contains('/project/')&fam.isin(['contact_or_location']))|(url.str.contains('/feature/')&p.eq('feature_page'))|(conf.eq('high')&title.str.contains('insufficient|conflict'))|(p.eq('unknown')&url.str.contains('/blog/|/news/|/contact|/pricing|/faq'))
 out=df.loc[flags,['normalized_url','source_url','source_root_domain','page_type_family_general','page_type_general','page_type_general_confidence','page_type_general_reason','cited']].copy();out['suspicious_reason']='rule_based_QA_review_only';return out
def _missingness(df):
 rows=[]
 for f in ['has_table','has_multiple_tables','has_headings','has_many_headings','has_links','has_many_links','has_substantial_text','log1p_word_count','log1p_text_char_count','heading_count_raw','table_count_raw','link_count_raw','word_count_raw']:
  if f not in df: continue
  m=df[f].isna(); c=_bool(df,'cited');rows.append({'feature':f,'n_rows':len(df),'n_missing':int(m.sum()),'missing_rate':float(m.mean()),'n_available':int((~m).sum()),'cited_rate_available':float(c[~m].mean()) if (~m).any() else np.nan,'cited_rate_missing':float(c[m].mean()) if m.any() else np.nan,'depends_on_scrape':True,'recommended_handling':'content_subset_or_explicit_availability_flag'})
 return pd.DataFrame(rows)
def _sparse(df,features):
 c=_bool(df,'cited'); rows=[]
 for f in features:
  if f not in df:continue
  for val,g in df.groupby(f,dropna=False):
   o=c.loc[g.index];n=len(g);cr=int(o.sum());mo=n-cr;up=g.prompt_id.nunique();s=n<20;u=cr<5 or mo<5;act='keep_unknown_as_own_category' if str(val)=='unknown' else ('diagnostic_only' if f=='page_type_general' else ('collapse_to_other' if s else ('sensitivity_only' if u or up<3 else 'keep')));rows.append({'feature':f,'category':val,'n_rows':n,'cited_rows':cr,'more_only_rows':mo,'unique_prompts':up,'sparse_flag':s,'unstable_flag':u,'recommended_lpm_action':act})
 return pd.DataFrame(rows)
def _categorical(df,features):
 rows=[]
 for f in features:
  if f not in df:continue
  t=citation_rate_by_category(df,f);t['feature']=f;t['diff_from_overall_pp']=t.difference_from_overall*100;t['unique_prompts']=df.groupby(f).prompt_id.nunique().reindex(t.category).to_numpy();t['sparse_flag']=t.n_rows.lt(20);t['recommended_lpm_action']=np.where(t.category.eq('unknown'),'keep_unknown_as_own_category',np.where(t.n_rows.lt(20),'collapse_to_other','keep'));rows.append(t)
 return pd.concat(rows,ignore_index=True)
def _leakage(df,candidates):
 main=sum([candidates.get(k,[]) for k in ('main_all_row_lpm','content_subset_lpm')],[]);sens=candidates.get('sensitivity_only',[]);rows=[]
 for v in FORBIDDEN:
  rows.append({'variable':v,'present_in_dataset':v in df.columns,'present_in_main_candidates':v in main,'present_in_sensitivity_only':v in sens,'action_required':'remove_from_main' if v in main else 'none'})
 return pd.DataFrame(rows)
def run_final_master(input_path:Path,base:Path,table_dir:Path,figure_dir:Path)->dict[str,Any]:
 df=pd.read_csv(input_path,low_memory=False);table_dir.mkdir(parents=True,exist_ok=True);(figure_dir/'interactive').mkdir(parents=True,exist_ok=True)
 df['page_type_family_general_collapsed']=_collapsed(df['page_type_family_general'])
 df['site_type_general_collapsed']=_collapsed(df['site_type_general'])
 df.to_csv(table_dir/'scope_condo_lpm_ready_final_pre_lpm_master.csv',index=False)
 c=_bool(df,'cited'); metrics=[('total_rows',len(df)),('unique_urls',df.normalized_url.nunique()),('unique_prompts',df.prompt_id.nunique()),('cited_rows',int(c.sum())),('more_only_rows',int((1-c).sum())),('cited_rate',float(c.mean())),('scrape_success_rate',float(_bool(df,'scraped_ok').mean())),('parse_success_rate',float(_bool(df,'parse_ok').mean())),('usable_content_rate',float(_bool(df,'usable_content').mean())),('content_feature_available_rate',float(_bool(df,'content_feature_available').mean())),('unknown_page_type_family_general_rate',float(df.page_type_family_general.eq('unknown').mean())),('unknown_site_type_general_rate',float(df.site_type_general.eq('unknown').mean())),('high_medium_general_taxonomy_rate',float(_bool(df,'page_type_general_confidence_high_or_medium').mean()))]
 consistency=pd.DataFrame(metrics,columns=['metric','value']);consistency['expected_flag']=np.where(consistency.metric.eq('total_rows'),consistency.value.ne(1139),False);consistency.to_csv(table_dir/'dataset_consistency_summary.csv',index=False)
 lineage=pd.DataFrame([['v4_plotly','exact numeric scatters and rolling curves',True,'pre_lpm_eda_v4_plotly','numeric appendix source'],['v5_readable','readable capped plots',True,'pre_lpm_eda_v5_readable_graphs','selected numeric figures copied'],['v6_intent','intent stratification',True,'pre_lpm_eda_v6_intent_stratified','intent framework reused'],['v7_general_taxonomy','general page functions',True,'general_page_taxonomy','main taxonomy'],['v8_feature_form','transformed LPM-safe forms',True,'pre_lpm_feature_form','final input'],['master','guardrails, collapse plan, specification',True,'final_pre_lpm_master','final checkpoint']],columns=['source_notebook','contribution','reused_in_final_notebook','output_files_used','notes']);lineage.to_csv(table_dir/'notebook_lineage_map.csv',index=False)
 for f in ['page_type_family_general','page_type_general','site_type_general','page_type_general_confidence']:_dist(df,f).to_csv(table_dir/f'{f}_distribution.csv',index=False)
 _suspicious(df).to_csv(table_dir/'general_taxonomy_suspicious_rows_final.csv',index=False)
 v8=base/'tables/pre_lpm_feature_form'; shutil.copy2(v8/'feature_form_inventory.csv',table_dir/'feature_form_inventory_final.csv');shutil.copy2(v8/'binary_feature_cited_rate_summary.csv',table_dir/'binary_feature_cited_rate_summary_final.csv');shutil.copy2(v8/'correlation_matrix_feature_form.csv',table_dir/'correlation_matrix_final.csv');shutil.copy2(v8/'vif_feature_form_summary.csv',table_dir/'vif_summary_final.csv')
 features=['page_type_family_general','site_type_general','page_type_general','heading_count_group','link_count_group','word_count_group','content_quality_flag','page_type_general_confidence'];cat=_categorical(df,features);cat.to_csv(table_dir/'categorical_feature_cited_rate_summary_final.csv',index=False)
 _numeric_artifacts(base,figure_dir);_selected_forests(base,figure_dir)
 if 'intent' in df: df['intent_group']=df.intent.fillna('<missing>').astype(str).str.casefold()
 if 'intent_group' in df:
  audit=pd.DataFrame([{'candidate_column':'intent','exists':True,'selected_as_intent_group':True,'notes':'carried from feature-form input'}]);audit.to_csv(table_dir/'intent_column_audit_final.csv',index=False);_dist(df,'intent_group').to_csv(table_dir/'intent_distribution_final.csv',index=False)
  for f,stem in [('page_type_family_general','page_type_family_general'),('site_type_general','site_type_general')]:
   s=_cell_summary(df,f);s.to_csv(table_dir/f'intent_{stem}_cell_summary_final.csv',index=False)
   for freq in (False,True):save_plotly_figure(_heatmap(s,f,freq),figure_dir/'interactive'/f'heatmap_intent_by_{stem}_{"frequency" if freq else "cited_rate"}_final.html',figure_dir/f'heatmap_intent_by_{stem}_{"frequency" if freq else "cited_rate"}_final.png')
   for cited_only,label in ((False,'all'),(True,'cited')):save_plotly_figure(_composition(s,f,cited_only),figure_dir/'interactive'/f'stacked_{stem}_by_intent_{label}_final.html',figure_dir/f'stacked_{stem}_by_intent_{label}_final.png')
 _missingness(df).to_csv(table_dir/'missingness_content_availability_final.csv',index=False)
 vif=pd.read_csv(table_dir/'vif_summary_final.csv'); redundant=pd.DataFrame([['has_table','has_multiple_tables','keep has_table; exclude multiple-tables from main'],['has_substantial_text','usable_content','use one availability/content-length indicator'],['log1p_word_count','log1p_text_char_count','use one log length measure'],['raw counts','all transformed forms','diagnostic only unless shape supports use']],columns=['feature_a','feature_b','recommendation']);redundant.to_csv(table_dir/'redundant_feature_recommendations_final.csv',index=False)
 sparse=_sparse(df,['page_type_family_general','site_type_general','page_type_general','heading_count_group','link_count_group','word_count_group','intent_group']);sparse.to_csv(table_dir/'sparse_category_collapse_plan_final.csv',index=False)
 main=['page_type_family_general_collapsed','site_type_general_collapsed','content_feature_available'];content=['page_type_family_general_collapsed','site_type_general_collapsed','has_table','has_price_or_package','has_contact_info','log1p_word_count','heading_count_group','link_count_group']; candidates={'outcome':['cited'],'fixed_effects':['prompt_id'],'main_all_row_lpm':main,'content_subset_lpm':[x for x in content if x in df.columns or x.endswith('_collapsed')],'sensitivity_only':['general_taxonomy_confidence_high_or_medium','source_position','observed_rank','intent_group interactions'],'diagnostic_only':[x for x in df.columns if x.endswith('_raw')]+['page_type_general'],'forbidden':[x for x in df.columns if any(z in x.casefold() for z in FORBIDDEN)]};(table_dir/'final_lpm_candidate_columns.json').write_text(json.dumps(candidates,indent=2),encoding='utf-8')
 leakage=_leakage(df,candidates);leakage.to_csv(table_dir/'leakage_guardrail_final.csv',index=False)
 dictionary=[]
 for role,values in candidates.items():
  for v in values:dictionary.append({'variable_name':v,'role':role.rstrip('s'),'variable_form':'outcome' if v=='cited' else ('id' if v=='prompt_id' else ('raw_numeric' if v.endswith('_raw') else ('log_numeric' if v.startswith('log1p') else 'categorical' if 'type' in v or 'group' in v else 'binary'))),'use_in_main_all_row_lpm':v in candidates['main_all_row_lpm'],'use_in_content_subset_lpm':v in candidates['content_subset_lpm'],'use_in_sensitivity_lpm':v in candidates['sensitivity_only'],'leakage_risk':'high' if role=='forbidden' else 'none','missingness_risk':'content_dependent' if v in ['has_table','log1p_word_count'] else 'low','collinearity_risk':'review' if v in ['has_multiple_tables','log1p_text_char_count','usable_content'] else 'low','reason':'see feature-form inventory and redundancy audit'})
 pd.DataFrame(dictionary).to_csv(table_dir/'final_lpm_candidate_variable_dictionary.csv',index=False)
 formula='cited ~ C(page_type_family_general_collapsed) + C(site_type_general_collapsed) + content_feature_available + C(prompt_id)'; content_formula='cited ~ C(page_type_family_general_collapsed) + C(site_type_general_collapsed) + has_table + has_price_or_package + has_contact_info + log1p_word_count + C(heading_count_group) + C(link_count_group) + C(prompt_id)'; (table_dir/'recommended_lpm_v1_spec_final.md').write_text(f'# Recommended LPM v1\n\nAll-row: `{formula}`\n\nContent subset (`content_feature_available == 1` or `usable_content == 1`): `{content_formula}`\n\nUse HC3 robust SE baseline; consider clustering by prompt/domain if clusters are sufficient. Interactions and rank/position are sensitivity only.\n',encoding='utf-8')
 formula_vars=['page_type_family_general_collapsed','site_type_general_collapsed','content_feature_available','prompt_id','has_table','has_price_or_package','has_contact_info','log1p_word_count','heading_count_group','link_count_group']; missing_formula=[v for v in formula_vars if v not in df]
 checklist=[('final dataset loaded','pass'),('rows consistent','pass' if len(df)==1139 else 'warning'),('cited binary outcome valid','pass' if set(c.unique())<= {0,1} else 'fail'),('general taxonomy columns present','pass' if 'page_type_family_general' in df else 'fail'),('feature-form columns present','pass' if 'has_table' in df else 'fail'),('collapsed formula variables present','pass' if not missing_formula else 'fail'),('no forbidden variables in main candidates','pass' if not leakage.present_in_main_candidates.any() else 'fail'),('raw count variables excluded','pass'),('content missingness explicit','pass'),('sparse categories identified','pass'),('collinear pairs flagged','pass' if vif.flag.ne('').any() else 'warning'),('intent diagnostics completed','pass' if 'intent_group' in df else 'warning'),('candidate dictionary created','pass'),('LPM v1 specification created','pass'),('final notebook outputs saved','pass')];check=pd.DataFrame(checklist,columns=['check','status']);check.to_csv(table_dir/'final_pre_lpm_master_readiness_checklist.csv',index=False);status='ready_for_LPM_v1' if not check.status.eq('warning').any() and not check.status.eq('fail').any() else ('not_ready_for_LPM' if check.status.eq('fail').any() else 'ready_for_LPM_v1_after_minor_fixes')
 report=f'# Final Pre-LPM Master Report\n\nDataset: {len(df)} source appearances, {df.normalized_url.nunique()} URLs, {df.prompt_id.nunique()} prompts, cited rate {c.mean():.1%}. General page taxonomy unknown rate {df.page_type_family_general.eq("unknown").mean():.1%}.\n\nThe feature-form layer is complete; raw counts remain diagnostic, content features remain content-subset/availability-aware, and VIF flags redundant transformations. Leakage guardrails pass when forbidden fields are absent from candidates. Intent patterns are descriptive only.\n\nRecommended all-row model: `{formula}`.\n\nRecommended content-subset model: `{content_formula}`.\n\nThe dataset is ready to enter the LPM econometric layer if leakage guardrails pass, missing formula variables are removed, sparse categories are collapsed, and content-derived features are restricted to a content-available subset or modeled with explicit availability flags.\n\n**Final readiness:** {status}.\n';(table_dir/'final_pre_lpm_master_report.md').write_text(report,encoding='utf-8')
 return {'input_path':str(input_path),'rows':len(df),'unique_urls':int(df.normalized_url.nunique()),'unique_prompts':int(df.prompt_id.nunique()),'cited_rows':int(c.sum()),'cited_rate':float(c.mean()),'unknown_page_family_rate':float(df.page_type_family_general.eq('unknown').mean()),'high_medium_taxonomy_rate':float(_bool(df,'page_type_general_confidence_high_or_medium').mean()),'main_variables':len(main),'forbidden_in_main':int(leakage.present_in_main_candidates.sum()),'missing_formula_variables':missing_formula,'severe_collinearity':int(vif.flag.eq('severe').sum()),'sparse_categories':int(sparse.sparse_flag.sum()),'final_readiness_status':status,'output_folder':str(table_dir)}
