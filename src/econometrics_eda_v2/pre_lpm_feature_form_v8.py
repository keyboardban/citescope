"""Leakage-safe feature forms and diagnostics for pre-LPM readiness."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.econometrics_eda_v2.metric_recheck import normalise_boolean
from src.econometrics_eda_v2.pre_lpm_diagnostics import citation_rate_by_category
from src.econometrics_eda_v2.pre_lpm_readable_graphs_v5 import apply_readable_plotly_layout, save_plotly_figure

FORBIDDEN = ("answer", "similarity", "source_group", "source_origin", "source_position", "observed_rank", "is_more_only", "cited_label")
RAW = ("word_count", "text_char_count", "heading_count", "table_count", "link_count", "image_count")

def _bool(df: pd.DataFrame, name: str) -> pd.Series:
    return normalise_boolean(df, name)[0].astype("Int64") if name in df else pd.Series(pd.NA, index=df.index, dtype="Int64")
def _num(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors="coerce") if name in df else pd.Series(np.nan, index=df.index)
def _attach(base: pd.DataFrame, eda: pd.DataFrame | None) -> pd.DataFrame:
    out=base.copy().reset_index(drop=True)
    if eda is None: return out
    eda=eda.reset_index(drop=True); keys=["prompt_id","normalized_url","cited"]
    same=len(out)==len(eda) and all(k in out and k in eda and out[k].astype(str).equals(eda[k].astype(str)) for k in keys)
    cols=["source_url","intent","word_count","text_char_count","heading_count","table_count","link_count","image_count","has_faq","has_bullets","has_author","has_date","has_schema","has_video","page_text_excerpt"]
    if same:
        for c in cols:
            if c in eda: out[c]=eda[c]
    return out
def _nullable_threshold(values: pd.Series, threshold: float) -> pd.Series:
    return pd.Series(np.where(values.notna(), (values>=threshold).astype(int), pd.NA), index=values.index, dtype="Int64")
def _nullable_positive(values: pd.Series) -> pd.Series: return _nullable_threshold(values, 1)
def _groups(values: pd.Series, bins: list[tuple[float,float,str]]) -> pd.Series:
    out=pd.Series(pd.NA,index=values.index,dtype="object")
    for low, high, label in bins: out.loc[values.ge(low)&values.lt(high)] = label
    return out
def _transform(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    # Prefer raw EDA word_count where present; content_word_count is its safe LPM-ready fallback.
    if out.get("word_count", pd.Series(index=out.index)).isna().all() and "content_word_count" in out: out["word_count"]=_num(out,"content_word_count")
    for c in RAW: out[f"{c}_raw"]=_num(out,c)
    wc,hc,tc,lc,cc=(_num(out,x) for x in ("word_count","heading_count","table_count","link_count","text_char_count"))
    out["has_table"]=_nullable_positive(tc); out["has_multiple_tables"]=_nullable_threshold(tc,2)
    out["has_headings"]=_nullable_positive(hc); out["has_many_headings"]=_nullable_threshold(hc,7)
    out["has_links"]=_nullable_positive(lc); out["has_many_links"]=_nullable_threshold(lc,9)
    out["has_substantial_text"]=_nullable_threshold(wc,300)
    out["heading_count_group"]=_groups(hc,[(0,2,"0–1 headings"),(2,7,"2–6 headings"),(7,13,"7–12 headings"),(13,np.inf,"13+ headings")])
    out["link_count_group"]=_groups(lc,[(0,4,"0–3 links"),(4,9,"4–8 links"),(9,np.inf,"9+ links")])
    out["word_count_group"]=_groups(wc,[(0,100,"very_short"),(100,300,"short"),(300,1000,"medium"),(1000,np.inf,"long")])
    out["log1p_word_count"]=np.log1p(wc.clip(lower=0)).where(wc.notna())
    out["log1p_text_char_count"]=np.log1p(cc.clip(lower=0)).where(cc.notna())
    out["scraped_ok"]=_bool(out,"scraped_ok").fillna(_bool(out,"scrape_success")); out["parse_ok"]=_bool(out,"parse_success")
    out["scraped_body_available"]=_bool(out,"scraped_body_available"); out["content_feature_available"]=_bool(out,"content_feature_available")
    out["usable_content"]=((out.get("content_quality_flag","").astype(str).str.casefold().eq("ok")) & wc.ge(300)).astype("Int64")
    out["taxonomy_confidence_high_or_medium"]=_bool(out,"taxonomy_confidence_high_or_medium")
    out["general_taxonomy_confidence_high_or_medium"]=_bool(out,"page_type_general_confidence_high_or_medium")
    out["numeric_content_features_available"]=(out["usable_content"].eq(1)|out["content_feature_available"].eq(1)).astype("Int64")
    for c in ("word_count","heading_count","table_count","link_count"): out[f"{c}_missing"]=_num(out,c).isna().astype("Int64")
    for c in ("has_faq","has_price_or_package","has_contact_info","has_bullets","has_image","has_author","has_date","has_schema","has_video"):
        source="image_count" if c=="has_image" else c
        if c=="has_image" and source in out: out[c]=_nullable_positive(_num(out,source))
        elif source in out: out[c]=_bool(out,source)
    return out
def _binary_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    cited=_bool(df,"cited"); rows=[]
    for f in features:
        values=pd.to_numeric(df[f],errors="coerce"); valid=values.isin([0,1]); n0=int((values[valid]==0).sum()); n1=int((values[valid]==1).sum()); r0=float(cited[valid&(values==0)].mean()) if n0 else np.nan; r1=float(cited[valid&(values==1)].mean()) if n1 else np.nan
        rows.append({"feature":f,"n_available":int(valid.sum()),"n_0":n0,"n_1":n1,"cited_rate_0":r0,"cited_rate_1":r1,"diff_pp":(r1-r0)*100 if n0 and n1 else np.nan,"min_group_size":min(n0,n1),"sparse_warning":n0<20 or n1<20,"imbalance_warning":min(n0,n1)/max(n0,n1)<.1 if n0 and n1 else True,"recommended_use":"main_or_sensitivity" if n0>=20 and n1>=20 else "diagnostic_only_sparse"})
    return pd.DataFrame(rows).sort_values("diff_pp",kind="stable")
def _binned_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows=[]
    for f in features:
        table=citation_rate_by_category(df,f); table["diff_from_overall_pp"]=table["difference_from_overall"]*100; table["sparse_flag"]=table.n_rows.lt(20); table["recommended_use"]="main_candidate" if f in {"page_type_family_general","site_type_general"} else ("diagnostic_only" if f=="page_type_general" else "sensitivity_only")
        rows.append(table.assign(feature=f))
    return pd.concat(rows,ignore_index=True)
def _vif(frame: pd.DataFrame) -> pd.DataFrame:
    cols=[c for c in frame if frame[c].nunique(dropna=True)>1]
    data=frame[cols].dropna()
    rows=[]
    for c in cols:
        y=data[c].to_numpy(float); others=data.drop(columns=c).to_numpy(float)
        if len(others) == 0: vif=np.nan
        else:
            x=np.column_stack([np.ones(len(y)),others]); pred=x@np.linalg.lstsq(x,y,rcond=None)[0]; denom=((y-y.mean())**2).sum(); r2=1-((y-pred)**2).sum()/denom if denom else 0; vif=np.inf if r2>=.999999 else 1/(1-r2)
        rows.append({"feature":c,"n_complete":len(data),"vif":vif,"flag":"severe" if vif>=10 else ("warning" if vif>=5 else "")})
    return pd.DataFrame(rows)
def _forest(table: pd.DataFrame, path: Path) -> None:
    fig=go.Figure(go.Scatter(x=table.diff_pp,y=table.feature,mode="markers+text",text=[f"min n={int(n)}" for n in table.min_group_size],textposition="middle right",marker={"color":np.where(table.sparse_warning|table.imbalance_warning,"#94a3b8","#287a8e"),"symbol":np.where(table.sparse_warning|table.imbalance_warning,"x","circle"),"size":10},hovertemplate="Feature: %{y}<br>Difference: %{x:+.1f} pp<extra></extra>")); fig.add_vline(x=0,line_dash="dash",line_color="#5d6670"); apply_readable_plotly_layout(fig,"Binary feature cited-rate difference","unadjusted descriptive association"); fig.update_xaxes(title="Cited rate (1) minus cited rate (0), percentage points"); fig.update_yaxes(automargin=True); save_plotly_figure(fig,path/"interactive/binary_feature_diff_forest.html",path/"binary_feature_diff_forest.png")
    plt.figure(figsize=(10,max(4,len(table)*.4+1))); plt.scatter(table.diff_pp,range(len(table)),c=np.where(table.sparse_warning|table.imbalance_warning,"#94a3b8","#287a8e")); plt.axvline(0,color="#5d6670",linestyle="--"); plt.yticks(range(len(table)),table.feature); plt.xlabel("Cited-rate difference (pp)"); plt.tight_layout(); (path/"preview").mkdir(exist_ok=True); plt.savefig(path/"preview/binary_feature_diff_forest.png",dpi=180); plt.close()
def _binned_plots(table: pd.DataFrame, path: Path) -> None:
    for feature, group in table.groupby("feature", sort=False):
        ordered=group.sort_values("diff_from_overall_pp"); labels=ordered.category.astype(str).str.replace("_"," ")
        fig=go.Figure(go.Scatter(x=ordered.diff_from_overall_pp,y=labels,mode="markers+text",text=[f"n={int(n)}" for n in ordered.n_rows],textposition="middle right",marker={"size":10,"color":np.where(ordered.sparse_flag,"#94a3b8","#287a8e"),"symbol":np.where(ordered.sparse_flag,"x","circle")},hovertemplate="Category: %{y}<br>Difference: %{x:+.1f} pp<extra></extra>")); fig.add_vline(x=0,line_dash="dash",line_color="#5d6670"); apply_readable_plotly_layout(fig,f"Cited-rate difference by {feature.replace('_',' ')}","unadjusted descriptive association"); fig.update_xaxes(title="Difference from overall cited rate (percentage points)"); fig.update_yaxes(automargin=True)
        save_plotly_figure(fig,path/"interactive"/f"cited_rate_by_{feature}.html",path/f"cited_rate_by_{feature}.png")
        plt.figure(figsize=(10,max(4,len(ordered)*.42+1.2))); plt.scatter(ordered.diff_from_overall_pp,range(len(ordered)),c=np.where(ordered.sparse_flag,"#94a3b8","#287a8e")); plt.axvline(0,color="#5d6670",linestyle="--"); plt.yticks(range(len(ordered)),labels); plt.xlabel("Difference from overall cited rate (pp)"); plt.tight_layout(); (path/"preview").mkdir(exist_ok=True); plt.savefig(path/"preview"/f"cited_rate_by_{feature}.png",dpi=180); plt.close()
def _inventory(df: pd.DataFrame) -> pd.DataFrame:
    specs=[("table_count","has_table","binary_dummy",True,False,True,False,"main interpretable table-presence form"),("table_count","table_count_raw","raw_diagnostic",False,False,False,True,"raw count diagnostic only"),("heading_count","has_many_headings","threshold_dummy",False,False,True,False,"sensitivity threshold"),("heading_count","heading_count_group","categorical_bins",False,False,True,False,"sensitivity bins"),("heading_count","heading_count_raw","raw_diagnostic",False,False,False,True,"raw count diagnostic only"),("word_count","log1p_word_count","log_transform",False,True,True,False,"content-subset only"),("word_count","word_count_raw","raw_diagnostic",False,False,False,True,"raw count diagnostic only"),("page_type_family_general","page_type_family_general","categorical_bins",True,False,False,False,"main cross-domain page function"),("site_type_general","site_type_general","categorical_bins",True,False,True,False,"main secondary site type"),("page_type_general","page_type_general","categorical_bins",False,False,False,True,"detailed category may be sparse")]
    rows=[]
    for orig,trans,form,main,subset,sens,diag,reason in specs:
        s=df[trans] if trans in df else pd.Series(dtype=float); rows.append({"original_feature":orig,"transformed_feature":trans,"feature_form":form,"use_in_main_lpm":main,"use_in_content_subset_lpm":subset,"use_in_sensitivity_only":sens,"diagnostic_only":diag,"leakage_risk":"none","missing_rate":float(s.isna().mean()) if len(s) else 1.0,"unique_values":int(s.nunique(dropna=True)) if len(s) else 0,"reason":reason})
    for c in df.columns:
        if any(token in c.casefold() for token in FORBIDDEN): rows.append({"original_feature":c,"transformed_feature":c,"feature_form":"forbidden","use_in_main_lpm":False,"use_in_content_subset_lpm":False,"use_in_sensitivity_only":False,"diagnostic_only":False,"leakage_risk":"high","missing_rate":float(df[c].isna().mean()),"unique_values":int(df[c].nunique(dropna=True)),"reason":"outcome, answer-derived, provenance, or rank/position field"})
    return pd.DataFrame(rows)
def run_feature_form_v8(input_path: Path, eda_path: Path|None, table_dir: Path, figure_dir: Path) -> dict[str,Any]:
    base=pd.read_csv(input_path,low_memory=False); eda=pd.read_csv(eda_path,low_memory=False) if eda_path and eda_path.exists() else None; df=_transform(_attach(base,eda)); table_dir.mkdir(parents=True,exist_ok=True); figure_dir.mkdir(parents=True,exist_ok=True); (figure_dir/"interactive").mkdir(exist_ok=True)
    binary=[c for c in ["has_table","has_multiple_tables","has_headings","has_many_headings","has_links","has_many_links","has_substantial_text","has_faq","has_price_or_package","has_contact_info","has_bullets","has_image","usable_content","content_feature_available","taxonomy_confidence_high_or_medium","general_taxonomy_confidence_high_or_medium"] if c in df]
    bins=[c for c in ["heading_count_group","link_count_group","word_count_group","page_type_family_general","site_type_general","page_type_general"] if c in df]
    binary_table=_binary_summary(df,binary); binary_table.to_csv(table_dir/"binary_feature_cited_rate_summary.csv",index=False); binned=_binned_summary(df,bins); binned.to_csv(table_dir/"binned_feature_cited_rate_summary.csv",index=False); _forest(binary_table,figure_dir); _binned_plots(binned,figure_dir)
    candidates=[c for c in ["has_table","has_multiple_tables","has_headings","has_many_headings","has_links","has_many_links","has_substantial_text","has_faq","has_price_or_package","has_contact_info","has_bullets","log1p_word_count","log1p_text_char_count","usable_content","content_feature_available"] if c in df]
    numeric=df[candidates].apply(pd.to_numeric,errors="coerce"); corr=numeric.corr(); corr.to_csv(table_dir/"correlation_matrix_feature_form.csv"); vif=_vif(numeric); vif.to_csv(table_dir/"vif_feature_form_summary.csv",index=False)
    inventory=_inventory(df); inventory.to_csv(table_dir/"feature_form_inventory.csv",index=False)
    output_cols=["cited","prompt_id","normalized_url","source_url","source_root_domain","intent","page_type_family_general","page_type_general","site_type_general","page_type_general_confidence","page_type_general_reason","page_type_general_confidence_high","page_type_general_confidence_medium","page_type_general_confidence_low","page_type_general_confidence_unknown","page_type_general_confidence_high_or_medium",*binary,"heading_count_group","link_count_group","word_count_group","log1p_word_count","log1p_text_char_count","scraped_ok","parse_ok","scraped_body_available","content_feature_available","usable_content","numeric_content_features_available","word_count_missing","heading_count_missing","table_count_missing","link_count_missing",*[f"{x}_raw" for x in RAW]]
    output_cols=[c for c in dict.fromkeys(output_cols) if c in df]; df[output_cols].to_csv(table_dir/"scope_condo_lpm_ready_general_taxonomy_feature_form.csv",index=False)
    candidates_json={"outcome":["cited"],"fixed_effects":["prompt_id"],"main_categorical":["page_type_family_general","site_type_general"],"main_binary":[c for c in ["has_table","has_faq","has_price_or_package","has_contact_info","content_feature_available"] if c in df],"content_subset_numeric":["log1p_word_count","heading_count_group","link_count_group"],"sensitivity_only":["has_many_headings","has_many_links","has_multiple_tables"],"diagnostic_only":[c for c in output_cols if c.endswith("_raw")]+["page_type_general"],"forbidden":[c for c in base.columns if any(t in c.casefold() for t in FORBIDDEN)]}
    (table_dir/"lpm_main_candidate_columns.json").write_text(json.dumps(candidates_json,indent=2),encoding="utf-8")
    validation={"has_table_created_correctly":bool((df.has_table.dropna()==(df.table_count_raw.dropna()>0).astype(int)).all()),"no_raw_count_in_main_candidates":not any(x.endswith("_raw") for x in candidates_json["main_binary"]+candidates_json["main_categorical"]),"missing_scraped_content_not_zero_imputed":bool(df.table_count_raw.isna().eq(df.has_table.isna()).all()),"binary_features_valid":all(set(pd.to_numeric(df[c],errors="coerce").dropna().unique()).issubset({0,1}) for c in binary),"sparse_binary_features_flagged":bool((~binary_table.sparse_warning | binary_table.recommended_use.eq("diagnostic_only_sparse")).all()),"high_collinearity_flagged":bool((corr.abs().where(~np.eye(len(corr),dtype=bool)).ge(.8).any().any()) or vif["flag"].ne("").any()),"forbidden_variables_excluded":not any(c in output_cols for c in candidates_json["forbidden"]),"output_lpm_ready_v2_exists":(table_dir/"scope_condo_lpm_ready_general_taxonomy_feature_form.csv").exists()}
    validation["status"]="pass" if all(validation.values()) else "warning"; (table_dir/"feature_form_validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    report="# Feature-Form Readiness Report\n\nRaw counts are not automatically wrong, but non-linear/zero-inflated counts are more interpretable as dummies, bins, or logs unless EDA supports a linear form. `table_count` is represented primarily by `has_table`.\n\n## Main LPM candidates\n`page_type_family_general`, `site_type_general`, `has_table`, available binary content flags, `content_feature_available`, and prompt fixed effects.\n\n## Content-subset only\n`log1p_word_count`, heading/link bins, restricted to content availability. Do not include both word and character logs without the correlation/VIF precheck.\n\n## Diagnostic and forbidden\nRaw count fields are diagnostic only. Answer-derived, outcome-derived, provenance, and rank/position fields are forbidden.\n\n## Suggested formulas\nMain v1: `cited ~ C(page_type_family_general) + C(site_type_general) + has_table + has_faq + has_price_or_package + has_contact_info + content_feature_available + C(prompt_id)`.\n\nContent subset: add `log1p_word_count + C(heading_count_group) + C(link_count_group)` after restricting to content availability.\n\n**Status:** ready_for_LPM_v1_after_feature_form_layer\n"; (table_dir/"feature_form_readiness_report.md").write_text(report,encoding="utf-8")
    return {"rows":len(df),"table_dir":str(table_dir),"figure_dir":str(figure_dir),"binary_features":len(binary),"status":validation["status"]}
