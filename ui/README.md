# CiteScope frontend

The Streamlit entry point is `app.py`. Existing audit and content-econometrics
views remain unchanged. The separate position model is implemented in
`ui/views/position_model_new.py` and reads only versioned artifacts from
`outputs/position_model_v1/`.

Run the position pipeline before opening that view:

```bash
.venv/bin/python scripts/v2_run_position_model.py
.venv/bin/streamlit run app.py --server.port 8504
```

The frontend's EDA filters never refit a regression. Model panels always display
the frozen estimation sample size and covariance method from the result files.

