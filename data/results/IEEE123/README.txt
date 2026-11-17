123-node supporting results

result_summary.json and results.csv contain the delivery outcome and resource accounting. delivered_plan.json records the complete 30-minute action sequence. selected_samples.csv and selected_sample_links.json retain the 91 selected sample summaries and identifiers; handoff_selection.json records candidate selection. frozen_task_settings.json and model_config.json describe the task and model.

Run python check_saved.py from this directory to check these processed results: the action sequence, energy calculation, peak current, 91 sample labels, 349 logical requests, 225 physical requests and 675 numerical entries. This is an internal consistency check of saved results; full raw simulator outputs are not included in this compact companion.

The feeder123_handoff entry in code/snapshots/index.json maps each original implementation file to its retained source copy under code/snapshots/. The earlier 123-node implementation and its no-gain outcome remain available for interpreting the candidate-handoff extension. Original run identifiers and source paths in JSON are provenance identifiers from the study workspace.
