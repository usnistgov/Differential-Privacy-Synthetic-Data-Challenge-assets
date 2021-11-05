RAW_FILE ?= IL_OH_10Y_PUMS.csv
OUTPUT_DIR ?= data/sprint2

${OUTPUT_DIR}/geo/ipums_puma_2010.zip:
	@mkdir -p $(dir $@)
	wget -O $@ https://usa.ipums.org/usa/resources/volii/shapefiles/ipums_puma_2010.zip

${OUTPUT_DIR}/geo/ipums_puma_2010.shp: ${OUTPUT_DIR}/geo/ipums_puma_2010.zip
	unzip -d $(dir $@) $<

${OUTPUT_DIR}/processed/public/clean.csv: ${OUTPUT_DIR}/raw/${RAW_FILE}
	@mkdir -p $(dir $@)
	python deid2/sprint2/clean_data.py $< --output-file $@

${OUTPUT_DIR}/processed/public/clean_with_individuals.csv: ${OUTPUT_DIR}/processed/public/clean.csv
	@mkdir -p $(dir $@)
	python deid2/sprint2/sim_individuals.py $< $@

${OUTPUT_DIR}/final/public/ground_truth.csv: ${OUTPUT_DIR}/processed/public/clean_with_individuals.csv
	@mkdir -p $(dir $@)
	python deid2/sprint2/finalize_data.py $< $@

${OUTPUT_DIR}/final/public/submission_format.csv: ${OUTPUT_DIR}/final/public/parameters.json
	@mkdir -p $(dir $@)
	python deid2/sprint2/baseline.py $< > $@

${OUTPUT_DIR}/final/public/parameters.json ${OUTPUT_DIR}/final/public/read_csv_kwargs.json: ${OUTPUT_DIR}/final/public/ground_truth.csv
	python deid2/sprint2/create_parameters_and_csv_kwargs.py $<
	@echo "testing that newly created parameters file is accurate..."
	pytest
	@echo "done"

.PHONY: data
data: ${OUTPUT_DIR}/final/public/submission_format.csv ${OUTPUT_DIR}/final/public/ground_truth.csv
